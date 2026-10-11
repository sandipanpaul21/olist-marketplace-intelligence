"""
Policy RAG (Part 3): chunk -> index -> retrieve -> answer -> verify.

FLOW (read this first)
  1. LOAD + CHUNK : policy .txt --load_policy--> lines --chunk_policy--> one chunk per clause
  2. INDEX        : chunks --PolicyIndex--> embeddings (meaning search) + BM25 (keyword search)
  3. RETRIEVE     : question --search_policy--> top-k clauses
  4. GENERATE     : question + clauses --answer_question--> JSON {answerable, answer, citations}
                    Two "not found" layers: (a) weak retrieval score, (b) the LLM says not answerable
  5. VERIFY       : answer --check_faithfulness--> citation check + number check + LLM judge
  6. EVALUATE     : evaluate_retrieval (recall@k), run_answers + summarize_answers (abstention)

Part 5 (agent) will call search_policy() at the bottom of this file as its tool.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Load the API key from a .env file in the project root, if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Settings (one place to change them)
# ----------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"          # small, runs locally on CPU
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge: used on queries only
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "claude-sonnet-5-5")
RRF_K = 60                                      # constant in reciprocal rank fusion
MIN_DENSE_SCORE = 0.35                          # abstention layer 1; tuned in notebook 05
DEFAULT_POLICY_PATH = (Path(__file__).resolve().parents[1]
                       / "notebooks" / "RAG" / "policy" / "marketplace_policy.txt")

# ----------------------------------------------------------------------------
# 1. LOAD + CHUNK  (one chunk per clause)
# ----------------------------------------------------------------------------
SECTION_RE = re.compile(r"^MOP-(\d+)\.\s+(.+)$")     # "MOP-1. Seller Performance Tiers"
CLAUSE_RE = re.compile(r"^(\d+)\.(\d+)\s+(.*)$")     # "1.1 Every active seller ..."
MOP_REF_RE = re.compile(r"MOP-(\d+)(?:\.(\d+))?")    # "MOP-2.2" or "MOP-7"
SECTION_WORD_RE = re.compile(r"Section (\d+)")       # "Section 6"
BARE_REF_RE = re.compile(r"\b(\d)\.(\d)\b")          # "under 2.2"


def load_policy(path):
    """Read the policy file as a list of non-empty lines."""
    text = Path(path).read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def find_refs(text, own_id):
    """Return clause/section IDs mentioned inside a clause, e.g. ['MOP-2.2', 'MOP-7']."""
    refs = set()
    for sec, cl in MOP_REF_RE.findall(text):
        refs.add(f"MOP-{sec}.{cl}" if cl else f"MOP-{sec}")
    for sec in SECTION_WORD_RE.findall(text):
        refs.add(f"MOP-{sec}")
    # Remove "MOP-x.y" first, so "MOP-2.2" is not counted again as a bare "2.2"
    stripped = MOP_REF_RE.sub(" ", text)
    for sec, cl in BARE_REF_RE.findall(stripped):
        refs.add(f"MOP-{sec}.{cl}")
    refs.discard(own_id)
    return sorted(refs)


def chunk_policy(lines):
    """Split the policy into one chunk per clause.

    A new chunk starts at every clause number. Table rows, bullets and numbered
    items are not clause starts, so they stay inside the clause above them.
    Lines before the first section header (title and note) are ignored.
    """
    chunks, section_no, section_title = [], None, None
    for ln in lines:
        m_sec = SECTION_RE.match(ln)
        if m_sec:                                   # section header: remember it, not a chunk
            section_no, section_title = int(m_sec.group(1)), m_sec.group(2)
            continue
        if section_no is None:                      # title/note at the top of the file
            continue
        m_cl = CLAUSE_RE.match(ln)
        if m_cl:                                    # clause start: open a new chunk
            chunks.append({
                "chunk_id": f"MOP-{m_cl.group(1)}.{m_cl.group(2)}",   # the citation ID
                "section_no": section_no,
                "section_title": section_title,
                "clause_section": int(m_cl.group(1)),
                "clause_no": int(m_cl.group(2)),
                "lines": [ln],
            })
        else:                                       # table row / bullet: add to previous clause
            chunks[-1]["lines"].append(ln)
    for c in chunks:
        c["text"] = "\n".join(c.pop("lines"))
        c["refs"] = find_refs(c["text"], c["chunk_id"])
    return chunks


# ----------------------------------------------------------------------------
# 2. INDEX  (meaning search with embeddings + keyword search with BM25)
# ----------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def embed_text(chunk, with_title=True):
    """The text that is embedded. The section title adds topic context to short clauses."""
    return f"{chunk['section_title']}. {chunk['text']}" if with_title else chunk["text"]


class PolicyIndex:
    """Holds the clause embeddings and the BM25 index. Built once, then searched many times."""

    def __init__(self, chunks, embedder=None, with_title=True):
        from rank_bm25 import BM25Okapi
        if embedder is None:                        # default: local sentence-transformers model
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(EMBED_MODEL)
        self.embedder = embedder
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        texts = [embed_text(c, with_title) for c in chunks]
        # normalize_embeddings=True -> dot product equals cosine similarity
        self.emb = np.asarray(embedder.encode(texts, normalize_embeddings=True))
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def dense_scores(self, query):
        """Cosine similarity between the question and every clause."""
        q = np.asarray(self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True))[0]
        return self.emb @ q

    def bm25_scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)))

    def search(self, query, k=3, mode="hybrid"):
        """Return the top-k clauses. mode: 'dense', 'bm25' or 'hybrid'."""
        dense, sparse = self.dense_scores(query), self.bm25_scores(query)
        if mode == "dense":
            order = np.argsort(-dense)
        elif mode == "bm25":
            order = np.argsort(-sparse)
        else:
            # Hybrid = reciprocal rank fusion: a clause ranked high by either search gets a high score.
            n = len(dense)
            rank_d, rank_s = np.empty(n, int), np.empty(n, int)
            rank_d[np.argsort(-dense)] = np.arange(n)
            rank_s[np.argsort(-sparse)] = np.arange(n)
            order = np.argsort(-(1 / (RRF_K + rank_d + 1) + 1 / (RRF_K + rank_s + 1)))
        return [{
            "rank": r + 1,
            "chunk_id": self.chunks[i]["chunk_id"],
            "section_title": self.chunks[i]["section_title"],
            "text": self.chunks[i]["text"],
            "dense_score": float(dense[i]),
            "bm25_score": float(sparse[i]),
        } for r, i in enumerate(order[:k])]


# ----------------------------------------------------------------------------
# 3. LLM helper
# ----------------------------------------------------------------------------
def call_llm(system, user, max_tokens=600):
    """One LLM call. Reads ANTHROPIC_API_KEY from the environment (or .env).

    No temperature is passed: the current anthropic SDK does not accept it in messages.create().
    Answers are saved to files by the notebooks, so a result can be reproduced from the saved file.
    """
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(model=LLM_MODEL, max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if b.type == "text")


def parse_json(text):
    """Take the first {...} block from the LLM reply. Return None if it is not valid JSON."""
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# 4. GENERATE  (cited answer, or "not found")
# ----------------------------------------------------------------------------
ANSWER_SYSTEM = """You answer questions about a marketplace operations policy.
Rules:
1. Use ONLY the policy clauses given in the user message. Never use outside knowledge.
2. If the clauses do not contain the answer, set "answerable" to false. Do not guess.
   If a clause says the topic is handled by another document, say that.
3. Cite every clause you used by its ID, for example MOP-2.2. Cite only IDs that were provided.
4. If the question needs a calculation, show the short working using numbers from the clauses.
Reply with JSON only, no other text, in exactly this form:
{"answerable": true or false, "answer": "...", "citations": ["MOP-1.2"]}
If answerable is false, "answer" must say briefly what the policy does not cover."""


def build_context(hits):
    return "\n\n".join(f"[{h['chunk_id']}]\n{h['text']}" for h in hits)


def answer_question(index, question, k=3, min_dense=MIN_DENSE_SCORE, mode="hybrid", llm=None):
    """Retrieve clauses, ask the LLM, and return the answer with its checks.

    abstained_by tells which layer said "not found":
      retrieval_score    - no clause was similar enough, the LLM was not called
      llm                - the LLM said the clauses do not answer the question
      no_valid_citation  - the LLM answered but cited no clause that was retrieved
      parse_error        - the LLM reply was not valid JSON
    """
    llm = llm or call_llm
    hits = index.search(question, k=k, mode=mode)
    top_dense = float(index.dense_scores(question).max())
    out = {"question": question, "retrieved_ids": [h["chunk_id"] for h in hits],
           "top_dense": top_dense, "abstained_by": None, "answerable": False,
           "answer": "", "citations": [], "invalid_citations": []}

    # Abstention layer 1: nothing in the policy looks like the question
    if top_dense < min_dense:
        out.update(abstained_by="retrieval_score",
                   answer="Not found in policy: no clause is similar enough to the question.")
        return out

    raw = llm(ANSWER_SYSTEM, f"Policy clauses:\n\n{build_context(hits)}\n\nQuestion: {question}")
    parsed = parse_json(raw)
    if parsed is None:
        out.update(abstained_by="parse_error", answer=raw[:300])
        return out

    cited = [str(c) for c in parsed.get("citations", [])]
    out["citations"] = [c for c in cited if c in out["retrieved_ids"]]
    out["invalid_citations"] = [c for c in cited if c not in out["retrieved_ids"]]
    out["answer"] = str(parsed.get("answer", ""))

    # Abstention layer 2: the LLM itself says the policy does not answer it
    if not parsed.get("answerable"):
        out["abstained_by"] = "llm"
    # An answer with no valid citation is not allowed through
    elif not out["citations"]:
        out["abstained_by"] = "no_valid_citation"
    else:
        out["answerable"] = True
    return out


# ----------------------------------------------------------------------------
# 5. VERIFY  (is the answer supported by the cited clauses?)
# ----------------------------------------------------------------------------
JUDGE_SYSTEM = """You check whether an answer is supported by policy clauses.
Verdicts:
SUPPORTED   - every claim is in the clauses, or is a simple calculation from them
PARTIAL     - some claims are not in the clauses
UNSUPPORTED - the answer contradicts the clauses or goes beyond them
Reply with JSON only: {"verdict": "SUPPORTED" or "PARTIAL" or "UNSUPPORTED", "reason": "one short sentence"}"""


def numbers_in(text):
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def check_faithfulness(index, result, llm=None):
    """Three checks on an answered question. Returns None if the question was not answered."""
    if not result["answerable"]:
        return None
    llm = llm or call_llm
    cited_text = "\n".join(index.by_id[c]["text"] for c in result["citations"])
    # Numbers in the answer should come from the cited clauses or from the question.
    allowed = numbers_in(cited_text) | numbers_in(result["question"])
    judge = parse_json(llm(JUDGE_SYSTEM, f"Clauses:\n{cited_text}\n\nAnswer:\n{result['answer']}")) or {}
    return {
        "citations_valid": not result["invalid_citations"],            # check 1: IDs were real
        "ungrounded_numbers": sorted(numbers_in(result["answer"]) - allowed),  # check 2: numbers
        "verdict": judge.get("verdict", "ERROR"),                      # check 3: LLM judge
        "reason": judge.get("reason", ""),
    }


# ----------------------------------------------------------------------------
# 6. EVALUATE
# ----------------------------------------------------------------------------
def parse_gold(s):
    """'MOP-1.2;MOP-1.3' -> ['MOP-1.2', 'MOP-1.3']"""
    return [x.strip() for x in str(s).split(";") if x.strip() and str(s) != "nan"]


def evaluate_retrieval(index, eval_df, ks=(1, 3, 5), mode="hybrid"):
    """Recall@k on the answerable questions.

    any@k : at least one correct clause is in the top-k
    all@k : every correct clause is in the top-k (matters for multi-clause questions)
    """
    rows = []
    for r in eval_df[eval_df["answerable"] == 1].itertuples():
        gold = set(parse_gold(r.gold_ids))
        ranked = [h["chunk_id"] for h in index.search(r.question, k=max(ks), mode=mode)]
        row = {"qid": r.qid}
        for k in ks:
            top = set(ranked[:k])
            row[f"any@{k}"], row[f"all@{k}"] = bool(gold & top), gold <= top
        row["first_gold_rank"] = next((i + 1 for i, c in enumerate(ranked) if c in gold), None)
        row["retrieved"] = ranked
        rows.append(row)
    per_q = pd.DataFrame(rows)
    summary = per_q[[c for c in per_q.columns if "@" in c]].mean()
    summary["MRR"] = (1 / per_q["first_gold_rank"]).fillna(0).mean()
    return per_q, summary


def run_answers(index, eval_df, **kwargs):
    """Answer every question in the eval set. Returns a list of result dicts."""
    results = []
    for r in eval_df.itertuples():
        res = answer_question(index, r.question, **kwargs)
        res.update(qid=r.qid, gold_answerable=int(r.answerable), gold_ids=parse_gold(r.gold_ids))
        results.append(res)
    return results


def summarize_answers(results):
    """Abstention and citation numbers for the final report."""
    df = pd.DataFrame(results)
    un, ans = df[df["gold_answerable"] == 0], df[df["gold_answerable"] == 1]
    cited_gold = ans.apply(lambda r: bool(set(r["citations"]) & set(r["gold_ids"])), axis=1)
    return pd.Series({
        "unanswerable abstained (correct)": f"{int((~un['answerable']).sum())} of {len(un)}",
        "answerable wrongly abstained": f"{int((~ans['answerable']).sum())} of {len(ans)}",
        "answerable cited a correct clause": f"{int(cited_gold.sum())} of {len(ans)}",
    })


# ----------------------------------------------------------------------------
# 7. TOOL FOR PART 5 (agent)
# ----------------------------------------------------------------------------
_INDEX = None


def get_index(policy_path=DEFAULT_POLICY_PATH):
    """Build the index once and reuse it."""
    global _INDEX
    if _INDEX is None:
        _INDEX = PolicyIndex(chunk_policy(load_policy(policy_path)))
    return _INDEX


def search_policy(query, k=3):
    """The tool the agent calls. Policy answers must come only from this."""
    return get_index().search(query, k=k)
