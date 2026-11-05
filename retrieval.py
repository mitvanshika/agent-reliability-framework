"""
FAQ retrieval for the demo agent, switchable from experiment settings
(settings.retrieval.method). Three methods:

  substring   the original behaviour, unchanged: store.search_faq(), i.e. the
              first FAQ whose topic appears literally in the query
              ("po box" must be typed as "po box").
  bm25        keyword ranking (written here, no dependency): stopwords removed,
              BM25 with the always-positive IDF used by Lucene. Finds FAQs
              that share words with the query, in any order.
  embeddings  semantic similarity with fastembed + BAAI/bge-small-en-v1.5 (a
              small ONNX model, no torch). Finds paraphrases with no shared
              words ("mailbox address" -> PO box). Optional dependency:
              pip install -r requirements-rag.txt

search() returns a list of {"topic", "text", "score"}, best first, at most
top_k, and NOTHING below min_score, so the agent can honestly say it doesn't
know or escalate instead of being handed an unrelated FAQ.

Deterministic: same query + same FAQs -> same output (ties broken by FAQ
order). Embedding vectors for the FAQs are computed once and cached on disk
under .cache/faq_vectors/, keyed by model + a hash of the FAQ content, so
editing an FAQ recomputes them automatically.
"""

import hashlib
import json
import math
import os
import re

import store

METHODS = ("substring", "bm25", "embeddings")
# Scores are on different scales per method; defaults chosen from measurements
# on the 5 FAQs (bm25: real matches 1.06-2.37; cosine: real 0.66-0.81, unrelated 0.48).
DEFAULT_MIN_SCORE = {"substring": None, "bm25": 0.5, "embeddings": 0.6}
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("RAG_CACHE_DIR") or os.path.join(REPO_DIR, ".cache")

STOPWORDS = set("""a an and are as at be by can could do does for from had has have how i if in is it its me my
of on or our so than that the their them then there these they this to us was we what when where which who why
will with would you your""".split())
BM25_K1, BM25_B = 1.5, 0.75


class RetrievalUnavailable(RuntimeError):
    """The requested method needs an optional dependency that is not installed."""


def load_faqs():
    """[(topic, answer), ...] in database order (same order store.search_faq scans)."""
    conn = store.get_connection()
    rows = conn.execute("SELECT topic, answer FROM faqs ORDER BY rowid").fetchall()
    conn.close()
    return [(r["topic"], r["answer"]) for r in rows]


def faq_hash(faqs) -> str:
    return hashlib.sha256(json.dumps(faqs, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- bm25

def tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def bm25_scores(query: str, faqs):
    docs = [tokenize(f"{topic} {answer}") for topic, answer in faqs]
    n = len(docs)
    if n == 0:
        return []
    avg_len = sum(map(len, docs)) / n
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(1 + (n - k + 0.5) / (k + 0.5)) for w, k in df.items()}   # always > 0
    q = tokenize(query)
    scores = []
    for d in docs:
        s = 0.0
        for w in q:
            if w in idf:
                tf = d.count(w)
                s += idf[w] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * len(d) / avg_len))
        scores.append(s)
    return scores


# ---------------------------------------------------------------- embeddings

_model = None
_vectors = {}   # (model, faq hash) -> normalised FAQ vectors (numpy array)


def embeddings_available() -> bool:
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RetrievalUnavailable(
                "retrieval.method 'embeddings' needs fastembed: pip install -r requirements-rag.txt") from e
        _model = TextEmbedding(EMBED_MODEL, cache_dir=os.path.join(CACHE_DIR, "fastembed_models"))
    return _model


def _embed(model, texts):
    import numpy as np
    return np.array(list(model.embed(texts)), dtype="float32")


def _normalise(m):
    import numpy as np
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def vector_cache_path(faqs) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", EMBED_MODEL)
    return os.path.join(CACHE_DIR, "faq_vectors", f"{safe}_{faq_hash(faqs)}.npy")


def faq_vectors(faqs):
    """FAQ vectors: memory -> disk cache -> compute once (and save)."""
    import numpy as np
    key = (EMBED_MODEL, faq_hash(faqs))
    if key in _vectors:
        return _vectors[key]
    path = vector_cache_path(faqs)
    if os.path.isfile(path):
        vecs = np.load(path)
    else:
        vecs = _normalise(_embed(_get_model(), [f"{t}. {a}" for t, a in faqs]))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npy"
        np.save(tmp, vecs)
        os.replace(tmp, path)
    _vectors[key] = vecs
    return vecs


def embedding_scores(query: str, faqs):
    import numpy as np
    docs = faq_vectors(faqs)
    q = _normalise(np.array(list(_get_model().query_embed(query)), dtype="float32"))[0]
    return [float(x) for x in docs @ q]


def warm_up():
    """Load the model and FAQ vectors now (so the first timed question doesn't pay for it)."""
    faq_vectors(load_faqs())
    _get_model()


# ---------------------------------------------------------------- search

def search(query: str, method: str = "substring", top_k: int = 1, min_score: float | None = None, faqs=None):
    """[{"topic", "text", "score"}], best first, at most top_k, none below min_score."""
    if method not in METHODS:
        raise ValueError(f"unknown retrieval method '{method}' (choose from {', '.join(METHODS)})")
    if method == "substring":
        # exactly store.search_faq(): first topic contained in the query, or nothing
        if faqs is None:
            answer = store.search_faq(query)
            if answer is None:
                return []
            topic = next((t for t, a in load_faqs() if a == answer and t in query.lower()), None)
            return [{"topic": topic, "text": answer, "score": 1.0}]
        for topic, answer in faqs:
            if topic in query.lower():
                return [{"topic": topic, "text": answer, "score": 1.0}]
        return []

    faqs = load_faqs() if faqs is None else list(faqs)
    scores = bm25_scores(query, faqs) if method == "bm25" else embedding_scores(query, faqs)
    threshold = DEFAULT_MIN_SCORE[method] if min_score is None else min_score
    ranked = sorted(range(len(faqs)), key=lambda i: (-scores[i], i))   # ties: FAQ order
    hits = [{"topic": faqs[i][0], "text": faqs[i][1], "score": round(scores[i], 4)}
            for i in ranked if scores[i] > 0 and scores[i] >= threshold]
    return hits[:top_k]
