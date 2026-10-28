import argparse
import re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from common import load_json, save_json

STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "and",
    "in",
    "for",
    "what",
    "how",
    "which",
    "why",
    "can",
    "does",
    "do",
    "on",
    "with",
    "be",
    "this",
    "that",
    "it",
    "we",
    "our",
    "by",
    "as",
    "should",
    "was",
    "were",
    "from",
    "at",
    "when",
}


def tokenize(text):
    return [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOP]


class Embedder:
    def __init__(self, path="models/minilm"):
        if not Path(path).is_dir():
            raise FileNotFoundError(
                "Run python -m scripts.download_assets --only models first"
            )
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).eval()

    def encode(self, texts):
        torch = self.torch
        results = []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = self.tokenizer(
                    texts[start : start + 16],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                tokens = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1)
                results.append(
                    torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()
                )
        return np.concatenate(results).astype(np.float32)


class Retriever:
    def __init__(self, index="data/index", semantic=True, embedder=None):
        self.index = Path(index)
        self.chunks = load_json(self.index / "chunks.json")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embedder = None
        self.vectors = None
        if semantic:
            self.embedder = embedder or Embedder()
            self.vectors = np.load(self.index / "embeddings.npy", allow_pickle=False)
            manifest = load_json(self.index / "manifest.json")
            if manifest["chunk_ids"] != [c["id"] for c in self.chunks] or len(
                self.vectors
            ) != len(self.chunks):
                raise ValueError("Index/chunk mismatch: rebuild index")

    def search(self, query, method="hybrid", k=3):
        if method not in {"bm25", "semantic", "hybrid"} or k < 1 or not query.strip():
            raise ValueError("Invalid search input")
        lexical = np.asarray(self.bm25.get_scores(tokenize(query)))
        semantic = None
        if method != "bm25":
            if self.embedder is None:
                raise ValueError("Semantic embeddings unavailable")
            semantic = self.vectors @ self.embedder.encode([query])[0]
        if method == "bm25":
            scores = lexical
        elif method == "semantic":
            scores = semantic
        else:
            # Reciprocal-rank fusion is scale independent. Stable sort breaks ties by corpus order.
            scores = np.zeros(len(self.chunks))
            for values in [lexical, semantic]:
                rank = np.argsort(-values, kind="stable")
                for pos, idx in enumerate(rank):
                    scores[idx] += 1 / (60 + pos + 1)
        order = np.argsort(-scores, kind="stable")[: min(k, len(self.chunks))]
        return [
            {
                **self.chunks[i],
                "score": float(scores[i]),
                "bm25_score": float(lexical[i]),
                "semantic_score": float(semantic[i]) if semantic is not None else None,
            }
            for i in order
        ]


def build_index(index="data/index"):
    folder = Path(index)
    chunks = load_json(folder / "chunks.json")
    embedder = Embedder()
    vectors = embedder.encode([c["text"] for c in chunks])
    np.save(folder / "embeddings.npy", vectors)
    save_json(
        folder / "manifest.json",
        {
            "chunk_ids": [c["id"] for c in chunks],
            "model": load_json("configs/models.json")["embedding"],
            "pooling": "attention-mask mean; L2 normalized",
            "max_tokens": 256,
            "dimension": vectors.shape[1],
        },
    )
    print(f"Saved {vectors.shape} embedding index")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--query")
    a = p.parse_args()
    if a.build:
        build_index()
    elif a.query:
        for c in Retriever().search(a.query):
            print(c)
