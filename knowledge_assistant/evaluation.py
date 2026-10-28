import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from common import load_json, save_json, hardware
from knowledge_assistant.retrieval import Retriever
from knowledge_assistant.generation import RAG, LocalGenerator


def evaluate(output="results/knowledge"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    questions = load_json("experiments/retrieval_questions.json")
    retriever = Retriever()
    generator = LocalGenerator()
    rag = RAG(retriever, generator)
    # Warm models, exclude startup from warm latency measurements.
    retriever.search("production throughput")
    generator.draft(
        "What is the scheduling objective?", retriever.search("scheduling objective")
    )
    rows = []
    answers = []
    for item in questions:
        for method in ["bm25", "semantic", "hybrid"]:
            begin = time.perf_counter()
            hits = retriever.search(item["question"], method, k=5)
            latency = 1000 * (time.perf_counter() - begin)
            relevant = set(item["documents"])
            found = []
            ranks = []
            for rank, hit in enumerate(hits, 1):
                if hit["document_id"] in relevant:
                    ranks.append(rank)
                    found.append(hit["document_id"])
            # Relevance is document-level; distinct relevant sources are counted once.
            row = {
                "id": item["id"],
                "question": item["question"],
                "method": method,
                "answerable": bool(relevant),
                "latency_ms": latency,
                "mrr_at5": 1 / min(ranks) if ranks else 0,
            }
            for k in [1, 3, 5]:
                observed = {h["document_id"] for h in hits[:k]}
                row[f"recall_at{k}"] = (
                    len(observed & relevant) / len(relevant) if relevant else None
                )
                row[f"hit_at{k}"] = int(bool(observed & relevant)) if relevant else None
            rows.append(row)
        result = rag.answer(item["question"])
        expected = set(item["documents"])
        correct = all(
            c["quote"].lower()
            in next(
                p["text"] for p in retriever.chunks if p["id"] == c["chunk_id"]
            ).lower()
            and c["document_id"] in expected
            for c in result["citations"]
        )
        text = result["answer"].lower()
        keyword_match = (
            any(
                all(word.lower() in text for word in alternative)
                for alternative in item["keywords"]
            )
            if expected
            else result["abstained"]
        )
        answers.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_documents": item["documents"],
                "expected_keywords": item["keywords"],
                "keyword_answer_correct": (
                    bool(keyword_match and not result["abstained"])
                    if expected
                    else bool(result["abstained"])
                ),
                "citation_source_and_span_correct": (
                    bool(correct) if result["citations"] else None
                ),
                **result,
            }
        )
        print(item["id"], result["answer_mode"], result["answer"][:110], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "retrieval.csv", index=False)
    save_json(out / "answers.json", answers)
    summaries = {}
    for method, g in df[df.answerable].groupby("method"):
        summaries[method] = {
            key: float(g[key].mean())
            for key in [
                "recall_at1",
                "recall_at3",
                "recall_at5",
                "hit_at3",
                "mrr_at5",
                "latency_ms",
            ]
        }
    answered = [a for a in answers if a["citations"]]
    known = [a for a in answers if a["expected_documents"]]
    unknown = [a for a in answers if not a["expected_documents"]]
    abstention_precision, abstention_recall, abstention_f1, _ = (
        precision_recall_fscore_support(
            [not bool(a["expected_documents"]) for a in answers],
            [a["abstained"] for a in answers],
            average="binary",
            zero_division=0,
        )
    )
    stats = {
        "questions": len(questions),
        "answerable_questions": len(known),
        "unanswerable_questions": len(unknown),
        "retrieval": summaries,
        "generation": {
            "abstention_detection_precision": float(abstention_precision),
            "abstention_detection_recall": float(abstention_recall),
            "abstention_detection_f1": float(abstention_f1),
            "keyword_answer_accuracy": float(
                np.mean([a["keyword_answer_correct"] for a in answers])
            ),
            "answerable_keyword_accuracy": float(
                np.mean([a["keyword_answer_correct"] for a in known])
            ),
            "unanswerable_abstention_rate": float(
                np.mean([a["abstained"] for a in unknown])
            ),
            "answerable_coverage": float(np.mean([not a["abstained"] for a in known])),
            "citation_source_and_span_accuracy": (
                float(
                    np.mean([a["citation_source_and_span_correct"] for a in answered])
                )
                if answered
                else None
            ),
            "validated_llm_span_count": sum(
                a["answer_mode"] == "validated_llm_exact_span" for a in answers
            ),
            "extractive_fallback_count": sum(
                a["answer_mode"] == "evidence_extract_after_rejected_draft"
                for a in answers
            ),
            "median_warm_latency_ms": float(
                np.median([a["latency_ms"] for a in answers])
            ),
        },
        "methodology": "Authored source-reviewed document-level relevance and keyword answer checks, not independent human adjudication or semantic entailment scoring. Exact-span citations must also match a gold relevant document. No held-out industrial corpus.",
        "hardware": hardware(),
    }
    save_json(out / "summary.json", stats)
    print(stats)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd.DataFrame(summaries).T[["recall_at1", "recall_at3", "mrr_at5"]].plot.bar(
        rot=0, ylim=(0, 1.1), figsize=(8, 4)
    )
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out / "retrieval.png", dpi=150)
    plt.close()
    return stats


if __name__ == "__main__":
    evaluate()
