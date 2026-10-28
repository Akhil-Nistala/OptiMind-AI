"""Local LLM draft + exact-span validation; unverified paraphrases are not asserted."""

import argparse
import re
import time
from pathlib import Path
from knowledge_assistant.retrieval import Retriever, tokenize


def sufficient(question, passages, min_similarity=0.30):
    terms = set(tokenize(question))
    identifiers = re.findall(r"\b[A-Za-z]{1,8}[-_]\d+\b", question)
    if any(
        not any(entity.lower() in p["text"].lower() for p in passages)
        for entity in identifiers
    ):
        return False
    if re.search(r"yesterday|live sensor|current reading|our exact", question, re.I):
        return False
    if not terms:
        return False
    for p in passages:
        overlap = len(terms & set(tokenize(p["text"])))
        if overlap >= 2 and (
            (
                p.get("semantic_score") is not None
                and p["semantic_score"] >= min_similarity
            )
            or (
                p.get("semantic_score") is None
                and overlap >= 3
                and p.get("bm25_score", 0) > 0
            )
        ):
            return True
    return False


class LocalGenerator:
    def __init__(self, path="models/flan-t5-small"):
        if not Path(path).is_dir():
            raise FileNotFoundError("Local language model missing; run assets download")
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).eval()

    def draft(self, question, passages):
        # Question first; selected context is bounded to the model's 512-token input.
        context = " ".join(p["text"] for p in passages[:2])
        prompt = f"Answer the question using only the context. Copy a short exact phrase from context. If unknown answer INSUFFICIENT. Question: {question} Context: {context}"
        batch = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )
        with self.torch.inference_mode():
            output = self.model.generate(
                **batch, max_new_tokens=80, do_sample=False, num_beams=1
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


class RAG:
    def __init__(self, retriever=None, generator=None):
        from common import load_json

        config = (
            load_json("configs/experiments.json")
            if Path("configs/experiments.json").exists()
            else {}
        )
        self.min_similarity = config.get("evidence_min_similarity", 0.30)
        self.retriever = retriever or Retriever()
        self.generator = generator or LocalGenerator()

    def answer(self, question, method="hybrid"):
        began = time.perf_counter()
        if not isinstance(question, str) or not 3 <= len(question.strip()) <= 2000:
            raise ValueError(
                "Question must be 3ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ2000 characters"
            )
        passages = self.retriever.search(question, method=method, k=8)
        if not sufficient(question, passages, self.min_similarity):
            return {
                "question": question,
                "answer": "Insufficient evidence in the indexed collection.",
                "abstained": True,
                "citations": [],
                "passages": passages,
                "raw_model_output": None,
                "answer_mode": "abstention",
                "latency_ms": 1000 * (time.perf_counter() - began),
            }
        # Sentence-level semantic/lexical reranking narrows context before generation.
        candidates = []
        terms = set(tokenize(question))
        anchors = [
            x
            for x in re.findall(r"\b[A-Z][A-Z0-9_-]{1,}\b", question)
            if x not in {"SQL"}
        ]
        for p in passages:
            for sentence in re.split(r"(?<=[.!?])\s+", p["text"]):
                if len(sentence.split()) >= 5:
                    overlap = len(terms & set(tokenize(sentence))) / max(1, len(terms))
                    anchor = sum(a.lower() in sentence.lower() for a in anchors)
                    candidates.append((sentence, p, overlap, anchor))
        if getattr(self.retriever, "embedder", None) is not None:
            vectors = self.retriever.embedder.encode(
                [c[0] for c in candidates] + [question]
            )
            similarities = vectors[:-1] @ vectors[-1]
        else:
            similarities = [0.0] * len(candidates)
        scored = sorted(
            [
                (float(sim) + 0.25 * c[2] + 0.4 * c[3], c[0], c[1])
                for sim, c in zip(similarities, candidates)
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        selected = []
        for _, sentence, p in scored:
            if sentence not in [x[0] for x in selected]:
                selected.append((sentence, p))
            if len(selected) == 2:
                break
        context_passages = [{**p, "text": sentence} for sentence, p in selected]
        draft = self.generator.draft(question, context_passages)
        norm = lambda text: re.sub(r"\s+", " ", text).strip().lower()
        citation = None
        answer = None
        mode = "validated_llm_exact_span"
        if len(tokenize(draft)) >= 4:
            for sentence, p in selected:
                if norm(draft) in norm(sentence):
                    citation = p
                    answer = draft
                    break
        if answer is None:
            mode = "evidence_extract_after_rejected_draft"
            answer, citation = selected[0]
        # Show supporting context even when a short exact model phrase was accepted.
        evidence = [
            {
                "quote": sentence,
                "chunk_id": p["id"],
                "document_id": p["document_id"],
                "title": p["title"],
                "page": p["page"],
                "path": p["path"],
                "source_url": p["source_url"],
            }
            for sentence, p in selected
        ]
        source = {
            "chunk_id": citation["id"],
            "document_id": citation["document_id"],
            "title": citation["title"],
            "page": citation["page"],
            "source_url": citation["source_url"],
            "path": citation["path"],
            "quote": answer,
        }
        return {
            "question": question,
            "answer": answer,
            "abstained": False,
            "citations": [source],
            "passages": passages,
            "raw_model_output": draft,
            "supporting_evidence": evidence,
            "answer_mode": mode,
            "latency_ms": 1000 * (time.perf_counter() - began),
        }


if __name__ == "__main__":
    import json

    p = argparse.ArgumentParser()
    p.add_argument("question")
    a = p.parse_args()
    print(json.dumps(RAG().answer(a.question), indent=2))
