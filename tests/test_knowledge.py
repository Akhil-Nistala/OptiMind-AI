from pathlib import Path
import numpy as np
import pytest
from knowledge_assistant.ingestion import normalize, chunk_text, ingest
from knowledge_assistant.generation import RAG, sufficient
from common import save_json


def test_chunking_and_metadata(tmp_path):
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, 20, 5)
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]
    assert normalize(" a \n b ") == "a b"
    (tmp_path / "sample.txt").write_text(text)
    result = ingest(tmp_path, tmp_path / "index.json")
    assert result[0]["page"] is None and result[0]["source_sha256"] and result[0]["id"]
    assert result == ingest(tmp_path, tmp_path / "index2.json")


def passage(text="A machine processes one operation at a time."):
    return {
        "id": "a",
        "document_id": "doc",
        "title": "Fixture",
        "text": text,
        "page": None,
        "path": "fixture.txt",
        "source_url": None,
        "semantic_score": 0.9,
        "score": 1,
    }


class FakeRetriever:
    def search(self, *args, **kwargs):
        return [passage()]


class FakeGenerator:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def draft(self, *args):
        self.calls += 1
        return self.text


def test_grounding_rejects_fabricated_answer():
    llm = FakeGenerator("The machine produces 900 units.")
    result = RAG(FakeRetriever(), llm).answer(
        "How many operations can one machine process?"
    )
    assert result["answer_mode"] == "evidence_extract_after_rejected_draft"
    assert "900" not in result["answer"]
    assert result["answer"] in passage()["text"]


def test_exact_span_and_insufficient_evidence():
    llm = FakeGenerator("machine processes one operation at a time")
    result = RAG(FakeRetriever(), llm).answer(
        "How many operations can one machine process?"
    )
    assert (
        result["answer_mode"] == "validated_llm_exact_span"
        and result["citations"][0]["chunk_id"] == "a"
    )
    before = llm.calls
    result = RAG(FakeRetriever(), llm).answer("Who won the cricket world championship?")
    assert result["abstained"] and llm.calls == before


def test_missing_entity_abstains():
    llm = FakeGenerator("one operation at a time")
    result = RAG(FakeRetriever(), llm).answer(
        "What machine pressure is safe for ZX-99?"
    )
    assert result["abstained"] and llm.calls == 0
