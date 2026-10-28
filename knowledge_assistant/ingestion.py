"""PDF page numbers are physical 1-based PDF pages; text has no page number."""

import re
import hashlib
from pathlib import Path
from pypdf import PdfReader
from common import load_json, save_json


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, words=110, overlap=20):
    if words < 10 or not 0 <= overlap < words:
        raise ValueError("Invalid chunk size/overlap")
    tokens = normalize(text).split()
    return [
        " ".join(tokens[i : i + words])
        for i in range(0, len(tokens), words - overlap)
        if len(tokens[i : i + words]) >= 8
    ]


def ingest(folder="data/documents", output="data/index/chunks.json"):
    folder = Path(folder)
    metadata = (
        {r["file"]: r for r in load_json(folder / "public_sources.json")}
        if (folder / "public_sources.json").exists()
        else {}
    )
    chunks = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt"}:
            continue
        meta = metadata.get(
            path.name,
            {
                "id": path.stem,
                "title": path.stem.replace("_", " "),
                "publisher": "OptiMind authored demonstration",
                "url": None,
                "license": "Project-authored synthetic operating procedure",
            },
        )
        pages = (
            [
                (i + 1, p.extract_text() or "")
                for i, p in enumerate(PdfReader(path).pages)
            ]
            if path.suffix.lower() == ".pdf"
            else [(None, path.read_text(encoding="utf-8"))]
        )
        for page, text in pages:
            # Exclude a third-party leakage table reproduced by permission in the DOE PDF.
            # Only agency narrative is used for the retrieval corpus.
            text = re.sub(
                r"Leakage rates a \(cfm\).*?Leak Detection",
                "Leak Detection",
                text,
                flags=re.S,
            )
            text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
            for ordinal, body in enumerate(chunk_text(text)):
                cid = hashlib.sha256(
                    f'{meta["id"]}|{page}|{ordinal}|{body}'.encode()
                ).hexdigest()[:16]
                chunks.append(
                    {
                        "id": cid,
                        "document_id": meta["id"],
                        "text": body,
                        "page": page,
                        "chunk": ordinal,
                        "title": meta["title"],
                        "source_url": meta.get("url"),
                        "path": str(path),
                        "publisher": meta["publisher"],
                        "license": meta["license"],
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    if not chunks:
        raise ValueError(
            "No extractable PDF/text chunks found; scanned PDFs require OCR (not implemented)"
        )
    save_json(output, chunks)
    print(
        f'Indexed {len(chunks)} chunks from {len({c["document_id"] for c in chunks})} documents'
    )
    return chunks


if __name__ == "__main__":
    ingest()
