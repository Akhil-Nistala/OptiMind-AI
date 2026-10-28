"""Prepare reproducible demo assets. Downloads about 400 MB of local model weights."""

import argparse
from scripts.download_assets import download_models, download_documents
from scripts.prepare_data import prepare
from knowledge_assistant.ingestion import ingest
from knowledge_assistant.retrieval import build_index

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-models", action="store_true")
    a = p.parse_args()
    prepare()
    download_documents()
    ingest()
    if not a.skip_models:
        download_models()
        build_index()
