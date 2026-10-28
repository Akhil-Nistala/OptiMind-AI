import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import load_json, save_json

DOCUMENTS = [
    {
        "id": "doe_air_cost",
        "file": "compressed_air_cost.pdf",
        "url": "https://www.energystar.gov/sites/default/files/buildings/tools/compressed_air1.pdf",
        "title": "Determine the Cost of Compressed Air for Your Plant",
        "publisher": "U.S. Department of Energy",
        "license": "DOE government information; public-domain notice, retain attribution",
    },
    {
        "id": "doe_air_leaks",
        "file": "compressed_air_leaks.pdf",
        "url": "https://www.energy.gov/sites/prod/files/2014/05/f16/compressed_air3.pdf",
        "title": "Minimize Compressed Air Leaks",
        "publisher": "U.S. Department of Energy / NREL",
        "license": "DOE technical tip sheet; retain attribution and original notices",
    },
]


def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def download_models():
    config = load_json("configs/models.json")
    records = []
    allowed = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.txt",
        "spiece.model",
        "README.md",
        "LICENSE",
        "LICENSE.txt",
    }
    for kind, model in config.items():
        info = json.loads(
            fetch(
                f'https://huggingface.co/api/models/{model["repo"]}/revision/{model["revision"]}?blobs=true'
            )
        )
        folder = Path(model["directory"])
        folder.mkdir(parents=True, exist_ok=True)

        def download(entry):
            name = entry["rfilename"]
            dest = folder / name
            content = (
                dest.read_bytes()
                if dest.exists()
                else fetch(
                    f'https://huggingface.co/{model["repo"]}/resolve/{model["revision"]}/{name}'
                )
            )
            sha = hashlib.sha256(content).hexdigest()
            expected = entry.get("lfs", {}).get("sha256")
            if expected and sha != expected:
                raise ValueError(f"Model checksum mismatch: {name}")
            if entry.get("size") is not None and len(content) != entry["size"]:
                raise ValueError(f"Model length mismatch: {name}")
            dest.write_bytes(content)
            print(f"{kind}: {name} ({len(content):,} bytes)", flush=True)
            return {"kind": kind, "file": name, "sha256": sha, "bytes": len(content)}

        with ThreadPoolExecutor(max_workers=3) as pool:
            records.extend(
                pool.map(
                    download, [x for x in info["siblings"] if x["rfilename"] in allowed]
                )
            )
    save_json("results/model_provenance.json", {"models": config, "files": records})


def download_documents():
    folder = Path("data/documents")
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for source in DOCUMENTS:
        dest = folder / source["file"]
        content = dest.read_bytes() if dest.exists() else fetch(source["url"])
        if not content.startswith(b"%PDF"):
            raise ValueError(f'Not a PDF: {source["url"]}')
        dest.write_bytes(content)
        records.append(
            {
                **source,
                "path": str(dest),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
        print(f'Document: {source["title"]}', flush=True)
    save_json("data/documents/public_sources.json", records)
    save_json(
        "results/document_provenance.json",
        {"copyright_policy": "https://www.energy.gov/web-policies", "sources": records},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["models", "documents", "all"], default="all")
    a = p.parse_args()
    if a.only in ["documents", "all"]:
        download_documents()
    if a.only in ["models", "all"]:
        download_models()
