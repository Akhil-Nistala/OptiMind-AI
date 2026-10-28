from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root.parent / "OptiMind-AI-project.zip"
    excluded = {".git", "__pycache__", ".pytest_cache", ".venv", "models"}
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and not any(p in excluded for p in relative.parts)
                and path.suffix.lower() != ".pdf"
            ):
                archive.write(path, Path("optimind-ai") / relative)
    print(f"{output} ({output.stat().st_size:,} bytes)")
