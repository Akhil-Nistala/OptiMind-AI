import json
import platform
from pathlib import Path
import numpy as np


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hardware():
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": str(torch.__version__),
        "cuda": torch.cuda.is_available(),
    }
