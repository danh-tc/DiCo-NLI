"""Environment detection: paths and precision differ between local and Kaggle."""

from __future__ import annotations

import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_REPO_NAME = "SemEval-2027-Task-2-DiCo-NLI"


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists()


def task_repo_dir() -> Path:
    """Clone of the organizers' repo (data + official scorer)."""
    if "DICO_TASK_REPO" in os.environ:
        return Path(os.environ["DICO_TASK_REPO"])
    return PROJECT_ROOT / "external" / TASK_REPO_NAME


def data_dir() -> Path:
    if "DICO_DATA_DIR" in os.environ:
        return Path(os.environ["DICO_DATA_DIR"])
    return task_repo_dir() / "final_data"


def runs_dir() -> Path:
    if "DICO_RUNS_DIR" in os.environ:
        return Path(os.environ["DICO_RUNS_DIR"])
    if is_kaggle():
        return Path("/kaggle/working/runs")
    return PROJECT_ROOT / "runs"


def precision() -> str:
    """bf16 on Ampere+ (RTX 5060), fp16 on older GPUs (Kaggle T4/P100), fp32 on CPU."""
    import torch

    if os.environ.get("DICO_PRECISION"):  # manual override: bf16 | fp16 | fp32
        return os.environ["DICO_PRECISION"]
    if not torch.cuda.is_available():
        return "fp32"
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return "bf16"
    if major >= 7:
        return "fp16"
    return "fp32"  # P100 (sm60): fp16 gives little speedup and risks overflow


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass


def describe() -> str:
    lines = [f"kaggle={is_kaggle()}", f"data_dir={data_dir()}", f"runs_dir={runs_dir()}"]
    try:
        import torch

        if torch.cuda.is_available():
            lines.append(f"gpu={torch.cuda.get_device_name(0)} precision={precision()}")
        else:
            lines.append("gpu=none")
    except ImportError:
        lines.append("torch not installed")
    return " | ".join(lines)
