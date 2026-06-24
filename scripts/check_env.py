from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn
import yaml

from src.utils import load_yaml, project_root, set_seed


def main() -> None:
    root = project_root()
    print(f"Project root: {root}")

    config_path = root / "configs" / "data.yaml"
    config = load_yaml(config_path)

    print(f"Project name: {config['project']['name']}")
    print(f"Numpy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"YAML loaded: {isinstance(config, dict)}")

    set_seed(config["project"]["seed"])
    print("Seed set successfully.")

    required_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "data/noisy",
        "notebooks",
        "kaggle",
        "src",
        "scripts",
        "outputs",
        "report",
    ]

    for d in required_dirs:
        path = root / d
        status = "OK" if path.exists() else "MISSING"
        print(f"{status}: {d}")

    print("Environment check completed.")


if __name__ == "__main__":
    main()