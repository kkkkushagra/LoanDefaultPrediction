import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


FILES = [
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
]


def main(data_dir: str | Path = "data") -> None:
    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FILES if not (destination / name).exists()]
    if not missing:
        return

    command = [
        sys.executable, "-m", "kaggle", "competitions", "download",
        "-c", "home-credit-default-risk", "-p", str(destination),
    ]
    subprocess.run(command, check=True)
    archive = destination / "home-credit-default-risk.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination)
        archive.unlink()

    absent = [name for name in missing if not (destination / name).exists()]
    if absent:
        raise FileNotFoundError(f"Kaggle download did not produce: {', '.join(absent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    main(parser.parse_args().data_dir)
