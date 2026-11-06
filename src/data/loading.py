from pathlib import Path

import pandas as pd

from .validation import validate_application_data


def load_application_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_application_data(frame)
