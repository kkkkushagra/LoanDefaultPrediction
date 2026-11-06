import numpy as np
import pandas as pd


SENSITIVE_COLUMNS = {"CODE_GENDER", "DAYS_BIRTH", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"}


def clean_application_data(
    frame: pd.DataFrame,
    sensitive_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    excluded = SENSITIVE_COLUMNS if sensitive_columns is None else sensitive_columns
    frame = frame.drop_duplicates(subset=["SK_ID_CURR"]).copy()
    target = frame["TARGET"].astype(int)
    features = frame.drop(columns=["TARGET", "SK_ID_CURR"])
    features = features.drop(columns=features.columns.intersection(excluded))
    return features.replace([np.inf, -np.inf], np.nan), target
