import numpy as np
import pandas as pd


AUDIT_COLUMNS = {"CODE_GENDER", "DAYS_BIRTH"}
SENSITIVE_COLUMNS = AUDIT_COLUMNS | {"FLAG_OWN_CAR", "FLAG_OWN_REALTY"}


def extract_audit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    audit = frame.reindex(columns=AUDIT_COLUMNS).copy()
    if "DAYS_BIRTH" in audit:
        audit["AGE_YEARS"] = (-audit["DAYS_BIRTH"] / 365.25).round(1)
    return audit


def clean_application_data(
    frame: pd.DataFrame,
    sensitive_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    excluded = SENSITIVE_COLUMNS if sensitive_columns is None else sensitive_columns
    frame = frame.drop_duplicates(subset=["SK_ID_CURR"]).copy()
    target = frame["TARGET"].astype(int)
    features = frame.drop(columns=["TARGET", "SK_ID_CURR"])
    features = features.drop(columns=features.columns.intersection(excluded))
    if "DAYS_EMPLOYED" in features:
        sentinel = features["DAYS_EMPLOYED"].eq(365243)
        features["DAYS_EMPLOYED_ANOMALY"] = sentinel.astype("int8")
        features.loc[sentinel, "DAYS_EMPLOYED"] = np.nan
    return features.replace([np.inf, -np.inf], np.nan), target
