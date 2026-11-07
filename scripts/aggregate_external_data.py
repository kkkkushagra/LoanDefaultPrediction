import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def float32_columns(path: Path, excluded: set[str] | None = None) -> tuple[list[str], dict[str, str]]:
    excluded = excluded or set()
    sample = pd.read_csv(path, nrows=100)
    columns = [
        column for column in sample.columns
        if column not in excluded
    ]
    dtype = {
        column: "float32"
        for column in columns
        if pd.api.types.is_numeric_dtype(sample[column])
    }
    return columns, dtype


def bureau_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(["SK_ID_CURR"])
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min"])
    result.columns = [f"BUREAU_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    result[f"BUREAU_ROW_COUNT"] = frame.groupby("SK_ID_CURR").size()
    return result.reset_index()


def previous_application_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(["SK_ID_CURR"])
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min"])
    result.columns = [f"PREV_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    result[f"PREV_ROW_COUNT"] = frame.groupby("SK_ID_CURR").size()
    if "NAME_CONTRACT_STATUS" in frame:
        status = pd.crosstab(frame["SK_ID_CURR"], frame["NAME_CONTRACT_STATUS"], normalize="index")
        for status_name in ("Approved", "Refused"):
            if status_name in status:
                result[f"PREV_{status_name.upper()}_SHARE"] = status[status_name]
    return result.reset_index()


def installment_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(["SK_ID_CURR"])
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min"])
    result.columns = [f"INSTALL_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    result[f"INSTALL_ROW_COUNT"] = frame.groupby("SK_ID_CURR").size()
    if {"DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"}.issubset(frame):
        result["INSTALL_LATE_PAYMENT_RATE"] = (
            frame["DAYS_ENTRY_PAYMENT"].gt(frame["DAYS_INSTALMENT"]).groupby(frame["SK_ID_CURR"]).mean()
        )
    return result.reset_index()


def main(data_dir: str | Path = "data", output_dir: str | Path = "data/processed") -> None:
    data = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("bureau.csv", bureau_aggregates, "bureau_agg.parquet"),
        ("previous_application.csv", previous_application_aggregates, "previous_application_agg.parquet"),
        ("installments_payments.csv", installment_aggregates, "installments_payments_agg.parquet"),
    ]
    for filename, aggregate, target in specifications:
        path = data / filename
        if not path.exists():
            continue
        if filename == "installments_payments.csv":
            columns = [
                "SK_ID_PREV",
                "SK_ID_CURR",
                "NUM_INSTALMENT_VERSION",
                "NUM_INSTALMENT_NUMBER",
                "DAYS_INSTALMENT",
                "DAYS_ENTRY_PAYMENT",
                "AMT_INSTALMENT",
                "AMT_PAYMENT",
            ]
            frame = pd.read_csv(
                path,
                usecols=columns,
                dtype={column: "float32" for column in columns if column != "SK_ID_CURR"},
            )
        else:
            columns, dtype = float32_columns(path, {"SK_ID_CURR", "NAME_CONTRACT_STATUS"})
            frame = pd.read_csv(path, usecols=columns, dtype=dtype)
        aggregate(frame).to_parquet(output / target, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
