import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bureau_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(["SK_ID_CURR"])
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min", "sum", "count"])
    result.columns = [f"BUREAU_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    return result.reset_index()


def previous_application_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(["SK_ID_CURR"])
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min", "sum", "count"])
    result.columns = [f"PREV_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    if "NAME_CONTRACT_STATUS" in frame:
        status = pd.crosstab(frame["SK_ID_CURR"], frame["NAME_CONTRACT_STATUS"], normalize="index")
        for status_name in ("Approved", "Refused"):
            if status_name in status:
                result[f"PREV_{status_name.upper()}_SHARE"] = status[status_name]
    return result.reset_index()


def installment_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(["SK_ID_CURR"])
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min", "sum", "count"])
    result.columns = [f"INSTALL_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
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
        aggregate(pd.read_csv(path)).to_parquet(output / target, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
