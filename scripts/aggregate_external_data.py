import argparse
import time
from pathlib import Path

import pandas as pd

ID_COLUMNS = ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]


def indexed_by_customer(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index() if frame.index.name == "SK_ID_CURR" else frame.copy()
    if "SK_ID_CURR" in result.columns:
        result = result.set_index("SK_ID_CURR")
    if result.index.name != "SK_ID_CURR":
        raise ValueError("Aggregator output must be indexed by SK_ID_CURR")
    return result.loc[:, ~result.columns.duplicated()].sort_index()


def float32_columns(
    path: Path,
    required: set[str] | None = None,
    excluded: set[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    required = required or set()
    excluded = excluded or set()
    sample = pd.read_csv(path, nrows=100)
    numeric = {
        c for c in sample.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(sample[c])
    }
    columns = [c for c in sample.columns if c in required or c in numeric]
    return columns, {c: "float32" for c in columns if c in numeric}


def _base_aggregate(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(ID_COLUMNS)
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min"])
    result.columns = [f"{prefix}_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    return indexed_by_customer(result)


def _row_count(frame: pd.DataFrame, id_column: str) -> pd.Series:
    if id_column in frame:
        return frame.groupby("SK_ID_CURR")[id_column].nunique()
    return frame.groupby("SK_ID_CURR").size()


def bureau_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    result = _base_aggregate(frame, "BUREAU")
    result["BUREAU_ROW_COUNT"] = _row_count(frame, "SK_ID_BUREAU")
    active = frame[frame["CREDIT_ACTIVE"].eq("Active")] if "CREDIT_ACTIVE" in frame else frame.iloc[0:0]
    if not active.empty and {"AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"}.issubset(active):
        totals = active.groupby("SK_ID_CURR")[["AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"]].sum()
        result["BUREAU_ACTIVE_DEBT_CREDIT_RATIO"] = (
            totals["AMT_CREDIT_SUM_DEBT"].div(totals["AMT_CREDIT_SUM"].replace(0, float("nan")))
        ).reindex(result.index)
        if "CREDIT_DAY_OVERDUE" in active:
            overdue = active["CREDIT_DAY_OVERDUE"].fillna(0).gt(0)
            result["BUREAU_ACTIVE_OVERDUE_MEAN"] = (
                overdue.groupby(active["SK_ID_CURR"]).mean().reindex(result.index)
            )
    return indexed_by_customer(result)


def previous_application_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    result = _base_aggregate(frame, "PREV")
    result["PREV_ROW_COUNT"] = _row_count(frame, "SK_ID_PREV")
    if "NAME_CONTRACT_STATUS" in frame:
        status = pd.crosstab(frame["SK_ID_CURR"], frame["NAME_CONTRACT_STATUS"], normalize="index")
        for name in ("Approved", "Refused"):
            if name in status:
                result[f"PREV_{name.upper()}_SHARE"] = status[name].reindex(result.index)
    if {"AMT_APPLICATION", "AMT_CREDIT"}.issubset(frame):
        ratio = frame["AMT_APPLICATION"].div(frame["AMT_CREDIT"].replace(0, float("nan")))
        result["PREV_APPLICATION_CREDIT_RATIO"] = (
            ratio.groupby(frame["SK_ID_CURR"]).mean().reindex(result.index)
        )
    if "NAME_CONTRACT_STATUS" in frame:
        for name in ("Approved", "Refused"):
            subset = frame[frame["NAME_CONTRACT_STATUS"].eq(name)]
            for column in ("AMT_APPLICATION", "AMT_CREDIT"):
                if not subset.empty and column in subset:
                    result[f"PREV_{name.upper()}_{column}_MEAN"] = (
                        subset.groupby("SK_ID_CURR")[column].mean().reindex(result.index)
                    )
    return indexed_by_customer(result)


def installment_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    result = _base_aggregate(frame, "INSTALL")
    result["INSTALL_ROW_COUNT"] = frame.groupby("SK_ID_CURR").size()
    customer = frame["SK_ID_CURR"]
    if {"DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"}.issubset(frame):
        late = frame["DAYS_ENTRY_PAYMENT"].gt(frame["DAYS_INSTALMENT"])
        result["INSTALL_LATE_PAYMENT_RATE"] = late.groupby(customer).mean().reindex(result.index)
        timing = frame["DAYS_ENTRY_PAYMENT"] - frame["DAYS_INSTALMENT"]
        result["INSTALL_DAYS_LATE_MEAN"] = timing.clip(lower=0).groupby(customer).mean().reindex(result.index)
        result["INSTALL_DAYS_EARLY_MEAN"] = timing.clip(upper=0).abs().groupby(customer).mean().reindex(result.index)
    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(frame):
        ratio = frame["AMT_PAYMENT"].div(frame["AMT_INSTALMENT"].replace(0, float("nan")))
        result["INSTALL_PAYMENT_RATIO_MEAN"] = ratio.groupby(customer).mean().reindex(result.index)
    return indexed_by_customer(result)


def pos_cash_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    result = _base_aggregate(frame, "POS")
    result["POS_ROW_COUNT"] = _row_count(frame, "SK_ID_PREV")
    for column in ("SK_DPD", "SK_DPD_DEF"):
        if column in frame:
            result[f"POS_{column}_LATE_RATE"] = (
                frame[column].gt(0).groupby(frame["SK_ID_CURR"]).mean().reindex(result.index)
            )
    return indexed_by_customer(result)


def credit_card_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    result = _base_aggregate(frame, "CREDIT_CARD")
    result["CREDIT_CARD_ROW_COUNT"] = _row_count(frame, "SK_ID_PREV")
    customer = frame["SK_ID_CURR"]
    if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(frame):
        utilisation = frame["AMT_BALANCE"].div(frame["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, float("nan")))
        result["CREDIT_CARD_UTILISATION_MEAN"] = utilisation.groupby(customer).mean().reindex(result.index)
    for column in ("SK_DPD", "SK_DPD_DEF"):
        if column in frame:
            result[f"CREDIT_CARD_{column}_LATE_RATE"] = (
                frame[column].gt(0).groupby(customer).mean().reindex(result.index)
            )
    return indexed_by_customer(result)


INSTALLMENT_COLUMNS = [
    "SK_ID_CURR",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
]
REQUIRED_STRING_COLUMNS = {
    "bureau.csv": {"SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE"},
    "previous_application.csv": {"SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS"},
}
NON_FLOAT_COLUMNS = {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "CREDIT_ACTIVE"}


def main(
    data_dir: str | Path = "data",
    output_dir: str | Path = "data/processed",
    nrows: int | None = None,
) -> None:
    data = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("bureau.csv", bureau_aggregates, "bureau_agg.parquet"),
        ("previous_application.csv", previous_application_aggregates, "previous_application_agg.parquet"),
        ("installments_payments.csv", installment_aggregates, "installments_payments_agg.parquet"),
        ("POS_CASH_balance.csv", pos_cash_aggregates, "pos_cash_agg.parquet"),
        ("credit_card_balance.csv", credit_card_aggregates, "credit_card_agg.parquet"),
    ]
    for filename, aggregate, target in specifications:
        path = data / filename
        if not path.exists():
            print(f"{filename}: not found, skipped")
            continue
        stage_start = time.perf_counter()
        if filename == "installments_payments.csv":
            frame = pd.read_csv(
                path,
                usecols=INSTALLMENT_COLUMNS,
                dtype={c: "float32" for c in INSTALLMENT_COLUMNS if c != "SK_ID_CURR"},
                nrows=nrows,
            )
        else:
            required = REQUIRED_STRING_COLUMNS.get(filename, {"SK_ID_CURR", "SK_ID_PREV"})
            columns, dtype = float32_columns(path, required=required, excluded=NON_FLOAT_COLUMNS)
            frame = pd.read_csv(path, usecols=columns, dtype=dtype, nrows=nrows)
        aggregated = aggregate(frame)
        aggregated.reset_index().to_parquet(output / target, index=False)
        elapsed = time.perf_counter() - stage_start
        print(
            f"{filename}: {elapsed:.2f}s, shape={aggregated.shape}, "
            f"columns={aggregated.shape[1]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--nrows", type=int, default=None, help="Read only this many rows per source table")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.nrows)