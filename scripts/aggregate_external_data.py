import argparse
import time
from pathlib import Path

import pandas as pd


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
    columns = [
        column for column in sample.columns
        if column in required or column not in excluded
    ]
    dtype = {
        column: "float32"
        for column in columns
        if column not in excluded
        if pd.api.types.is_numeric_dtype(sample[column])
    }
    return columns, dtype


def bureau_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(
        ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]
    )
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min"])
    result.columns = [f"BUREAU_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    result = indexed_by_customer(result)
    result["BUREAU_ROW_COUNT"] = (
        frame.groupby("SK_ID_CURR")["SK_ID_BUREAU"].nunique()
        if "SK_ID_BUREAU" in frame
        else frame.groupby("SK_ID_CURR").size()
    )
    active = frame[frame.get("CREDIT_ACTIVE").eq("Active")] if "CREDIT_ACTIVE" in frame else frame.iloc[0:0]
    if not active.empty and {"AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"}.issubset(active):
        active_totals = active.groupby("SK_ID_CURR")[["AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"]].sum()
        result["BUREAU_ACTIVE_DEBT_CREDIT_RATIO"] = (
            active_totals["AMT_CREDIT_SUM_DEBT"].div(active_totals["AMT_CREDIT_SUM"].replace(0, float("nan")))
        ).reindex(result.index)
        if "CREDIT_DAY_OVERDUE" in active:
            result["BUREAU_ACTIVE_OVERDUE_MEAN"] = (
                active.assign(overdue=active["CREDIT_DAY_OVERDUE"].fillna(0).gt(0)).groupby("SK_ID_CURR")["overdue"].mean()
            ).reindex(result.index)
    return indexed_by_customer(result)


def previous_application_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(
        ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]
    )
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min"])
    result.columns = [f"PREV_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    result = indexed_by_customer(result)
    result["PREV_ROW_COUNT"] = (
        frame.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
        if "SK_ID_PREV" in frame
        else frame.groupby("SK_ID_CURR").size()
    )
    if "NAME_CONTRACT_STATUS" in frame:
        status = pd.crosstab(frame["SK_ID_CURR"], frame["NAME_CONTRACT_STATUS"], normalize="index")
        for status_name in ("Approved", "Refused"):
            if status_name in status:
                result[f"PREV_{status_name.upper()}_SHARE"] = status[status_name]
    if {"AMT_APPLICATION", "AMT_CREDIT"}.issubset(frame):
        ratios = frame.assign(
            ratio=frame["AMT_APPLICATION"].div(frame["AMT_CREDIT"].replace(0, float("nan")))
        ).groupby("SK_ID_CURR")["ratio"].mean()
        result["PREV_APPLICATION_CREDIT_RATIO"] = ratios.reindex(result.index)
    if "NAME_CONTRACT_STATUS" in frame:
        for status_name in ("Approved", "Refused"):
            subset = frame[frame["NAME_CONTRACT_STATUS"].eq(status_name)]
            if not subset.empty and "AMT_APPLICATION" in subset:
                result[f"PREV_{status_name.upper()}_AMT_APPLICATION_MEAN"] = (
                    subset.groupby("SK_ID_CURR")["AMT_APPLICATION"].mean().reindex(result.index)
                )
            if not subset.empty and "AMT_CREDIT" in subset:
                result[f"PREV_{status_name.upper()}_AMT_CREDIT_MEAN"] = (
                    subset.groupby("SK_ID_CURR")["AMT_CREDIT"].mean().reindex(result.index)
                )
    result = result.reset_index()
    return indexed_by_customer(result)


def installment_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(
        ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]
    )
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min"])
    result.columns = [f"INSTALL_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    result = indexed_by_customer(result)
    result[f"INSTALL_ROW_COUNT"] = frame.groupby("SK_ID_CURR").size()
    if {"DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"}.issubset(frame):
        result["INSTALL_LATE_PAYMENT_RATE"] = (
            frame["DAYS_ENTRY_PAYMENT"].gt(frame["DAYS_INSTALMENT"]).groupby(frame["SK_ID_CURR"]).mean()
        )
    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(frame):
        ratio = frame.assign(
            ratio=frame["AMT_PAYMENT"].div(frame["AMT_INSTALMENT"].replace(0, float("nan")))
        ).groupby("SK_ID_CURR")["ratio"].mean()
        result["INSTALL_PAYMENT_RATIO_MEAN"] = ratio.reindex(result.index)
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(frame):
        timing = frame["DAYS_ENTRY_PAYMENT"] - frame["DAYS_INSTALMENT"]
        result["INSTALL_DAYS_LATE_MEAN"] = timing.clip(lower=0).groupby(frame["SK_ID_CURR"]).mean().reindex(result.index)
        result["INSTALL_DAYS_EARLY_MEAN"] = timing.clip(upper=0).abs().groupby(frame["SK_ID_CURR"]).mean().reindex(result.index)
    result = result.reset_index()
    return indexed_by_customer(result)


def pos_cash_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(
        ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"]
    )
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min"])
    result.columns = [f"POS_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    result = indexed_by_customer(result)
    result["POS_ROW_COUNT"] = (
        frame.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
        if "SK_ID_PREV" in frame
        else frame.groupby("SK_ID_CURR").size()
    )
    for column in ("SK_DPD", "SK_DPD_DEF"):
        if column in frame:
            result[f"POS_{column}_LATE_RATE"] = frame[column].gt(0).groupby(frame["SK_ID_CURR"]).mean()
    return indexed_by_customer(result)


def credit_card_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns.difference(
        ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"]
    )
    result = frame.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "min"])
    result.columns = [f"CREDIT_CARD_{column}_{stat.upper()}_AGG" for column, stat in result.columns]
    result = indexed_by_customer(result)
    result["CREDIT_CARD_ROW_COUNT"] = (
        frame.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
        if "SK_ID_PREV" in frame
        else frame.groupby("SK_ID_CURR").size()
    )
    if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(frame):
        utilisation = frame["AMT_BALANCE"].div(frame["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, float("nan")))
        result["CREDIT_CARD_UTILISATION_MEAN"] = utilisation.groupby(frame["SK_ID_CURR"]).mean()
    for column in ("SK_DPD", "SK_DPD_DEF"):
        if column in frame:
            result[f"CREDIT_CARD_{column}_LATE_RATE"] = frame[column].gt(0).groupby(frame["SK_ID_CURR"]).mean()
    return indexed_by_customer(result)


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
            continue
        stage_start = time.perf_counter()
        if filename == "installments_payments.csv":
            columns = [
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
                nrows=nrows,
            )
        else:
            count_id = (
                "SK_ID_BUREAU"
                if filename == "bureau.csv"
                else "SK_ID_PREV"
            )
            columns, dtype = float32_columns(
                path,
                required={"SK_ID_CURR", count_id},
                excluded={"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "NAME_CONTRACT_STATUS"},
            )
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
