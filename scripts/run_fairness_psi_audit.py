import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.service import ScoringService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
NUMERIC_FIELDS = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
]
CATEGORICAL_FIELDS = ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE"]
SCORING_FIELDS = [*NUMERIC_FIELDS, *CATEGORICAL_FIELDS]
MIN_GROUP_SIZE = 100
PSI_EPSILON = 0.5
SIMULATED_INCOME_MULTIPLIER = 1.25
AGE_GROUPS = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]


def _normalized_categories(values: pd.Series) -> pd.Series:
    categories = values.astype("string").fillna("Missing / unknown")
    return categories.astype(object)


def population_stability_index(
    reference: pd.Series,
    comparison: pd.Series,
    numeric: bool,
    epsilon: float = PSI_EPSILON,
) -> float:
    if reference.empty or comparison.empty:
        raise ValueError("PSI requires non-empty reference and comparison samples")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    reference_values = reference.copy()
    comparison_values = comparison.copy()
    if numeric:
        reference_values = pd.to_numeric(reference_values, errors="coerce")
        comparison_values = pd.to_numeric(comparison_values, errors="coerce")
        finite_reference = reference_values[np.isfinite(reference_values)]
        if finite_reference.empty:
            raise ValueError("Numeric PSI reference has no finite observations")
        edges = np.unique(finite_reference.quantile(np.linspace(0.1, 0.9, 9)).to_numpy())
        edges = edges[np.isfinite(edges)]
        reference_bins = pd.cut(
            reference_values,
            bins=np.concatenate(([-np.inf], edges, [np.inf])),
            include_lowest=True,
        ).astype("string").fillna("Missing / unknown")
        comparison_bins = pd.cut(
            comparison_values,
            bins=np.concatenate(([-np.inf], edges, [np.inf])),
            include_lowest=True,
        ).astype("string").fillna("Missing / unknown")
    else:
        reference_bins = _normalized_categories(reference_values)
        comparison_bins = _normalized_categories(comparison_values)

    reference_bins = reference_bins.astype(str)
    comparison_bins = comparison_bins.astype(str)
    if not numeric:
        reference_categories = set(reference_bins.unique())
        comparison_bins = comparison_bins.where(
            comparison_bins.isin(reference_categories),
            "Unseen category",
        )
        reference_bins = reference_bins.where(
            reference_bins.isin(reference_categories),
            "Unseen category",
        )
    buckets = sorted(set(reference_bins.unique()) | set(comparison_bins.unique()))
    reference_counts = reference_bins.value_counts().reindex(buckets, fill_value=0).to_numpy(dtype=float)
    comparison_counts = comparison_bins.value_counts().reindex(buckets, fill_value=0).to_numpy(dtype=float)
    reference_share = (reference_counts + epsilon) / (
        reference_counts.sum() + epsilon * len(buckets)
    )
    comparison_share = (comparison_counts + epsilon) / (
        comparison_counts.sum() + epsilon * len(buckets)
    )
    return float(np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share)))


def psi_rows(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    snapshot: str,
) -> list[dict]:
    rows = []
    for field in SCORING_FIELDS:
        psi = population_stability_index(
            reference[field],
            comparison[field],
            numeric=field in NUMERIC_FIELDS,
        )
        rows.append({
            "snapshot": snapshot,
            "field": field,
            "psi": psi,
            "interpretation": psi_interpretation(psi),
        })
    return rows


def psi_interpretation(value: float) -> str:
    if value < 0.10:
        return "little change"
    if value < 0.25:
        return "moderate change"
    return "large change"


def age_group(age: float) -> str:
    if pd.isna(age):
        return "Missing / unknown"
    if age < 18:
        return "Under 18"
    for lower, upper in zip((18, 25, 35, 45, 55, 65), (25, 35, 45, 55, 65, np.inf)):
        if lower <= age < upper:
            return AGE_GROUPS[(18, 25, 35, 45, 55, 65).index(lower)]
    return "Missing / unknown"


def group_metrics(
    population: pd.DataFrame,
    group_column: str,
    groups: list[str],
    threshold: float,
    minimum_group_size: int = MIN_GROUP_SIZE,
) -> list[dict]:
    rows = []
    for group in groups:
        subset = population.loc[population[group_column] == group]
        size = len(subset)
        if not size:
            rows.append({
                "group": group,
                "group_size": 0,
                "status": "missing",
                "observed_default_rate": None,
                "flag_rate": None,
                "recall": None,
                "false_positive_rate": None,
            })
            continue

        positive = subset[TARGET_COLUMN].to_numpy(dtype=int) == 1
        flagged = subset["probability"].to_numpy() >= threshold
        positives = int(positive.sum())
        negatives = size - positives
        true_positive = int((positive & flagged).sum())
        false_positive = int((~positive & flagged).sum())
        rows.append({
            "group": group,
            "group_size": size,
            "status": "small" if size < minimum_group_size else "ok",
            "observed_default_rate": float(positive.mean()),
            "flag_rate": float(flagged.mean()),
            "recall": true_positive / positives if positives else None,
            "false_positive_rate": false_positive / negatives if negatives else None,
        })
    return rows


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[ID_COLUMN, TARGET_COLUMN])
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{path} must have non-null, unique applicant IDs")
    return frame


def load_raw_applications(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=[ID_COLUMN, *SCORING_FIELDS],
    )


def join_applications(split: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    joined = split.merge(raw, on=ID_COLUMN, how="left", validate="one_to_one")
    if joined[SCORING_FIELDS].isna().all(axis=1).any():
        raise ValueError("Some split IDs did not match raw application records")
    joined["DAYS_EMPLOYED"] = joined["DAYS_EMPLOYED"].replace(365243, np.nan)
    return joined


def score_test_set(service: ScoringService, applications: pd.DataFrame) -> np.ndarray:
    features = applications.reindex(columns=[ID_COLUMN, *SCORING_FIELDS]).copy()
    return service.predict_calibrated(service.transform(features))


def prepare_fairness_population(
    applications: pd.DataFrame,
    audit: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    population = applications[[ID_COLUMN, TARGET_COLUMN]].merge(
        audit[[ID_COLUMN, "CODE_GENDER", "AGE_YEARS"]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    population["probability"] = probabilities
    population["gender_group"] = _normalized_categories(population["CODE_GENDER"])
    population["age_group"] = population["AGE_YEARS"].map(age_group)
    return population


def make_report(result: dict, command: str) -> str:
    lines = [
        "# Task 7 Fairness and PSI Audit",
        "",
        f"Reproduce with: `{command}`",
        "",
        "Uses the saved fitted model, preprocessor, and feature pipeline. It does not train "
        "or refit model artifacts. Gender and age are joined from the audit frame by "
        "`SK_ID_CURR` and are used only to calculate reporting groups; they are not sent "
        "to the scorer. Review flags use the model artifact's current cutoff.",
        "",
        "A group is labeled `small` when it has fewer than "
        f"{result['fairness']['minimum_group_size']} applicants; its metrics are shown but "
        "should be interpreted cautiously. Missing groups are listed explicitly, with "
        "`n/a` metrics where no denominator exists.",
        "",
        f"- Held-out test applications: {result['fairness']['sample_size']:,}",
        f"- Positive cases: {result['fairness']['positive_count']:,}",
        f"- Current review threshold: {result['fairness']['threshold']:.9f}",
        "",
    ]
    for group_type, title in (("gender", "Gender groups"), ("age", "Age groups")):
        lines += [
            f"## {title}",
            "",
            "| Group | N | Status | Observed default rate | Flag rate | Recall | False-positive rate |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
        for row in result["fairness"][group_type]:
            metric = lambda key: "n/a" if row[key] is None else f"{row[key]:.2%}"
            lines.append(
                f"| {row['group']} | {row['group_size']:,} | {row['status']} | "
                f"{metric('observed_default_rate')} | {metric('flag_rate')} | "
                f"{metric('recall')} | {metric('false_positive_rate')} |"
            )
        lines.append("")

    lines += [
        "## Population Stability Index",
        "",
        "No comparable post-training scoring snapshot was available. The saved training "
        "split is the reference; the saved held-out test split is the baseline comparison. "
        "The drift demo is synthetic: test `AMT_INCOME_TOTAL` values are multiplied by "
        f"{result['psi']['simulated_income_multiplier']:.2f}; all other fields are unchanged. "
        "Numeric PSI uses ten reference quantile bins plus a separate missing bin; "
        "categorical PSI reports unseen comparison categories separately. A 0.5 count "
        "pseudocount avoids zero-frequency log terms. Interpretation bands are common "
        "heuristics, not project acceptance criteria. The Home Credit employed-day sentinel "
        "365243 is treated as missing, matching scoring.",
        "",
        "| Snapshot | Field | PSI | Heuristic interpretation |",
        "|---|---|---:|---|",
    ]
    for row in result["psi"]["fields"]:
        lines.append(
            f"| {row['snapshot']} | {row['field']} | {row['psi']:.6f} | "
            f"{row['interpretation']} |"
        )
    lines += [
        "",
        f"- Train-vs-test mean PSI: {result['psi']['baseline_mean']:.6f}",
        f"- Train-vs-test max PSI: {result['psi']['baseline_max']:.6f}",
        f"- Simulated-shift mean PSI: {result['psi']['simulated_mean']:.6f}",
        f"- Simulated-shift max PSI: {result['psi']['simulated_max']:.6f}",
        "",
        "The simulated snapshot is a demonstration only and does not indicate observed "
        "production drift.",
        "",
    ]
    return "\n".join(lines)


def run_audit(
    train_path: Path,
    test_path: Path,
    raw_path: Path,
    audit_path: Path,
    output_json: Path,
    output_report: Path,
    minimum_group_size: int = MIN_GROUP_SIZE,
) -> dict:
    if minimum_group_size < 1:
        raise ValueError("minimum_group_size must be positive")
    train_membership = load_split(train_path)
    test_membership = load_split(test_path)
    if train_membership[ID_COLUMN].isin(test_membership[ID_COLUMN]).any():
        raise ValueError("Training and test applicant IDs must be disjoint")
    raw = load_raw_applications(raw_path)
    if raw[ID_COLUMN].isna().any() or raw[ID_COLUMN].duplicated().any():
        raise ValueError(f"{raw_path} must have non-null, unique applicant IDs")

    train_applications = join_applications(train_membership, raw)
    test_applications = join_applications(test_membership, raw)
    audit = pd.read_parquet(audit_path)
    required_audit_columns = {ID_COLUMN, "CODE_GENDER", "AGE_YEARS"}
    if not required_audit_columns.issubset(audit.columns):
        raise ValueError(f"Audit file must contain {sorted(required_audit_columns)}")
    if audit[ID_COLUMN].isna().any() or audit[ID_COLUMN].duplicated().any():
        raise ValueError(f"{audit_path} must have non-null, unique applicant IDs")

    reference = train_applications[SCORING_FIELDS]
    comparison = test_applications[SCORING_FIELDS]
    shifted = comparison.copy()
    shifted["AMT_INCOME_TOTAL"] = shifted["AMT_INCOME_TOTAL"] * SIMULATED_INCOME_MULTIPLIER
    psi_fields = [
        *psi_rows(reference, comparison, "Train vs held-out test baseline"),
        *psi_rows(reference, shifted, "Simulated test shift (+25% income)"),
    ]

    service = ScoringService(PROJECT_ROOT)
    probabilities = score_test_set(service, test_applications)
    population = prepare_fairness_population(test_applications, audit, probabilities)
    gender_groups = sorted(set(population["gender_group"].unique()) | {"Missing / unknown"})
    age_groups = [
        *AGE_GROUPS,
        "Under 18",
        "Missing / unknown",
    ]
    fairness = {
        "sample_size": len(population),
        "positive_count": int(population[TARGET_COLUMN].sum()),
        "threshold": service.cost_threshold,
        "minimum_group_size": minimum_group_size,
        "gender": group_metrics(
            population,
            "gender_group",
            gender_groups,
            service.cost_threshold,
            minimum_group_size,
        ),
        "age": group_metrics(
            population,
            "age_group",
            age_groups,
            service.cost_threshold,
            minimum_group_size,
        ),
    }
    baseline_values = [row["psi"] for row in psi_fields if row["snapshot"] == "Train vs held-out test baseline"]
    simulated_values = [row["psi"] for row in psi_fields if row["snapshot"] == "Simulated test shift (+25% income)"]
    result = {
        "fairness": fairness,
        "psi": {
            "reference": str(train_path),
            "baseline_comparison": str(test_path),
            "simulated_income_multiplier": SIMULATED_INCOME_MULTIPLIER,
            "fields": psi_fields,
            "baseline_mean": float(np.mean(baseline_values)),
            "baseline_max": float(np.max(baseline_values)),
            "simulated_mean": float(np.mean(simulated_values)),
            "simulated_max": float(np.max(simulated_values)),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    command = ".\\.venv\\Scripts\\python.exe -m scripts.run_fairness_psi_audit"
    output_report.write_text(make_report(result, command), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out fairness and PSI baseline audits")
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/processed/train.parquet")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data/processed/test.parquet")
    parser.add_argument("--raw-path", type=Path, default=PROJECT_ROOT / "data/application_train.csv")
    parser.add_argument("--audit-path", type=Path, default=PROJECT_ROOT / "data/processed/audit.parquet")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "docs/fairness_psi_audit.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "docs/fairness_psi_audit.md",
    )
    parser.add_argument("--minimum-group-size", type=int, default=MIN_GROUP_SIZE)
    arguments = parser.parse_args()
    result = run_audit(
        train_path=arguments.train_path,
        test_path=arguments.test_path,
        raw_path=arguments.raw_path,
        audit_path=arguments.audit_path,
        output_json=arguments.output_json,
        output_report=arguments.output_report,
        minimum_group_size=arguments.minimum_group_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
