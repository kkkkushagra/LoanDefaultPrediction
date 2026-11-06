from pathlib import Path

import joblib
import pandas as pd

from .loading import load_application_data
from .preprocessing import clean_application_data, extract_audit_frame
from src.features import FeaturePipeline


def main(
    input_path: str | Path = "data/application_train.csv",
    output_dir: str | Path = "data/processed",
):
    source = load_application_data(input_path)
    audit = extract_audit_frame(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(output / "audit.parquet", index=False)

    features, target = clean_application_data(source)
    from .pipeline import DataPreprocessor
    from sklearn.model_selection import train_test_split

    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    preprocessor = DataPreprocessor(random_state=42)
    train_processed = preprocessor.fit_transform(train_x)
    test_processed = preprocessor.transform(test_x)

    bureau_agg = output / "bureau_agg.parquet"
    bureau_source = Path("data/bureau.csv")
    if bureau_source.exists() and not bureau_agg.exists():
        bureau_raw = pd.read_csv(bureau_source)
        numeric = bureau_raw.select_dtypes(include="number").columns
        numeric = [column for column in numeric if column != "SK_ID_CURR"]
        bureau = bureau_raw.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "sum"])
        bureau.columns = [f"BUREAU_{column}_{stat}" for column, stat in bureau.columns]
        bureau.reset_index().to_parquet(bureau_agg, index=False)

    feature_source = features.copy()
    feature_source["SK_ID_CURR"] = source["SK_ID_CURR"].reindex(feature_source.index)
    feature_pipeline = FeaturePipeline(
        bureau_path=bureau_agg if bureau_agg.exists() else None
    ).fit(feature_source.loc[train_x.index], train_y.loc[train_x.index])
    feature_train = feature_pipeline.transform(feature_source.loc[train_x.index])
    feature_test = feature_pipeline.transform(feature_source.loc[test_x.index])

    train_ids = source.loc[train_x.index, "SK_ID_CURR"]
    test_ids = source.loc[test_x.index, "SK_ID_CURR"]
    train_matrix = pd.concat(
        [train_ids.reset_index(drop=True).rename("SK_ID_CURR"), train_processed.reset_index(drop=True), feature_train.reset_index(drop=True)],
        axis=1,
    )
    test_matrix = pd.concat(
        [test_ids.reset_index(drop=True).rename("SK_ID_CURR"), test_processed.reset_index(drop=True), feature_test.reset_index(drop=True)],
        axis=1,
    )
    pd.concat([train_matrix, train_y.reset_index(drop=True).rename("TARGET")], axis=1).to_parquet(
        output / "train.parquet", index=False
    )
    pd.concat([test_matrix, test_y.reset_index(drop=True).rename("TARGET")], axis=1).to_parquet(
        output / "test.parquet", index=False
    )
    joblib.dump(preprocessor, output / "preprocessor.joblib")
    joblib.dump(feature_pipeline, output / "feature_pipeline.joblib")
    return preprocessor


if __name__ == "__main__":
    main()
