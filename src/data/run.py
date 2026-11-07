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

    from scripts.aggregate_external_data import main as aggregate_external
    aggregate_external("data", output)

    bureau_agg = output / "bureau_agg.parquet"
    feature_source = features.copy()
    feature_source["SK_ID_CURR"] = source["SK_ID_CURR"].reindex(feature_source.index)
    feature_pipeline = FeaturePipeline(
        bureau_path=bureau_agg if bureau_agg.exists() else None,
        external_paths={
            "bureau": bureau_agg,
            "previous": output / "previous_application_agg.parquet",
            "installments": output / "installments_payments_agg.parquet",
            "pos_cash": output / "pos_cash_agg.parquet",
            "credit_card": output / "credit_card_agg.parquet",
        },
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
