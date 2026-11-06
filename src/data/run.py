from pathlib import Path

import joblib
import pandas as pd

from .loading import load_application_data
from .preprocessing import clean_application_data, extract_audit_frame


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
    train_x = preprocessor.fit_transform(train_x)
    test_x = preprocessor.transform(test_x)

    pd.concat([train_x, train_y.rename("TARGET")], axis=1).to_parquet(
        output / "train.parquet", index=False
    )
    pd.concat([test_x, test_y.rename("TARGET")], axis=1).to_parquet(
        output / "test.parquet", index=False
    )
    joblib.dump(preprocessor, output / "preprocessor.joblib")
    return preprocessor


if __name__ == "__main__":
    main()
