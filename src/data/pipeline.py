from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .loading import load_application_data
from .preprocessing import clean_application_data


class DataPreprocessor:
    def __init__(self, outlier_quantiles=(0.01, 0.99), random_state=42):
        self.outlier_quantiles = outlier_quantiles
        self.random_state = random_state
        self.numeric_medians_: dict[str, float] = {}
        self.categorical_modes_: dict[str, object] = {}
        self.outlier_bounds_: dict[str, tuple[float, float]] = {}
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.encoder_: OneHotEncoder | None = None

    def fit(self, frame: pd.DataFrame):
        frame = frame.replace([np.inf, -np.inf], np.nan)
        self.numeric_columns_ = frame.select_dtypes(include=np.number).columns.tolist()
        self.categorical_columns_ = frame.select_dtypes(exclude=np.number).columns.tolist()

        for column in self.numeric_columns_:
            values = frame[column].dropna()
            bounds = values.quantile(self.outlier_quantiles).tolist()
            self.outlier_bounds_[column] = (float(bounds[0]), float(bounds[1]))
            self.numeric_medians_[column] = float(values.median())

        for column in self.categorical_columns_:
            modes = frame[column].dropna().mode()
            self.categorical_modes_[column] = modes.iloc[0] if not modes.empty else "Unknown"

        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(self._categorical(frame))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame[self.numeric_columns_].copy()
        for column in self.numeric_columns_:
            numeric[column] = numeric[column].clip(*self.outlier_bounds_[column])
            numeric[column] = numeric[column].fillna(self.numeric_medians_[column])

        encoded = self.encoder_.transform(self._categorical(frame))
        names = self.encoder_.get_feature_names_out(self.categorical_columns_)
        encoded_frame = pd.DataFrame(encoded, index=frame.index, columns=names)
        return pd.concat([numeric.astype(float), encoded_frame], axis=1)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def _categorical(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.reindex(columns=self.categorical_columns_).copy()
        for column in self.categorical_columns_:
            result[column] = result[column].fillna(self.categorical_modes_[column]).astype(str)
        return result


def prepare_data(path: str | Path, test_size=0.2, random_state=42):
    features, target = clean_application_data(load_application_data(path))
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    preprocessor = DataPreprocessor(random_state=random_state)
    return (
        preprocessor.fit_transform(train_x),
        preprocessor.transform(test_x),
        train_y,
        test_y,
    )
