import pandera.pandas as pa
import pandas as pd


APPLICATION_SCHEMA = pa.DataFrameSchema(
    {
        "SK_ID_CURR": pa.Column(int, pa.Check.ge(0), nullable=False, unique=True),
        "TARGET": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    coerce=True,
    strict=False,
)


def validate_application_data(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Application data is empty")
    return APPLICATION_SCHEMA.validate(frame, lazy=True)
