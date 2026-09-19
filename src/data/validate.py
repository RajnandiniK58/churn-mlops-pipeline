"""Validate a DataFrame against a schema; used as a DVC stage and importable."""
import json
import sys

import pandas as pd
import pandera.pandas as pa

from src.config import RAW_DATA_PATH, REPORTS_DIR
from src.data.schemas import CLEAN_SCHEMA, RAW_SCHEMA


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema = RAW_SCHEMA) -> pd.DataFrame:
    """Return df if valid; raise SchemaErrors listing EVERY problem if not."""
    return schema.validate(df, lazy=True)


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    return validate(df, CLEAN_SCHEMA)


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "validation.json"

    try:
        validate(df, RAW_SCHEMA)
    except pa.errors.SchemaErrors as err:
        print("[validate] FAILED. Problems found:\n")
        print(err.failure_cases.to_string())
        report_path.write_text(json.dumps({"passed": 0, "rows": len(df)}, indent=2))
        sys.exit(1)  # non-zero exit => `dvc repro` stops here

    report = {"passed": 1, "rows": len(df), "columns": df.shape[1],
              "churn_rate": round(float((df["Churn"] == "Yes").mean()), 4)}
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[validate] OK -> {report}")


if __name__ == "__main__":
    main()
