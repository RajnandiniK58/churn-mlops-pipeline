"""Optional: print what the encode + scale pipeline produces."""
import pandas as pd

from src.config import TARGET, TRAIN_PATH
from src.features.preprocess import build_preprocessor

train = pd.read_csv(TRAIN_PATH)
X = train.drop(columns=[TARGET])

pre = build_preprocessor()
X_t = pre.fit_transform(X)

print("Before:", X.shape, "| After:", X_t.shape)
print("\nOutput columns:")
for i, c in enumerate(X_t.columns):
    print(f"  {i:2d}. {c}")
print("\nScaled columns (mean ~0, std ~1):")
print(X_t[["tenure", "MonthlyCharges", "TotalCharges", "n_addon_services"]].describe().loc[["mean", "std"]].round(3))
print("\nFirst 3 rows:")
print(X_t.head(3).T.round(2).to_string())
