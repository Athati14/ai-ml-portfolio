"""Population Stability Index drift check against the training reference."""
import numpy as np, pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e = np.histogram(expected, quantiles)[0] / len(expected)
    a = np.histogram(actual, quantiles)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def report(reference_path: str, live: pd.DataFrame, threshold: float = 0.2) -> dict:
    ref = pd.read_parquet(reference_path)
    out = {}
    for col in ref.columns:
        if col in live.columns:
            score = psi(ref[col].values, live[col].values)
            out[col] = {"psi": round(score, 4), "drift": score > threshold}
    return out


if __name__ == "__main__":
    ref = pd.read_parquet("reference_features.parquet")
    print(report("reference_features.parquet", ref.sample(frac=0.5, random_state=1)))
