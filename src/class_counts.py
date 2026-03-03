import pandas as pd
from pathlib import Path


LABEL_COLUMN = "SBS"
CLASS_ORDER = ["unburned", "low", "moderate", "high"]


def normalize_label(label: str) -> str:
    """
    Normalize raw class labels to one of:
    'unburned', 'low', 'moderate', 'high'
    """
    if label is None:
        return "unknown"

    s = str(label).strip().lower()

    # Handle numeric encodings if they appear
    numeric_map = {
        "0": "unburned",
        "1": "low",
        "2": "moderate",
        "3": "high",
    }
    if s in numeric_map:
        return numeric_map[s]

    # Handle common text variants
    if s == "mod":
        return "moderate"

    if s in CLASS_ORDER:
        return s

    return s or "unknown"


def compute_class_counts(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found in {csv_path}")

    labels = df[LABEL_COLUMN].map(normalize_label)
    counts = labels.value_counts()

    # Reindex to consistent order, keep any unexpected labels at the end
    ordered = counts.reindex(CLASS_ORDER).fillna(0).astype(int)

    # Append any remaining labels not in CLASS_ORDER
    for label, count in counts.items():
        if label not in CLASS_ORDER:
            ordered.loc[label] = count

    return ordered


def print_class_counts(csv_path: Path, title: str | None = None) -> None:
    path = Path(csv_path)
    if title is None:
        title = path.name

    print(f"\n===== Class counts for {title} =====")
    counts = compute_class_counts(path)
    total = counts.sum()

    for cls in counts.index:
        print(f"{cls:9}: {counts[cls]:6d}")
    print(f"Total: {total}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    datasets = [
        (
            "Real fires (original points)",
            project_root / "data" / "real_all_fires_complete_covariates_fixed_1229.csv",
        ),
        (
            "Real fires (upsampled points)",
            project_root / "data" / "real_all_fires_upsampled_points_with_covariates_fixed.csv",
        ),
    ]

    for title, path in datasets:
        print_class_counts(path, title)


if __name__ == "__main__":
    main()

