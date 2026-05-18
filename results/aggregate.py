import csv
from pathlib import Path


RESULTS_DIR = Path("results/tables")
RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_result(model_name, metrics):
    """
    Save one model benchmark result.
    """
    ensure_results_dir()

    file_exists = RESULTS_FILE.exists()

    with open(
        RESULTS_FILE,
        mode="a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "model",
                    "hr@5",
                    "ndcg@5",
                    "hr@10",
                    "ndcg@10",
                    "hr@20",
                    "ndcg@20",
                ]
            )

        writer.writerow(
            [
                model_name,
                metrics.get("hr@5", 0.0),
                metrics.get("ndcg@5", 0.0),
                metrics.get("hr@10", 0.0),
                metrics.get("ndcg@10", 0.0),
                metrics.get("hr@20", 0.0),
                metrics.get("ndcg@20", 0.0),
            ]
        )


def load_results():
    """
    Load benchmark results from CSV.
    """
    ensure_results_dir()

    if not RESULTS_FILE.exists():
        return []

    rows = []

    with open(
        RESULTS_FILE,
        mode="r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def aggregate_results():
    """
    Return all benchmark rows.
    """
    return load_results()


def export_results_csv():
    """
    Return benchmark CSV path.
    """
    ensure_results_dir()
    return RESULTS_FILE
