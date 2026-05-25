from pathlib import Path
import urllib.request


REVIEW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/"
    "data/amazon_2023/raw/review_categories/"
    "Cell_Phones_and_Accessories.jsonl.gz"
)

META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/"
    "data/amazon_2023/raw/meta_categories/"
    "meta_Cell_Phones_and_Accessories.jsonl.gz"
)


def download_file(url, save_path):
    if save_path.exists():
        print(f"Already exists: {save_path}")
        return

    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved to: {save_path}")


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    review_path = raw_dir / "Cell_Phones_and_Accessories.jsonl.gz"
    meta_path = raw_dir / "meta_Cell_Phones_and_Accessories.jsonl.gz"

    download_file(REVIEW_URL, review_path)
    download_file(META_URL, meta_path)

    print("Download completed.")


if __name__ == "__main__":
    main()
