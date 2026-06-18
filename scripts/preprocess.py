# scripts/preprocess.py
from datasets.preprocess import preprocess_and_save


def main():
    preprocess_and_save()
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
