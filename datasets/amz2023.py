import gzip
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Abstract base dataset for Amazon review + metadata datasets.

    Expected files:
        - review_file: review JSONL.GZ
        - meta_file: metadata JSONL.GZ
    """

    def __init__(
        self,
        review_file: str,
        meta_file: str,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Args:
            review_file: Path to Amazon review file.
            meta_file: Path to Amazon metadata file.
            max_samples: Optional sample limit for debugging.
        """
        self.review_file = Path(review_file)
        self.meta_file = Path(meta_file)
        self.max_samples = max_samples

        self.reviews = self._load_jsonl_gz(self.review_file)
        self.meta = self._load_jsonl_gz(self.meta_file)

        self._build_index()

    def _load_jsonl_gz(self, file_path: Path) -> List[Dict]:
        """
        Load compressed JSONL.GZ file.
        """
        data = []

        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data.append(json.loads(line))

                if self.max_samples and idx + 1 >= self.max_samples:
                    break

        return data

    def _build_index(self) -> None:
        """
        Build metadata lookup by parent ASIN.
        """
        self.meta_lookup = {}

        for item in self.meta:
            asin = item.get("parent_asin")
            if asin:
                self.meta_lookup[asin] = item

    def get_review_with_meta(self, idx: int) -> Dict:
        """
        Return merged review + metadata record.
        """
        review = self.reviews[idx]
        parent_asin = review.get("parent_asin")

        meta = self.meta_lookup.get(parent_asin, {})

        return {
            "review": review,
            "meta": meta,
        }

    def __len__(self) -> int:
        return len(self.reviews)

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Must be implemented in child dataset class.
        """
        pass
