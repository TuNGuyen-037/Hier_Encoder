# models/__init__.py
from .gru4rec import GRU4Rec
from .hier_gru import HierGRU
from .sasrec import SASRec
from .hier_sasrec import HierSASRec
from .nextitnet import NextItNet
from .hier_nextitnet import HierNextItNet


MODEL_REGISTRY = {
    "gru4rec": GRU4Rec,
    "hier_gru": HierGRU,
    "sasrec": SASRec,
    "hier_sasrec": HierSASRec,
    "nextitnet": NextItNet,
    "hier_nextitnet": HierNextItNet,
}


ALL_MODELS = list(MODEL_REGISTRY.keys())


__all__ = [
    "GRU4Rec",
    "HierGRU",
    "SASRec",
    "HierSASRec",
    "NextItNet",
    "HierNextItNet",
    "MODEL_REGISTRY",
    "ALL_MODELS",
]
