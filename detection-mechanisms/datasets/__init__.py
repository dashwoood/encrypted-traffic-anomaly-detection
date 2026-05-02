"""Dataset preparation: canonical schema, Kaggle download, benchmark normalization."""
from .schema import CANONICAL_FEATURES, CANONICAL_LABEL, CANONICAL_NUM_FEATURES, canonical_header
from .prepare import prepare_datasets
from .sequences import prepare_sequence_data, flows_to_sequences

__all__ = [
    "CANONICAL_FEATURES",
    "CANONICAL_LABEL",
    "CANONICAL_NUM_FEATURES",
    "canonical_header",
    "prepare_datasets",
    "prepare_sequence_data",
    "flows_to_sequences",
]
