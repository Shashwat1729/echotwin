"""Dataset package: construction, splitting, retrieval, manifests."""

from .builder import (
    BuiltDataset,
    ChatTurn,
    DatasetConfig,
    ExampleMetadata,
    TrainingExample,
    build_examples,
    compute_dataset_stats,
    is_low_information,
    load_jsonl,
    save_jsonl,
    segment_messages,
)
from .retrieval import (
    BM25Index,
    RetrievalConfig,
    StyleExemplar,
    build_index_from_examples,
    build_index_from_messages,
    retrieve_exemplars,
)
from .splits import (
    DatasetManifest,
    DatasetSplit,
    SplitConfig,
    SplitReport,
    make_manifest,
    split_dataset,
)

__all__ = [
    "BM25Index",
    "BuiltDataset",
    "ChatTurn",
    "DatasetConfig",
    "DatasetManifest",
    "DatasetSplit",
    "ExampleMetadata",
    "RetrievalConfig",
    "SplitConfig",
    "SplitReport",
    "StyleExemplar",
    "TrainingExample",
    "build_examples",
    "build_index_from_examples",
    "build_index_from_messages",
    "compute_dataset_stats",
    "is_low_information",
    "load_jsonl",
    "make_manifest",
    "retrieve_exemplars",
    "save_jsonl",
    "segment_messages",
    "split_dataset",
]
