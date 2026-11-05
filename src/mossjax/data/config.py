import os
import tomllib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import cache as cache_mod
from .buckets import BucketManager

DEFAULT_RESOLUTION = (1024, 1024)


@dataclass
class SubsetConfig:
    image_dir: str
    num_repeats: int = 1
    caption_dropout_rate: Optional[float] = None
    caption_extension: str = ".txt"
    metadata_file: Optional[str] = None
    flip_aug: bool = False


@dataclass
class DatasetSpec:
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION
    batch_size: int = 1
    enable_bucket: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = False
    subsets: List[SubsetConfig] = field(default_factory=list)


@dataclass
class DatasetConfig:
    datasets: List[DatasetSpec]
    general: Dict[str, Any]
    path: str


def parse_resolution(value) -> Tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError(f"invalid resolution: {value!r}")
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) == 1:
            return (int(parts[0]), int(parts[0]))
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"invalid resolution: {value!r}")


def _resolve_path(base: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def load_dataset_config(path: str) -> DatasetConfig:
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    general = {key: value for key, value in raw.items() if key != "datasets"}
    general.update(raw.get("general", {}))

    datasets: List[DatasetSpec] = []
    for entry in raw.get("datasets", []):
        settings = {**general, **{key: value for key, value in entry.items() if key != "subsets"}}
        spec = DatasetSpec(
            resolution=parse_resolution(settings.get("resolution", DEFAULT_RESOLUTION)),
            batch_size=int(settings.get("batch_size", 1)),
            enable_bucket=bool(settings.get("enable_bucket", False)),
            min_bucket_reso=int(settings.get("min_bucket_reso", 256)),
            max_bucket_reso=int(settings.get("max_bucket_reso", 1024)),
            bucket_reso_steps=int(settings.get("bucket_reso_steps", 64)),
            bucket_no_upscale=bool(settings.get("bucket_no_upscale", False)),
        )
        for subset_entry in entry.get("subsets", []):
            if "image_dir" not in subset_entry:
                raise ValueError(f"subset without image_dir in {path}")
            dropout = subset_entry.get("caption_dropout_rate", settings.get("caption_dropout_rate"))
            spec.subsets.append(
                SubsetConfig(
                    image_dir=_resolve_path(base, str(subset_entry["image_dir"])),
                    num_repeats=int(subset_entry.get("num_repeats", 1)),
                    caption_dropout_rate=None if dropout is None else float(dropout),
                    caption_extension=str(subset_entry.get("caption_extension", settings.get("caption_extension", ".txt"))),
                    metadata_file=None
                    if subset_entry.get("metadata_file") is None
                    else _resolve_path(base, str(subset_entry["metadata_file"])),
                    flip_aug=bool(subset_entry.get("flip_aug", False)),
                )
            )
        datasets.append(spec)

    if not datasets:
        raise ValueError(f"no [[datasets]] entries found in {path}")
    return DatasetConfig(datasets=datasets, general=general, path=path)


def build_records(dataset_config: DatasetConfig) -> List[Dict]:
    records: List[Dict] = []
    for dataset in dataset_config.datasets:
        manager = None
        if dataset.enable_bucket:
            manager = BucketManager(
                dataset.bucket_no_upscale,
                dataset.resolution,
                dataset.min_bucket_reso,
                dataset.max_bucket_reso,
                dataset.bucket_reso_steps,
            )
            manager.make_buckets()
        for subset in dataset.subsets:
            if not os.path.isdir(subset.image_dir):
                raise FileNotFoundError(f"image_dir does not exist: {subset.image_dir}")
            pairs = cache_mod.find_cache_pairs(
                subset.image_dir, num_repeats=subset.num_repeats, caption_dropout_rate=subset.caption_dropout_rate
            )
            for record in pairs:
                record["flip_aug"] = subset.flip_aug
                record["subset"] = subset
                if manager is not None:
                    size = cache_mod.read_latents_original_size(record["latents"][0])
                    if size is not None:
                        reso, _, _ = manager.select_bucket(int(size[0]), int(size[1]))
                        record["bucket"] = (int(reso[0]), int(reso[1]))
                records.append(record)
    if not records:
        raise ValueError("no cached samples found for the dataset config")
    return records
