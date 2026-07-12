"""Loaders, generators and windowing utilities for weather time series."""

from taranis.data.meteofrance import (
    CANAUX_MF,
    CANAUX_MF_RICH,
    CODES_ORAGE_WMO,
    build_storm_labels,
    build_storm_labels_combined,
    build_storm_labels_from_ww,
    prepare_station,
    read_synop_csv,
    resample_regular,
    summarize_stations,
)
from taranis.data.synthetic import generate
from taranis.data.windows import (
    CANAUX,
    WindowedDataset,
    channel_stats,
    chronological_split,
    make_windows,
    normalize,
)

__all__ = [
    "CANAUX",
    "CANAUX_MF",
    "CANAUX_MF_RICH",
    "CODES_ORAGE_WMO",
    "WindowedDataset",
    "build_storm_labels",
    "build_storm_labels_combined",
    "build_storm_labels_from_ww",
    "channel_stats",
    "chronological_split",
    "generate",
    "make_windows",
    "normalize",
    "prepare_station",
    "read_synop_csv",
    "resample_regular",
    "summarize_stations",
]
