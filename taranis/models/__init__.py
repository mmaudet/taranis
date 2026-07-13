"""Taranis models: physics baseline and TS-JEPA."""

from taranis.models.baseline_3ch import Baseline3ch, Baseline3chHGB
from taranis.models.baseline_enriched import BaselineEnriched, BaselineEnrichedHGB
from taranis.models.baseline_physics import BaselinePhysics
from taranis.models.tsjepa import (
    TSJEPA,
    EMAWrapper,
    PatchEmbed,
    TransformerBlock,
    TransformerEncoder,
    TSJEPAConfig,
    embedding_stats,
    jepa_loss,
    sample_block_mask,
)

__all__ = [
    "Baseline3ch",
    "Baseline3chHGB",
    "BaselineEnriched",
    "BaselineEnrichedHGB",
    "BaselinePhysics",
    "EMAWrapper",
    "PatchEmbed",
    "TransformerBlock",
    "TransformerEncoder",
    "TSJEPA",
    "TSJEPAConfig",
    "embedding_stats",
    "jepa_loss",
    "sample_block_mask",
]
