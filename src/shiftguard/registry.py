"""Which ShiftGuard variant is the reference model, and its ablations.

The reference model was selected on validation data only (see
scripts/validation_model_selection.py): disabling reliability-weighted graph
propagation gave the lowest stressed validation MAE averaged over both
datasets. Artifact directories keep their original variant names.
"""

from __future__ import annotations

REFERENCE_MODEL = "shiftguard_noreliability"
# First version of ShiftGuard, i.e. the reference plus reliability weighting.
LEGACY_MODEL = "shiftguard"

# Ablations of the reference model; the legacy model is its "+ reliability" row.
REFERENCE_ABLATIONS = (
    LEGACY_MODEL,
    *(
        f"{REFERENCE_MODEL}_{flag}"
        for flag in (
            "staticgraph",
            "adaptivegraph",
            "norouter",
            "noaugmentation",
            "independentaugmentation",
        )
    ),
)
# Ablations of the legacy model, run before the reference was selected.
LEGACY_ABLATIONS = (
    "shiftguard_noreliability",
    "shiftguard_staticgraph",
    "shiftguard_adaptivegraph",
    "shiftguard_norouter",
    "shiftguard_noaugmentation",
    "shiftguard_independentaugmentation",
)
SHIFTGUARD_FLAGS = (
    "noreliability",
    "staticgraph",
    "adaptivegraph",
    "norouter",
    "noaugmentation",
    "independentaugmentation",
)


def shiftguard_flags(name: str) -> set[str]:
    """Parse ``shiftguard[_flag...]`` into its set of ablation flags."""

    if name != "shiftguard" and not name.startswith("shiftguard_"):
        raise ValueError(name)
    flags = set(name.split("_")[1:])
    unknown = flags - set(SHIFTGUARD_FLAGS)
    if unknown:
        raise ValueError(f"unknown ShiftGuard flags {sorted(unknown)} in {name}")
    if {"staticgraph", "adaptivegraph"} <= flags:
        raise ValueError(name)
    return flags
