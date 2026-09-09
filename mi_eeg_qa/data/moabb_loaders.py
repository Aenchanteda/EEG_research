"""Optional MOABB/MNE loaders with graceful dependency skips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MOABBRequest:
    dataset: str = "BNCI2014_001"  # BCIC IV 2a in MOABB naming.
    subjects: Sequence[int] = (1,)
    paradigm: str = "left_right_hand"


def _require_moabb():
    try:
        from moabb.datasets import BNCI2014_001, BNCI2014_004, PhysionetMI  # type: ignore
        from moabb.paradigms import MotorImagery, LeftRightImagery  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "MOABB/MNE loaders require optional dependencies. Install with `pip install -e .[moabb]`."
        ) from exc
    return {
        "BNCI2014_001": BNCI2014_001,
        "BNCI2014_004": BNCI2014_004,
        "PhysionetMI": PhysionetMI,
        "MotorImagery": MotorImagery,
        "LeftRightImagery": LeftRightImagery,
    }


def load_moabb_epochs(request: MOABBRequest):
    """Load epochs through MOABB when optional dependencies are installed.

    This function intentionally returns MOABB's `(X, y, metadata)` tuple and avoids
    becoming a hard dependency for tests or smoke runs.
    """

    mods = _require_moabb()
    if request.dataset not in {"BNCI2014_001", "BNCI2014_004", "PhysionetMI"}:
        raise ValueError(f"Unsupported dataset {request.dataset!r}")
    dataset = mods[request.dataset]()
    paradigm_cls = mods["LeftRightImagery"] if request.paradigm == "left_right_hand" else mods["MotorImagery"]
    paradigm = paradigm_cls()
    return paradigm.get_data(dataset=dataset, subjects=list(request.subjects))
