"""
Standalone loss functions for distillation methods.

Available modules:
- ``taid``: Timestep-Aware Interpolation Distillation
"""

__all__ = [
    "compute_taid_loss",
]


def __getattr__(name):
    """Lazy imports to avoid errors when not all loss modules exist yet."""
    if name == "compute_taid_loss":
        from .taid import compute_taid_loss

        return compute_taid_loss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
