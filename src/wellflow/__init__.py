import shutil
from pathlib import Path


def copy_examples(destination="."):
    """Copy example notebooks and data files to a local directory.

    Args:
        destination: Directory to copy examples into. Defaults to current directory.
    """
    src = Path(__file__).parent / "examples"
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, dest / f.name)
    print(f"Examples copied to {dest.resolve()}")


from .io import (
    convert_excel_col_to_index,
    read_plate_measurements,
    read_plate_layout,
    merge_with_layout,
)
from .qc import (
    read_flagged_wells,
    add_flag_column,
    drop_flags,
    drop_col,
    drop_row,
    drop_well,
)
from .preprocess import (
    add_blank_correction,
    add_smoothed_od,
)
from .growth import (
    add_growth_rate,
    estimate_od_threshold,
    compute_mu_max,
)

__all__ = [
    "copy_examples",
    "convert_excel_col_to_index",
    "read_plate_measurements",
    "read_plate_layout",
    "merge_with_layout",
    "read_flagged_wells",
    "add_flag_column",
    "drop_flags",
    "drop_col",
    "drop_row",
    "drop_well",
    "add_blank_correction",
    "add_smoothed_od",
    "add_growth_rate",
    "estimate_od_threshold",
    "compute_mu_max",
]
