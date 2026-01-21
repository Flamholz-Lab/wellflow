from .plate import (
    convert_excel_col_to_index,
    read_plate_measurements,
    read_plate_layout,
    merge_measurements_and_conditions,
    read_flagged_wells,
    add_flag_column,
    drop_flags,
    drop_col,
    drop_row,
    drop_well,
    with_blank_corrected_od,
    with_smoothed_od,
    add_growth_rate,
    estimate_early_od_threshold,
    summarize_mu_max,
)

# Backwards-compatible aliases for older API names
add_blank_value = with_blank_corrected_od
add_smoothed_od = with_smoothed_od
mu_max_create = summarize_mu_max
read_plate_design = read_plate_layout

__all__ = [
    "convert_excel_col_to_index",
    "read_plate_measurements",
    "read_plate_layout",
    "read_plate_design",
    "merge_measurements_and_conditions",
    "read_flagged_wells",
    "add_flag_column",
    "drop_flags",
    "drop_col",
    "drop_row",
    "drop_well",
    "with_blank_corrected_od",
    "with_smoothed_od",
    "add_smoothed_od",
    "add_blank_value",
    "add_growth_rate",
    "estimate_early_od_threshold",
    "summarize_mu_max",
    "mu_max_create",
]
