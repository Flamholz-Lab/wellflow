# wellflow

A Python package for processing plate reader growth data and estimating microbial growth rates.

`wellflow` provides a reproducible pipeline for:
- Converting plate reader exports into tidy data
- Blank correction and smoothing
- Growth rate estimation via rolling log-linear regression
- Extracting maximum growth rates per well or experimental condition

---

## Installation

```bash
pip install git+https://github.com/flamholz-lab/wellflow.git
```

Requires Python ≥ 3.11.

---

## Getting the examples

After installing, copy the example notebooks and data files to your current directory:

```python
import wellflow as wf
wf.copy_examples()
```

This gives you `workflow.ipynb` (full pipeline walkthrough) and `plot.ipynb` (plotting), along with the example data files they use. Open and run them from the same directory.

---

## Quick start

```python
import wellflow as wf

# 1. Load plate reader output (Synergy H1, wide format)
raw = wf.read_plate_measurements(
    path="plate_reader_output.xlsx",
    reader_model="Synergy H1",
    data_format="wide",
    timepoint_cols=["Time", "T° 600"],
    header_row=44,
    last_row=237,
    start_col="B",
)

# 2. Load plate layout and merge
layout = wf.read_plate_layout(path="plate_layout.xlsx", format="column_blocks")
df = wf.merge_with_layout(measurements=raw, layout=layout)

# 3. Remove bad wells (optional)
df = wf.drop_flags(df, flags="flagged_wells.xlsx")

# 4. Blank correction and smoothing
df = wf.add_blank_correction(df, window=4)
df = wf.add_smoothed_od(df, window=5)

# 5. Per-timepoint growth rate
df = wf.add_growth_rate(df, window=9)

# 6. Maximum growth rate per well
mu_max = wf.compute_mu_max(df, window=8)
```

For a detailed walkthrough of each step and plotting, see the example notebooks (`wf.copy_examples()`).
