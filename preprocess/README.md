# Preprocessing Pipeline – Azure VM Criticality Project

This folder contains the **end-to-end preprocessing pipeline** that turns ~119 GB of raw Azure VM logs into a compact, ML-ready dataset with:

- one row per **VM request**
- a binary label `critical` (0/1)
- **user/tenant history** features at request time
- a **time-based train/val/test split**

Run the notebooks in order:

> `00_config_and_utils` → `01_build_vm_static` → `02_aggregate_vm_cpu` →  
> `03_build_vm_with_label` → `04_tenant_history_features` → `05_create_splits_and_eda`

---

## 1. Directory Layout & Raw Data

Assume this folder is `root/preprocess/`.

```text
preprocess/
  README.md                 # this file
  01_build_vm_static.ipynb
  02_aggregate_vm_cpu.ipynb
  03_build_vm_with_label.ipynb
  04_tenant_history_features.ipynb
  05_create_splits_and_eda.ipynb
  data_raw/
    subscriptions.csv.gz    # or subscriptions/subscriptions.csv.gz
    deployment.csv.gz       # or deployment/deployment.csv.gz
    vmtable.csv.gz          # or vmtable/vmtable.csv.gz
    vm_cpu/                 # 125 vm CPU shards (or vm_cpu_readings/)
      part-00000.csv.gz
      ...
  data_intermediate/        # created by notebooks
  data_final/               # created by notebooks
```

### Raw data placement

Download the Azure Public Dataset CSVs and place them under `preprocess/data_raw/`:

* `subscriptions.csv.gz` – subscription-level info
  (or `subscriptions/subscriptions.csv.gz`; adjust paths in `01_build_vm_static` if needed)
* `deployment.csv.gz` – deployment-level info
  (or `deployment/deployment.csv.gz`)
* `vmtable.csv.gz` – VM-level metadata
  (or `vmtable/vmtable.csv.gz`)
* `vm_cpu/` – folder containing all ~125 `csv.gz` shards of VM CPU readings

  * By default, `02_aggregate_vm_cpu` expects `vm_cpu_readings/`.

    * **Option A:** rename your folder to `vm_cpu_readings`.
    * **Option B:** edit `CPU_DIR = RAW_DATA_DIR / "vm_cpu_readings"` to `"vm_cpu"`.

No need to unzip the `.csv.gz` files; Polars reads gzipped CSVs directly.

---

## 2. Notebooks & Workflow

### 01_build_vm_static.ipynb – static VM table

**Input (from `data_raw/`)**

* `subscriptions.csv.gz`
* `deployment.csv.gz`
* `vmtable.csv.gz`

**What it does**

* Reads subscription, deployment, and VM tables.
* Interprets integer timestamps as seconds since window start.
* Adds VM-level features:

  * `day_idx` (0–29), `hour_of_day` (0–23) based on `ts_vm_created`
  * `lifetime_sec`, `lifetime_hours`
  * `vm_mem_per_core`
* Joins deployment and subscription info:

  * `deployment_size`, `log_deployment_size`
  * `ts_first_vm_created`, `count_vms_created`
  * `sub_first_day`, `sub_first_hour`

**Output (in `data_intermediate/`)**

* `vm_static.parquet`
  One row per VM, static/request-time features only.

**Knobs to tune**

* None critical here; this is mostly fixed feature engineering.

---

### 02_aggregate_vm_cpu.ipynb – CPU log aggregation (119 GB → per-VM stats)

**Input (from `data_raw/`)**

* `vm_cpu/` (or `vm_cpu_readings/`), containing ~125 gzipped CSV shards with:

  * `timestamp`, `vm_id`, `min_cpu`, `max_cpu`, `avg_cpu`

**What it does**

Two-pass aggregation with progress bars:

1. **Pass 1 (per-shard partials)**

   * For each CSV shard:

     * compute per-VM partial aggregates:

       * counts, sums, sq-sums:

         * `n_readings`, `sum_avg`, `sum_avg_sq`
       * extremes:

         * `max_cpu`
       * counts above thresholds:

         * `cnt_gt_60`, `cnt_gt_80` (avg CPU > 60/80%)
       * day/night sums and counts:

         * `sum_day`, `cnt_day`, `sum_night`, `cnt_night`
       * per-hour sums and counts (0–23):

         * `sum_hour_h`, `cnt_hour_h`
     * write each shard’s result to `data_intermediate/cpu_parts/cpu_part_XXX.parquet`.

2. **Pass 2 (combine partials)**

   * Reads all `cpu_part_*.parquet`, aggregates by `vm_id` again.
   * Computes final per-VM usage features:

     * `cpu_mean`, `cpu_std`
     * `cpu_frac_gt_60`, `cpu_frac_gt_80`
     * `day_cpu_mean`, `night_cpu_mean`, `day_night_ratio`
     * `cpu_hour_0_mean` … `cpu_hour_23_mean` (hourly means)

**Output (in `data_intermediate/`)**

* `cpu_parts/cpu_part_*.parquet` – intermediate per-shard partials.
* `vm_usage_agg.parquet` – final per-VM CPU/diurnal stats.

**Knobs to tune**

* None for label semantics; this is pure aggregation.
* You can change which hours are considered “day” vs “night” inside this notebook if needed.

---

### 03_build_vm_with_label.ipynb – join static + usage + label

**Input**

* `data_intermediate/vm_static.parquet`
* `data_intermediate/vm_usage_agg.parquet`

**What it does**

* Joins static VM features with usage features on `vm_id`.
* Filters out VMs with too few readings:

  * `n_readings >= MIN_READINGS` (default 12).
* Computes label components:

  * `long_lived` from `lifetime_hours` (≥ `LONG_LIVED_HOURS`)
  * `sustained_high` from `cpu_frac_gt_60` and/or `p95_max_cpu`
  * `periodic_enough` from the hourly CPU means (`cpu_hour_*_mean`), using a simple FFT energy ratio (24h vs 12h/8h)
  * `strong_diurnal` (legacy, still kept) from `day_night_ratio` and `day_cpu_mean`
* Defines binary label:

  * `critical = periodic_enough & (long_lived | sustained_high)` cast to `0/1`.

**Output**

* `data_intermediate/vm_full_labeled.parquet`
  One row per VM, static + usage features + `critical` label and components.

**Knobs to tune (important)**

* `MIN_READINGS` – how many CPU readings a VM must have to be kept.
* Label thresholds:

  * `LONG_LIVED_HOURS` (default 24h)
  * `THRESH_FRAC_GT_60` (fraction of time with >60% CPU)
  * `THRESH_P95` (CPU p95 threshold; remember to divide percents by 100)
  * `PERIODICITY_MIN_ENERGY`, `PERIODICITY_RATIO_8_12` (24h energy dominance)
  * `THRESH_DAY_NIGHT_RATIO`, `THRESH_DAY_MEAN` (legacy `strong_diurnal`, kept for analysis)
* Changing these thresholds **directly changes** the positive rate of `critical` VMs.

---

### 04_tenant_history_features.ipynb – time-aware tenant (user) history

**Input**

* `data_intermediate/vm_full_labeled.parquet`

**What it does**

* Sorts by `subscription_id`, then `ts_vm_created` (VM creation time).
* Uses Polars window + cumulative sums to compute, for each VM, **history before this VM**:

  * `hist_n_vms` – number of previous VMs for this tenant
  * `hist_n_critical` – number of previous critical VMs
  * `hist_critical_frac` – `hist_n_critical / hist_n_vms`
  * `hist_lifetime_mean`, `hist_lifetime_std`
  * `hist_cpu_mean_mean`
  * `hist_p95_mean`
  * `hist_frac_gt60_mean`
  * `hist_day_night_ratio_mean`
  * `hist_has_past` – 1 if `hist_n_vms > 0`, else 0
* Optionally keeps only VMs with **enough history**:

  * `hist_n_vms >= K_MIN_HISTORY` (default 3).

**Output (in `data_final/`)**

* `vm_request_table_all.parquet`
  All VMs with history features (even if `hist_n_vms == 0`).
* `vm_request_table.parquet`
  Filtered to VMs whose tenant has at least `K_MIN_HISTORY` previous VMs.

**Knobs to tune (important)**

* `K_MIN_HISTORY` – how many previous VMs a tenant must have for the VM to be included in the main dataset.

  * Lower `K_MIN_HISTORY` → more rows with weaker history.
  * Higher `K_MIN_HISTORY` → fewer rows but stronger history signals.

---

### 05_create_splits_and_eda.ipynb – time-based splits & sanity checks

**Input**

* `data_final/vm_request_table.parquet`

**What it does**

* Computes `day_idx = ts_vm_created // 86400`.
* Assigns **time-based splits**:

  * `train`: `day_idx <= 19` (first 20 days)
  * `val`:   `20 <= day_idx <= 24`
  * `test`:  `day_idx >= 25` (last 5 days)
* Writes combined and per-split Parquet files.
* Prints basic sanity checks:

  * class balance (`critical` rate) per split
  * unique tenants per split
  * simple feature distributions (e.g., `vm_category`, `vm_virtual_core_count`).

**Output (in `data_final/`)**

* `vm_request_table_with_split.parquet`
  Main ML dataset; one row per VM, with:

  * static features
  * usage features
  * history features
  * label `critical`
  * `split ∈ {"train","val","test"}`
* `vm_train.parquet`, `vm_val.parquet`, `vm_test.parquet`
  Convenience per-split files.

**Knobs to tune**

* Split boundaries:

  * if you want different durations (e.g. 70/15/15 split), change the `day_idx` conditions.
* You can also introduce **earlier/late** splits if you want extra validation schemes.

---

## 3. Using the final dataset

Typical usage in a modeling notebook:

```python
import polars as pl

df = pl.read_parquet("data_final/vm_request_table_with_split.parquet")

train = df.filter(pl.col("split") == "train")
val   = df.filter(pl.col("split") == "val")
test  = df.filter(pl.col("split") == "test")

# Drop IDs and high-leakage columns before training
X_train = train.drop(["vm_id", "subscription_id", "deployment_id", "critical", "split"])
y_train = train["critical"]
```

From here, you can feed `X_train`, `y_train` into sklearn / XGBoost / LightGBM, using **class weights** to handle the imbalance in `critical`.
