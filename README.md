# Azure VM Criticality – Project Overview

This repository implements our full pipeline for predicting whether a new VM from a tenant is **“critical”** based on:

- its **VM configuration & type**
- its **aggregated CPU behavior**
- the tenant’s **historical VM behavior**, using only information available **up to request time**
- a **time-based train/val/test split** over a 30-day window.

The heavy preprocessing lives under [`preprocess/`](./preprocess), and produces a single ML-ready table:

> **`preprocess/data_final/vm_request_table_with_split.parquet`**

Each row in this final dataset is **one VM request**.

- ~**N rows** (depends on filters like `MIN_READINGS` and `K_MIN_HISTORY`)
- **70 columns**:
  - identifiers & timing
  - static VM metadata
  - coarse CPU stats (from Azure vmtable)
  - detailed CPU usage & diurnal features (from vm_cpu shards)
  - label components + final `critical` label
  - tenant history features
  - `split` (train/val/test)

If you want details on the preprocessing steps and intermediate files, see [`preprocess/README.md`](./preprocess/README.md).

---

## Final Dataset: Column Dictionary

### Location

Final modeling dataset:

```text
preprocess/data_final/vm_request_table_with_split.parquet
```

Each row: one VM request at creation time; label uses full 30-day behavior, features use only data available at request time (plus pre-window tenant info).

---

### 1. Identifiers & timestamps

* **`vm_id`**

  * **Type:** `str` (hashed VM ID)
  * **Range:** arbitrary hash strings
  * **Role:** unique VM identifier; drop before training.

* **`subscription_id`**

  * **Type:** `str` (hashed tenant/user ID)
  * **Range:** arbitrary hash strings
  * **Role:** groups VMs by tenant; used to define tenant history.

* **`deployment_id`**

  * **Type:** `str` (hashed deployment ID)
  * **Range:** arbitrary hash strings
  * **Role:** groups VMs launched as part of the same deployment/batch.

* **`ts_vm_created`**

  * **Type:** `int64` (seconds)
  * **Range:** `[0, ~2_592_000]` (0 to ~30 days)
  * **Role:** VM creation time relative to dataset window start.

* **`ts_vm_deleted`**

  * **Type:** `int64` (seconds)
  * **Range:** `[0, ~2_592_000]`
  * **Role:** VM deletion time; if missing, treated as active until window end.

* **`ts_first_vm_created`**

  * **Type:** `int64` (seconds)
  * **Range:** `[0, ~2_587_200]`
  * **Role:** first-ever VM creation time for this subscription.

* **`day_idx`**

  * **Type:** `int32`
  * **Range:** `[0, 29]`
  * **Definition:** `floor(ts_vm_created / 86400)`
  * **Role:** day index within the 30-day window; used for time split.

* **`hour_of_day`**

  * **Type:** `int32`
  * **Range:** `[0, 23]`
  * **Definition:** `(ts_vm_created % 86400) / 3600`
  * **Role:** hour of day when the VM was requested.

* **`lifetime_sec`**

  * **Type:** `int64`
  * **Range:** `[0, ~2_592_000]`
  * **Definition:** `max(ts_vm_deleted – ts_vm_created, 0)` (clipped to ≥ 0).

* **`lifetime_hours`**

  * **Type:** `float64`
  * **Range:** `[0, ~720]` (30 days)
  * **Definition:** `lifetime_sec / 3600`.

---

### 2. Static VM metadata & coarse stats

From `vmtable` and joined subscription/deployment tables:

* **`vm_category`**

  * **Type:** `str`
  * **Values:** `"Delay-insensitive"`, `"Interactive"`, `"Unknown"`
  * **Role:** basic workload type.

* **`vm_virtual_core_count`**

  * **Type:** `int64`
  * **Range:** positive integers
  * **Role:** number of virtual cores requested.

* **`vm_memory_gb`**

  * **Type:** `float64` (GB)
  * **Range:** `> 0`
  * **Role:** VM memory capacity.

* **`vm_mem_per_core`**

  * **Type:** `float64` (GB / core)
  * **Definition:** `vm_memory_gb / max(vm_virtual_core_count, 1)`
  * **Range:** `> 0`.

* **`max_cpu`**

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Source:** `vmtable` coarse max CPU.

* **`avg_cpu`**

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Source:** `vmtable` coarse average CPU.

* **`p95_max_cpu`**

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Role:** p95 of max CPU; used in “sustained high load” part of label.

* **`deployment_size`**

  * **Type:** `int64`
  * **Range:** `≥ 1`
  * **Role:** number of VMs in this deployment; proxy for batch size.

* **`log_deployment_size`**

  * **Type:** `float64`
  * **Definition:** `log(1 + deployment_size)`
  * **Range:** `[0, ln(1 + max_deployment_size)]`.

* **`count_vms_created`**

  * **Type:** `int64`
  * **Range:** `≥ 1`
  * **Role:** total number of VMs ever created for this subscription.

* **`sub_first_day`**

  * **Type:** `int32`
  * **Range:** `[0, 29]`
  * **Definition:** `ts_first_vm_created // 86400`.

* **`sub_first_hour`**

  * **Type:** `int32`
  * **Range:** `[0, 23]`
  * **Definition:** `(ts_first_vm_created % 86400) // 3600`.

---

### 3. Aggregated CPU usage features (per VM)

From fully aggregated `vm_cpu` readings (our `vm_usage_agg`):

* **`n_readings`**

  * **Type:** `int64`
  * **Range:** `≥ MIN_READINGS` (e.g. 12)
  * **Role:** number of CPU samples aggregated for this VM.

* **`max_cpu_right`**

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Source:** max of per-reading `max_cpu` from `vm_cpu` logs (distinct from `max_cpu` from `vmtable`).

* **`cpu_mean`**

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Definition:** mean of per-reading `avg_cpu` over this VM’s life.

* **`cpu_std`**

  * **Type:** `float64` (percent)
  * **Range:** `≥ 0`
  * **Role:** std dev of per-reading `avg_cpu`.

* **`cpu_frac_gt_60`**

  * **Type:** `float64`
  * **Range:** `[0, 1]`
  * **Definition:** fraction of readings where `avg_cpu > 60%`.

* **`cpu_frac_gt_80`**

  * **Type:** `float64`
  * **Range:** `[0, 1]`
  * **Definition:** fraction of readings where `avg_cpu > 80%`.

---

### 4. Diurnal & hourly CPU pattern features

* **`day_cpu_mean`**

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Definition:** mean `avg_cpu` during “day” hours (default: 08:00–19:59).

* **`night_cpu_mean`**

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Definition:** mean `avg_cpu` during “night” hours (default: 20:00–07:59).

* **`day_night_ratio`**

  * **Type:** `float64`
  * **Range:** `≥ 0` (≈ `day_cpu_mean / (night_cpu_mean + 1e-3)`)
  * **Role:** strength of day–night pattern; used in label.

* **`cpu_hour_0_mean` … `cpu_hour_23_mean`** (24 columns)

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Definition:** mean `avg_cpu` for each hour of day (UTC) across the VM’s life.
  * **Role:** fine-grained diurnal profile (also used to derive periodicity).

---

### 5. Label components & final label

These are computed in `03_build_vm_with_label` and used to define `critical`.

* **`long_lived`**

  * **Type:** `bool`
  * **Definition:** `lifetime_hours >= LONG_LIVED_HOURS` (e.g. ≥ 24h).

* **`sustained_high`**

  * **Type:** `bool`
  * **Definition:**

    * `(cpu_frac_gt_60 >= THRESH_FRAC_GT_60)` **OR**
    * `(p95_max_cpu / 100.0 >= THRESH_P95)`.

* **`periodicity_energy_24`**

  * **Type:** `float64`
  * **Definition:** fraction of FFT energy at the 24h frequency, using the hourly means vector.

* **`periodicity_ratio_24_vs_8_12`**

  * **Type:** `float64`
  * **Definition:** `energy_24h / max(energy_12h, energy_8h)` from the same FFT.

* **`periodic_enough`**

  * **Type:** `bool`
  * **Definition:** `(periodicity_energy_24 >= PERIODICITY_MIN_ENERGY) AND (periodicity_ratio_24_vs_8_12 >= PERIODICITY_RATIO_8_12)`.

* **`critical`**

  * **Type:** `int8`
  * **Values:** `{0, 1}`
  * **Definition:** `(periodic_enough AND (long_lived OR sustained_high))` cast to int.
  * **Role:** main supervised target.

---

### 6. Tenant (subscription) history features

Computed time-aware in `04_tenant_history_features`: for each VM, we look at **only earlier VMs** of the same subscription and summarize their behavior.

* **`hist_n_vms`**

  * **Type:** `int64`
  * **Range:** `≥ 0`
  * **Definition:** number of **previous** VMs for this subscription.

* **`hist_n_critical`**

  * **Type:** `int64`
  * **Range:** `≥ 0`
  * **Definition:** number of previous VMs labeled `critical`.

* **`hist_has_past`**

  * **Type:** `int8`
  * **Values:** `{0, 1}`
  * **Definition:** `1` if `hist_n_vms > 0`, else `0`.

* **`hist_critical_frac`**

  * **Type:** `float64`
  * **Range:** `[0, 1]`
  * **Definition:** `hist_n_critical / hist_n_vms` (0 if `hist_n_vms == 0`).

* **`hist_lifetime_mean`**

  * **Type:** `float64` (hours)
  * **Range:** `≥ 0`
  * **Definition:** mean `lifetime_hours` over previous VMs.

* **`hist_lifetime_std`**

  * **Type:** `float64` (hours)
  * **Range:** `≥ 0`
  * **Definition:** std dev of `lifetime_hours` over previous VMs.

* **`hist_cpu_mean_mean`**

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Definition:** mean `cpu_mean` over previous VMs.

* **`hist_p95_mean`**

  * **Type:** `float64` (percent)
  * **Range:** `[0, 100]`
  * **Definition:** mean `p95_max_cpu` over previous VMs.

* **`hist_frac_gt60_mean`**

  * **Type:** `float64`
  * **Range:** `[0, 1]`
  * **Definition:** mean `cpu_frac_gt_60` over previous VMs.

* **`hist_day_night_ratio_mean`**

  * **Type:** `float64`
  * **Range:** `≥ 0`
  * **Definition:** mean `day_night_ratio` over previous VMs.

---

### 7. Split column (time-based train/val/test)

* **`split`**

  * **Type:** `str`
  * **Values:** `"train"`, `"val"`, `"test"`
  * **Definition:** based on `day_idx`:

    * `train` if `day_idx <= 19`
    * `val` if `20 <= day_idx <= 24`
    * `test` if `day_idx >= 25`
  * **Role:** prevents temporal leakage; use this to slice data.

---

### Feature availability for **new incoming VMs**

For modeling “**will this new VM be critical?**” at request time, it’s crucial to separate:

- Features that are **available at VM arrival time** (what the scheduler can reasonably know)
- Features that are **post-hoc / offline only**, because they use the *future behavior of that VM* (and are used only to build labels or analyze behavior)

Below we list **every column** in `vm_request_table_with_split.parquet` and classify it.

---

#### Features we conceptually HAVE at VM arrival time

These can be known when the VM is requested, **without looking into that VM’s future**. Some are direct inputs, some are simple transforms, some are history over prior VMs of the same tenant.

**Identifiers & timing**

- `vm_id`  
  Unique VM identifier (we will **not** use as a predictive feature, but it exists).
- `subscription_id`  
  Tenant/user ID; used to define history, and can be used as a grouping key (not as a raw feature).
- `deployment_id`  
  Deployment/job ID (batch of VMs requested together).
- `ts_vm_created`  
  Request time (seconds since window start).
- `day_idx`  
  Day index derived from `ts_vm_created` (0–29).
- `hour_of_day`  
  Hour-of-day derived from `ts_vm_created` (0–23).

**Static VM config & deployment metadata**

- `vm_category`  
  Application class: `"Delay-insensitive"`, `"Interactive"`, `"Unknown"`.  
  Assumed to be known metadata if Azure tags workloads.
- `vm_virtual_core_count`  
  Number of virtual cores for this VM.
- `vm_memory_gb`  
  Requested memory (GB).
- `vm_mem_per_core`  
  Derived: `vm_memory_gb / vm_virtual_core_count` (with core count clipped ≥ 1).
- `deployment_size`  
  Number of VMs in this deployment (known from the deployment spec).
- `log_deployment_size`  
  Transform: `log(1 + deployment_size)`.

**Tenant “static” info (from past activity)**

These come from **all VMs created so far** for the subscription (conceptually available from historical logs):

- `ts_first_vm_created`  
  Time of the first VM for this subscription (within the trace window).
- `count_vms_created`  
  Total number of VMs created for this subscription (up to “now” conceptually).
- `sub_first_day`  
  Day index of `ts_first_vm_created`.
- `sub_first_hour`  
  Hour-of-day of `ts_first_vm_created`.

**Tenant history features (based only on previous VMs)**

These are computed from **prior VMs of the same `subscription_id`**, not from the future of the *current* VM. Conceptually, a real system could maintain these online using historical telemetry:

- `hist_n_vms`  
  Number of **previous** VMs for this tenant.
- `hist_n_critical`  
  Number of previous VMs that were labeled `critical` (based on their full history).
- `hist_has_past`  
  Indicator (0/1) for whether this tenant has any history (`hist_n_vms > 0`).
- `hist_critical_frac`  
  Fraction of previous VMs that were critical.
- `hist_lifetime_mean`  
  Mean lifetime (hours) of previous VMs.
- `hist_lifetime_std`  
  Std dev of lifetime (hours) of previous VMs.
- `hist_cpu_mean_mean`  
  Mean `cpu_mean` over previous VMs.
- `hist_p95_mean`  
  Mean `p95_max_cpu` over previous VMs.
- `hist_frac_gt60_mean`  
  Mean `cpu_frac_gt_60` over previous VMs.
- `hist_day_night_ratio_mean`  
  Mean `day_night_ratio` over previous VMs.

> **Note:** In the offline trace, these history features are computed using the full 30-day window. For a real scheduler, they would be maintained incrementally from past logs. For modeling, we treat them as **arrival-time features** because they never use the future of the *current* VM.

---

#### Features we do NOT have at arrival time (offline/post-hoc)

These depend on the **observed behavior of this VM during the window** and are **not available** when the VM is first requested. They are used to:

- define the label `critical`, and/or
- inspect behavior offline

but **must not** be used as predictive features for “new incoming VM” models.

**This VM’s future lifecycle**

- `ts_vm_deleted`  
  Deletion time (end of VM life).
- `lifetime_sec`  
  Duration in seconds (`ts_vm_deleted - ts_vm_created`).
- `lifetime_hours`  
  Duration in hours.

**This VM’s aggregated CPU usage**

All of these require observing the VM’s CPU time series over its life:

- `max_cpu`  
  Coarse max CPU from `vmtable` over the VM’s lifetime.
- `avg_cpu`  
  Coarse average CPU from `vmtable`.
- `p95_max_cpu`  
  p95 of max CPU (over this VM’s life).
- `n_readings`  
  Number of CPU samples for this VM in `vm_cpu`.
- `max_cpu_right`  
  Max of per-reading `max_cpu` from `vm_cpu` logs.
- `cpu_mean`  
  Mean `avg_cpu` over all readings for this VM.
- `cpu_std`  
  Std dev of `avg_cpu` over this VM’s readings.
- `cpu_frac_gt_60`  
  Fraction of readings with `avg_cpu > 60%` for this VM.
- `cpu_frac_gt_80`  
  Fraction of readings with `avg_cpu > 80%` for this VM.

**This VM’s diurnal / hourly pattern**

These features summarize **the VM’s own** time-of-day behavior; they are inherently post-hoc:

- `day_cpu_mean`  
  Mean CPU during “day” hours for this VM.
- `night_cpu_mean`  
  Mean CPU during “night” hours for this VM.
- `day_night_ratio`  
  `day_cpu_mean / (night_cpu_mean + ε)` for this VM.
- `cpu_hour_0_mean` … `cpu_hour_23_mean` (24 columns)  
  Mean CPU in each hour-of-day bucket for this VM.

**Label components & label**

Derived directly from this VM’s full life; **targets**, not inputs:

- `long_lived`  
  Boolean flag (e.g. `lifetime_hours >= 24`).
- `sustained_high`  
  Boolean: high `cpu_frac_gt_60` and/or high `p95_max_cpu`.
- `periodic_enough`  
  Boolean: 24h FFT energy dominates 12h/8h and clears a minimum energy floor.
- `strong_diurnal`  
  Legacy boolean: strong `day_night_ratio` and high `day_cpu_mean` (kept for analysis).
- `critical`  
  Final label: `1` if `periodic_enough & (long_lived | sustained_high)`, else `0`.

**Split metadata**

- `split`  
  `"train"`, `"val"`, or `"test"` based on `day_idx`.  
  This exists only in the offline dataset to define **evaluation splits** and is **never** a model feature.

---

#### Summary for modeling

- **Use as features for new incoming VMs:**  
  Static config, deployment info, request time, and **tenant history** (`hist_*`, `count_vms_created`, etc.).
- **Do *not* use as features:**  
  Anything that summarizes the **future behavior of this VM** (`lifetime_*`, `cpu_mean`, `cpu_frac_gt_*`, `day_*`, `cpu_hour_*`, label comps, `critical`, `split`).

When building new models, always construct your feature list from the ✅ group above to avoid label leakage and to match the “predict at VM arrival” scenario.


## Quick usage example

```python
import polars as pl

df = pl.read_parquet("preprocess/data_final/vm_request_table_with_split.parquet")

# Split
train = df.filter(pl.col("split") == "train")
val   = df.filter(pl.col("split") == "val")
test  = df.filter(pl.col("split") == "test")

# Drop IDs and non-feature columns
feature_cols = [
    c for c in df.columns
    if c not in {"vm_id", "subscription_id", "deployment_id", "critical", "split"}
]
X_train = train.select(feature_cols)
y_train = train["critical"]
```

This is the dataset you’ll train models on in the modeling notebooks.

```
```
