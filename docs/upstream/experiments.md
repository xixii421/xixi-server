# Profiling and Request Replay

Profile model capacity, prepare replay inputs, and collect service metrics. Assumes setup is done (see README). Paper ablation/scoring scripts and plotting notebooks have been removed.

## 1. Fetch the trace
Downloads the official Azure LMM trace ([CC-BY](https://github.com/Azure/AzurePublicDataset))
and samples a window to `traces/<TRACE>.csv`.

```bash
pixi run python utils/fetch_trace.py                  # first hour
pixi run python utils/fetch_trace.py --start-hour 5 --hours 1
```

## 2. Profile capacity (once per model)
Measures encoder + backend `M_C` → `profiles/<model>/profile.json` (used by autoscaling).

```bash
pixi run python -m lmm_serving.profiler.run
```

## 3. Generate per-QPM traces
Slices a window from a source trace and scales the rate (higher factor = lighter load).
Writes `traces/experiment_traces/<trace>_scale<factor>.csv` (gitignored).

```bash
pixi run python utils/create_experiment_qps_traces.py <source.csv> <factor> [window_min]
```

## 4. Run one experiment
`<setup>` = `monolith` | `modserve`; `<scales>` = comma list of trace factors.

```bash
pixi run python -m lmm_serving.launcher run modserve 2.0 --nodes-config internvl_3box.yaml
pixi run python -m lmm_serving.launcher run modserve 8.0,4.0,2.0 --dry-run   # plan only
```
Metrics land in `metrics/<setup>/<run_id>/` (auto-merged across nodes).
Historical upstream experiment CSVs are not bundled. The launcher recreates result
directories as needed; generated metrics and legacy `online_serving/results/` outputs
are ignored by Git. Small replay traces remain bundled. Profiles under `profiles/`
are upstream references, not measurements of your machine; reprofile before using
their capacity estimates for autoscaling.

## 5. Autoscaling replay
Same launcher with `AUTOSCALE=1`, replaying a bursty window for monolith and modserve.

```bash
AUTOSCALE=1 pixi run python -m lmm_serving.launcher run modserve <scale> --nodes-config llama_3box.yaml
```
