# Running on NERSC Perlmutter (Slurm)

Only relevant on a Slurm cluster. The toolkit itself has no job scripts; applications keep
their own `sbatch` files. This page lists what such scripts and interactive work need.

## Login nodes

Login nodes are shared. Allowed there: single-threaded work (`OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`) of up to about 20 min, one such process at a time,
a few GB of memory. That covers test files, the full suite (~12 min), `pytest -m 'not slow'`
(~5 min), `scripts/smoke.py`, `ruff` and config validation.

Everything else goes to a compute node: longer runs, multi-threaded work, training on real data,
large GP fits. Interactive: `salloc --account=<project> --constraint=cpu --qos=interactive
--time=01:00:00 --nodes=1`; batch: a `debug` job (below).

## CPU jobs (`train`, `predict`, `mfgp`, `binomial`)

```bash
#SBATCH --account=<project>
#SBATCH --constraint=cpu
#SBATCH --qos=debug          # smoke/debug only (<= 30 min); real runs: --qos=regular
#SBATCH --time=00:15:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --cpu-bind=none .venv/bin/python -u -m core.surrogates train CONFIG.yaml
```

- `debug` is for checking that a pipeline runs and how long a step takes; real runs use
  `regular` with the time they need.
- Large GP fits may use more BLAS threads (e.g. 32); training keeps one Torch thread (the CLI
  sets it).
- Give only the slow stage a long `--time`: a pending job's limit can be lowered with
  `scontrol update`, but usually not raised.
- Chain stages with `--dependency=afterok:JOBID` (train → predict → mfgp).

## GPU jobs (`stream`, `training.device: cuda`)

The `.venv` Torch is CPU-only, so GPU jobs use the NERSC PyTorch module with this repository
on `PYTHONPATH`:

```bash
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1 --ntasks-per-node=1

module load pytorch
export PYTHONPATH="/path/to/RESUM_FLEX${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4     # data-loader threads per rank
srun --cpu-bind=none python -u -m core.surrogates stream CONFIG.yaml
# several GPUs: srun --cpu-bind=none torchrun --standalone --nproc-per-node=4 -m core.surrogates stream CONFIG.yaml
```

## Accounting

`compute.json` records the Slurm job id, allocated cores and GPUs and `core_hours`
(wall time × allocated physical cores), so runs can be compared by cost.
