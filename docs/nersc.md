# Running on NERSC Perlmutter (Slurm)

Only relevant on a Slurm cluster. The toolkit itself has no job scripts; applications keep
their own `sbatch` files. This page lists what such scripts and interactive work need.

## Login nodes

Never train, fit GPs or run the full test suite on login nodes. Fine there: single test files,
`--help`, config validation, the synthetic `scripts/phase1_plot_ground_truth.py`.

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

## GPU jobs (`photon`, `training.device: cuda`)

The `.venv` Torch is CPU-only, so GPU jobs use the NERSC PyTorch module with this repository
on `PYTHONPATH`:

```bash
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1 --ntasks-per-node=1

module load pytorch
export PYTHONPATH="/path/to/RESUM_FLEX${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4     # data-loader threads per rank
srun --cpu-bind=none python -u -m core.surrogates photon CONFIG.yaml
# several GPUs: srun --cpu-bind=none torchrun --standalone --nproc-per-node=4 -m core.surrogates photon CONFIG.yaml
```

## Accounting

`compute.json` records the Slurm job id, allocated cores and GPUs and `core_hours`
(wall time × allocated physical cores), so runs can be compared by cost.
