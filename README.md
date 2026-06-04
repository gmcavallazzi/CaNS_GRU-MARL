# CaNS_GRU-MARL

Code accompanying the paper

> **Drag reduction or reward hacking? Recurrent multi-agent reinforcement learning that earns its reward**
> G. M. Cavallazzi, M. Pérez Cuadrado, A. Pinelli.

It contains a modified version of the [CaNS](https://github.com/CaNS-World/CaNS)
finite-difference Navier–Stokes solver, extended with a wall-actuation interface,
and the multi-agent reinforcement-learning framework (GRU-MARL) used to train and
evaluate the wall controller, together with the trained policy reported in the paper.

## Credit and licence

This repository is a **modified version of CaNS**, the canonical Navier–Stokes
solver by Pedro Costa and the CaNS contributors
(<https://github.com/CaNS-World/CaNS>), reused here with the author's permission.
CaNS is distributed under the MIT licence, which is retained in [`LICENSE`](LICENSE);
the modifications and the DRL framework in this repository are released under the
same licence.

If you use this code, please cite the CaNS solver

> P. Costa, *A FFT-based finite-difference solver for massively-parallel direct
> numerical simulations of turbulent flows*, Computers & Mathematics with
> Applications 76, 1853–1862 (2018). doi:10.1016/j.camwa.2018.07.034

and the paper above for the controller.

The changes to CaNS are confined to the solver's actuation hooks (principally
`src/drl.f90` and `src/opposition.f90`, which apply a Python-supplied wall-normal
velocity as a time-dependent boundary condition); the numerics are unchanged.

## Repository layout

```
src/             modified CaNS Fortran solver (actuation interface in drl.f90, opposition.f90)
configs/         CaNS compiler / flag / library makefiles and defaults
dependencies/    external.mk (the decomposition libraries are obtained separately, see Build)
Makefile         CaNS build entry point
build.conf       compiler / GPU / precision selection
drl/             multi-agent DRL framework (PyTorch + PettingZoo)
  stwStart_gru.py, stwEnv_gru.py, models_gru.py, replay_buffer.py, utils_gru.py
  eval_checkpoint.py, config_gru_refined.yaml, *.nml, launch_*.sh
  model/checkpoint_ep_108.pth   trained GRU-MARL policy used in the paper
requirements.txt
LICENSE
```

## Build (CaNS solver)

**System requirements** (as for upstream CaNS):

- an MPI implementation and a Fortran compiler (GNU `gfortran`, NVIDIA `nvfortran`, or Intel);
- FFTW3 for CPU runs, or cuFFT for GPU runs;
- for GPU runs: the NVIDIA HPC SDK (`nvfortran`), CUDA, and CMake (to build cuDecomp).

**Decomposition libraries.** CaNS relies on 2DECOMP&FFT (CPU/GPU) and cuDecomp
(GPU), which are *not* redistributed here. Obtain them exactly as upstream CaNS
does and place them under `dependencies/`:

```bash
git clone --recursive https://github.com/CaNS-World/CaNS /tmp/CaNS
cp -r /tmp/CaNS/dependencies/2decomp-fft dependencies/
cp -r /tmp/CaNS/dependencies/cuDecomp   dependencies/
```

**Compile.** Select the compiler and options in [`build.conf`](build.conf)
(`FCOMP`, `GPU`, `SINGLE_PRECISION`, …). For GPU builds, set the cuDecomp CUDA
compute-capability list in `dependencies/external.mk`
(`CUDECOMP_CUDA_CC_LIST`) to match your hardware. Then, from the repository root:

```bash
make libs && make
```

This builds the 2DECOMP&FFT / cuDecomp libraries and produces the `cans` executable.

## Python environment (DRL framework)

```bash
conda create -n cans_drl python=3.12
conda activate cans_drl
pip install -r requirements.txt   # mpi4py needs the system MPI on PATH
```

## Running

The DRL framework launches CaNS as an MPI co-process and exchanges observations
and actions through the solver's actuation interface.

- **CFD only** — from a run directory holding `input.nml` and `drl.nml`: `./cans`
- **Train GRU-MARL** (the refined configuration used in the paper):

  ```bash
  mpirun -n 1 python drl/stwStart_gru.py --config drl/config_gru_refined.yaml : -n 1 ./cans
  ```

- **Evaluate the trained policy** on one episode:

  ```bash
  mpirun -n 1 python drl/eval_checkpoint.py \
      --checkpoint drl/model/checkpoint_ep_108.pth \
      --config drl/config_gru_refined.yaml : -n 1 ./cans
  ```

`drl/model/checkpoint_ep_108.pth` is the trained GRU-MARL policy reported in the paper.

The `drl/launch_*.sh` scripts wrap the commands above for a SLURM cluster
(NVIDIA HPC SDK + conda). They are specific to the environment they were written
for and should be adapted (module loads, partitions, the conda environment name)
before use elsewhere.
