# Installation

This section describes how to set up the OpusDistillery pipeline locally, as well as on three of our supported clusters.

## Locally

### System Requirements

- Ubuntu 18.04 (it can work on other Linux distributions, but might require `setup` scripts fixes; see more details in [marian installation instructions](https://marian-nmt.github.io/quickstart/)).
- One or several Nvidia GPUs with CUDA drivers installed and at least 8 GB of memory.
- CUDNN installed
- At least 16 CPU cores ( some steps of the pipeline utilize multiple cores pretty well, so the more the better).
- 64 GB RAM (128 GB+ might be required for bigger datasets)
- 200+ GB of disk space ( mostly for datasets and transformations ). 
  It depends on chosen datasets and can be significantly higher.
  
### Installation

0. Clone the repo:
``` 
git clone https://github.com/Helsinki-NLP/OpusDistillery/
cd OpusDistillery
```
1. Choose a [Snakemake profile](https://github.com/Snakemake-Profiles) from `profiles/` or create a new one 
2. Adjust paths in the `Makefile` if needed and set `PROFILE` variable to the name of your profile
3. Adjust Snakemake and workflow settings in the `profiles/<profile>/config.yaml`, see [Snakemake CLI reference](https://snakemake.readthedocs.io/en/stable/executing/cli.html) for details
4. Configure experiment and datasets in `configs/config.prod.yml` (or `configs/config.test.yml` for test run)
5. Change source code if needed for the experiment
6. **(Cluster mode)** Adjust cluster settings in the cluster profile.
   For `slurm-moz`: `profiles/slurm-moz/config.cluster.yml`
   You can also modify `profiles/slurm-moz/submit.sh` or create a new Snakemake [profile](https://github.com/Snakemake-Profiles).
7. **(Cluster mode)** It might require further tuning of requested resources in `Snakemake` file:
    - Use `threads` for a rule to adjust parallelism
    - Use `resources: mem_mb=<memory>` to adjust total memory requirements per task 
      (default is set in `profile/slurm-moz/config.yaml`)
8. Install Mamba - fast Conda package manager

```
make conda
```

9. Install Snakemake

```
make snakemake
```

10. Update git submodules

```
make git-modules
```

11. Install requirements:
```
source ../mambaforge/etc/profile.d/conda.sh ; conda activate ; conda activate snakemake
pip install -r requirements.txt 
```

You are all set!

## On a Cluster

### System Requirements

- Slurm cluster with CPU and Nvidia GPU nodes
- CUDA 11.2 ( it was also tested on 11.5)
- CUDNN library installed
- Singularity module if running with containerization (recommended)
- If running without containerization, there is no procedure to configure the environment automatically.
  All the required modules (for example `parallel`) should be preinstalled and loaded in ~/.bashrc

## Installation on Puhti and Mahti
1. Clone the repository.
2. Download the Ftt.sif container to the repository root (ask [Ona](ona.degibert@helsinki.fi))
3. Create a virtual Python environment for Snakemake (e.g. in the parent dir of the repository):
    1. The environment needs to be created with a non-containerized python, as otherwise Apptainer integration will not work. On puhti and mahti, the python executables in /usr/bin/ should work: `/usr/bin/python3.9 -m venv snakemake_env`.
    2. Activate the virtual environment: `source ./snakemake_env/bin/activate`.
    3. Install snakemake: `pip install snakemake`.
4. Install micromamba (e.g. in the parent dir of the repository): `curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba`
5. Return to the repository directory and update Git submodules: `make git-modules`
6. Create a _data_ directory (e.g. in the parent dir of the repository) and create a _tmp_ dir in it.
7. If the data directory is not located in the parent directory of the repository, edit _profiles/slurm-puhti/config.yaml_ or _profiles/slurm-mahti/config.yaml_ and change the bindings in the singularity-args section to point to your data directory, and also enter the _data_ directory path as the _root_ value of the _config_ section.
8. Edit profiles/slurm-puhti/config.cluster.yaml to change the CSC account to one you have access to. 
9. Load cuda modules: module load gcc/9.4.0 cuda cudnn
10. Run pipeline: `make run-hpc PROFILE="slurm-puhti"` or `make run PROFILE="slurm-mahti"`. More information in [Basic Usage](usage.md).

## Installation on Lumi
1. Clone the repository.
2. Download the Ftt.sif container to the repository root (ask [Ona](ona.degibert@helsinki.fi))
3. Create a virtual Python environment for Snakemake (e.g. in the parent dir of the repository):
    1. The environment needs to be created with a non-containerized python, as otherwise Apptainer integration will not work. On lumi, use the _cray-python_ module (it is not containerized): `module load cray-python; python -m venv snakemake_env`.
    2. Activate the virtual environment: `source ./snakemake_env/bin/activate`.
    3. Install snakemake: `pip install snakemake`.
4. Install micromamba (e.g. in the parent dir of the repository): `curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba`
5. Return to the repository directory and update Git submodules: `make git-modules`
6. Create a _data_ directory (e.g. in the parent dir of the repository) and create a _tmp_ dir in it.
7. If the data directory is not located in the parent directory of the repository, edit profiles/slurm-lumi/config.yaml and change the bindings in the singularity-args section to point to your data directory, and also enter the _data_ directory path as the _root_ value of the _config_ section.
8. Edit profiles/slurm-puhti/config.cluster.yaml to change the CSC account to one you have access to. 
9. Load rocm module: module load rocm.
10. Copy the marian executables to _3rd_party/lumi-marian/build_ (compiling lumi-marian is currently hacky, so this workaround makes things easier) from `/scratch/project_462000447/lumi-marian`
11. Enter _export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH_ to make sure Marian can find all the libraries when it runs containerized.
12. Run pipeline: `make run-hpc PROFILE="slurm-lumi"`.  More information in [Basic Usage](usage.md).