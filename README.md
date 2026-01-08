# OpusDistillery

OpusDistillery is an end-to-end pipeline to perform systematic multilingual distillation of MT models. It is built on top of the [Firefox Translations Training pipeline](https://github.com/mozilla/firefox-translations-training), originally developed within the [Bergamot project](https://browser.mt), for training efficient NMT models that can run locally in a web browser.

Read our [docs](https://helsinki-nlp.github.io/OpusDistillery/).

We have implemented the use of pre-trained OPUS-MT models, the tracking of GPU utilisation and multilinguality support.

* **OPUS-MT models**: We have added the option to simply provide the URL of an existing OPUS-MT model. Our tool is also able to select the best available OpusMT model per language pair.

* **GPU Utilisation** With the hope of moving towards greener NLP and NMT, we have added GPU utilisation tracking so that we can report the amount of hours and energy consumed by the pipeline.

* **Multilinguality Support**: The pipeline supports training multilingual models. This covers two aspects: support for using any combination of multilingual and bilingual teachers, as well as support for multilingual student training.

# Multilingual Training

This branch is based on the main branch and allows for the distilling multilingual students from OPUS-MT models.
The different possible distilling scenarios that we envision and that are covered are the following (o2m: one2many, m2o: many2one, m2m: many2many):

|ID | Configuration         | Teacher | Student | Example config                              | Tested? |
|---|-----------------------|---------|---------|---------------------------------------------|---------|  
| 1 | bilingual - bilingual | en-et   | en-et   | [Config file](configs/config.1.o2o.o2o.yml) | y       | 
| 2 | o2m - bilingual       | eng-fiu | en-et   | [Config file](configs/config.2.o2m.o2o.yml) | y       |
| 3 | o2m - o2m             | eng-fiu | eng-fiu | [Config file](configs/config.3.o2m.o2m.yml) | y       |
| 4 | m2o - bilingual       | fiu-eng | et-en   | [Config file](configs/config.4.m2o.o2o.yml) | y       |
| 5 | m2o - m2o             | fiu-eng | fiu-eng | [Config file](configs/config.5.m2o.m2o.yml) | n       |
| 6 | m2m - bilingual       | fiu-gmw | et-en   | [Config file](configs/config.6.m2m.o2o.yml) | y       |
| 7 | m2m - o2m             | gmw-fiu | eng-fiu | [Config file](configs/config.7.m2m.o2m.yml) | y       |
| 8 | m2m - m2o             | fiu-gmw | fiu-eng | [Config file](configs/config.8.m2m.m2o.yml) | n       |
| 9 | m2m - m2m             | gmw-fiu | gmw-fiu | [Config file](configs/config.9.m2m.m2m.yml) | n       |

Some things have changed in the configuration file:

- Languages: you can either specify the languages you want to train by `src` and `trg` if the model is bilingual. If the model is multilingual of any kind, you need to specify `langpairs`, you can see how in [this example](configs/config.1.o2o.o2o.yml#L10). 
- Mulilingual configuration: now you need to specify if either the teacher, the backward or the student is a one2many model, so that we can handle language tags appropietly. We created `one2many-teacher`, `one2many-backward` and `one2many-student` options to hanlde this. You can see how in [this example](configs/config.1.o2o.o2o.yml#L21).
- `max-parallel-sents`: this allows you to define the maximum parallel sentences you want to download per language pair in the case of multilingual models.
- `dirname`: usually the directory structure relies on the source and target languages, in case of a multilingual model of any kind, you can specify the name of the directory you want to use. You can see how in [this example](configs/config.1.o2o.o2o.yml#L8). 

TO DO:
- Download different datasets per language pair, right now it only downloads the same dataset for all language pairs. If a dataset doesn't exist for a given language pair, it creates dummy files.
- Downloading monolingual datasets. The use of monolingual data is not implemented, currently only supports the use of bilingual data.

Not implemented:
- Multiple teachers or backward models: currenlty only multilingual models can be used, not individual models.
- Multilingual Teacher training, at the moment only takes opusmt as teacher
- mono src and trg are not working
- At the moment, if you specify an opus-mt model as a teacher, it will be download for as many language pairs as you have.

# OpusFilter

We have added support for using [OpusFilter](https://github.com/Helsinki-NLP/OpusFilter), a tool for filtering and combining parallel corpora. For data filtering, instead of the default cleaning or using bicleaner, you can choose to use opusfilter with a default configuration or with a specific configuration you provide.

In the configuration file, if you want to use a [default](pipeline/clean/run-opusfilter.py#13) configuration, you can see how in [this example](configs/opusfilter/config.fiu-eng.opusfilter.yml#L33). Otherwise, you can specify the path to a specific file with an Opusfilter configuration such as [this one](configs/opusfilter/config.opusfilter.yml).

# OpusTrainer

We have also added support for using [OpusTrainer](https://github.com/hplt-project/OpusTrainer), a tool for curriculum training and data augmentation. 

In the configuration file, you can specify a path to the OpusTrainer configuration as in [here](configs/opustrainer/config.fiu-eng.opustrainer.yml#L37). However, this assumes that you already now the final paths of the data as specified in [here](configs/opustrainer/config.fiu-eng.opustrainer.stages.yml).

At the moment, this is only implement for student training. For future work, we would like to implement it as well for teacher and backward training.

# OPUS-MT integration

This fork makes it possible to use OPUS-MT models as teacher and backward models in the _firefox-translations-training_ pipeline (FTT). Other additions are profiles for running jobs on CSC supercomputers (*puhti*, *lumi* and *mahti*) and code for monitoring the power usage of jobs.

# Workflow changes
- Added download rule for Tatoeba-Challenge data.
- Added download rule for OPUS-MT models (tested with Tatoeba-Challenge models, old models might need some changes)
- Added config parameters for specifying OPUS-MT models as teacher and/or backward model.
- Added subword segmentation and desegmentation rules.

# Subword segmentation issues
The biggest incompatibility with OPUS-MT models and FTT is in subword segmentation: default FTT trains models that use the in-built sentencepiece support in Marian, while OPUS-MT models expect data to be pre-segmented. To make it possible to use both the default FTT training and pre-built OPUS-MT models, segmentation and desegmentation steps have been added around marian-specific rules. This causes some clutter, but it's probably the best solution (instead of e.g. doing the segmentation/desegmentation inside the marian scripts), since it also makes it possible to easily implement other subword segmentation methods in the workflow. 


# Snakemake and conda on HPC
FTT is based on Snakemake, which has many benefits in terms of reproducibility and existing support. Among other things, Snakemake supports HPC environments and SLURM out of the box, which should make it ideal for CSC machines. However, Snakemake also makes heavy use of conda, which has been deprecated on CSC machines due to its unsuitability for HPC file systems (https://docs.csc.fi/computing/usage-policy/#conda-installations), and FTT specifically relies on several conda environments. Fortunately, Snakemake has a functionality for containerizing conda environments, so all the conda environments needed by FTT can be provided in an Apptainer container (Ftt.sif).

Containerization does not entirely solve the conda problem, since the Snakemake program itself requires conda to run. CSC provides a snakemake module, but problematically these modules are container-based, and since containers cannot be nested on CSC machines, it is not possible to use containerized conda environments with the CSC snakemake modules. This can be solved by installing Snakemake with pip (this is discouraged in the Snakemake documentation, but I have seen no problems so far).

# Non-containerized software
FTT uses software that is not included in the containerized conda environments, including several marian installations and other NLP tools. These are automatically built as part of the pipeline. The Ftt.sif container includes the prerequisites for the software components. It's also possible to provide paths to separately built software installations. 

# Getting started on CSC's puhti and mahti
1. Clone the repository.
2. Download the Ftt.sif container to the repository root.
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
10. Run pipeline: `make run-hpc PROFILE="slurm-puhti"` or `make run PROFILE="slurm-mahti"`

# Getting started on CSC's lumi
1. Clone the repository.
2. Download the Ftt.sif container to the repository root.
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
10. Copy the marian executables to _3rd_party/lumi-marian/build_ (compiling lumi-marian is currently hacky, so this workaround makes things easier).
11. Enter _export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH_ to make sure Marian can find all the libraries when it runs containerized.
12. Run pipeline: `make run-hpc PROFILE="slurm-puhti"`

# Testing
Since running the whole pipeline for a high-resource language pair will take a long time, there is a test config available for testing that everything works as it should. The test config is used by default, you can change into the full config by modifying the Makefile and changing config.opusmt-test.yml to config.opusmt.yml. You can also provide the config on the command line as the CONFIG parameter with make. Note that even the test config will take a long time if the training corpus is large (since translating the training data will take time). So to do a quick functionality check, pick a language pair with as little data as possible in Tatoeba-Challenge (while still having trained forward and backward models). The default epo-afr is good for quick checking (although note that bicleaner step will be skipped, as there are no bicleaner packs for those languages).

You can test the pipeline without running it by using make dry-run. If you want to build a specific file or rule, you can use the TARGET parameter with make. 

# Acknowledgements

This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546]
