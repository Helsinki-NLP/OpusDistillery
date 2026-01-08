# Configuration Files

The configuration files for OpusDistillery are written in [YAML](https://yaml.org/) format and are divided into two main sections:

- **`experiment`**: Contains the general setup and parameters for the experiment, excluding dataset information.
- **`datasets`**: Specifies the datasets used for training, development, and evaluation. Details about datasets can be found in [Dataset Importers](downloading_and_selecting_data.md).

### Experiment Setup

In the `experiment` section, the following key parameters must be defined:

- **`dirname`**: The directory where all experiment outputs will be stored.
- **`name`**: The name of the experiment. All generated data and models will be saved under `dirname`/`name`.
- **`langpairs`**: A list of language pairs for the student model, using ISO two-letter language codes.

Example configuration:
```yaml
experiment:
  dirname: test
  name: fiu-eng
  langpairs:
    - et-en
    - fi-en
    - hu-en
```

## Data processing

### OpusFilter

OpusDistillery supports [OpusFilter](https://github.com/Helsinki-NLP/OpusFilter), a tool for filtering and combining parallel corpora. Instead of the default cleaning, you can choose to filter data using OpusFilter with either a default configuration or a custom configuration that you provide.

In the configuration file, if you want to use a default configuration, see this [example](https://github.com/Helsinki-NLP/OpusDistillery/blob/multi-ftt/configs/pipeline/clean/run-opusfilter.py#13). 
Otherwise, you can specify the path to a custom OpusFilter configuration file such as [this one](https://github.com/Helsinki-NLP/OpusDistillery/blob/multi-ftt/configs/configs/opusfilter/config.opusfilter.yml).

```yaml
  opusfilter:
    config: default # # Or specify the path to an OpusFilter configuration file
```

### Bicleaner AI

Currently, Bicleaner AI is not operational. Do you want to collaborate? Feel free to work on this [issue](https://github.com/Helsinki-NLP/OpusDistillery/issues/3).

## Teacher models

You can select a teacher model from OPUS-MT or Hugging Face.

### OPUS-MT Teachers

To specify an OPUS-MT teacher, use:

* `opusmt-teacher`

It can be one of the following:

1. A URL to an OPUS-MT model:

```yaml
  opusmt-teacher: "https://object.pouta.csc.fi/Tatoeba-MT-models/fiu-eng/opus4m-2020-08-12.zip"
```

2. A path to a local OPUS-MT model:

```yaml
  opusmt-teacher: "/path/to/opus-mt/model"
```

3. A list of OPUS-MT models:

```yaml
  opusmt-teacher:
    - "https://object.pouta.csc.fi/Tatoeba-MT-models/gem-gem/opus-2020-10-04.zip"
    - "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-swe/opus+bt-2021-04-14.zip"
```


4. For multilingual students, specify different teachers for each language pair:

```yaml
  opusmt-teacher:
    en-uk: "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-ukr/opus+bt-2021-04-14.zip"
    en-ru: "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-rus/opus+bt-2021-04-14.zip"
    en-be: "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-bel/opus+bt-2021-03-07.zip"
```

5. Use the `best` option to automatically select the best teacher for each language pair, based on FLORES200+ scores from the [OPUS-MT dashboard](https://opus.nlpl.eu/dashboard/).

```yaml
  opusmt-teacher: "best"
```

### Hugging Face Teachers

You can also use a [Hugging Face](https://huggingface.co/) model as a teacher.

* `modelname`: The model identifier from the Hugging Face hub.
* `modelclass`: The class of the model being loaded.

```yaml
  huggingface:
    modelname: "Helsinki-NLP/opus-mt-mul-en"
    modelclass: "transformers.AutoModelForSeq2SeqLM"
```

You can also configure the decoding options:

```yaml
  huggingface:
    modelname: "HPLT/translate-et-en-v1.0-hplt_opus"
    modelclass: "transformers.AutoModelForSeq2SeqLM"
    config:
      top_k: 50
      top_p: 0.90
      temperature: 0.1
      max_new_tokens: 128
```

For models that use language tags, additional parameters are required:

* `lang_info`: Set to True if language tags are needed.
* `lang_tags`: A mapping of language codes to the tags used by the model.

```yaml
  huggingface:
    modelname: "facebook/nllb-200-distilled-600M"
    modelclass: "transformers.AutoModelForSeq2SeqLM"
    lang_info: True
    lang_tags:
      en: eng_Latn
      et: est_Latn
```

Finally, for models requiring a prompt, you can define it like this:

```yaml
  huggingface:
    modelname: "google-t5/t5-small"
    modelclass: "transformers.AutoModelForSeq2SeqLM"
    lang_tags:
      en: English
      de: German
    prompt: "Translate {src_lang} to {tgt_lang}: {source}"
```

In this case, the lang_tags mapping will be used in the prompt.

Note: When using a Hugging Face model as a teacher, there is no scoring or cross-entropy filtering.

## Backward models

Currently, only OPUS-MT models are available as backward models for scoring translations.

To specify a backward model, use:

* `opusmt-backward`: The URL or path to an OPUS-MT model. Like the teacher models, this can also be a dictionary for multilingual students or `best`.

```yaml
  opusmt-backward:
    uk-en: "https://object.pouta.csc.fi/Tatoeba-MT-models/ukr-eng/opus+bt-2021-04-30.zip"
    ru-en: "https://object.pouta.csc.fi/Tatoeba-MT-models/rus-eng/opus+bt-2021-04-30.zip"
    be-en: "https://object.pouta.csc.fi/Tatoeba-MT-models/bel-eng/opus+bt-2021-04-30.zip"
```

If left empty, the cross-entropy filtering step will be skipped.

## Multilinguality
Specify whether the teacher, backward, and student models are many-to-one to properly handle language tags. By default, this is set to `False`.

* `one2many-teacher`: `True` or `False` (default). If `opusmt-teacher` is set to `best`, this should also be `best`.
* `one2many-backward`: `True` or `False` (default). If `opusmt-backward` is set to `best`, this should also be `best`.
* `one2many-student`: `True` or `False` (default). 

```yaml
# Specify if the teacher and the student are one2many
  one2many-teacher: True
  one2many-student: True
```
## Training

### Marian arguments
You can override default pipeline settings with [Marian-specific settings](https://marian-nmt.github.io/docs/cmd/marian/).

You can use the following options: `training-teacher`, `decoding-teacher`,`training-backward`, `decoding-backward`,`training-student`, `training-student-finetuned`.

```yaml
  marian-args:
  # These configs override pipeline/train/configs
  training-student:
    dec-depth: 3
    enc-depth: 3
    dim-emb: 512
    tied-embeddings-all: true
    transformer-decoder-autoreg: rnn
    transformer-dim-ffn: 2048
    transformer-ffn-activation: relu
    transformer-ffn-depth: 2
    transformer-guided-alignment-layer: last
    transformer-heads: 8
    transformer-postprocess: dan
    transformer-preprocess: ""
    transformer-tied-layers: []
    transformer-train-position-embeddings: false
    type: transformer
```

### Opustrainer

OpusDistillery supports [OpusTrainer](https://github.com/hplt-project/OpusTrainer) for curriculum training and data augmentation.

You can specify a path to the OpusTrainer configuration, such as in [this example](https://github.com/Helsinki-NLP/OpusDistillery/blob/multi-ftt/configs/opustrainer/config.fiu-eng.opustrainer.yml#L37).
This assumes you know the final paths of the data, as defined in [this file](https://github.com/Helsinki-NLP/OpusDistillery/blob/multi-ftt/configs/opustrainer/config.fiu-eng.opustrainer.stages.yml).

Currently, this is implemented only for student training.

```yaml
  opustrainer:
    model: student # Ideally, could be teacher or backward
    path: "configs/opustrainer/config.fiu-eng.opustrainer.stages.yml" # This assumes you already know the paths to the data
```

## Exporting

The final student model is exported in the Bergamot format, which uses shortlists for training. Shortlists are trained using alignments, so there's an option to train a student without guided alignment using the tiny architecture. To disable this, specify export in the configuration file:

```yaml
  export: "no"
```

### Other

* `parallel-max-sentences`: Maximum parallel sentences to download from each dataset.
* `split-length`: The number of sentences into which you want to split your training data for forward translation.
* `best-model`: Metric used to select the best model.
* `spm-sample-size`: Sample size for training the student’s SPM vocabulary.
* `spm-vocab-size`: Vocabulary size for training the student’s SPM vocabulary.
* `student-prefix`: To train multiple students with the same data, add a prefix to the student name, which will allow multiple students to be trained under the same directory structure with the same data. More details on the directory structure can be found [here](../pipeline/dir_structure.md).
