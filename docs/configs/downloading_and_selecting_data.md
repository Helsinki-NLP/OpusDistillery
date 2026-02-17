# Dataset importers

Dataset importers can be used in `datasets` sections of the [training config](https://github.com/Helsinki-NLP/OpusDistillery/tree/main/configs/test.yml#L66).

Example:
```
datasets:
  train:
    - tc_Tatoeba-Challenge-v2023-09-26
  devtest:
    - flores_dev
  test:
    - flores_devtest
```

Data source | Prefix | Name examples | Type | Comments
--- | --- | --- | ---| ---
[MTData](https://github.com/thammegowda/mtdata) | mtdata | newstest2017_ruen | corpus | Supports many datasets. Run `mtdata list -l ru-en` to see datasets for a specific language pair.
[OPUS](opus.nlpl.eu/) | opus | ParaCrawl__v7.1 | corpus | Many open source datasets. Go to the website, choose a language pair, check links under Moses column to see what names and version is used in a link. The version should be separated by a double _.
[SacreBLEU](https://github.com/mjpost/sacrebleu) | sacrebleu | wmt20 | corpus | Official evaluation datasets available in SacreBLEU tool. Recommended to use in `datasets:test` config section. Look up supported datasets and language pairs in `sacrebleu.dataset` python module.
[Flores](https://github.com/facebookresearch/flores) | flores | dev, devtest | corpus | Evaluation dataset from Facebook that supports 100 languages.
[Bouquet](https://huggingface.co/datasets/facebook/bouquet) | bouquet | dev, test | corpus | Evaluation dataset from Facebook that supports several language across diferent domains.
Custom parallel | url | `https://storage.googleapis.com/releng-translations-dev/data/en-ru/pytest-dataset.[LANG].zst` | corpus | A custom zst compressed parallel dataset, for instance uploaded to GCS. The language pairs should be split into two files. the `[LANG]` will be replaced with the `to` and `from` language codes.
[Paracrawl](https://paracrawl.eu/) | paracrawl-mono | paracrawl8 | mono | Datasets that are crawled from the web. Only [mono datasets](https://paracrawl.eu/index.php/moredata) are used in this importer. Parallel corpus is available using opus importer.
[News crawl](http://data.statmt.org/news-crawl) | news-crawl | news.2019 | mono | Some news monolingual datasets from [WMT21](https://www.statmt.org/wmt21/translation-task.html)
[Common crawl](https://commoncrawl.org/) | commoncrawl | wmt16 | mono | Huge web crawl datasets. The links are posted on [WMT21](https://www.statmt.org/wmt21/translation-task.html)
Custom mono | url | `https://storage.googleapis.com/releng-translations-dev/data/en-ru/pytest-dataset.ru.zst` | mono | A custom zst compressed monolingual dataset, for instance uploaded to GCS.

## Find datasets

You can also use [find-corpus](https://github.com/Helsinki-NLP/OpusDistillery/tree/main/utils/find-corpus.py) tool to find all datasets for an importer and get them formatted to use in config.

Set up a local [poetry](https://python-poetry.org/) environment.
```
task find-corpus -- en ru
```

Make sure to check licenses of the datasets before using them.

## Adding a new importer

Just add a shell script to [corpus](https://github.com/Helsinki-NLP/OpusDistillery/tree/main/pipeline/data/importers/corpus) or [mono](https://github.com/Helsinki-NLP/OpusDistillery/tree/main/pipeline/data/importers/mono) which is named as `<prefix>.sh` and accepts the same parameters as the other scripts from the same folder.

## Issues
* Currently, it is not possible to download specific datasets per language pair; the tool downloads the same dataset for all language pairs. If a dataset doesn't exist for a given language pair, dummy files are created. Do you want to collaborate? Feel free to work on this [issue](https://github.com/Helsinki-NLP/OpusDistillery/issues/1).
* There is currently no support for downloading monolingual datasets. The use of monolingual data is not fully implemented; only bilingual data is supported at this time. Do you want to collaborate? Feel free to work on this [issue](https://github.com/Helsinki-NLP/OpusDistillery/issues/2).