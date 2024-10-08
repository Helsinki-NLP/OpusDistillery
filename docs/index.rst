OpusDistillery
=====================

Welcome to OpusDistillery's documentation!

OpusDistillery is an end-to-end pipeline to perform systematic multilingual distillation of MT models.
It is built on top of the `Firefox Translations Training pipeline <https://github.com/mozilla/firefox-translations-training>`_,
originally developed within the `Bergamot project <https://browser.mt>`_, for training efficient NMT models that can run locally in a web browser.

The pipeline is capable of training a translation model for any language pair(s) end to end.
Translation quality depends on the chosen datasets, data cleaning procedures and hyperparameters.
Some settings, especially low resource languages might require extra tuning.

We use `Marian <https://marian-nmt.github.io/>`_, the fast neural machine translation engine.

New features:

* **OPUS-MT models**: We have added the option to simply provide the URL of an existing OPUS-MT model. Our tool is also able to select the best available OpusMT model per language pair.
* **Hugging Face models**: You can also automatically distill from an existing model on Hugging Face.
* **Multilinguality Support**: The pipeline supports training multilingual models. This covers two aspects: support for using any combination of multilingual and bilingual teachers, as well as support for multilingual student training.
* **GPU Utilisation** With the hope of moving towards greener NLP and NMT, we have added GPU utilisation tracking so that we can report the amount of hours and energy consumed by the pipeline.

.. toctree::
   :caption: Get started
   :maxdepth: 1

   installation.md
   usage.md
   quickstart.md

.. toctree::
   :caption: Setting up your experiment
   :name: configs/
   :maxdepth: 1

   configs/configuration_files.md
   configs/downloading_and_selecting_data.md
   configs/examples.md

.. toctree::
   :caption: The pipeline
   :name: pipeline/
   :maxdepth: 1

   pipeline/steps.md
   pipeline/dir_structure.md

.. toctree::
   :caption: Other information
   :maxdepth: 1

   troubleshooting.md
   references.md