localrules: ensemble_models, self_ensemble_model

ruleorder: self_ensemble_model > ensemble_models > train_model

wildcard_constraints:
    src="\w{2,3}",
    trg="\w{2,3}",
    train_vocab="train_joint_spm_vocab[^/]+",
    training_type="[^/]+",
    model_type="[^/_]+",
    index_type="[^-]+"

gpus_num=config["gpus-num"]

rule convert_marian_to_hf:
    message: "Converting Marian model for use with HuggingFace transformers library"
    log: "{preprocessing}/convert_to_hf/convert.log"
    threads: 1
    input:
        marian_model=f'{{preprocessing}}/final.model.npz.best-{config["best-model-metric"]}.npz'
    output:
        hf_model="{preprocessing}/convert_to_hf/model.safetensors"
    params:
        model_name=f'final.model.npz.best-{config["best-model-metric"]}.npz'
    shell: '''python pipeline/hf/convert_marian_to_pytorch.py --src {wildcards.preprocessing} --dest {wildcards.preprocessing}/convert_to_hf --model_name {params.model_name}'''
        

rule ensemble_models:
    wildcard_constraints:
        model1="train_model[^\+]+(?=\+)",
        model2="(?<=\+)train_model[^\+/]+"
    message: "Creating an ensemble model decoder config"
    log: "{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/{model1}+{model2}.log"
    conda: "envs/base.yml"
    threads: 1
    input:
        model1_decoder_config=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/{{model1}}/final.model.npz.best-{config["best-model-metric"]}.npz.decoder.yml',
        model2_decoder_config=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/{{model2}}/final.model.npz.best-{config["best-model-metric"]}.npz.decoder.yml',
        vocab=ancient("{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm")
    output:
        decoder_config=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/{{model1}}+{{model2}}/final.model.npz.best-{config["best-model-metric"]}.npz.decoder.yml'
    params:
        decoder_1_weight=0.5
    shell: '''python pipeline/train/ensemble.py --decoder_file_1 "{input.model1_decoder_config}" --decoder_file_2 "{input.model2_decoder_config}" --output_decoder_file {output.decoder_config} --vocab_file {input.vocab} --decoder_1_weight {params.decoder_1_weight} >> {log} 2>&1'''

#This ensembles the chfr and ce-mean-words models
use rule ensemble_models as self_ensemble_model with:
    log: "{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/{model1}_selfensemble.log"
    input:
        model1_decoder_config=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/{{model1}}/final.model.npz.best-{config["best-model-metric"]}.npz.decoder.yml',
        model2_decoder_config=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/{{model1}}/model.npz.best-ce-mean-words.npz.decoder.yml',
        vocab=ancient("{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm")
    output:
        decoder_config=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/{{model1}}+selfensemble/final.model.npz.best-{config["best-model-metric"]}.npz.decoder.yml'

rule train_model:
    message: "Training a model"
    log: "{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{index_type}-{model_type}-{training_type}_train_model.log"
    conda: "envs/base.yml"
    envmodules:
        "LUMI/22.08",
        "partition/G",
        "rocm/5.3.3"
    threads: gpus_num*3
    resources: gpu=gpus_num,mem_mb=64000
    input:
        dev_source="{project_name}/{src}-{trg}/{preprocessing}/{index_type}-cleandev.{src}.gz",
        dev_target="{project_name}/{src}-{trg}/{preprocessing}/{index_type}-cleandev.{trg}.gz",
        train_source="{project_name}/{src}-{trg}/{preprocessing}/{index_type}-train.{src}.gz",
        train_target="{project_name}/{src}-{trg}/{preprocessing}/{index_type}-train.{trg}.gz",
        marian=ancient(config["marian"]),
        vocab=ancient("{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm")
    output: 
    	model=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/train_model_{{index_type}}-{{model_type}}-{{training_type}}/final.model.npz.best-{config["best-model-metric"]}.npz',
        decoder_config=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/train_model_{{index_type}}-{{model_type}}-{{training_type}}/final.model.npz.best-{config["best-model-metric"]}.npz.decoder.yml'
    params:
        args=config["training-teacher-args"]
    shell: f'''bash pipeline/train/train.sh \
                {{wildcards.model_type}} {{wildcards.training_type}} {{wildcards.src}} {{wildcards.trg}} "{{input.train_source}}" "{{input.train_target}}" "{{input.dev_source}}" "{{input.dev_target}}" "{{output.model}}" "{{input.vocab}}" "{config["best-model-metric"]}" {{params.args}} >> {{log}} 2>&1'''

use rule train_model as train_student_model with:
    message: "Training a student model"
    log: "{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/{postprocessing}/train_model_{model_type}-{training_type}/train_model.log"
    input:
        dev_source="{project_name}/{src}-{trg}/{preprocessing}/dev.{src}.gz",
        dev_target="{project_name}/{src}-{trg}/{preprocessing}/dev.{trg}.gz",
        train_source="{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/{postprocessing}/train.{src}.gz",
        train_target="{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/{postprocessing}/train.{trg}.gz",
        marian=ancient(config["marian"]),
        vocab="{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm"
    output: 
    	model=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/{{postprocessing}}/train_model_{{model_type}}-{{training_type}}/final.model.npz.best-{config["best-model-metric"]}.npz'
        

localrules: ct2_conversion

rule ct2_conversion:
    message: "Converting Marian model for ctranslate2 use"
    log: "{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/convert_model.log"
    conda: None
    container: None
    threads: 1
    input:
    	model=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/train_model_{{model_type}}-{{training_type}}/final.model.npz.best-{config["best-model-metric"]}.npz',
        vocab="{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm"
    output: 
    	model='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/ct2_conversion/model.bin'
    params:
        text_vocab="{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.vocab",
        yml_vocab="{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/conversion_vocab.yml",
        conversion_dir="{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/ct2_conversion"
    shell:
        """
            python pipeline/train/convert_vocab.py --input_vocab {params.text_vocab} --output_vocab {params.yml_vocab} >> {log} 2>&1 && \
            ct2-marian-converter --force --model_path {input.model} --vocab_paths {params.yml_vocab} {params.yml_vocab} --output_dir {params.conversion_dir} >> {log} 2>&1
        """ 

