wildcard_constraints:
    src="\w{2,3}",
    trg="\w{2,3}",
    train_vocab="train_joint_spm_vocab[^/]+",
    model_name="[^/]+",
    seg="(|_seg)"

gpus_num=config["gpus-num"]

localrules: download_opus_model, ct2_conversion
rule download_opus_model:
    message: "Downloading OPUS-MT teacher model"
    log: "{datadir}/models/{src}-{trg}/{model_name}/download_model_{model_name}.log"
    conda: None
    container: None
    threads: 1
    output: 
        model=f'{{datadir}}/models/{{src}}-{{trg}}/{{model_name}}/final.model.npz.best-{config["best-model-metric"]}.npz',
        vocab=f'{{datadir}}/models/{{src}}-{{trg}}/{{model_name}}/vocab.yml',
        src_spm='{datadir}/models/{src}-{trg}/{model_name}/source.spm', 
        trg_spm='{datadir}/models/{src}-{trg}/{model_name}/target.spm'
    params: 
        model_dir='{datadir}/models/{src}-{trg}/{model_name}' 
    shell: '''bash pipeline/opusmt/download-model.sh \
                "{wildcards.model_name}" "{params.model_dir}" "{output.model}" "{wildcards.src}" "{wildcards.trg}" >> {log} 2>&1'''

# Prepare corpus for finetuning, this is separate from the finetune rule as that runs on GPU and this does longish CPU prep
rule prepare_finetune_opusmt:
    message: "Preparing corpus for finetuning OPUS-MT model"
    log: "{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_{epochs}_{model_name}/finetune.log"
    conda: "envs/base.yml"
    wildcard_constraints:
        learning_rate="\d+"
    threads: 1
    input:
        basemodeldir="{datadir}/models/{src}-{trg}/{model_name}",
        dev_source="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/dev.{src}.gz",
        dev_target="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/dev.{trg}.gz",
        train_source="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/train.{src}.gz",
        train_target="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/train.{trg}.gz",
        marian=ancient(config["marian"]),
        #vocab="{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm"
    output: 
        dev_source="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_{epochs}_{model_name}/valid.sp.{src}.gz",
        dev_target="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_{epochs}_{model_name}/valid.sp.{trg}.gz",
        train_source="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_{epochs}_{model_name}/train.sp.{src}.gz",
        train_target="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_{epochs}_{model_name}/train.sp.{trg}.gz",
        vocab="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_{epochs}_{model_name}/vocab.yml",
    params: 
        prefix_train="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/train",
        prefix_dev="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/dev",
        segmented_input=lambda wildcards: "true" if wildcards.seg else "false",
        modeldir="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_{epochs}_{model_name}/"
    shell: '''bash pipeline/opusmt/prep_finetune.sh {wildcards.src} {wildcards.trg} "{params.prefix_train}" "{params.prefix_dev}" "{params.modeldir}" "{input.basemodeldir}" "{params.segmented_input}" >> {log} 2>&1'''

rule finetune_opusmt:
    message: "Finetune OPUS-MT model on corpus"
    log: "{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_{epochs}_{model_name}/finetune.log"
    conda: "envs/base.yml"
    wildcard_constraints:
        learning_rate="\d+"
    threads: gpus_num*7
    resources: gpu=gpus_num
    input:
        model=lambda wildcards: f'{wildcards.datadir}/models/{wildcards.src}-{wildcards.trg}/{wildcards.model_name}/final.model.npz.best-{config["best-model-metric"]}.npz' if wildcards.epochs == "1" else f'{wildcards.datadir}/{wildcards.project_name}/{wildcards.src}-{wildcards.trg}/{wildcards.preprocessing}/finetune{wildcards.seg}_{wildcards.learning_rate}_{int(wildcards.epochs)-1}_{wildcards.model_name}/final.model.npz.best-{config["best-model-metric"]}.npz',
        dev_source="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_1_{model_name}/valid.sp.{src}.gz",
        dev_target="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_1_{model_name}/valid.sp.{trg}.gz",
        train_source="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_1_{model_name}/train.sp.{src}.gz",
        train_target="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_1_{model_name}/train.sp.{trg}.gz",
        marian=ancient(config["marian"]),
        vocab="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune{seg}_{learning_rate}_1_{model_name}/vocab.yml",
    output: 
        model=f'{{datadir}}/{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/finetune{{seg}}_{{learning_rate}}_{{epochs}}_{{model_name}}/final.model.npz.best-{config["best-model-metric"]}.npz'
    params: 
        args=config["finetune-args"],
        best_metric=config["best-model-metric"],
        segmented_input=lambda wildcards: "true" if wildcards.seg else "false" 
    shell: '''bash pipeline/opusmt/finetune.sh {wildcards.src} {wildcards.trg} "{output.model}" "{input.model}" "{params.best_metric}" {threads} "0.{wildcards.learning_rate}" "{wildcards.epochs}" "{params.segmented_input}" "{input.vocab}" {params.args} >> {log} 2>&1'''


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

