wildcard_constraints:
    src="\w{2,3}",
    trg="\w{2,3}",
    train_vocab="train_joint_spm_vocab[^/]+",
    learn_rate="\d+",
    epochs="\d+",
    nocrawled="(|_nocrawled)",
    min_score="\d\.\d"

gpus_num=config["gpus-num"]

def find_domain_sets(wildcards, checkpoint):
    checkpoint_output = checkpoint.get(src=wildcards.src,trg=wildcards.trg,project_name=wildcards.project_name,download_tc_dir=wildcards.download_tc_dir,min_score=wildcards.min_score,nocrawled=wildcards.nocrawled).output["subcorpora"]
    return glob_wildcards(os.path.join(checkpoint_output,f"{{domain,.*}}.{wildcards.src}.gz")).domain

def find_translate_sets(wildcards, checkpoint):
    checkpoint_output = checkpoint.get(**wildcards).output["output_dir"]
    return glob_wildcards(os.path.join(checkpoint_output,f"{{domain,.*}}.{wildcards.src}.gz")).domain

#TODO: for domeval, only the fuzzy sentences need to be translated for each index. The non-fuzzies can be reused from a common non-fuzz translation file (generate this separately). Otherwise translation takes ages.

# This translates the domeval sets with various indexes in an economincal fashion, i.e. only translating fuzzies
# TODO: this could also be done with single-file rules, if the non-fuzzy file were to be translated
# first. Would make the process cleaner, with no need for the output dir
checkpoint translate_domeval:
    message: "Translating domain evaluation data"
    log: "{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/{preprocessing}/{train_vocab}/{train_model}/eval/translate_domeval.log"
    conda: None
    container: None
    resources: gpu=gpus_num
    envmodules:
        "LUMI/22.08",
        "partition/G",
        "rocm/5.3.3"
    threads: 1
    priority: 50
    input:
        decoder=ancient(config["marian-decoder"]),
    	domain_src=lambda wildcards: expand("{{project_name}}/{{src}}-{{trg}}/{{download_tc_dir}}/extract_tc_scored_{{min_score}}{{nocrawled}}/{{preprocessing}}/{domain}-domeval.{{src}}.gz", domain=find_domain_sets(wildcards, checkpoints.extract_tc_scored)),
        train_src="{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/{preprocessing}/train-domeval.{src}.gz",
        all_filtered_src="{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/{preprocessing}/all_filtered-domeval.{src}.gz",
        decoder_config=f'{{project_name}}/{{src}}-{{trg}}/{{download_tc_dir}}/extract_tc_scored_{{min_score}}{{nocrawled}}/{{preprocessing}}/{{train_vocab}}/{{train_model}}/final.model.npz.best-{config["best-model-metric"]}.npz.decoder.yml' 
    output:
        output_dir=directory("{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/{preprocessing}/{train_vocab}/{train_model}/eval/domeval")
    params:
        domain_index_src_dir="{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/{preprocessing}",
	uses_bands=lambda wildcards: "false" if "nobands" in wildcards.train_model else "true"
    shell: '''pipeline/eval/translate-domeval.sh {params.domain_index_src_dir} {output.output_dir} {wildcards.src} {wildcards.trg} {input.decoder} {input.decoder_config} {params.uses_bands} --mini-batch 128 --workspace 20000 >> {log} 2>&1'''

# This evaluates the translations generated with translate_domeval
rule eval_domeval:
    message: "Evaluating domain translation quality"
    log: "{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/{preprocessing}/{train_vocab}/{train_model}/eval/evaluate_domains.log"
    conda: None
    container: None
    threads: 1
    priority: 50
    input:
        domain_index_trg=lambda wildcards: expand("{{project_name}}/{{src}}-{{trg}}/{{download_tc_dir}}/extract_tc_scored_{{min_score}}{{nocrawled}}/{{preprocessing}}/{{train_vocab}}/{{train_model}}/eval/domeval/{domain}-domeval.{{trg}}.gz", domain=find_translate_sets(wildcards, checkpoints.translate_domeval))
        #baseline_translations="{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/subset_5M/baseline_preprocessing_2000/train_joint_spm_vocab_50000_prepend/train_model_train-baseteacher-train/eval/domeval.{trg}"
    output:
        report('{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/{preprocessing}/{train_vocab}/{train_model}/eval/domeval.done',
            category='evaluation', subcategory='model', caption='reports/evaluation.rst')
    params:
        input_dir="{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/{preprocessing}/{train_vocab}/{train_model}/eval/domeval",
        domeval_ids="{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/domeval.ids.gz",
        system_id="{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/{preprocessing}/{train_vocab}/{train_model}",
        baseline_translations="{project_name}/{src}-{trg}/{download_tc_dir}/extract_tc_scored_{min_score}{nocrawled}/subset_5M/baseline_preprocessing_2000/train_joint_spm_vocab_50000_prepend/train_model_train-baseteacher-train/eval/domeval.{trg}"
    shell: '''python pipeline/eval/score-domeval.py  --input_dir {params.input_dir} --report {output} --src_lang {wildcards.src} --trg_lang {wildcards.trg} --system_id {params.system_id} --domeval_ids {params.domeval_ids} --baseline_translations {params.baseline_translations} >> {log} 2>&1'''

# Add a rule that combines domeval dirs from two models, and uses one of the models as short sentence backup (or just hardcode this)

rule evaluate:
    message: "Evaluating a model"
    log: "{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/eval/evaluate_{dataset}.log"
    conda: "envs/base.yml"
    threads: 7
    resources: gpu=1
    priority: 50
    group: "evaluate"
        
    input:
        ancient(config["marian-decoder"]),
        eval_source='{project_name}/{src}-{trg}/{preprocessing}/{dataset}.{src}.gz',
        eval_target='{project_name}/{src}-{trg}/{preprocessing}/{dataset}.{trg}.gz',
        src_spm='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm',
        trg_spm='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm',
    	model=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/train_model_{{model_type}}-{{training_type}}/final.model.npz.best-{config["best-model-metric"]}.npz'
    output:
        metrics=report('{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/eval/{dataset}.metrics',
            category='evaluation', subcategory='{model}', caption='reports/evaluation.rst'),
        translations='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/eval/{dataset}.{trg}'
    params:
        dataset_prefix='{project_name}/{src}-{trg}/{preprocessing}/{dataset}',
        res_prefix='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/eval/{dataset}',
        src_lng=lambda wildcards: wildcards.src if "backward" not in wildcards.model_type else wildcards.trg,
        trg_lng=lambda wildcards: wildcards.trg if "backward" not in wildcards.model_type else wildcards.src,
        decoder_config=f'{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/{{train_vocab}}/train_model_{{model_type}}-{{training_type}}/final.model.npz.best-{config["best-model-metric"]}.npz.decoder.yml',
	decoder=config["marian-decoder"]
    shell: '''bash pipeline/eval/eval-gpu.sh "{params.res_prefix}" "{params.dataset_prefix}" {params.src_lng} {params.trg_lng} {params.decoder} "{params.decoder_config}" >> {log} 2>&1'''


rule evaluate_ct2:
    message: "Evaluating a model using ctranslate2"
    log: "{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/eval/evaluate_ct2_{dataset}.log"
    #conda: "envs/base.yml"
    conda: None
    container: None
    threads: workflow.cores
    #resources: gpu=1
    priority: 50
    wildcard_constraints:
        model="[\w-]+"
    input:
        ancient(config["marian-decoder"]),
        eval_source='{project_name}/{src}-{trg}/{preprocessing}/{dataset}.{src}.gz',
        eval_target='{project_name}/{src}-{trg}/{preprocessing}/{dataset}.{trg}.gz',
        src_spm='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm',
        trg_spm='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/vocab.spm',
    	model='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/ct2_conversion/model.bin'
    output:
        report('{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/eval/{dataset}.ct2.metrics',
            category='evaluation', subcategory='{model}', caption='reports/evaluation.rst')
    params:
        dataset_prefix='{project_name}/{src}-{trg}/{preprocessing}/{dataset}',
        res_prefix='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/eval/{dataset}',
        src_lng=lambda wildcards: wildcards.src if "backward" not in wildcards.model_type else wildcards.trg,
        trg_lng=lambda wildcards: wildcards.trg if "backward" not in wildcards.model_type else wildcards.src,
        ct2_model_dir='{project_name}/{src}-{trg}/{preprocessing}/{train_vocab}/train_model_{model_type}-{training_type}/ct2_conversion'
    shell: '''bash pipeline/eval/eval-ct2.sh "{params.res_prefix}" "{params.dataset_prefix}" {params.src_lng} {params.trg_lng} {params.ct2_model_dir} "{input.src_spm}" {threads} 1 >> {log} 2>&1'''


#TODO: combine model evaluation rules by storing vocabs in model dir with normally trained models as well
rule evaluate_opus_model:
    message: "Evaluating an OPUS model"
    log: "{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune_{learn_rate}_{epochs}_{model_name}/eval/evaluate_{modeltype}{dataset}.log"
    conda: None
    container: None
    threads: 7
    resources: gpu=1
    priority: 50
    wildcard_constraints:
        modeltype="(basemodel-|)"
    input:
        ancient(config["marian-decoder"]),
        eval_source='{datadir}/{project_name}/{src}-{trg}/{preprocessing}/{dataset}.{src}.gz',
        eval_target='{datadir}/{project_name}/{src}-{trg}/{preprocessing}/{dataset}.{trg}.gz',
        model=f'{{datadir}}/{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/finetune_{{learn_rate}}_{{epochs}}_{{model_name}}/final.model.npz.best-{config["best-model-metric"]}.npz'
    output:
        report('{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune_{learn_rate}_{epochs}_{model_name}/eval/{modeltype}{dataset}.metrics',
            category='evaluation', subcategory='{model}', caption='reports/evaluation.rst')
    params:
        dataset_prefix='{datadir}/{project_name}/{src}-{trg}/{preprocessing}/{dataset}',
        res_prefix='{datadir}/{project_name}/{src}-{trg}/{preprocessing}/finetune_{learn_rate}_{epochs}_{model_name}/eval/{modeltype}{dataset}',
        decoder_config=
            lambda wildcards: f'{wildcards.datadir}/models/{wildcards.src}-{wildcards.trg}/{wildcards.model_name}/decoder.yml' if wildcards.modeltype=="basemodel-" else f'{wildcards.datadir}/{wildcards.project_name}/{wildcards.src}-{wildcards.trg}/{wildcards.preprocessing}/finetune_{wildcards.learn_rate}_{wildcards.epochs}_{wildcards.model_name}/final.model.npz.best-{config["best-model-metric"]}.npz.decoder.yml',
    	decoder=config["marian-decoder"]
    shell: '''bash pipeline/eval/eval-gpu.sh "{params.res_prefix}" "{params.dataset_prefix}" {wildcards.src} {wildcards.trg} {params.decoder} "{params.decoder_config}" >> {log} 2>&1'''
