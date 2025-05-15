import yaml
import os
import glob
import hashlib

from snakemake.utils import min_version

containerized: 'Ftt.sif'

min_version("6.6.1")
# include statement will include the code in the file as is, into the same variable scope. This is why the configuration (specifying directories etc.) is done with include, those configuration settings need to be in the global scope in the main Snakefile (but it's cleaner to have them in a separate file to reduce clutter). 
include: "./configuration.smk" 

### helper functions

def find_parts(wildcards, checkpoint):
    checkpoint_output = checkpoint.get(**wildcards).output[0]
    return glob_wildcards(os.path.join(checkpoint_output,"file.{part,\d+}")).part

def find_annotation_parts(wildcards, checkpoint):
    checkpoint_output = checkpoint.get(**wildcards).output[0]
    return glob_wildcards(os.path.join(checkpoint_output,"file.src.{part,\d+}.gz")).part

def dataset_norm(name: str):
    return name.replace('/','_')

def get_args(section):
    return marian_args.get(section) or ""

wildcard_constraints:
    src="\w{2,3}",
    trg="\w{2,3}"

#Sub-workflows are included as modules, which have their own variable scope, they don't inherit variables from the main Snakefile. Now it would be possible to also include the configuration.smk in the sub-workflow files, but that seems like a bad practise since most sub-workflows only use a couple of the global settings, they don't need access to the whole configuration.

#There are examples of sub-workflows as modules below. The module rat is a semi-complete example of how I think we should proceed. The compile_deps and data modules use a different approach which I decided not to pursue, so ignore them.

rat_config = {
    "fuzzy-match-cli": f"{bin}/FuzzyMatch-cli"}

# The prefix value is a directory that will be appended to all the relative paths in the module, so effectively it's the output dir value. So we control input using the configuration file, and output using the prefix value in the module statement.
module rat:
    snakefile: "./rat.smk"
    config: rat_config

use rule * from rat as *

vocab_config = {
    "spm-train": f"{marian_dir}/spm_train",
    "user-defined-symbols": ",".join(["FUZZY_BREAK","SRC_FUZZY_BREAK"] + [f"FUZZY_BREAK_{bucket}" for bucket in range(0,10)]),
    "spm-sample-size": 1000000,
    "spm-character-coverage": 1.0
    }

module vocab:
    snakefile: "./vocab.smk"
    config: vocab_config

use rule * from vocab

terms_config = {
    "spm-encoder": f"{marian_dir}/spm_encode",
    "fast-align": f"{bin}/fast_align",
    "atools": f"{bin}/atools"
    }

module terms:
    snakefile: "./terms.smk"
    config: terms_config

use rule * from terms


train_config = {
    "marian": f"{marian_dir}/marian",
    "gpus-num": gpus_num,
    "best-model-metric": best_model_metric,
    "training-teacher-args": get_args("training-teacher")}


module train:
    snakefile: "./train.smk"
    config: train_config

use rule * from train

opusmt_config = {
    "marian": f"{marian_dir}/marian",
    "gpus-num": gpus_num,
    "best-model-metric": best_model_metric,
    "finetune-args": get_args("finetune")}

module opusmt:
    snakefile: "./opusmt.smk"
    config: opusmt_config

use rule * from opusmt

eval_config = {
    "marian-decoder": f"{marian_dir}/marian-decoder",
    "gpus-num": gpus_num,
    "best-model-metric": best_model_metric}


module eval:
    snakefile: "./eval.smk"
    config: eval_config

use rule * from eval

translate_config = {
    "decoder": f"{marian_dir}/marian-decoder",
    "gpus-num": gpus_num,
    "best-model-metric": best_model_metric,
    "decoding-teacher-args": get_args("decoding-teacher")}


module translate:
    snakefile: "./translate.smk"
    config: translate_config

use rule * from translate

#Ignore these modules, they use a dead-end approach, will change it later.
module compile_deps:
    snakefile: "./compile_deps.smk"
    config: config

use rule * from compile_deps

module data:
    snakefile: "./data.smk"
    config: config

use rule * from data

# set common environment variables
envs = f'''SRC={src} TRG={trg} MARIAN="{marian_dir}" BMT_MARIAN="{bmt_marian_dir}" GPUS="{gpus}" WORKSPACE={workspace} \
BIN="{bin}" CUDA_DIR="{cuda_dir}" CUDNN_DIR="{cudnn_dir}" ROCM_PATH="{rocm_dir}" '''
# CUDA_VISIBLE_DEVICES is used by bicleaner ai. slurm sets this variable
# it can be overriden manually by 'gpus' config setting to split GPUs in local mode
# Note that this will also work with AMD GPUs, they recognize this env variable
envs += f' CUDA_VISIBLE_DEVICES="{gpus}" '

### workflow options
results = []

if train_student:
    results.extend([
        f'{experiment_dir}/config.yml',
        *expand(f'{eval_student_dir}/{{dataset}}.metrics',dataset=eval_datasets)
        ])

if quantize_student:
    results.extend([
        f'{exported_dir}/model.{src}{trg}.intgemm.alphas.bin.gz',
        f'{exported_dir}/lex.50.50.{src}{trg}.s2t.bin.gz',
        f'{exported_dir}/vocab.{src}{trg}.spm.gz',
        *expand(f'{eval_student_finetuned_dir}/{{dataset}}.metrics',dataset=eval_datasets),
        *expand(f'{eval_speed_dir}/{{dataset}}.metrics',dataset=eval_datasets)
    ])

#if rat:
    	
mixture_hash = None
if wmt23_termtask:
    finetune_teacher_with_terms = wmt23_termtask.get('finetune-teacher-with-terms') 
    train_term_teacher = wmt23_termtask.get('train-term-teacher') 
    mixture_of_models = wmt23_termtask.get('mixture-of-models')

    annotation_schemes = wmt23_termtask['annotation-schemes']
    term_ratios = wmt23_termtask['term-ratios']
    sents_per_term_sents = wmt23_termtask['sents-per-term-sents']

    #results.extend(expand(f'{eval_res_dir}/teacher-base0-{{ens}}/wmt23_termtask.score',ens=ensemble))
    results.extend(expand(f'{eval_res_dir}/teacher-base0-{{ens}}/evalsets_terms.score',ens=ensemble))
    results.extend(expand(f'{eval_res_dir}/teacher-base0-{{ens}}/evalsets_terms.noterms.score',ens=ensemble))

    if finetune_teacher_with_terms: 
        #results.extend(expand(f'{eval_res_dir}/teacher-base-finetuned-term-{{annotation_scheme}}-{{term_ratio}}-{{sents_per_term_sent}}{{omit}}/wmt23_termtask.score',
        #    annotation_scheme=annotation_schemes,
        #    term_ratio=term_ratios,
        #    sents_per_term_sent=sents_per_term_sents,
        #    omit=omit_unannotated))

        #results.extend(expand(f'{eval_res_dir}/teacher-base-finetuned-term-{{annotation_scheme}}-{{term_ratio}}-{{sents_per_term_sent}}{{omit}}/evalsets_terms.score',
        #    annotation_scheme=annotation_schemes,
        #    term_ratio=term_ratios,
        #    sents_per_term_sent=sents_per_term_sents,
        #    omit=omit_unannotated))

        results.extend(expand(f'{eval_res_dir}/teacher-base-finetuned-term-{{annotation_scheme}}-{{term_ratio}}-{{sents_per_term_sent}}{{omit}}/evalsets_terms.score',
            annotation_scheme=annotation_schemes,
            term_ratio=term_ratios,
            sents_per_term_sent=sents_per_term_sents,
            omit=omit_unannotated))
        
        results.extend(expand(f'{eval_res_dir}/teacher-base-finetuned-term-{{annotation_scheme}}-{{term_ratio}}-{{sents_per_term_sent}}{{omit}}/evalsets_terms.noterms.score',
            annotation_scheme=annotation_schemes,
            term_ratio=term_ratios,
            sents_per_term_sent=sents_per_term_sents,
            omit=omit_unannotated))

	results.extend(expand(f'{eval_res_dir}/teacher-base-finetuned-term-{{annotation_scheme}}-{{term_ratio}}-{{sents_per_term_sent}}{{omit}}/{{dataset}}.metrics',
            annotation_scheme=annotation_schemes,
            term_ratio=term_ratios,
            sents_per_term_sent=sents_per_term_sents,
            omit=omit_unannotated,
            dataset=eval_datasets))

    if mixture_of_models:
        mixture_hash = hashlib.md5(("+".join(mixture_of_models)).encode("utf-8")).hexdigest()
        #results.extend(expand(f'{eval_res_dir}/mixture-{mixture_hash}/testset_terms.mixture.{trg}'))
        #results.extend(expand(f'{eval_res_dir}/mixture-{mixture_hash}/blindset_terms.mixture.{trg}'))
        results.extend(expand(f'{eval_res_dir}/mixture-{mixture_hash}/evalsets_terms.mixture.{trg}'))
    else:
        mixture_hash = None    

    if train_term_teacher:
#            results.extend(expand(f'{eval_res_dir}/teacher-base-term-{{annotation_scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/wmt23_termtask.score',
#                annotation_scheme=annotation_schemes,
#                term_ratio=term_ratios,
#                sents_per_term_sent=sents_per_term_sents))

        results.extend(expand(f'{eval_res_dir}/teacher-base-term-{{annotation_scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/{{dataset}}.metrics',
            annotation_scheme=annotation_schemes,
            term_ratio=term_ratios,
            sents_per_term_sent=sents_per_term_sents,dataset=eval_datasets))


    if train_student:
        results.extend([f'{eval_student_dir}/wmt23_termtask.score'])
        
        results.extend(expand(f'{eval_student_dir}-term-{{annotation_scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/wmt23_termtask.score',
            annotation_scheme=annotation_schemes,
            term_ratio=term_ratios,
            sents_per_term_sent=sents_per_term_sents))

        results.extend(expand(f'{eval_student_dir}-term-{{annotation_scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/{{dataset}}.metrics',
            annotation_scheme=annotation_schemes,
            term_ratio=term_ratios,
            sents_per_term_sent=sents_per_term_sents,dataset=eval_datasets))
         


results.extend(expand(f'{eval_res_dir}/teacher-base0-{{ens}}/{{dataset}}.metrics',ens=ensemble, dataset=eval_datasets))

if len(ensemble) > 1:
    results.extend(expand(f'{eval_teacher_ens_dir}/{{dataset}}.metrics', dataset=eval_datasets))

if install_deps:
    results.append("/tmp/flags/setup.done")

#three options for backward model: pretrained path, url to opus-mt, or train backward
if backward_pretrained:
    do_train_backward = False
    backward_dir = backward_pretrained
elif opusmt_backward:
    do_train_backward = False 
else:
    # don't evaluate pretrained model
    if train_student:
        results.extend(expand(f'{eval_backward_dir}/{{dataset}}.metrics',dataset=eval_datasets))
        do_train_backward=True
    else:
        do_train_backward=False

# augmentation

if mono_trg_datasets and not (opusmt_teacher or forward_pretrained):
    teacher_corpus = f'{augmented}/corpus'
    augment_corpus = True
    final_teacher_dir = teacher_finetuned_dir
    results.extend(expand(f'{eval_res_dir}/teacher-finetuned0-{{ens}}/{{dataset}}.metrics',ens=ensemble, dataset=eval_datasets))
else:
    augment_corpus = False
    final_teacher_dir = teacher_base_dir


### rules

shell.prefix(f"{envs} ")

model = (config["experiment"]["opusmt-teacher"].split("/")[-1]).replace(".zip","")

# results = expand(f"{data_root_dir}/{experiment}/{src}-{trg}/corpus_{{corpus}}/finetune_{{learning_rate}}_{{epochs}}_{model}/eval/eval-{{dataset}}.metrics", corpus=config["datasets"]["train"], learning_rate=config["experiment"]["finetune"]["learning-rates"],epochs=range(1,config["experiment"]["finetune"]["epochs"]+1), dataset=eval_datasets)

# For base model, only generate the metrics once
# results.extend(expand(f"{data_root_dir}/{experiment}/{src}-{trg}/corpus_{{corpus}}/finetune_{{learning_rate}}_{{epochs}}_{model}/eval/basemodel-eval-{{dataset}}.metrics", corpus=config["datasets"]["train"], learning_rate=config["experiment"]["finetune"]["learning-rates"][0], epochs=1, dataset=eval_datasets))

#print(results)

rule all:
    input: results

wildcard_constraints:
    term_ratio="\d+",
    sents_per_term_sent="\d+"

localrules: experiment

rule experiment:
    message: "Saving experiment metadata"
    output: f'{experiment_dir}/config.yml'
    priority: 100   
    run:
        os.makedirs(experiment_dir, exist_ok=True)
        with open(f'{experiment_dir}/config.yml', 'w') as f:
            yaml.dump(config, f)

# todo: fix jobs grouping in cluster mode

# setup

if install_deps:
    rule setup:
        message: "Installing dependencies"
        log: f"{log_dir}/install-deps.log"
        conda: "envs/base.yml"
        priority: 99
        # group: 'setup'
        output: touch("/tmp/flags/setup.done")  # specific to local machine
        shell: 'bash pipeline/setup/install-deps.sh >> {log} 2>&1'

# augmentation and teacher training


if do_train_backward: 
    mono_trg_file = f'{translated}/mono_trg/file.{{part}}'
    deseg_mono_trg_outfile = f'{mono_trg_file}.out'
    
    rule train_backward:
        message: "Training backward model"
        log: f"{log_dir}/train_backward.log"
        conda: "envs/base.yml"
        threads: gpus_num * 2
        resources: gpu=gpus_num
        #group 'backward'
        input:
            rules.merge_devset.output, train_src=clean_corpus_src,train_trg=clean_corpus_trg,
            bin=ancient(trainer), vocab=vocab_path,
        output:  model=f'{backward_dir}/{best_model}'
        params: prefix_train=clean_corpus_prefix,prefix_test=f"{original}/devset",
                args=get_args("training-backward")
        shell: '''bash pipeline/train/train.sh \
                    backward train {trg} {src} "{params.prefix_train}" "{params.prefix_test}" "{backward_dir}" \
                    "{input.vocab}" "{best_model_metric}" {params.args} >> {log} 2>&1'''

elif opusmt_backward:
    mono_trg_file = f'{translated}/mono_trg/file.{{part}}.{{model_index}}.opusmt'
    deseg_mono_trg_outfile = f'{mono_trg_file}.out.deseg'
    
    rule download_opusmt_backward:
        message: "Downloading OPUS-MT backward model"
        log: f"{log_dir}/download_backward.log"
        conda: "envs/base.yml"
        output:  model=f'{backward_dir}/{best_model}',vocab=f'{backward_dir}/vocab.yml', model_dir=directory({backward_dir})
        shell: '''bash pipeline/opusmt/download-model.sh \
                    "{opusmt_backward}" "{backward_dir}" "{best_model}" {trg_three_letter} {src_three_letter} >> {log} 2>&1''' 


if augment_corpus:
    checkpoint split_mono_trg:
        message: "Splitting monolingual trg dataset"
        log: f"{log_dir}/split_mono_trg.log"
        conda: "envs/base.yml"
        threads: 1
        input: corpora=f"{clean}/mono.{trg}.gz", bin=ancient(deduper)
        output: directory(f'{translated}/mono_trg')
        shell: 'bash pipeline/translate/split-mono.sh {input.corpora} {output} {split_length} >> {log} 2>&1'

    #TODO: make it possible to use multiple backward models, add filtering for backtranslations
    #TODO: add preprocessing and deseg for OPUS-MT backward model backtranslation, currently works only with trained backward model
    rule translate_mono_trg:
        message: "Translating monolingual trg dataset with backward model"
        log: f"{log_dir}/translate_mono_trg/{{part}}.log"
        conda: "envs/base.yml"
        threads: gpus_num * 2
        resources: gpu=gpus_num
        input:
            bin=ancient(decoder), file=mono_trg_file,
            vocab=vocab_path, model=f'{backward_dir}/{best_model}'
        output: file=f'{mono_trg_file}.out'
        params: args = get_args("decoding-backward")
        shell: '''bash pipeline/translate/translate.sh "{input.file}" "{output.file}" "{input.vocab}" {input.model} {params.args} \
                >> {log} 2>&1'''

    rule collect_mono_trg:
        message: "Collecting translated mono trg dataset"
        log: f"{log_dir}/collect_mono_trg.log"
        conda: "envs/base.yml"
        threads: 4
        #group 'mono_trg'
        input:
            lambda wildcards: expand(deseg_mono_trg_outfile,
                part=find_parts(wildcards, checkpoints.split_mono_trg))
        output: f'{translated}/mono.{src}.gz'
        params: src_mono=f"{clean}/mono.{trg}.gz",dir=directory(f'{translated}/mono_trg')
        shell: 'bash pipeline/translate/collect.sh "{params.dir}" "{output}" "{params.src_mono}" "" >> {log} 2>&1'

    rule merge_augmented:
        message: "Merging augmented dataset"
        log: f"{log_dir}/merge_augmented.log"
        conda: "envs/base.yml"
        threads: 4
        #group 'mono_trg'
        input:
            src1=clean_corpus_src,
            src2=rules.collect_mono_trg.output,
            trg1=clean_corpus_trg,
            trg2=rules.split_mono_trg.input.corpora,
            bin=ancient(deduper)
        output: res_src=f'{augmented}/corpus.{src}.gz',res_trg=f'{augmented}/corpus.{trg}.gz'
        shell: '''bash pipeline/translate/merge-corpus.sh \
                    "{input.src1}" "{input.src2}" "{input.trg1}" "{input.trg2}" "{output.res_src}" "{output.res_trg}" "" \
                      >> {log} 2>&1'''






    #rule teacher_alignments:
    #    message: 'Training word alignment and lexical shortlists'
    #    log: f"{log_dir}/alignments.log"
    #    conda: "envs/base.yml"
    #    threads: workflow.cores
    #    input:
    #        ancient(spm_encoder), ancient(spm_exporter),
    #        src_corpus=f'{teacher_corpus}.{src}.gz',trg_corpus=f'{teacher_corpus}.{trg}.gz',
    #        vocab=vocab_path,
    #        fast_align=ancient(rules.fast_align.output.fast_align), atools=ancient(rules.fast_align.output.atools),
    #        extract_lex=ancient(rules.extract_lex.output)
    #    output: alignment=f'{teacher_align_dir}/corpus.aln.gz',shortlist=f'{teacher_align_dir}/lex.s2t.pruned.gz'
    #    params: input_prefix=teacher_corpus
    #    shell: '''bash pipeline/alignment/generate-alignment-and-shortlist.sh \
    #                "{params.input_prefix}" "{input.vocab}" "{teacher_align_dir}" {threads} >> {log} 2>&1'''

if augment_corpus:
    rule finetune_teacher:
        message: "Finetune teacher on parallel corpus"
        log: f"{log_dir}/finetune_teacher0-{{ens}}.log"
        conda: "envs/base.yml"
        threads: gpus_num * 2
        resources: gpu=gpus_num
        input:
            rules.merge_devset.output, model=f'{teacher_base_dir}0-{{ens}}/{best_model}',
            train_src=clean_corpus_src, train_trg=clean_corpus_trg,
            bin=ancient(trainer), vocab=vocab_path
        output: model=f'{teacher_finetuned_dir}0-{{ens}}/{best_model}'
        params: prefix_train=clean_corpus_prefix, prefix_test=f"{original}/devset",
                dir=directory(f'{teacher_finetuned_dir}0-{{ens}}'),
                args=get_args("training-teacher-finetuned")
        shell: '''bash pipeline/train/train.sh \
                    teacher train {src} {trg} "{params.prefix_train}" "{params.prefix_test}" "{params.dir}" \
                    "{input.vocab}" "{best_model_metric}" --pretrained-model "{input.model}" {params.args} >> {log} 2>&1'''

### translation with teacher

"""checkpoint split_corpus:
    message: "Splitting the corpus to translate"
    log: f"{log_dir}/split_corpus.log"
    conda: "envs/base.yml"
    threads: 1
    input: corpus_src=clean_corpus_src,corpus_trg=clean_corpus_trg
    output: directory(f"{translated}/corpus")
    shell: '''bash pipeline/translate/split-corpus.sh \
                {input.corpus_src} {input.corpus_trg} {output} {split_length} >> {log} 2>&1'''

if opusmt_teacher:
    teacher_source_file = f'{translated}/corpus/file.{{part}}.{{model_index}}.opusmt'
    teacher_target_file = f'{translated}/corpus/file.{{part}}.{{model_index}}.opusmt.nbest'
    teacher_mono_source_file = f'{translated}/mono_src/file.{{part}}.{{model_index}}.opusmt'
    teacher_mono_target_file = f'{translated}/mono_src/file.{{part}}.{{model_index}}.opusmt.out'
    translated_mono_src_extension = "opusmt.out"
    deseg_nbest_file = f'{teacher_target_file}.deseg'
    
    rule opusmt_deseg_translation:
        message: "Desegmenting OPUS-MT model translation"
        log: f"{log_dir}/opusmt_deseg_mono_translation/{{part}}.{{model_index}}.log"
        threads: 1
        wildcard_constraints:
            model_index="\d+"
        input: f'{translated}/mono_src/file.{{part}}.{{model_index}}.opusmt.out'
        output: f'{translated}/mono_src/file.{{part}}.{{model_index}}.out'
        run: 
            with open(input[0], "rt", encoding="utf8") as infile,open(output[0], "wt", encoding="utf8") as outfile:
                for line in infile:
                    deseg_line = line.replace(" ","").replace("▁"," ")
                    outfile.write(deseg_line)

    #This is an optional rule that only applies when OPUS-MT model is used as teacher.
    #Required due to OPUS-MT models not using the integrated SentencePiece in Marian
    rule opusmt_preprocess_corpus:
        message: "Preprocessing source file for OPUS-MT model"
        log: f"{log_dir}/opusmt_preprocess_corpus/{{corpus}}.{{part}}.{{model_index}}.log"
        conda: "envs/base.yml"
        threads: 1
        input: 
            file=f'{translated}/{{corpus}}/file.{{part}}', 
            teacher_model=f"{final_teacher_dir}{{model_index}}-0/{best_model}",
            spm_encoder=ancient(spm_encoder)
        output: f'{translated}/{{corpus}}/file.{{part}}.{{model_index}}.opusmt'
        shell: '''bash pipeline/translate/opusmt-preprocess.sh \
                    {input.file} {input.teacher_model} src "source.spm" {input.spm_encoder} {target_language_token} {wildcards.model_index} >> {log} 2>&1'''
    rule opusmt_deseg_nbest:
        message: "Desegmenting OPUS-MT model nbest list"
        log: f"{log_dir}/opusmt_deseg_nbest/{{part}}.{{model_index}}.log"
        threads: 1
        input: nbest=f"{teacher_source_file}.nbest"
        output: temp(deseg_nbest_file)
        run: 
            with open(input[0], "rt", encoding="utf8") as infile,open(output[0], "wt", encoding="utf8") as outfile:
                for line in infile:
                    line_split = line.split(" ||| ")
                    line_split[1] = line_split[1].replace(" ","").replace("▁"," ")
                    outfile.write(" ||| ".join(line_split))
else:    
    teacher_source_file = f'{translated}/corpus/file.{{part}}'
    teacher_target_file = f'{translated}/corpus/file.{{part}}.{{model_index}}.nbest'
    teacher_mono_source_file = f'{translated}/mono_src/file.{{part}}'
    teacher_mono_target_file = f'{translated}/mono_src/file.{{part}}.{{model_index}}.out'
    translated_mono_src_extension = ".out"
    deseg_nbest_file = teacher_target_file


     
rule translate_corpus:
    message: "Translating corpus with teacher"
    log: f"{log_dir}/translate_corpus/{{part}}.{{model_index}}.log"
    conda: "envs/base.yml"
    threads: gpus_num*2
    resources: gpu=gpus_num
    input:
        ancient(decoder),
        file=teacher_source_file,
        vocab=vocab_path,
        teacher_models=expand(f"{final_teacher_dir}{{{{model_index}}}}-{{ens}}/{best_model}",ens=ensemble)
    output: file=teacher_target_file
    params: args=get_args('decoding-teacher')
    shell: '''bash pipeline/translate/translate-nbest.sh \
                "{input.file}" "{output.file}" "{input.vocab}" {input.teacher_models} {params.args} >> {log} 2>&1'''

rule extract_best:
    message: "Extracting best translations for the corpus"
    log: f"{log_dir}/extract_best/{{part}}.{{model_index}}.log"
    conda: "envs/base.yml"
    threads: 1
    #group 'translate_corpus'
    input: nbest=deseg_nbest_file, ref=f"{translated}/corpus/file.{{part}}.ref"
    output: f"{translated}/corpus/file.{{part}}.nbest.{{model_index}}.out"
    shell: 'python pipeline/translate/bestbleu.py -i {input.nbest} -r {input.ref} -m bleu -o {output} >> {log} 2>&1'

model_indices = list(range(len(opusmt_teacher))) if opusmt_teacher else [0]

rule collect_corpus:
    message: "Collecting translated corpus"
    log: f"{log_dir}/collect_corpus_{{model_index}}.log"
    conda: "envs/base.yml"
    threads: 4
    #group 'translate_corpus'
    input: lambda wildcards: expand(f"{translated}/corpus/file.{{part}}.nbest.{wildcards.model_index}.out", part=find_parts(wildcards, checkpoints.split_corpus))
    output: trg_corpus=f'{translated}/corpus.{{model_index}}.{trg}.gz'
    params: src_corpus=clean_corpus_src
    shell: 'bash pipeline/translate/collect.sh {translated}/corpus {output} {params.src_corpus} {wildcards.model_index} >> {log} 2>&1'

# mono

checkpoint split_mono_src:
    message: "Splitting monolingual src dataset"
    log: f"{log_dir}/split_mono_src.log"
    conda: "envs/base.yml"
    threads: 1
    input: corpora=f"{clean}/mono.{src}.gz", bin=ancient(deduper)
    output: directory(f'{translated}/mono_src')
    shell: 'bash pipeline/translate/split-mono.sh {input.corpora} {output} {split_length} >> {log} 2>&1'
    
rule translate_mono_src:
    message: "Translating monolingual src dataset with teacher"
    log: f"{log_dir}/translate_mono_src/{{part}}.{{model_index}}.log"
    conda: "envs/base.yml"
    threads: gpus_num*2
    wildcard_constraints:
        model_index="\d+"
    resources: gpu=gpus_num
    input:
        file=teacher_mono_source_file,vocab=vocab_path,
        teacher_models=expand(f"{final_teacher_dir}{{{{model_index}}}}-{{ens}}/{best_model}",ens=ensemble),
        bin=ancient(decoder)
    output: file=teacher_mono_target_file
    params: args=get_args('decoding-teacher')
    shell: '''bash pipeline/translate/translate.sh "{input.file}" "{output.file}" "{input.vocab}" {input.teacher_models} \
              {params.args} >> {log} 2>&1'''

#If there are no mono src datasets, create dummy output files, since the merge step
#expects translated mono src files (TODO: separate deduping and shuffling from merge script
#to remove the need for this workaround)
if mono_src_datasets is None:
    rule collect_mono_src_dummy:
        message: "Collecting translated mono src dataset (dummy rule, used in case where no mono src datasets)"
        log: f"{log_dir}/collect_mono_src.{{model_index}}.log"
        conda: "envs/base.yml"
        threads: 1
        #group 'mono_src'
        params: src_mono=f"{clean}/mono.{src}.gz",dir=f'{translated}/mono_src'
        output: trg_mono=f'{translated}/mono.{{model_index}}.{trg}.gz'
        shell: 'touch {output.trg_mono}  >> {log} 2>&1'
    rule mono_src_dummy:
        message: "Creating mono src dataset (dummy rule, used in case where no mono src datasets)"
        log: f"{log_dir}/create_mono_src.log"
        conda: "envs/base.yml"
        threads: 1
        #group 'mono_src'
        params: src_mono=f"{clean}/mono.{src}.gz",dir=f'{translated}/mono_src'
        output: src_mono=f"{clean}/mono.{src}.gz"
        shell: 'touch {output.src_mono} >> {log} 2>&1'
else:
    rule collect_mono_src:
        message: "Collecting translated mono src dataset"
        log: f"{log_dir}/collect_mono_src.{{model_index}}.log"
        conda: "envs/base.yml"
        threads: 4
        wildcard_constraints:
           model_index="\d+"
        #group 'mono_src'
        input:
           lambda wildcards: expand(f'{translated}/mono_src/file.{{part}}.{wildcards.model_index}.out',part=find_parts(wildcards, checkpoints.split_mono_src))
        output: f'{translated}/mono.{{model_index}}.{trg}.gz'
        params: src_mono=f"{clean}/mono.{src}.gz",dir=f'{translated}/mono_src'
        shell: 'bash pipeline/translate/collect-mono.sh "{params.dir}" "{output}" "{params.src_mono}" {wildcards.model_index} >> {log} 2>&1'
    
# merge

rule merge_translated:
    message: "Merging translated datasets"
    log: f"{log_dir}/merge_translated.log"
    conda: "envs/base.yml"
    threads: 4
    resources: mem_mb=64000
    #group 'mono_src'
    input:
        src1=clean_corpus_src,
        src2=f"{clean}/mono.{src}.gz",
        trg1=lambda wildcards: expand(f"{translated}/corpus.{{model_index}}.{trg}.gz",model_index=model_indices),
        trg2=lambda wildcards: expand(f"{translated}/mono.{{model_index}}.{trg}.gz",model_index=model_indices),
        bin=ancient(deduper)
    output: res_src=f'{merged}/corpus.{src}.gz',res_trg=f'{merged}/corpus.{trg}.gz'
    params:
        trg1_template=f"{translated}/corpus.model_index.{trg}.gz",
        trg2_template=f"{translated}/mono.model_index.{trg}.gz"
    shell: '''bash pipeline/translate/merge-corpus.sh \
                "{input.src1}" "{input.src2}" "{params.trg1_template}" "{params.trg2_template}" \
                "{output.res_src}" "{output.res_trg}" {model_indices} >> {log} 2>&1'''

# train student 

# preprocess source and target when scoring with opusmt model (note that deseg is not required, since
# scoring produces just scores)
if opusmt_backward:
    score_source = f"{merged}/corpus.{src}.opusmt.gz"
    score_target = f"{merged}/corpus.{trg}.opusmt.gz"
else:    
    score_source = rules.merge_translated.output.res_src
    score_target = rules.merge_translated.output.res_trg

#preprocess corpus before scoring, note that since the scoring is done with the
#backward model, source should be segmented with target.spm and vice versa
rule opusmt_preprocess_for_scoring:
    message: "Preprocessing source file for OPUS-MT model"
    log: f"{log_dir}/opusmt_preprocess_corpus/preprocess_for_scoring.log"
    conda: "envs/base.yml"
    threads: 1
    resources: mem_mb=64000
    input: 
        res_src=rules.merge_translated.output.res_src,
        res_trg=rules.merge_translated.output.res_trg,
        model=f'{backward_dir}/{best_model}',
        spm_encoder=ancient(spm_encoder)
    output: opusmt_source=f"{merged}/corpus.{src}.opusmt.gz",
            opusmt_target=f"{merged}/corpus.{trg}.opusmt.gz"
    shell: '''bash pipeline/translate/opusmt-preprocess.sh \
              {input.res_src} {input.model} src "target.spm" {input.spm_encoder} {target_language_token} && \
              bash pipeline/translate/opusmt-preprocess.sh \
              {input.res_trg} {input.model} trg "source.spm" {input.spm_encoder} {source_language_token} >> {log} 2>&1'''

rule score:
    message: "Scoring"
    log: f"{log_dir}/score.log"
    conda: "envs/base.yml"
    threads: gpus_num*2
    resources: gpu=gpus_num
    input:
        ancient(scorer),
        model=f'{backward_dir}/{best_model}', vocab=backward_vocab,
        src_corpus=score_source, trg_corpus=score_target
    output: f"{filtered}/scores.txt"
    params: input_prefix=f'{merged}/corpus'
    shell: '''bash pipeline/cefilter/score.sh \
                "{input.model}" "{input.vocab}" "{input.src_corpus}" "{input.trg_corpus}" "{output}" >> {log} 2>&1'''

rule ce_filter:
    message: "Cross entropy filtering"
    log: f"{log_dir}/ce_filter.log"
    conda: "envs/base.yml"
    threads: workflow.cores
    resources: mem_mb=workflow.cores*5000
    input:
        src_corpus=rules.merge_translated.output.res_src,trg_corpus=rules.merge_translated.output.res_trg,
        scores=rules.score.output
    output: src_corpus=f"{filtered}/corpus.{src}.gz",trg_corpus=f"{filtered}/corpus.{trg}.gz"
    params: input_prefix=f'{merged}/corpus',output_prefix=f'{filtered}/corpus'
    shell: '''bash pipeline/cefilter/ce-filter.sh \
                "{params.input_prefix}" "{params.output_prefix}" "{input.scores}" >> {log} 2>&1'''

rule alignments:
    message: 'Training word alignment and lexical shortlists'
    log: f"{log_dir}/alignments.log"
    conda: "envs/base.yml"
    threads: workflow.cores
    input:
        ancient(spm_encoder), ancient(spm_exporter),
        src_corpus=rules.ce_filter.output.src_corpus,trg_corpus=rules.ce_filter.output.trg_corpus,
        vocab=vocab_path,
        fast_align=ancient(rules.fast_align.output.fast_align), atools=ancient(rules.fast_align.output.atools),
        extract_lex=ancient(rules.extract_lex.output)
    output: alignment=f'{align_dir}/corpus.aln.gz',shortlist=f'{align_dir}/lex.s2t.pruned.gz'
    params: input_prefix=f'{filtered}/corpus'
    shell: '''bash pipeline/alignment/generate-alignment-and-shortlist.sh \
                "{params.input_prefix}" "{input.vocab}" "{align_dir}" {threads} >> {log} 2>&1'''

rule train_student:
    message: "Training student"
    log: f"{log_dir}/train_student.log"
    conda: "envs/base.yml"
    threads: gpus_num*3
    resources: gpu=gpus_num
    #group 'student'
    input:
        rules.merge_devset.output, ancient(trainer),
        train_src=rules.ce_filter.output.src_corpus, train_trg=rules.ce_filter.output.trg_corpus,
        alignments=rules.alignments.output.alignment,
        vocab=vocab_path
    output: model=f'{student_dir}/{best_model}'
    params: prefix_train=rules.ce_filter.params.output_prefix,prefix_test=f"{original}/devset",
            args=get_args("training-student")
    shell: '''bash pipeline/train/train-student.sh \
                "{input.alignments}" student train {src} {trg} "{params.prefix_train}" "{params.prefix_test}" \
                "{student_dir}" "{input.vocab}" "{best_model_metric}" {params.args} >> {log} 2>&1'''

# quantize

rule finetune_student:
    message: "Fine-tuning student"
    log: f"{log_dir}/finetune_student.log"
    conda: "envs/base.yml"
    threads: gpus_num*2
    resources: gpu=gpus_num
    #group 'student-finetuned'
    input:
        rules.merge_devset.output, ancient(trainer),
        train_src=rules.ce_filter.output.src_corpus, train_trg=rules.ce_filter.output.trg_corpus,
        alignments=rules.alignments.output.alignment, student_model=rules.train_student.output.model,
        vocab=vocab_path
    output: model=f'{student_finetuned_dir}/{best_model}'
    params: prefix_train=rules.ce_filter.params.output_prefix,prefix_test=f"{original}/devset",
            args=get_args("training-student-finetuned")
    shell: '''bash pipeline/train/train-student.sh \
                "{input.alignments}" student finetune {src} {trg} "{params.prefix_train}" "{params.prefix_test}" \
                "{student_finetuned_dir}" "{input.vocab}" "{best_model_metric}" --pretrained-model "{input.student_model}" {params.args} >> {log} 2>&1'''

rule quantize:
    message: "Quantization"
    log: f"{log_dir}/quantize.log"
    conda: "envs/base.yml"
    threads: 1
    input:
        ancient(bmt_decoder), ancient(bmt_converter),
        shortlist=rules.alignments.output.shortlist, model=rules.finetune_student.output.model,
        vocab=vocab_path, devset=f"{original}/devset.{src}.gz"
    output: model=f'{speed_dir}/model.intgemm.alphas.bin'
    shell: '''bash pipeline/quantize/quantize.sh \
                "{input.model}" "{input.vocab}" "{input.shortlist}" "{input.devset}" "{speed_dir}" >> {log} 2>&1'''

rule export:
    message: "Exporting models"
    log: f"{log_dir}/export.log"
    conda: "envs/base.yml"
    #group 'export'
    threads: 1
    input:
        model=rules.quantize.output.model,shortlist=rules.alignments.output.shortlist,
        vocab=vocab_path,marian=bmt_converter
    output:
        model=f'{exported_dir}/model.{src}{trg}.intgemm.alphas.bin.gz',
        shortlist=f'{exported_dir}/lex.50.50.{src}{trg}.s2t.bin.gz',
        vocab=f'{exported_dir}/vocab.{src}{trg}.spm.gz'
    shell:
        'bash pipeline/quantize/export.sh "{speed_dir}" "{input.shortlist}" "{input.vocab}" "{exported_dir}" >> {log} 2>&1'


### evaluation

rule evaluate:
    message: "Evaluating a model"
    log: f"{log_dir}/eval/eval_{{model}}_{{dataset}}.log"
    conda: "envs/base.yml"
    threads: 7
    resources: gpu=1
    #group '{model}'
    priority: 50
    wildcard_constraints:
        model="[\w-]+"
    input:
        ancient(decoder),
        data=multiext(f'{eval_data_dir}/{{dataset}}',f".{src}.gz",f".{trg}.gz"),
        src_spm=f'{teacher_base_dir}0-0/source.spm',
        trg_spm=f'{teacher_base_dir}0-0/target.spm',
        models=lambda wildcards: f'{models_dir}/{wildcards.model}/model.npz'
                                    if "finetuned-term" in wildcards.model
                                    else f'{models_dir}/{wildcards.model}/{best_model}'
                                    #TODO: handle ensembling better
                                    #if wildcards.model != 'teacher-ensemble'
                                    #else [f'{final_teacher_dir}0-{ens}/{best_model}' for ens in ensemble]
    output:
        report(f'{eval_res_dir}/{{model}}/{{dataset}}.metrics',
            category='evaluation', subcategory='{model}', caption='reports/evaluation.rst')
    params:
        dataset_prefix=f'{eval_data_dir}/{{dataset}}',
        vocab=lambda wildcards: f'{models_dir}/{wildcards.model}/vocab.yml',
        res_prefix=f'{eval_res_dir}/{{model}}/{{dataset}}',
        src_lng=lambda wildcards: src if wildcards.model != 'backward' else trg,
        trg_lng=lambda wildcards: trg if wildcards.model != 'backward' else src,
        decoder_config=lambda wildcards: f'{models_dir}/{wildcards.model}/model.npz.decoder.yml'
                            if "finetuned-term" in wildcards.model
                            else f'{models_dir}/{wildcards.model}/decoder.yml'
                            #if wildcards.model != 'teacher-ensemble'
                            #else f'{final_teacher_dir}0-0/{best_model}.decoder.yml'
    shell: '''bash pipeline/opusmt/eval.sh "{params.res_prefix}" "{params.dataset_prefix}" \
             {params.src_lng} {params.trg_lng} "{input.src_spm}" \
             "{input.trg_spm}" "{params.vocab}" "{params.decoder_config}" {input.models} >> {log} 2>&1'''

rule eval_quantized:
    message: "Evaluating qunatized student model"
    log: f"{log_dir}/eval_quantized_{{dataset}}.log"
    conda: "envs/base.yml"
    #group 'export'
    threads: 1
    priority: 50
    input:
        ancient(bmt_decoder),
        data=multiext(f'{eval_data_dir}/{{dataset}}',f".{src}.gz",f".{trg}.gz"),
        model=rules.quantize.output.model,
        shortlist=rules.alignments.output.shortlist,
        vocab=vocab_path
    output:
        report(f'{eval_speed_dir}/{{dataset}}.metrics', category='evaluation',
            subcategory='quantized', caption='reports/evaluation.rst')
    params:
        dataset_prefix=f'{eval_data_dir}/{{dataset}}',
        res_prefix=f'{eval_speed_dir}/{{dataset}}',
        decoder_config='../quantize/decoder.yml'
    shell: '''bash pipeline/eval/eval-quantized.sh "{input.model}" "{input.shortlist}" "{params.dataset_prefix}" \
            "{input.vocab}" "{params.res_prefix}" "{params.decoder_config}" >> {log} 2>&1'''

rule annotate_evalsets: 
    message: "Annotating evalsets with term information"
    log: f"{log_dir}/annotate_evalset.log"
    conda: "envs/base.yml"
    threads: workflow.cores
    #This should run on CPU, there are not that many sentences usually. If using a big evalset, uncomment this.
    #resources: gpu=1,mem_mb=128000
    #group 'student'
    input:
        evalsets_aln=f'{eval_data_dir}/evalsets.aln.gz',
        evalsets_src=f'{eval_data_dir}/evalsets.spm.src.gz',
        evalsets_trg=f'{eval_data_dir}/evalsets.spm.trg.gz',
        src_vocab=f'{teacher_base_dir}0-0/source.spm',
        trg_vocab=f'{teacher_base_dir}0-0/target.spm'
    output:
        evalsets_terms_src=f'{eval_data_dir}/evalsets_terms.src',
        evalsets_terms_trg=f'{eval_data_dir}/evalsets_terms.trg',
        evalsets_terms_aln=f'{eval_data_dir}/evalsets_terms.aln',
        evalsets_sgm_src=f'{eval_data_dir}/evalsets.src.sgm',
        evalsets_sgm_trg=f'{eval_data_dir}/evalsets.trg.sgm'	
	
    params: 
        evalsets_terms_src_gz=f'{eval_data_dir}/evalsets_terms.src.gz',
        evalsets_terms_trg_gz=f'{eval_data_dir}/evalsets_terms.trg.gz',
        evalsets_terms_aln_gz=f'{eval_data_dir}/evalsets_terms.aln.gz',
    shell: '''bash pipeline/opusmt/annotate_evalset.sh "{input.src_vocab}" "{input.trg_vocab}" \
                "{src}" "{trg}" "{input.evalsets_src}" "{input.evalsets_trg}" "{input.evalsets_aln}" \
                "{params.evalsets_terms_src_gz}" "{params.evalsets_terms_trg_gz}" \
                "{params.evalsets_terms_aln_gz}" "{output.evalsets_sgm_src}" \
                "{output.evalsets_sgm_trg}" "{params.evalsets_terms_src_gz}" "{output.evalsets_terms_src}" \
                "{params.evalsets_terms_trg_gz}" "{output.evalsets_terms_trg}" \
                "{params.evalsets_terms_aln_gz}" "{output.evalsets_terms_aln}" >> {log} 2>&1'''

rule eval_termscore: 
    message: "Scoring evalsets based on recognized terms"
    log: f"{log_dir}/eval/eval_{{model}}_termscore.log"
    conda: "envs/base.yml"
    threads: 8
    resources: gpu=1
    #group '{model}'
    priority: 50
    wildcard_constraints:
        model="[\w-]+"
    input:
        ancient(decoder),
        eval_src=rules.annotate_evalsets.output.evalsets_terms_src,
        eval_trg=rules.annotate_evalsets.output.evalsets_terms_trg,
        evalsets_sgm_src=f'{eval_data_dir}/evalsets.src.sgm',
        evalsets_sgm_trg=f'{eval_data_dir}/evalsets.trg.sgm',
        models=lambda wildcards: f'{models_dir}/{wildcards.model}/model.npz'
                                    if "finetuned-term" in wildcards.model
                                    else f'{models_dir}/{wildcards.model}/{best_model}'
                                    #TODO: handle ensembling better
                                    #if wildcards.model != 'teacher-ensemble'
                                    #else [f'{final_teacher_dir}0-{ens}/{best_model}' for ens in ensemble]
    output: 
        with_terms=f'{eval_res_dir}/{{model}}/evalsets_terms.score',
        without_terms=f'{eval_res_dir}/{{model}}/evalsets_terms.noterms.score'

    params:
        res_prefix=f'{eval_res_dir}/{{model}}/evalsets_terms',
        decoder_config=lambda wildcards: f'{models_dir}/{wildcards.model}/model.npz.decoder.yml'
                            if "finetuned-term" in wildcards.model
                            else f'{models_dir}/{wildcards.model}/decoder.yml',
        vocab=lambda wildcards: f'{models_dir}/{wildcards.model}/vocab.yml'
                            #if wildcards.model != 'teacher-ensemble'
                            #else f'{final_teacher_dir}0-0/{best_model}.decoder.yml'
    shell: '''bash pipeline/opusmt/eval-termscore.sh "{input.eval_src}" "{src}" "{trg}" \
            "{params.decoder_config}" {input.models} {params.vocab} {params.res_prefix} \
            "{input.evalsets_sgm_src}" "{input.evalsets_sgm_trg}" >> {log} 2>&1'''


rule testset_mixture_termscore: 
    message: "Scoring testset based on recognized terms using mixture of models"
    log: f"{log_dir}/eval/testset_mixture-{mixture_hash}_termscore.log"
    conda: "envs/base.yml"
    threads: 16
    resources: gpu=8
    #group '{model}'
    priority: 50
    wildcard_constraints:
        models="[+\w-]+"
    input:
        ancient(decoder),
        eval_src=f"{data_root_dir}/wmt23_term_devtest/test/test.{src}-{trg}.{src}",
        eval_dict=f"{data_root_dir}/wmt23_term_devtest/test/test.{src}-{trg}.dict.jsonl",
        vocab=vocab_path, 
        models=lambda wildcards: [f'{models_dir}/{model}/model.npz'
                                    if "finetuned-term" in model
                                    else f'{models_dir}/{model}/{best_model}' for model in mixture_of_models] 
                                    #TODO: handle ensembling better
                                    #if wildcards.model != 'teacher-ensemble'
                                    #else [f'{final_teacher_dir}0-{ens}/{best_model}' for ens in ensemble]
    output: f'{eval_res_dir}/mixture-{mixture_hash}/testset_terms.mixture.{trg}'
    params:
        models=lambda wildcards: " ".join([f'{models_dir}/{model}/model.npz'
                                    if "finetuned-term" in model
                                    else f'{models_dir}/{model}/{best_model}' for model in mixture_of_models]), 
        res_prefix=f'{eval_res_dir}/mixture-{mixture_hash}/testset_terms',
        decoder_config=lambda wildcards: [f'{models_dir}/{model}/model.npz.decoder.yml'
                            if "finetuned-term" in model
                            else f'{models_dir}/{model}/{best_model}.decoder.yml' for model in mixture_of_models][0] 
                            #if wildcards.model != 'teacher-ensemble'
                            #else f'{final_teacher_dir}0-0/{best_model}.decoder.yml'
    shell: '''bash pipeline/wmt23_termtask/term_mixture.sh "{input.eval_src}" "{input.eval_dict}" "{src}" "{trg}" \
            "{params.decoder_config}" {input.vocab} {params.res_prefix} 8 {params.models} >> {log} 2>&1'''


rule blindset_mixture_termscore: 
    message: "Scoring blindset based on recognized terms using mixture of models"
    log: f"{log_dir}/eval/blindset_mixture-{mixture_hash}_termscore.log"
    conda: "envs/base.yml"
    threads: 16
    resources: gpu=8
    #group '{model}'
    priority: 50
    wildcard_constraints:
        models="[+\w-]+"
    input:
        ancient(decoder),
        eval_src=f"{data_root_dir}/wmt23_blind/blind_terminology_{src}_{trg}.txt.{src}",
        eval_dict=f"{data_root_dir}/wmt23_blind/blind_terminology_{src}_{trg}.txt.jsonl",
        vocab=vocab_path, 
        models=lambda wildcards: [f'{models_dir}/{model}/model.npz'
                                    if "finetuned-term" in model
                                    else f'{models_dir}/{model}/{best_model}' for model in mixture_of_models] 
                                    #TODO: handle ensembling better
                                    #if wildcards.model != 'teacher-ensemble'
                                    #else [f'{final_teacher_dir}0-{ens}/{best_model}' for ens in ensemble]
    output: f'{eval_res_dir}/mixture-{mixture_hash}/blindset_terms.mixture.{trg}'
    params:
        models=lambda wildcards: " ".join([f'{models_dir}/{model}/model.npz'
                                    if "finetuned-term" in model
                                    else f'{models_dir}/{model}/{best_model}' for model in mixture_of_models]), 
        res_prefix=f'{eval_res_dir}/mixture-{mixture_hash}/blindset_terms',
        decoder_config=lambda wildcards: [f'{models_dir}/{model}/model.npz.decoder.yml'
                            if "finetuned-term" in model
                            else f'{models_dir}/{model}/{best_model}.decoder.yml' for model in mixture_of_models][0] 
                            #if wildcards.model != 'teacher-ensemble'
                            #else f'{final_teacher_dir}0-0/{best_model}.decoder.yml'
    shell: '''bash pipeline/wmt23_termtask/term_mixture.sh "{input.eval_src}" "{input.eval_dict}" "{src}" "{trg}" \
            "{params.decoder_config}" {input.vocab} {params.res_prefix} 8 {params.models} >> {log} 2>&1'''

rule align_evalsets:
    message: 'Training word alignment for evalsets'
    log: f"{log_dir}/evalset_alignments.log"
    conda: "envs/base.yml"
    threads: workflow.cores
    input:
        ancient(spm_encoder), ancient(spm_exporter),
        evalset_src=expand(f'{eval_data_dir}/{{dataset}}.{src}.gz',dataset=eval_datasets),
        evalset_trg=expand(f'{eval_data_dir}/{{dataset}}.{trg}.gz',dataset=eval_datasets),
        src_corpus=ancient(f'{teacher_corpus}.{src}.gz'),trg_corpus=ancient(f'{teacher_corpus}.{trg}.gz'),
        src_vocab=f'{teacher_base_dir}0-0/source.spm',
        trg_vocab=f'{teacher_base_dir}0-0/target.spm',
        fast_align=ancient(rules.fast_align.output.fast_align), atools=ancient(rules.fast_align.output.atools),
    output:
        evalsets_aln=f'{eval_data_dir}/evalsets.aln.gz',
        evalsets_src=f'{eval_data_dir}/evalsets.spm.src.gz',
        evalsets_trg=f'{eval_data_dir}/evalsets.spm.trg.gz'
    params: output_dir=eval_data_dir 
    shell: '''bash -c \'cat {input.evalset_src} > {output.evalsets_src} && \
             cat {input.evalset_trg} > {output.evalsets_trg} && \
             bash pipeline/eval/generate-alignment.sh \
                    "{input.evalset_src}" "{input.evalset_trg}" "{input.src_corpus}" "{input.trg_corpus}" "{input.src_vocab}" "{input.trg_vocab}" "{params.output_dir}" {threads} >> {log} 2>&1\''''

rule wmt23_termtask_score: 
    message: "Scoring wmt23 termtask dev data"
    log: f"{log_dir}/eval/eval_{{model}}_wmt23_termtask.log"
    conda: "envs/base.yml"
    threads: 8
    resources: gpu=1
    #group '{model}'
    priority: 50
    wildcard_constraints:
        model="[\w-]+"
    input:
        ancient(decoder),
        wmt23_dev_src=f"{data_root_dir}/wmt23_term_devtest/dev/dev.{src}-{trg}.{src}",
        wmt23_dev_dict=f"{data_root_dir}/wmt23_term_devtest/dev/dev.{src}-{trg}.dict.jsonl",
        vocab=vocab_path, 
        models=lambda wildcards: f'{models_dir}/{wildcards.model}/model.npz'
                                    if "finetuned-term" in wildcards.model
                                    else f'{models_dir}/{wildcards.model}/{best_model}'
                                    #TODO: handle ensembling better
                                    #if wildcards.model != 'teacher-ensemble'
                                    #else [f'{final_teacher_dir}0-{ens}/{best_model}' for ens in ensemble]
    output: f'{eval_res_dir}/{{model}}/wmt23_termtask.score'
    params:
        res_prefix=f'{eval_res_dir}/{{model}}/wmt23_termtask',
        decoder_config=lambda wildcards: f'{models_dir}/{wildcards.model}/model.npz.decoder.yml'
                            if "finetuned-term" in wildcards.model
                            else f'{models_dir}/{wildcards.model}/{best_model}.decoder.yml'
                            #if wildcards.model != 'teacher-ensemble'
                            #else f'{final_teacher_dir}0-0/{best_model}.decoder.yml'
    shell: '''bash pipeline/wmt23_termtask/eval.sh "{input.wmt23_dev_src}" "{input.wmt23_dev_dict}" "{src}" "{trg}" \
            "{params.decoder_config}" {input.models} {input.vocab} {params.res_prefix} >> {log} 2>&1'''
"""
