from langcodes import Language

spm_encoder = config["spm-encoder"]
fast_align = config["fast-align"]
atools = config["atools"]

localrules: split_corpus_for_annotation, collect_term_annotations, copy_valid_set

wildcard_constraints:
    src="\w{2,3}",
    trg="\w{2,3}",
    max_terms_per_sent="\d+",
    term_ratio="\d+",
    sents_per_term_sent="\d+"
    

def find_annotation_parts(wildcards, checkpoint):
    checkpoint_output = checkpoint.get(**wildcards).output[0]
    return glob_wildcards(os.path.join(checkpoint_output,"file.src.{part,\d+}.gz")).part

rule copy_valid_set:
    message: "Copying validation set"
    log: "{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}/copy_valid_set.log"
    conda: None
    container: None
    threads: 1
    #resources: gpu=1
    input:
        valid_src="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/dev.{src}.gz",
        valid_trg="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/dev.{trg}.gz",
    output:
        valid_src="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}/dev.{src}.gz",
        valid_trg="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}/dev.{trg}.gz",
    shell: '''cp {input.valid_src} {output.valid_src} && cp {input.valid_trg} {output.valid_trg}  >> {log} 2>&1'''

rule opusmt_term_alignments:
    message: 'Training word alignment and generating sp corpora for pseudo-term annotation'
    log: "{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/align_and_sp.log"
    conda: "envs/base.yml"
    threads: workflow.cores
    input:	
        ancient(spm_encoder),
        source="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/train.{src}.gz", 
        target="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/train.{trg}.gz",
        src_vocab='{datadir}/models/{src}-{trg}/{model_name}/source.spm',
        trg_vocab='{datadir}/models/{src}-{trg}/{model_name}/target.spm',
        fast_align=ancient(fast_align),
        atools=ancient(atools),
    output: 
        alignments="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/train.aln.gz",
        train_src="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/train.spm.{src}.gz",
        train_trg="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/train.spm.{trg}.gz"
    params: 
        output_dir="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}"
    shell: '''bash pipeline/opusmt/generate-alignment-and-sp.sh \
                "{input.source}" "{input.target}" "{input.src_vocab}" \
                "{input.trg_vocab}" "{params.output_dir}" {threads} "{wildcards.src}" "{wildcards.trg}" >> {log} 2>&1'''

rule annotate_terms: 
    message: "Annotating corpus with term information"
    log: "{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/annotate_{part}.log"
    conda: None
    container: None
    threads: 16
    #resources: gpu=1
    input:
        alignments="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/file.aln.{part}.gz",
        train_src="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/file.src.{part}.gz",
        train_trg="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/file.trg.{part}.gz",
        src_vocab='{datadir}/models/{src}-{trg}/{model_name}/source.spm',
        trg_vocab='{datadir}/models/{src}-{trg}/{model_name}/target.spm',
    output:
        annotated_src="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/annotated.src.{part}.gz",
        annotated_trg="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/annotated.trg.{part}.gz",
        annotated_alignments="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/annotated.aln.{part}.gz"
    params:
        # Stanza wants two letter codes
        src=lambda wildcards: Language.get(wildcards.src).to_tag(),
        trg=lambda wildcards: Language.get(wildcards.trg).to_tag()
    shell: '''python 3rd_party/soft-term-constraints/src/softconstraint.py \
                --source_spm "{input.src_vocab}" --target_spm "{input.trg_vocab}" --annotation_method {wildcards.scheme} \
                --term_start_tag augmentsymbol0 --term_end_tag augmentsymbol1 --trans_end_tag augmentsymbol2 \
                --mask_tag augmentsymbol3 --source_lang "{params.src}" --target_lang "{params.trg}" \
                --source_corpus "{input.train_src}" --target_corpus "{input.train_trg}" \
                --alignment_file "{input.alignments}" --terms_per_sent_ratio {wildcards.term_ratio} \
                --sents_per_term_sent {wildcards.sents_per_term_sent} --max_terms_per_sent {wildcards.max_terms_per_sent} \
                --source_output_path "{output.annotated_src}" --target_output_path "{output.annotated_trg}" \
                --sp_input --sp_output \
                --alignment_output_path "{output.annotated_alignments}" \
                --batch_size 1000 >> {log} 2>&1'''

checkpoint split_corpus_for_annotation:
    message: "Splitting the corpus for term annotation"
    log: "{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split.log"
    conda: None
    container: None
    threads: 1
    input:
        train_src="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/train.spm.{src}.gz",
        train_trg="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/train.spm.{trg}.gz",
        alignments="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/train.aln.gz"
    output: directory("{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split")
    shell: '''pipeline/opusmt/split-corpus-and-aln.sh \
                {input.train_src} {input.train_trg} {input.alignments} {output} 1000000 >> {log} 2>&1'''

rule collect_term_annotations:
    message: "Collecting term-annotated data"
    log: "{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/split.log"
    conda: "envs/base.yml"
    threads: 4
    input:
        src=lambda wildcards: expand("{{datadir}}/{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/annotate_terms_{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}-{{max_terms_per_sent}}/split/annotated.src.{part}.gz", part=find_annotation_parts(wildcards, checkpoints.split_corpus_for_annotation)),
        trg=lambda wildcards: expand("{{datadir}}/{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/annotate_terms_{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}-{{max_terms_per_sent}}/split/annotated.trg.{part}.gz", part=find_annotation_parts(wildcards, checkpoints.split_corpus_for_annotation)),
        alignments=lambda wildcards: expand("{{datadir}}/{{project_name}}/{{src}}-{{trg}}/{{preprocessing}}/annotate_terms_{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}-{{max_terms_per_sent}}/split/annotated.aln.{part}.gz", part=find_annotation_parts(wildcards, checkpoints.split_corpus_for_annotation))
    output:
        annotated_src="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/train.{src}.gz",
        annotated_trg="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/train.{trg}.gz",
        annotated_alignments="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/train.aln.gz",
        annotated_omit_src="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/train-omit.{src}.gz",
        annotated_omit_trg="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/train-omit.{trg}.gz",
        annotated_omit_alignments="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/train-omit.aln.gz"
    params:
        src_prefix="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/annotated.src",
        trg_prefix="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/annotated.trg",
        aln_prefix="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/annotate_terms_{scheme}-{term_ratio}-{sents_per_term_sent}-{max_terms_per_sent}/split/annotated.aln"

    shell: '''bash pipeline/wmt23_termtask/collect.sh "{params.src_prefix}" "{params.trg_prefix}" "{params.aln_prefix}" \
            "{output.annotated_src}" "{output.annotated_trg}" "{output.annotated_alignments}" \
            "{output.annotated_omit_src}" "{output.annotated_omit_trg}" "{output.annotated_omit_alignments}" >> {log} 2>&1'''
