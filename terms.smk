spm_encoder = config["spm-encoder"]
fast_align = config["fast-align"]
atools = config["atools"]

wildcard_constraints:
    src="\w{2,3}",
    trg="\w{2,3}",

rule opusmt_finetune_alignments:
    message: 'Training word alignment and generating sp corpora for pseudo-term annotation'
    log: "{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/align_and_sp.log"
    conda: "envs/base.yml"
    threads: workflow.cores
    input:	
        ancient(spm_encoder),
        source="{project_name}/{src}-{trg}/{preprocessing}/train.{src}.gz", 
        target="{project_name}/{src}-{trg}/{preprocessing}/train.{trg}.gz",
        src_vocab='{datadir}/models/{src}-{trg}/{model_name}/source.spm',
        trg_vocab='{datadir}/models/{src}-{trg}/{model_name}/target.spm',
        fast_align=ancient(fast_align),
        atools=ancient(atools),
    output: 
        alignment="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/corpus.aln.gz",
        train_src="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/corpus.spm.{src}.gz",
        train_trg="{datadir}/{project_name}/{src}-{trg}/{preprocessing}/terms_align_and_sp_{model_name}/corpus.spm.{trg}.gz"
    params: 
        input_prefix="{project_name}/{src}-{trg}/{preprocessing}/train.{src}"
    shell: '''bash pipeline/opusmt/generate-alignment-and-sp.sh \
                "{params.input_prefix}" "{input.src_vocab}" "{input.trg_vocab}" "{teacher_align_dir}" {threads} >> {log} 2>&1'''

"""
rule finetune_opusmt_with_terms:
    message: "Finetune OPUS-MT model on term annotated corpus"
    log: f"{log_dir}/finetune_teacher-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}{{omit}}.log"
    wildcard_constraints:
        omit="(.{0}|-omit)"
    conda: "envs/base.yml"
    threads: gpus_num * 2
    resources: gpu=gpus_num
    input:
        rules.merge_devset.output, ancient(trainer),
        model=f'{teacher_base_dir}0-0/{best_model}',
        train_src=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus{{omit}}.{src}.gz",
        train_trg=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus{{omit}}.{trg}.gz",
        alignments=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus{{omit}}.aln.gz"
    output: model=f'{teacher_base_dir}-finetuned-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}{{omit}}/model.npz'
    params: 
        prefix_train=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus{{omit}}",
        prefix_test=f"{original}/devset",
        args=get_args("finetune-teacher-with-terms"),
        teacher_term_dir=f"{teacher_base_dir}-finetuned-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}{{omit}}"
    shell: '''bash pipeline/opusmt/term-finetune.sh \
                {src} {trg} "{params.prefix_train}" "{params.prefix_test}" \
                "{params.teacher_term_dir}" "{input.model}" "{best_model_metric}" {params.args} >> {log} 2>&1'''
    #shell: '''bash pipeline/train/train-student.sh \
    #            "{input.alignments}" teacher train {src} {trg} "{params.prefix_train}" "{params.prefix_test}" \
    #            "{params.teacher_term_dir}" "{input.vocab}" "{best_model_metric}" --pretrained-model "{input.model}" "{params.args}" >> {log} 2>&1'''

rule annotate_teacher_terms: 
    message: "Annotating corpus with term information"
    log: f"{log_dir}/annotate_teacher_terms/{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}.{{part}}.log"
    conda: "envs/base.yml"
    threads: 7
    resources: gpu=1
    #group 'student'
    input:
        train_src=f"{term_data_dir}/teacher_corpus/file.src.{{part}}.gz",
        train_trg=f"{term_data_dir}/teacher_corpus/file.trg.{{part}}.gz",
        alignments=f"{term_data_dir}/teacher_corpus/file.aln.{{part}}.gz",
        src_vocab=f'{teacher_base_dir}0-0/source.spm',
        trg_vocab=f'{teacher_base_dir}0-0/target.spm',	
    output:
        annotated_src=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.src.{{part}}.gz",
        annotated_trg=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.trg.{{part}}.gz",
        annotated_alignments=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.aln.{{part}}.gz"
    shell: '''python 3rd_party/soft-term-constraints/src/softconstraint.py \
                --source_spm "{input.src_vocab}" --target_spm "{input.trg_vocab}" --annotation_method {wildcards.scheme} \
                --term_start_tag augmentsymbol0 --term_end_tag augmentsymbol1 --trans_end_tag augmentsymbol2 \
                --mask_tag augmentsymbol3 --source_lang "{src}" --target_lang "{trg}" \
                --source_corpus "{input.train_src}" --target_corpus "{input.train_trg}" \
                --alignment_file "{input.alignments}" --terms_per_sent_ratio {wildcards.term_ratio} \
                --sents_per_term_sent {wildcards.sents_per_term_sent}  \
                --source_output_path "{output.annotated_src}" --target_output_path "{output.annotated_trg}" \
                --sp_input --sp_output \
                --alignment_output_path "{output.annotated_alignments}" >> {log} 2>&1'''


rule collect_teacher_term_annotations:
    message: "Collecting term-annotated data"
    log: f"{log_dir}/annotate_teacher_terms/{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}.collect.log"
    wildcard_constraints:
        omit="(.{0}|-omit)"
    conda: "envs/base.yml"
    threads: 4
    input:
        src=lambda wildcards: expand(f"{term_data_dir}/teacher-term-{wildcards.scheme}-{wildcards.term_ratio}-{wildcards.sents_per_term_sent}/annotated.src.{{part}}.gz",
            part=find_annotation_parts(wildcards, checkpoints.split_teacher_corpus_for_annotation)),
        trg=lambda wildcards: expand(f"{term_data_dir}/teacher-term-{wildcards.scheme}-{wildcards.term_ratio}-{wildcards.sents_per_term_sent}/annotated.trg.{{part}}.gz",
            part=find_annotation_parts(wildcards, checkpoints.split_teacher_corpus_for_annotation)),
        alignment=lambda wildcards: expand(f"{term_data_dir}/teacher-term-{wildcards.scheme}-{wildcards.term_ratio}-{wildcards.sents_per_term_sent}/annotated.aln.{{part}}.gz",
            part=find_annotation_parts(wildcards, checkpoints.split_teacher_corpus_for_annotation))
    output:
        annotated_src=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus.{src}.gz",
        annotated_trg=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus.{trg}.gz",
        annotated_alignments=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus.aln.gz",
        annotated_omit_src=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus-omit.{src}.gz",
        annotated_omit_trg=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus-omit.{trg}.gz",
        annotated_omit_alignments=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus-omit.aln.gz"
    params:
        src_prefix=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.src",
        trg_prefix=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.trg",
        aln_prefix=f"{term_data_dir}/teacher-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.aln"

    shell: '''bash pipeline/wmt23_termtask/collect.sh "{params.src_prefix}" "{params.trg_prefix}" "{params.aln_prefix}" \
            "{output.annotated_src}" "{output.annotated_trg}" "{output.annotated_alignments}" \
            "{output.annotated_omit_src}" "{output.annotated_omit_trg}" "{output.annotated_omit_alignments}" >> {log} 2>&1'''



checkpoint split_corpus_for_annotation:
    message: "Splitting the corpus for term annotation"
    log: f"{log_dir}/split_corpus_for_annotation.log"
    conda: "envs/base.yml"
    threads: 1
    input:
        train_src=rules.ce_filter.output.src_corpus,
        train_trg=rules.ce_filter.output.trg_corpus,
        alignments=rules.alignments.output.alignment
    output: directory(f"{term_data_dir}/corpus")
    shell: '''bash pipeline/wmt23_termtask/split-corpus.sh \
                {input.train_src} {input.train_trg} {input.alignments} {output} {split_length} >> {log} 2>&1'''

rule annotate_terms: 
    message: "Annotating corpus with term information"
    log: f"{log_dir}/annotate_terms/{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}.{{part}}.log"
    conda: "envs/base.yml"
    threads: 7
    resources: gpu=1,mem_mb=128000
    #group 'student'
    input:
        train_src=f"{term_data_dir}/corpus/file.src.{{part}}.gz",
        train_trg=f"{term_data_dir}/corpus/file.trg.{{part}}.gz",
        alignments=f"{term_data_dir}/corpus/file.aln.{{part}}.gz",
        vocab=vocab_path
    output:
        annotated_src=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.src.{{part}}.gz",
        annotated_trg=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.trg.{{part}}.gz",
        annotated_alignments=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.aln.{{part}}.gz"
    shell: '''python 3rd_party/soft-term-constraints/src/softconstraint.py \
                --source_spm "{input.vocab}" --target_spm "{input.vocab}" --annotation_method {wildcards.scheme} \
                --term_start_tag augmentsymbol0 --term_end_tag augmentsymbol1 --trans_end_tag augmentsymbol2 \
                --mask_tag augmentsymbol3 --source_lang "{src}" --target_lang "{trg}" \
                --source_corpus "{input.train_src}" --target_corpus "{input.train_trg}" \
                --alignment_file "{input.alignments}" --terms_per_sent_ratio {wildcards.term_ratio} \
                --sents_per_term_sent {wildcards.sents_per_term_sent}  \
                --source_output_path "{output.annotated_src}" --target_output_path "{output.annotated_trg}" \
                --alignment_output_path "{output.annotated_alignments}" >> {log} 2>&1'''


rule collect_term_annotations:
    message: "Collecting term-annotated data"
    log: f"{log_dir}/annotate_terms/{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}.collect.log"
    conda: "envs/base.yml"
    threads: 4
    input:
        src=lambda wildcards: expand(f"{term_data_dir}/student-term-{wildcards.scheme}-{wildcards.term_ratio}-{wildcards.sents_per_term_sent}/annotated.src.{{part}}.gz",
            part=find_annotation_parts(wildcards, checkpoints.split_corpus_for_annotation)),
        trg=lambda wildcards: expand(f"{term_data_dir}/student-term-{wildcards.scheme}-{wildcards.term_ratio}-{wildcards.sents_per_term_sent}/annotated.trg.{{part}}.gz",
            part=find_annotation_parts(wildcards, checkpoints.split_corpus_for_annotation)),
        alignment=lambda wildcards: expand(f"{term_data_dir}/student-term-{wildcards.scheme}-{wildcards.term_ratio}-{wildcards.sents_per_term_sent}/annotated.aln.{{part}}.gz",
            part=find_annotation_parts(wildcards, checkpoints.split_corpus_for_annotation))
    output:
        annotated_src=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus.{src}.gz",
        annotated_trg=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus.{trg}.gz",
        annotated_alignments=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus.aln.gz"
    params:
        src_prefix=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.src",
        trg_prefix=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.trg",
        aln_prefix=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/annotated.aln"

    shell: '''bash pipeline/wmt23_termtask/collect.sh "{params.src_prefix}" "{params.trg_prefix}" "{params.aln_prefix}" \
            "{output.annotated_src}" "{output.annotated_trg}" "{output.annotated_alignments}" >> {log} 2>&1'''

rule train_term_student:
    message: "Training student with term constraints"
    log: f"{log_dir}/train_student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}.log"
    conda: "envs/base.yml"
    threads: gpus_num*3
    resources: gpu=gpus_num
    #group 'student'
    input:
        rules.merge_devset.output, ancient(trainer),
        train_src=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus.{src}.gz",
        train_trg=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus.{trg}.gz",
        alignments=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus.aln.gz",
        vocab=vocab_path
    output: model=f'{student_dir}-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/{best_model}'
    params: prefix_train=f"{term_data_dir}/student-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}/corpus",prefix_test=f"{original}/devset",
            args=get_args("training-student"),student_term_dir=f"{student_dir}-term-{{scheme}}-{{term_ratio}}-{{sents_per_term_sent}}"
    shell: '''bash pipeline/train/train-student.sh \
                "{input.alignments}" student train {src} {trg} "{params.prefix_train}" "{params.prefix_test}" \
                "{params.student_term_dir}" "{input.vocab}" "{best_model_metric}" {params.args} >> {log} 2>&1'''


#TODO: These are copy pasted from the similar rules for student training, because of deadline.
#Make generic rules for annotation when time for that
checkpoint split_teacher_corpus_for_annotation:
    message: "Splitting the teacher corpus for term annotation"
    log: f"{log_dir}/split_teacher_corpus_for_annotation.log"
    conda: "envs/base.yml"
    threads: 1
    input:
        train_src=f'{teacher_corpus}.{src}.gz',train_trg=f'{teacher_corpus}.{trg}.gz',
        alignments=rules.opusmt_finetune_alignments.output.alignment
    output: directory(f"{term_data_dir}/teacher_corpus")
    shell: '''bash pipeline/wmt23_termtask/split-corpus.sh \
                {input.train_src} {input.train_trg} {input.alignments} {output} {split_length} >> {log} 2>&1'''
"""
