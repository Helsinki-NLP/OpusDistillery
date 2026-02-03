from langcodes import *

if config["ct2"] == True:
    ruleorder: translate_corpus_hf_ct2 > translate_corpus_hf
else:
    ruleorder: translate_corpus_hf > translate_corpus_hf_ct2

rule translate_corpus_hf:
    message: "Translating corpus with Hugging Face teacher"
    log: f"{config['log_dir']}/translate_corpus/{{langpair}}/{{part}}.{{model_index}}.log"
    conda: "../envs/hf.yml"
    threads: config["gpus_num"] * 2
    resources: gpu=config["gpus_num"]
    input:
        file=config["teacher_source_file"]
    output: file=config["teacher_target_file"]
    params: src=lambda wildcards: wildcards.langpair.split('-')[0],
            trg=lambda wildcards: wildcards.langpair.split('-')[1],
            model_dir=config["final_teacher_dir"],
            teacher=config["hf_teacher"],
            modelclass=config["modelclass"],
            langinfo=config["langinfo"],
            prompt={config["prompt"]},
            langtags=config["langtags"],
            decoder_config=config["decoder_config"],
            batch_size=config["batch_size"],
            token=config["token"]
    shell: '''
        bash pipeline/translate/translate_hf.sh \
            "{input.file}" "{output.file}" "{params.teacher}" "{params.model_dir}" "{params.src}" "{params.trg}" \
            "{params.modelclass}" "{params.langinfo}" "{params.prompt}" '{params.langtags}' "{params.decoder_config}" "{params.batch_size}" "{params.token}" {log} >> {log} 2>&1
        '''

rule translate_corpus_hf_ct2:
    message: "Translating corpus with Hugging Face teacher with Ctranslate2"
    log: f"{config['log_dir']}/translate_corpus/{{langpair}}/{{part}}.{{model_index}}.log"
    conda: "../envs/hf.yml"
    threads: config["gpus_num"] * 2
    resources: gpu=config["gpus_num"]
    input:
        file=config["teacher_source_file"]
    output: file=config["teacher_target_file"]
    params: src=lambda wildcards: wildcards.langpair.split('-')[0],
            trg=lambda wildcards: wildcards.langpair.split('-')[1],
            model_dir=config["final_teacher_dir"],
            teacher=config["hf_teacher"],
            langinfo=config["langinfo"],
            prompt={config["prompt"]},
            langtags=config["langtags"],
            decoder_config=config["decoder_config"],
            batch_size=config["batch_size"]
    shell: '''
        bash pipeline/translate/translate_ctranslate.sh \
            "{input.file}" "{output.file}" "{params.teacher}" "{params.model_dir}" "{params.src}" "{params.trg}" \
            "{params.langinfo}" "{params.prompt}" '{params.langtags}' "{params.decoder_config}" "{params.batch_size}" {log} >> {log} 2>&1
        '''