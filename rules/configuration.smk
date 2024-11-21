
### configuration

containerized: 'Ftt.sif'

install_deps = config['deps'] == 'true'
data_root_dir = config.get('root', srcdir("../data"))
cuda_dir = config.get('cuda', os.environ.get("CUDA_INSTALL_ROOT","")) 
cudnn_dir = config.get('cudnn', os.environ.get("CUDNN_INSTALL_ROOT",""))
rocm_dir = config.get('rocm',os.environ.get("ROCM_PATH",""))

gpus_num = config['numgpus']
# marian occupies all GPUs on a machine if `gpus` are not specified
gpus = config['gpus'] if config['gpus'] else ' '.join([str(n) for n in range(int(gpus_num))])
workspace = config['workspace']
marian_cmake = config['mariancmake']
marian_version = config.get('marianversion','marian-dev')

# experiment parameters
experiment = config['experiment']['name']
dirname = config['experiment']['dirname']

# Read langpairs from config
langpairs = config['experiment']['langpairs']
if not langpairs:
    langpairs = [f"{src}-{trg}"]

# multilinguality 
o2m_teacher = config['experiment'].get('one2many-teacher',False)
o2m_student = config['experiment'].get('one2many-student',False)
o2m_backward = config['experiment'].get('one2many-backward',False)


# Modified variables to fit naming
src="source"
trg="target"

mono_max_sent_src = config['experiment'].get('mono-max-sentences-src')
mono_max_sent_trg = config['experiment'].get('mono-max-sentences-trg')
parallel_max_sents = config['experiment'].get('parallel-max-sentences',"inf")



backward_pretrained = config['experiment'].get('backward-model')
backward_pretrained_vocab = config['experiment'].get('backward-vocab')
vocab_pretrained = config['experiment'].get('vocab')
forward_pretrained = config['experiment'].get('forward-model')

experiment_dir=f"{data_root_dir}/experiments/{dirname}/{experiment}"

# override marian configs
marian_args = {name: ' '.join([f'--{k} {v}' for k,v in conf.items() ])
               for name, conf in config.get('marian-args',{}).items()}

# There can be multiple opus teachers, but a single teacher can also be provided
# as string, so convert it to list here
opusmt_teacher = config['experiment'].get('opusmt-teacher')
if opusmt_teacher and not isinstance(opusmt_teacher,list):
    opusmt_teacher = [opusmt_teacher]
opusmt_backward = config['experiment'].get('opusmt-backward')

# datasets
train_datasets = config['datasets']['train']
valid_datasets = config['datasets']['devtest']
eval_datasets = config['datasets']['test']
mono_src_datasets = config['datasets'].get('mono-src')
mono_trg_datasets = config['datasets'].get('mono-trg')
mono_datasets = {src: mono_src_datasets, trg: mono_trg_datasets}
mono_max_sent = {src: mono_max_sent_src, trg: mono_max_sent_trg}

# parallelization

ensemble = list(range(config['experiment'].get('teacher-ensemble',0)))

split_length = config['experiment']['split-length']

# logging
log_dir = f"{data_root_dir}/logs/{dirname}/{experiment}"
reports_dir = f"{data_root_dir}/reports/{dirname}/{experiment}"

# binaries
cwd = os.getcwd()
third_party_dir = f'{cwd}/3rd_party'

if marian_version == 'lumi-marian':
    marian_dir = f'{third_party_dir}/lumi-marian/build/'
else:
    marian_dir = f'{third_party_dir}/marian-dev/build/'
    
bmt_marian_dir = f'{third_party_dir}/browsermt-marian-dev/build'
trainer = f'{marian_dir}marian'
decoder = f'{marian_dir}marian-decoder'
scorer = f'{marian_dir}marian-scorer'
spm_encoder = f'{marian_dir}spm_encode'
spm_trainer = f'{marian_dir}spm_train'
spm_exporter = f'{marian_dir}spm_export_vocab'
bmt_decoder = f'{bmt_marian_dir}/marian-decoder'
bmt_converter = f'{bmt_marian_dir}/marian-conv'

kenlm = f'{third_party_dir}/kenlm'
fast_align_build = f'{third_party_dir}/fast_align/build'
extract_lex_build = f'{third_party_dir}/extract-lex/build'
preprocess_build_dir=f'{third_party_dir}/preprocess/build'
bin = f'{cwd}/bin'
deduper = f'{cwd}/bin/dedupe'

# data
data_dir = f"{data_root_dir}/data/{dirname}/{experiment}"
clean = f"{data_dir}/clean"
biclean = f"{data_dir}/biclean"
opusfiltered = f"{data_dir}/opusfilter"
cache_dir = f"{data_dir}/cache"
original = f"{data_dir}/original"
translated = f"{data_dir}/translated"
augmented = f"{data_dir}/augmented"
merged = f"{data_dir}/merged"
filtered = f'{data_dir}/filtered'
align_dir = f"{data_dir}/alignment"

# models
student_prefix = config['experiment'].get('student-prefix')
models_dir = f"{data_root_dir}/models/{dirname}/{experiment}"
# Teacher dir
teacher_base_dir = f"{models_dir}/teacher-base"
if opusmt_teacher:
    teacher_base_dir = f"{models_dir}/{{langpair}}/teacher-base"
teacher_finetuned_dir = f"{models_dir}/teacher-finetuned"
if student_prefix:
    student_dir = f"{models_dir}/"+student_prefix+"_student"
    student_finetuned_dir = f"{models_dir}/"+student_prefix+"_student-finetuned"
    speed_dir = f"{models_dir}/"+student_prefix+"_speed"
    exported_dir = f"{models_dir}/"+student_prefix+"_exported"
else:
    student_dir = f"{models_dir}/student"
    student_finetuned_dir = f"{models_dir}/student-finetuned"
    speed_dir = f"{models_dir}/speed"
    exported_dir = f"{models_dir}/exported"
best_model_metric = config['experiment']['best-model']
best_model = f"final.model.npz.best-{best_model_metric}.npz"
backward_dir = f'{models_dir}/backward'
if opusmt_backward:
    backward_dir = f"{models_dir}/{{langpair}}/backward"
spm_sample_size=config['experiment'].get('spm-sample-size')
spm_vocab_size=config['experiment'].get('spm-vocab-size',"32000")

#forward pretrained models are trained with sentencepiece integration, the value is a path to the directory
if forward_pretrained:
    teacher_base_dir = forward_pretrained
    #this means that the when the model dirs are expanded, the result is only the teacher_base_dir
    ensemble = [""] 


#default vocab path used with base ftt
vocab_path = vocab_pretrained or f"{models_dir}/vocab/vocab.spm"

if opusmt_backward:
   backward_vocab = f"{backward_dir}/vocab.yml"
else:
   backward_vocab = vocab_path

#evaluation
eval_data_dir = f"{original}/{{langpair}}/eval"
eval_res_dir = f"{models_dir}/evaluation"
eval_backward_dir = f'{eval_res_dir}/backward'
if student_prefix:
    eval_student_dir = f'{eval_res_dir}/'+student_prefix+'_student'
    eval_student_finetuned_dir = f'{eval_res_dir}/'+student_prefix+'_student-finetuned'
    eval_speed_dir = f'{eval_res_dir}/'+student_prefix+'_speed'
else:
    eval_student_dir = f'{eval_res_dir}/student'
    eval_student_finetuned_dir = f'{eval_res_dir}/student-finetuned'
    eval_speed_dir = f'{eval_res_dir}/speed'
eval_teacher_ens_dir = f'{eval_res_dir}/teacher-ensemble'


# bicleaner


if 'bicleaner' in config['experiment']:
    bicl_default_threshold = config['experiment']['bicleaner']['default-threshold']
    bicl_dataset_thresholds = config['experiment']['bicleaner']['dataset-thresholds']

    bicleaner_type = packs.find(src, trg)
else:
    bicleaner_type = None    

if bicleaner_type == 'bicleaner-ai':
    if marian_version == 'lumi-marian':
        bicleaner_env = 'envs/bicleaner-ai-lumi.yml'
    else:
        bicleaner_env = 'envs/bicleaner-ai.yml'
else:
    bicleaner_env = 'envs/bicleaner.yml' 

if bicleaner_type:
    clean_corpus_prefix = f'{biclean}/corpus'
    teacher_corpus = f'{biclean}/corpus'
    use_bicleaner = True
else:
    clean_corpus_prefix = f'{clean}/corpus'
    teacher_corpus = f'{clean}/corpus'
    use_bicleaner = False


# opusfilter

if 'opusfilter' in config['experiment']:
    opusfilter_config = config['experiment']['opusfilter'].get('config')
    if not opusfilter_config:
        opusfilter_config = "default"
    use_opusfilter = True
    clean_corpus_prefix=f'{opusfiltered}/{{langpair}}/corpus'
    teacher_corpus = f'{opusfiltered}/corpus'
else:
    use_opusfilter = False
    clean_corpus_prefix = f'{clean}/{{langpair}}/corpus'
    teacher_corpus = f'{clean}/corpus'

clean_corpus_src = f'{clean_corpus_prefix}.source.gz'
if opusmt_teacher:
    clean_corpus_src = f'{clean_corpus_prefix}.source.langtagged.gz'

clean_corpus_trg = f'{clean_corpus_prefix}.target.gz'

# opustrainer

if 'opustrainer' in config['experiment']:
    opustrainer_model = config['experiment']['opustrainer']['model']
    opustrainer_config = config['experiment']['opustrainer']['path']
    if opustrainer_model == 'student': # Should be modified to include teacher and backward model training with OpusTrainer
        do_train_student_opustrainer = True
    else:
        do_train_student_opustrainer = False
else:
        do_train_student_opustrainer = False

# huggingface

if "huggingface" in config["experiment"]:
    hf_teacher = config['experiment']['huggingface'].get('modelname')
    hf_modelclass = config['experiment']['huggingface'].get('modelclass')
    hf_langinfo = config['experiment']['huggingface'].get('lang_info','False')
    hf_prompt = config['experiment']['huggingface'].get('prompt','{source}')
    hf_config = config['experiment']['huggingface'].get('config','default')
    hf_batchsize = config['experiment']['huggingface'].get('batch_size','8')
    hf_langtags = config['experiment']['huggingface'].get('lang_tags',dict())
    hf_ct2 = config['experiment']['huggingface'].get('ct2','False')
    huggingface = True
    train_student_dir = f"{merged}/{{langpair}}"
else:
    huggingface = False
    train_student_dir = f"{filtered}/{{langpair}}"
