from opusfilter.opusfilter import OpusFilter
import yaml
import sys

input_files = sys.argv[1].split()
filtered_files = sys.argv[2].split()
dedup_files = [filtered_file.replace(".gz",".dedup.gz") for filtered_file in filtered_files]
src_lang = sys.argv[3]
tgt_lang = sys.argv[4]
config_file = sys.argv[5]
threads=sys.argv[6]

if config_file == "default": #if a no configuration is given, use default
	filter_params = [
    {'AlphabetRatioFilter': {}},
    {'LanguageIDFilter': {
        'id_method': 'cld2',
        'languages': [src_lang, tgt_lang]
    }},
    {'LengthRatioFilter': {
        'name': 'word',
        'unit': 'word'
    }},
    {'NonZeroNumeralsFilter': {}},
    {'TerminalPunctuationFilter': {}},
    {'RepetitionFilter': {}},
    {'LengthFilter': {
        'min_length': 3,
		'max_length': 100
    }}
	]
else:
	with open(config_file, 'r') as file:
		filter_params = yaml.safe_load(file)
	
config = {
    'common': {
        'default_n_jobs': int(threads)
    },
    'steps': [
        {
            'type': 'remove_duplicates',
            'parameters': {
                'inputs': input_files,
                'outputs': dedup_files
            }
        },
        {
            'type': 'filter',
            'parameters': {
                'inputs': dedup_files,
                'outputs': filtered_files,
                'filters': filter_params
            }
        }
    ]
}

of = OpusFilter(config)
of.execute_steps()
