import os
import argparse
import sacrebleu
import csv
import gzip
import bsbleu
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Generate BLEU and chrF scores for domain-specific translations.")
    
    parser.add_argument("--input_dir", help="Input directory containing the test files.")
    parser.add_argument("--report_file", help="Output report file.")
    parser.add_argument("--src_lang", help="Three-letter source language code.")
    parser.add_argument("--trg_lang", help="Three-letter target language code.")
    parser.add_argument("--system_id", help="ID of the system used to translate domeval.")
    parser.add_argument("--domeval_ids", help="Path to TSV file containing domain evaluation IDs (with domain names).")
    parser.add_argument("--baseline_translations", help="Path to translations with a baseline system.")
    parser.add_argument("--uses_bands", action="store_true", help="Whether to use nobands files or normal files.")
    
    return parser.parse_args()

def read_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

#generate pseudorefs from fuzzy matches for copy rate calculation, use the bsbleu code with modifications (lists instead of files, and BP 0)
def split_on_fuzzy_break(source, target):
    results = []

    #check if fuzzy indexes are used (this is hacky and it should be an argument to the script, but time is short)
    fuzzy_indexes = re.search("FUZZY_BREAK_\d",source[0])

    # Process each sentence pair

    for src_sent, tgt_sent in zip(source, target):
        pseudo_refs = []
        # Split the source sentence by 'FUZZY_BREAK'
        if fuzzy_indexes:
            parts = re.split("FUZZY_BREAK_\d", src_sent)
        else:
            parts = src_sent.split('FUZZY_BREAK')
        for index,fuzzy in enumerate(parts[0:-1]):
            pseudo_refs.append(fuzzy)
        results.append(pseudo_refs)
    max_fuzzy_count = max([len(x) for x in results])
    # fill up all result lines to the max fuzzy count to construct full pseudorefs
    for res in results:
        if len(res) < max_fuzzy_count:
            for i in range(0, max_fuzzy_count-len(res)):
                res.append(res[-1])

    pseudo_files = []
    for fuzzy_count in range(0,max_fuzzy_count):
        pseudo_files.append([])
        pseudo_files[fuzzy_count] = [x[fuzzy_count] for x in results]
    return pseudo_files

class MetricSummary:
    def __init__(self, bleu, chrf, copyrate):
        self.bleu = bleu
        self.chrf = chrf
        self.copyrate = copyrate

def calculate_metrics(reference, translations, source=None):
    bleu_score = sacrebleu.corpus_bleu(translations, [reference]).score
    chrf_score = sacrebleu.corpus_chrf(translations, [reference]).score
    
    if source:
        pseudo_refs = split_on_fuzzy_break(source,translations)
        copyrate_metric = sacrebleu.CopyRate()
        copyrate = copyrate_metric.corpus_score(translations, pseudo_refs)
        copyrate_score = copyrate.score
    else:
        copyrate_score = None

    return MetricSummary(bleu=bleu_score, chrf=chrf_score, copyrate=copyrate_score)

def evaluate_baseline(input_dir, trg_lang, ref_lines, report_file, system_id, baseline_translations):
    baseline_metrics = calculate_metrics(ref_lines, baseline_translations)
    report_file.write(f"full_domeval\tbaseline\t{baseline_metrics.bleu}\t{baseline_metrics.chrf}\tN/A\tall\t0\t{system_id}\n")

def combine_baseline_and_rat_domain(domain_fuzzy_ids, baseline_lines, rat_lines, domain_ids):
    combined_lines = []
    for index in domain_ids:
        if index in domain_fuzzy_ids:
            combined_lines.append(rat_lines[index])
        else:
            combined_lines.append(baseline_lines[index])
    return combined_lines

# this combines fuzzy translations with baseline translations for a given set
def combine_baseline_and_rat(fuzzy_line_nums, baseline_lines, rat_lines):
    combined_lines = []
    for index,line in enumerate(baseline_lines):
        # fuzzy linenums file start from 1 
        if index+1 in fuzzy_line_nums:
            combined_lines.append(rat_lines[index])
        else:
            combined_lines.append(baseline_lines[index])
    return combined_lines

def evaluate_full_domeval(domain, input_dir, trg_lang, ref_lines, report_file, system_id, nofuzzies_trg_lines, baseline_trg_lines):
    trg_file = os.path.join(input_dir, f"{domain}-domeval.{trg_lang}")
    source_fuzzies_file = os.path.join(input_dir, f"{domain}-domeval.fuzzies")
    translated_fuzzies_file = os.path.join(input_dir, f"{domain}-domeval.translated_fuzzies")
    linenum_file = os.path.join(input_dir, f"{domain}-domeval.linenum")
    
    fuzzy_source_lines = read_file_lines(source_fuzzies_file)
    fuzzy_lines = read_file_lines(translated_fuzzies_file)
    linenum_lines = [int(line.split(":")[0].strip()) for line in read_file_lines(linenum_file)]
    fuzzy_ref_lines = [ref_lines[linenum-1] for linenum in linenum_lines]
    nofuzzy_lines = [nofuzzies_trg_lines[linenum-1] for linenum in linenum_lines]
    baseline_lines = [baseline_trg_lines[linenum-1] for linenum in linenum_lines]

    # Process trg file (full test set translation). This is slow, so don't run for all domains, just for
    # big indices
    if domain in ["train","all_filtered","nobands_train","nobands_all_filtered"]:
        trg_lines = read_file_lines(trg_file)
        combined_lines = combine_baseline_and_rat(linenum_lines, baseline_trg_lines, trg_lines) 
        full_baselinemix_metrics = calculate_metrics(ref_lines, combined_lines)
        report_file.write(f"full_domeval\tbaselinemix_{domain}\t{full_baselinemix_metrics.bleu}\t{full_baselinemix_metrics.chrf}\tN/A\tall\t{len(fuzzy_lines)}\t{system_id}\n")

        full_domain_metrics = calculate_metrics(ref_lines, trg_lines)
        report_file.write(f"full_domeval\t{domain}\t{full_domain_metrics.bleu}\t{full_domain_metrics.chrf}\tN/A\tall\t{len(fuzzy_lines)}\t{system_id}\n")

    fuzzy_metrics = calculate_metrics(fuzzy_ref_lines, fuzzy_lines, fuzzy_source_lines)
    report_file.write(f"all_{domain}_fuzzies\t{domain}\t{fuzzy_metrics.bleu}\t{fuzzy_metrics.chrf}\t{fuzzy_metrics.copyrate}\tonly_fuzzies\t{len(fuzzy_lines)}\t{system_id}\n")
    
    nofuzzy_metrics = calculate_metrics(fuzzy_ref_lines, nofuzzy_lines)
    report_file.write(f"all_{domain}_fuzzies\tno_fuzzies\t{nofuzzy_metrics.bleu}\t{nofuzzy_metrics.chrf}\tN/A\tonly_fuzzies\t{len(fuzzy_lines)}\t{system_id}\n")

    baseline_metrics = calculate_metrics(fuzzy_ref_lines, baseline_lines)
    report_file.write(f"all_{domain}_fuzzies\tbaseline\t{baseline_metrics.bleu}\t{baseline_metrics.chrf}\tN/A\tonly_fuzzies\t{len(fuzzy_lines)}\t{system_id}\n")
    
def evaluate_domeval_domains(input_dir, trg_lang, tsv_file, all_ref_lines, nofuzzies_trg_lines, baseline_lines, report_file, system_id):
    # Read the TSV file into a dictionary mapping each line number to its domain
    id_to_domain_dict = {}
    domain_to_id_dict = {}
    with gzip.open(tsv_file, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            domain = row[0]
            id_to_domain_dict[idx] = domain
            if domain not in domain_to_id_dict:
                domain_to_id_dict[domain] = []
            domain_to_id_dict[domain].append(idx)

    all_trg_lines = {}
    
    domains = set(id_to_domain_dict.values())

    # Initialize a dictionary to hold domain-specific sentences for evaluation
    domain_specific_refs = {}
    domain_specific_trans = {}
    domain_specific_nofuzzies = {}
    domain_specific_baselines = {}
    
    # the crawled corpora are not included as indexes, but they exist in the data, so only add
    # the tranlations if the translation file exists
    index_domains = []
    for domain in domain_to_id_dict.keys():
        domain_trg_file = os.path.join(input_dir, f"{domain}-domeval.{trg_lang}")
        if os.path.exists(domain_trg_file):
            index_domains.append(domain)

    for index_domain in index_domains:
        domain_trg_file = os.path.join(input_dir, f"{index_domain}-domeval.{trg_lang}")
        if os.path.exists(domain_trg_file):
            all_trg_lines[index_domain] = read_file_lines(domain_trg_file)

    domain_to_fuzzy_id = {}
    indexdomain_to_fuzzy_src = {}

    for index_domain in index_domains:
        domain_to_fuzzy_id[index_domain] = {}
        with open(os.path.join(input_dir, f"{index_domain}-domeval.linenum"),'r') as linenum_file:
            linenum_lines = {int(line.split(":")[0].strip())-1 for line in linenum_file.readlines()}
            # get those ids from domain that have fuzzies with the given index domain
            for domain in domains:
                domain_to_fuzzy_id[index_domain][domain] = linenum_lines.intersection(domain_to_id_dict[domain])
        with open(os.path.join(input_dir, f"{index_domain}-domeval.fuzzies"),'r') as fuzzy_file:
            indexdomain_to_fuzzy_src[index_domain] = zip(linenum_lines, fuzzy_file.readlines())
            
    for idx, domain in id_to_domain_dict.items():
        if domain not in all_trg_lines.keys():
            continue
        #new domain initialization
        if domain not in domain_specific_refs:
            domain_specific_refs[domain] = []
            domain_specific_nofuzzies[domain] = []
            domain_specific_baselines[domain] = []
            for index_domain in index_domains:
                if index_domain not in domain_specific_trans:
                    domain_specific_trans[index_domain] = {}
                if domain not in domain_specific_trans[index_domain]:
                    domain_specific_trans[index_domain][domain] = []

        domain_specific_refs[domain].append(all_ref_lines[idx])
        domain_specific_nofuzzies[domain].append(nofuzzies_trg_lines[idx])
        domain_specific_baselines[domain].append(baseline_lines[idx])

        for index_domain in index_domains:
            domain_specific_trans[index_domain][domain].append(all_trg_lines[index_domain][idx])

    # Now calculate sacrebleu for domain-specific translations
    for domain in domain_specific_refs:
        print(f"processing {domain}")
        domain_ref_lines = domain_specific_refs[domain]        
        domain_baseline_lines = domain_specific_baselines[domain]

        # calculate baseline score for domain
        baseline_metrics = calculate_metrics(domain_ref_lines, domain_baseline_lines)
        report_file.write(f"{domain}\tbaseline\t{baseline_metrics.bleu}\t{baseline_metrics.chrf}\tN/A\tall\t0\t{system_id}\n")

        for index_domain in index_domains:
            domain_fuzzies = domain_to_fuzzy_id[index_domain][domain]
            fuzzy_count = len(domain_fuzzies)
            # don't bother to evaluate if there aren't that many fuzzies for this combo of index and domain
            if fuzzy_count < 20:
                continue
            domain_trg_lines = domain_specific_trans[index_domain][domain]
            domain_all_metrics = calculate_metrics(domain_ref_lines, domain_trg_lines)
            report_file.write(f"{domain}\t{index_domain}\t{domain_all_metrics.bleu}\t{domain_all_metrics.chrf}\tN/A\tall\t{fuzzy_count}\t{system_id}\n")

            domain_fuzzy_src = [src for (index, src) in indexdomain_to_fuzzy_src[index_domain] if index in domain_to_fuzzy_id[index_domain][domain]]
            fuzzy_ref_lines = [all_ref_lines[linenum] for linenum in domain_fuzzies]
            fuzzy_trg_lines = [all_trg_lines[index_domain][linenum] for linenum in domain_fuzzies]

            # TODO: the src is out of sync, FIX
            domain_fuzzy_metrics = calculate_metrics(fuzzy_ref_lines, fuzzy_trg_lines, domain_fuzzy_src)
            report_file.write(f"{domain}\t{index_domain}\t{domain_fuzzy_metrics.bleu}\t{domain_fuzzy_metrics.chrf}\t{domain_fuzzy_metrics.copyrate}\tonly_fuzzies\t{fuzzy_count}\t{system_id}\n")

            baseline_mix_lines = combine_baseline_and_rat_domain(domain_fuzzies, baseline_lines, all_trg_lines[index_domain], domain_ids=domain_to_id_dict[domain])
            
            domain_baseline_mix_metrics = calculate_metrics(domain_ref_lines, baseline_mix_lines)
            report_file.write(f"{domain}\tbaselinemix_{index_domain}\t{domain_baseline_mix_metrics.bleu}\t{domain_baseline_mix_metrics.chrf}\tN/A\tall\t{fuzzy_count}\t{system_id}\n")

        nofuz_trg_lines = domain_specific_nofuzzies[domain]
        domain_nofuzzy_metrics = calculate_metrics(domain_ref_lines, nofuz_trg_lines)
        report_file.write(f"{domain}\tnofuzzies\t{domain_nofuzzy_metrics.bleu}\t{domain_nofuzzy_metrics.chrf}\tN/A\tall\t0\t{system_id}\n")

def main():
    # Parse the arguments
    args = parse_args()
    
    # Prepare the report content

    with open(args.report_file,'wt') as report_file:

        # Read the reference file domeval.[trg].ref
        ref_file_path = os.path.join(args.input_dir, f"domeval.{args.trg_lang}.ref")
        ref_lines = read_file_lines(ref_file_path)
        baseline_lines = read_file_lines(args.baseline_translations)
        nofuzzies_trg_path = os.path.join(args.input_dir, f"nofuzzies.{args.trg_lang}")
        nofuzzies_trg_lines = read_file_lines(nofuzzies_trg_path)

        evaluate_domeval_domains(args.input_dir, args.trg_lang, args.domeval_ids, ref_lines,  nofuzzies_trg_lines, baseline_lines, report_file, args.system_id)

        for file_name in os.listdir(args.input_dir):
            if file_name.endswith(f"-domeval.{args.trg_lang}"):
                domain = file_name.replace(f"-domeval.{args.trg_lang}","")
                evaluate_full_domeval(domain, args.input_dir, args.trg_lang, ref_lines, report_file, args.system_id, nofuzzies_trg_lines, baseline_lines)

        evaluate_baseline(args.input_dir, args.trg_lang, ref_lines, report_file, args.baseline_translations, baseline_lines)


if __name__ == "__main__":
    main()
