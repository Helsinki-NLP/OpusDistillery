import argparse
import gzip
import os
import re
from random import shuffle, seed

def get_fuzzy_bucket(score):
    return int(score*10)

def main(args):
    print("Augmenting sentences")

    # open all the files for line-by-line processing
    with gzip.open(args.src_file_path,'rt') as src_input_file, \
        gzip.open(args.trg_file_path,'rt') as trg_input_file, \
        gzip.open(args.src_output_path,'wt') as src_output_file, \
        gzip.open(args.trg_output_path,'wt') as trg_output_file, \
        gzip.open(args.score_file_path,'rt') as score_file:

        fuzzy_buckets = {}
        mix_score_line = None

        if not f"{args.src_lang}-{args.trg_lang}" in os.path.basename(args.score_file_path):
            reverse_sents = True
        else:
            reverse_sents = False

        mix_score_file = None
        if args.mix_score_file_path:
            mix_score_file = gzip.open(args.mix_score_file_path,'rt')
            if not f"{args.src_lang}-{args.trg_lang}" in os.path.basename(args.mix_score_file_path):
                reverse_mix_sents = True
            else:
                reverse_mix_sents = False

            
        augmented_count = 0
        for index, src_sentence in enumerate(src_input_file):
            if args.lines_to_augment and augmented_count == args.lines_to_augment:
                break
            
            src_sentence = src_sentence.strip()
            trg_sentence = trg_input_file.readline().strip()
            score_line = score_file.readline().strip()

            if mix_score_file:
                mix_score_line = mix_score_file.readline().strip()

            if score_line:
                matches = re.findall("(?P<score>1|\d\.\d+)\t\d+=(?P<index_src>.+?) \|\|\| (?P<index_trg>[^\t]+)",score_line)
                # set seed to make shuffle deterministic
                seed(0)
                # shuffle to avoid too many high fuzzies
                shuffle(matches)
                # if index lang pair is different from the args lang pair, switch source and target

                if reverse_sents:
                    filtered_matches = [(float(score),trg,src) for (score,src,trg) in matches if float(score) >= args.min_score and float(score) <= args.max_score][0:args.max_fuzzies]
                else:
                    filtered_matches = [(float(score),src,trg) for (score,src,trg) in matches if float(score) >= args.min_score and float(score) <= args.max_score][0:args.max_fuzzies]
            else:
                filtered_matches = []

            # mix means using one match from an alternative source (used with targetsim to have one match that is guaranteed to be reflected on the target side)
            if mix_score_line:
                mix_matches = re.findall("(?P<score>\d\.\d+)\t\d+=(?P<index_src>.+?) \|\|\| (?P<index_trg>[^\t]+)",mix_score_line)

                if reverse_mix_sents:
                    filtered_mix_matches = [(float(score),trg,src) for (score,src,trg) in mix_matches if float(score) >= args.min_score and float(score) <= args.max_score][0:args.max_fuzzies]
                else:
                    filtered_mix_matches = [(float(score),src,trg) for (score,src,trg) in mix_matches if float(score) >= args.min_score and float(score) <= args.max_score][0:args.max_fuzzies]
                
                if filtered_mix_matches:
                    if filtered_matches:
                        filtered_matches[0] = filtered_mix_matches[0]
                    else:
                        filtered_matches = filtered_mix_matches

            #mix up matches to prevent the model from learning an order of sourcesim and targetsim
            shuffle(filtered_matches)

            # keep track of fuzzy counts
            for fuzzy in filtered_matches:
                fuzzy_bucket = get_fuzzy_bucket(fuzzy[0])
                if fuzzy_bucket in fuzzy_buckets:
                    fuzzy_buckets[fuzzy_bucket] += 1
                else:
                    fuzzy_buckets[fuzzy_bucket] = 1

            if len(filtered_matches) >= args.min_fuzzies:
                if args.include_source:
                    fuzzy_string = "".join([f"{match[1]}{args.source_separator}{match[2]}{args.target_separator}{get_fuzzy_bucket(match[0])}" for match in filtered_matches])    
                else:
                    fuzzy_string = "".join([f"{match[2]}{args.target_separator}{get_fuzzy_bucket(match[0])}" for match in filtered_matches])    
                src_output_file.write(f"{fuzzy_string}{src_sentence}\n")
                trg_output_file.write(trg_sentence+"\n")
                augmented_count += 1
            else:
                if not args.exclude_non_augmented:
                    src_output_file.write(f"{src_sentence}\n")
                    trg_output_file.write(trg_sentence+"\n")
                    augmented_count += 1

        if args.mix_score_file_path:
            mix_score_file.close()

    print(fuzzy_buckets)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Augment data with fuzzies from index.")
    parser.add_argument("--src_file_path", help="Path to the file containing the source sentences that should be augmented with fuzzies.")
    parser.add_argument("--trg_file_path", help="Path to the file containing the target sentences that should be augmented with fuzzies.")
    parser.add_argument("--src_lang", help="Source lang code.")
    parser.add_argument("--trg_lang", help="Target lang code.")
    parser.add_argument("--score_file_path", help="Path to the file containing the indices of fuzzies found for each sentence in the sentence file")
    parser.add_argument("--mix_score_file_path", help="Path to the file containing the indices of the mix fuzzies found for each sentence in the sentence file. One of these fuzzies is used alongside the normal fuzzies when augmenting a sentence")
    parser.add_argument("--src_output_path", help="Path to save the source file augmented with fuzzies.")
    parser.add_argument("--trg_output_path", help="Path to save the target file augmented with fuzzies.")
    parser.add_argument("--source_separator", default="SRC_FUZZY_BREAK", help="Separator token that separates the source side of fuzzies from other fuzzies and the source sentence")
    parser.add_argument("--target_separator", default="FUZZY_BREAK", help="Separator token that separates the target side of fuzzies from other fuzzies and the source sentence")
    parser.add_argument("--min_score", type=float, help="Only consider fuzzies that have a score equal or higher than this")
    parser.add_argument("--max_score", type=float, default=1, help="Only consider fuzzies that have a score equal or lower than this")
    parser.add_argument("--min_fuzzies", type=int, help="Augment sentence if it has at least this many fuzzies")
    parser.add_argument("--max_fuzzies", type=int, help="Augment the sentence with at most this many fuzzies (use n best matches if more than max fuzzies found)") 
    parser.add_argument("--lines_to_augment", type=int, default=-1, help="Augment this many lines, default is all lines") 
    parser.add_argument("--include_source", action="store_true", help="Also include source in the augmented line") 
    parser.add_argument("--exclude_non_augmented", action="store_true", help="Do not include non-augmented in the output") 
    # Parse the arguments
    args = parser.parse_args()
    print(args)
    main(args)
