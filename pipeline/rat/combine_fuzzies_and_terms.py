import argparse
import gzip
import sys

def process_files(fuzzy_file_path, term_file_path, output_file_path, combosim_doubled):
    with gzip.open(fuzzy_file_path, 'rt', encoding='utf-8') as fuzzy_file, \
         gzip.open(term_file_path, 'rt', encoding='utf-8') as term_file, \
         gzip.open(output_file_path, 'wt', encoding='utf-8') as output_file:
        
        for line_index,line1 in enumerate(fuzzy_file):
        
            # in combosim files the fuzzy lines are doubled, so read the term line only every other fuzzy line
            if combosim_doubled and line_index % 2 == 0:
                line2 = term_file.readline()
            elif not combosim_doubled:
                line2 = term_file.readline()
            # Extract prefix from fuzzy file (before augmentsymbol)
            line1_split = line1.rsplit(r'augmentsymbol', 1)
            # remember to restore the augmentsymbol index which is the first char of the second part after the split
            if len(line1_split) > 1:
                prefix = line1_split[0] + "augmentsymbol" + line1_split[1][0]
            else:
                prefix = ""
            
            # replace augmentsymbol1 (used with nobands models) with augmentsymbol3, since 1 is used in the term annotations for a different purpose
            prefix = prefix.replace("augmentsymbol1","augmentsymbol3")
            
            # Process term line: detok sentencepiece
            detok_term_line = line2.replace(' ', '').replace('▁', ' ').strip()
            
            # Combine and print
            output_file.write(prefix + detok_term_line+"\n")
            
    
def main():
    parser = argparse.ArgumentParser(
        description="Merge term and fuzzy corpora to get unified rat training corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--fuzzy_src_corpus', help="Gzipped corpus annotated with fuzzies")
    parser.add_argument('--term_src_corpus', help="Gzipped corpus annotated with terms")
    parser.add_argument('--output_src_corpus', help="Gzipped output corpus annotated with both terms and fuzzies")
    parser.add_argument('--combosim_doubled', action="store_true", help="Indicates whether corpus has been doubled with combosim")
    
    args = parser.parse_args()
    print(args)
    process_files(args.fuzzy_src_corpus, args.term_src_corpus, args.output_src_corpus, args.combosim_doubled)

if __name__ == '__main__':
    main()
