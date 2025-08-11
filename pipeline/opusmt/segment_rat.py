import argparse
import gzip
import re
import sentencepiece as spm

def parse_args():
    parser = argparse.ArgumentParser(description="Segment text split by augmentsymbolN using two SentencePiece models.")
    parser.add_argument("--input", required=True, help="Input gzipped text file.")
    parser.add_argument("--output", required=True, help="Output gzipped text file.")
    parser.add_argument("--model1", required=True, help="SentencePiece model for the last segment.")
    parser.add_argument("--model2", required=True, help="SentencePiece model for the other segments.")
    parser.add_argument("--line_count", type=int, default=None, help="Optional number of lines to process (stop after N lines).")
    parser.add_argument("--terms_appended", action="store_true", help="If terms have been appended, term augmentsymbols (0-3) need special processing.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load SentencePiece models
    sp1 = spm.SentencePieceProcessor()
    sp2 = spm.SentencePieceProcessor()
    sp1.load(args.model1)
    sp2.load(args.model2)

    # Regular expression to find augmentsymbolN markers
    pattern = re.compile(r'(augmentsymbol[0-9])')

    line_num = 0
    with gzip.open(args.input, 'rt', encoding='utf-8') as fin, \
         gzip.open(args.output, 'wt', encoding='utf-8') as fout:
        
        for line in fin:
            if args.line_count is not None and line_num >= args.line_count:
                break

            line = line.strip()
            if not line:
                fout.write("\n")
                line_num += 1
                continue

            if args.terms_appended and "augmentsymbol0" in line:
                # Split the line into fuzzy and term portions
                parts = line.split("augmentsymbol0",1)
                prefix_split = pattern.split(parts[0])
                source_sent_prefix = prefix_split[-1]
                fuzzy_split = prefix_split[0:-1] 
                
                segmented_fuzzy = []
                if fuzzy_split:
                    text_segments = fuzzy_split[::2]      # Actual text
                    symbols = fuzzy_split[1::2]           # augmentsymbolN parts
                    
                    # Tokenize each text segment
                    tokenized_segments = []
                    for i, segment in enumerate(text_segments):
                        tokenized = sp2.encode(segment, out_type=str)
                        tokenized_segments.append(" ".join(tokenized))
                    
                    # Reconstruct line by alternating tokenized segments and symbols
                    for i in range(len(symbols)):
                        segmented_fuzzy.append(tokenized_segments[i])
                        segmented_fuzzy.append(symbols[i])
                    segmented_fuzzy.append(tokenized_segments[-1])
                
                term_source_sent = source_sent_prefix + "augmentsymbol0" + parts[1]
                term_split = pattern.split(term_source_sent)
                
                text_segments = term_split[::2]      # Actual text
                symbols = term_split[1::2]           # augmentsymbolN parts
                
                segmented_terms = []
                # Tokenize each text segment
                tokenized_segments = []
                for i, segment in enumerate(text_segments):
                    if i < len(symbols):
                        next_symbol = symbols[i]
                    else:
                        next_symbol = None
                    if next_symbol == "augmentsymbol2":
                        tokenized = sp2.encode(segment, out_type=str)
                    else:
                        tokenized = sp1.encode(segment, out_type=str)
                    tokenized_segments.append(" ".join(tokenized))
                    
                # Reconstruct line by alternating tokenized segments and symbols
                for i in range(len(symbols)):
                    segmented_terms.append(tokenized_segments[i])
                    segmented_terms.append(symbols[i])
                segmented_terms.append(tokenized_segments[-1])
                
                fout.write(" ".join(segmented_fuzzy+segmented_terms) + "\n")
                
            else:
                # Split the line and keep the delimiters
                parts = pattern.split(line)
                text_segments = parts[::2]      # Actual text
                symbols = parts[1::2]           # augmentsymbolN parts
    
                # Tokenize each text segment
                tokenized_segments = []
                for i, segment in enumerate(text_segments):
                    if i == len(text_segments) - 1:
                        tokenized = sp1.encode(segment, out_type=str)
                    else:
                        tokenized = sp2.encode(segment, out_type=str)
                    tokenized_segments.append(" ".join(tokenized))
    
                # Reconstruct line by alternating tokenized segments and symbols
                result = []
                for i in range(len(symbols)):
                    result.append(tokenized_segments[i])
                    result.append(symbols[i])
                result.append(tokenized_segments[-1])
    
                fout.write(" ".join(result) + "\n")
                line_num += 1

if __name__ == "__main__":
    main()

