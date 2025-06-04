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

