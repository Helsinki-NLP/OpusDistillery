import argparse
import gzip
import os
import re

def process_files(combination_factor, source_path, target_path, source_sim_path, target_sim_path, source_output_path, target_output_path, source_nobands_output_path, target_nobands_output_path):

    # Regex pattern to replace FUZZY_BREAK_[0-9] with FUZZY_BREAK
    fuzzy_break_pattern = re.compile(r"FUZZY_BREAK_\d")

    # Open all files, reading the input files one line at a time
    with gzip.open(source_path, 'rt') as source_file, \
         gzip.open(source_sim_path, 'rt') as source_sim_file, \
         gzip.open(target_path, 'rt') as target_file, \
         gzip.open(target_sim_path, 'rt') as target_sim_file, \
         gzip.open(source_output_path, 'wt') as output_source_file, \
         gzip.open(target_output_path, 'wt') as output_target_file, \
         gzip.open(source_nobands_output_path, 'wt') as nobands_output_source_file, \
         gzip.open(target_nobands_output_path, 'wt') as nobands_output_target_file:

        # Read and combine source and target_sim source files
        for i, (source_line, source_sim_line) in enumerate(zip(source_file, source_sim_file)):
            if combination_factor == 1 or (combination_factor == 2 and i % 2 == 0):
                # For factor 1, include all lines
                # For factor 2, include only even lines from source
                output_source_file.write(source_line)
                nobands_output_source_file.write(fuzzy_break_pattern.sub("FUZZY_BREAK", source_line))
                
            if combination_factor == 1 or (combination_factor == 2 and i % 2 == 1):
                # For factor 1, include all lines
                # For factor 2, include only odd lines from target_sim source
                output_source_file.write(source_sim_line)
                nobands_output_source_file.write(fuzzy_break_pattern.sub("FUZZY_BREAK", source_sim_line))
                
        # Read and combine target and target_sim target files
        for i, (target_line, target_sim_line) in enumerate(zip(target_file, target_sim_file)):
            if combination_factor == 1 or (combination_factor == 2 and i % 2 == 0):
                # For factor 1, include all lines
                # For factor 2, include only even lines from target
                output_target_file.write(target_line)
                nobands_output_target_file.write(fuzzy_break_pattern.sub("FUZZY_BREAK", target_line))
                
            if combination_factor == 1 or (combination_factor == 2 and i % 2 == 1):
                # For factor 1, include all lines
                # For factor 2, include only odd lines from target_sim target
                output_target_file.write(target_sim_line)
                nobands_output_target_file.write(fuzzy_break_pattern.sub("FUZZY_BREAK", target_sim_line))

                
def main():
    parser = argparse.ArgumentParser(description="Combine and process gzipped files based on combination factor.")
    parser.add_argument('--combination_factor', type=int, required=True, help="Combination factor (1 or 2)")
    parser.add_argument('--source', type=str, required=True, help="Path to the source file (gzipped)")
    parser.add_argument('--target', type=str, required=True, help="Path to the target file (gzipped)")
    parser.add_argument('--source_targetsim', type=str, required=True, help="Path to the targetsim source file (gzipped)")
    parser.add_argument('--target_targetsim', type=str, required=True, help="Path to the targetsim target file (gzipped)")
    parser.add_argument('--source_output', type=str, required=True, help="Path to the source output file (gzipped)")
    parser.add_argument('--target_output', type=str, required=True, help="Path to the target output file (gzipped)")
    parser.add_argument('--source_nobands_output', type=str, required=True, help="Path to the source nobands output file (gzipped)")
    parser.add_argument('--target_nobands_output', type=str, required=True, help="Path to the target nobands output file (gzipped)")

    args = parser.parse_args()

    # Validate combination factor
    if args.combination_factor not in [1, 2]:
        raise ValueError("Combination factor must be either 1 or 2.")

    # Process files with the provided combination factor
    process_files(args.combination_factor, args.source, args.target, args.source_targetsim, args.target_targetsim, args.source_output, args.target_output, args.source_nobands_output, args.target_nobands_output)

if __name__ == "__main__":
    main()

