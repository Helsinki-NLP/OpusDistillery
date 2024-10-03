import argparse
import os
import torch
import time
import ast
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import importlib

torch.cuda.empty_cache()  # Clear unused memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer

import importlib

def parse_args():
    parser = argparse.ArgumentParser(description="Translate text using Hugging Face pipeline.")
    parser.add_argument('filein', type=str, help='Input file name')
    parser.add_argument('fileout', type=str, help='Output file name')
    parser.add_argument('modelname', type=str, help='Model name')
    parser.add_argument('modeldir', type=str, help='Model directory')
    parser.add_argument('src', type=str, help='Source language prefix')
    parser.add_argument('trg', type=str, help='Target language prefix')
    parser.add_argument('modelclass', type=str, help='Model class string')
    parser.add_argument('langinfo',  type=str, help="Specify if source and target languages are required")
    parser.add_argument('prompt',  type=str, help="Prompt to use for decoding")
    parser.add_argument('langtags',  type=str, help="Language tag mapping specific to the model")
    parser.add_argument('config',  type=str, help="Specific configuration for decoding")
    parser.add_argument('batchsize',  type=str, help="Batch size for decoding. Should be small for large models.")
    return parser.parse_args()

def convert_simple_dict(d):
    """Convert numeric strings to integers or floats in a flat dictionary."""
    return {key: ast.literal_eval(value) if isinstance(value, str) and value.isdigit() else value for key, value in d.items()}

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_inputs, sentence_ids):
        self.tokenized_inputs = tokenized_inputs
        self.sentence_ids = sentence_ids

    def __len__(self):
        return len(self.tokenized_inputs['input_ids'])

    def __getitem__(self, idx):
        # Return both the tokenized input and the sentence ID
        item = {key: val[idx] for key, val in self.tokenized_inputs.items()}
        item['sentence_id'] = self.sentence_ids[idx]
        return item

def merge_temp_files(output_file, num_processes, num_return_sequences):
    from collections import defaultdict

    all_sentences = defaultdict(list)

    # Collect sentences by their ID
    for rank in range(num_processes):
        temp_file = f"{output_file}.rank{rank}.tmp"
        with open(temp_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                sentence_id = line.split(' ||| ')[0]  # Get the sentence ID
                all_sentences[sentence_id].append(line.strip())
        os.remove(f"{output_file}.rank{rank}.tmp")

    # Write the sorted sentences to the output file, keeping only the first 8 for each ID
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for sentence_id, sentences in sorted(all_sentences.items(), key=lambda x: int(x[0])):
            for sentence in sentences[:num_return_sequences]:  # Keep only the first 8 sentences for each ID
                outfile.write(f"{sentence}\n")


def main():
    #os.environ['HF_HOME'] = args.modeldir
    args = parse_args()

    # Create a DataLoaderConfiguration object
    dataloader_config = DataLoaderConfiguration(split_batches=True)

    print(f"Translating {args.filein} from {args.src} to {args.trg} with {args.modelname}...")

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPUs available:", torch.cuda.device_count())

    model_name=args.modelname
    prompt=args.prompt
    lang_tags=ast.literal_eval(args.langtags)

    # Split the module and class names
    module_name, class_name = args.modelclass.rsplit(".", 1)
    # Import the module
    module = importlib.import_module(module_name)
    # Get the class from the module
    model_class = getattr(module, class_name)

    model = model_class.from_pretrained(model_name, trust_remote_code=True)

    # Mapping target languages
    src_lang = lang_tags.get(args.src, None)
    tgt_lang = lang_tags.get(args.trg, None)

    if args.langinfo in ["True","true","1"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, src_lang=src_lang, tgt_lang=tgt_lang, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    num_return_sequences=8

    if args.config == "default":
        config=dict()
    else:
        config=convert_simple_dict(ast.literal_eval(args.config))

    print("Tokenizing...")

    # Read the input text
    with open(args.filein, 'r', encoding='utf-8') as infile:
        text = infile.readlines()

    if "indictrans" in model_name:
        print("IndicTrans model found! Preprocessing sentences with IndicProcessor...")
        from IndicTransToolkit import IndicProcessor
        ip = IndicProcessor(inference=True)
        text = ip.preprocess_batch(text, src_lang=src_lang, tgt_lang=tgt_lang)
        print("Done!")
    else:
        # Format sentences with prompt
        text = [prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, source=t) for t in text]

    # Tokenize all the inputs at once
    tokenized_inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    
    sentence_ids = list(range(len(text)))  # Create a list of sentence IDs (0, 1, 2, ...)
    
    # Create the dataset with tokenized inputs and sentence IDs
    dataset = TokenizedDataset(tokenized_inputs, sentence_ids)

    # Pass the config to the Accelerator
    accelerator = Accelerator(device_placement=True, dataloader_config=dataloader_config)

    # Get the rank of the process (0 to 3 for 4 GPUs)
    rank = accelerator.process_index
    temp_file = f"{args.fileout}.rank{rank}.tmp"  # Create a temporary file for each process

    dataloader = DataLoader(dataset, batch_size=ast.literal_eval(args.batchsize), shuffle=False)

    dataloader = accelerator.prepare(dataloader) # Prepare dataset for distributed inference
    model = accelerator.prepare(model)  # Prepare model for distributed inference

    print("Starting translations...")

    # Accumulate multiple sentences in memory and write them to the file in larger batches
    buffer_size = 1000000
    buffer = []

    with open(temp_file, 'w', encoding='utf-8') as outfile:
        start_time = time.time()

        for batch in dataloader:
            sentence_ids = batch.pop('sentence_id').tolist()  # Extract sentence IDs from the batch

            augmented_sentence_ids = [x for x in sentence_ids for _ in range(num_return_sequences)]
            
            with torch.no_grad():
                if "indictrans" in model_name:
                    # Generate output
                    print("Translating...!")
                    translated_batch = model.generate(
                        **batch,
                        num_return_sequences=num_return_sequences,
                        num_beams=num_return_sequences,
                        **config,
                    )
                    print("Translations done!")
                    print("Detokenizing...")
                    with tokenizer.as_target_tokenizer():
                        translated_batch = tokenizer.batch_decode(translated_batch.detach().cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    print("Done!")
                else:
                    # Generate output
                    translated_batch = model.module.generate(
                        **batch,
                        num_return_sequences=num_return_sequences,
                        num_beams=num_return_sequences,
                        **config,
                    )
                
                    # Decode the output
                    translated_batch = tokenizer.batch_decode(translated_batch, skip_special_tokens=True)
            
            print(translated_batch)
            # Write each translated sentence to the buffer
            for id, sentence in zip(augmented_sentence_ids,translated_batch):
                #source_id = round(i/num_return_sequences)+sentence_counter
                curr_prompt = prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, source=text[id])
                sentence = sentence.replace(curr_prompt, "")

                # Add to buffer
                buffer.append(f"{id} ||| {sentence}\n")
            
                # When buffer is full, write it to file and clear the buffer
                if len(buffer) >= buffer_size:
                    outfile.writelines(buffer)  # Write buffer to file
                    buffer = []  # Clear the buffer

        # If there are any remaining sentences in the buffer, flush them to the file
        if buffer:
            outfile.writelines(buffer)

        end_time = time.time()
        total_time = end_time - start_time
        translations_per_second = len(text) / total_time if total_time > 0 else float('inf')

    # Final progress print
    print(f"Translation complete. Translating {len(text)} sentences took {total_time:.2f} seconds.")
    print(f"{translations_per_second:.2f} translations/second")

    accelerator.wait_for_everyone()

    # Ensure all processes are done, then merge
    if accelerator.is_main_process:  # Only do the merging from the main process
        merge_temp_files(args.fileout, accelerator.num_processes, num_return_sequences)
        print(f"Merged all files into {args.fileout}")


if __name__ == "__main__":
    main()
