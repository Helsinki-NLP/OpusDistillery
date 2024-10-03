import argparse
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, DataLoaderConfiguration
import ast
import time

torch.cuda.empty_cache()  # Clear unused memory


def parse_args():
    parser = argparse.ArgumentParser(description="Translate text using IndicTrans2.")
    parser.add_argument('filein', type=str, help='Input file with sentences to translate')
    parser.add_argument('fileout', type=str, help='Output file for translated sentences')
    parser.add_argument('modelname', type=str, help='Model name')
    parser.add_argument('modeldir', type=str, help='Model directory')
    parser.add_argument('src', type=str, help='Source language prefix')
    parser.add_argument('trg', type=str, help='Target language prefix')
    parser.add_argument('langinfo', type=str, help="Specify if source and target languages are required")
    parser.add_argument('langtags', type=str, help="Language tag mapping specific to the model")
    parser.add_argument('config', type=str, help="Specific configuration for decoding")
    parser.add_argument('batchsize', type=int, help="Batch size for decoding. Should be small for large models.")
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
    args = parse_args()
    #os.environ['HF_HOME'] = args.modeldir

    # Create a DataLoaderConfiguration object
    dataloader_config = DataLoaderConfiguration(split_batches=True)

    print(f"Translating {args.filein} from {args.src} to {args.trg} with {args.modelname}...")

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPUs available:", torch.cuda.device_count())

    model_name = args.modelname
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

    num_return_sequences = 8

    if args.config == "default":
        config = dict()
    else:
        config = convert_simple_dict(ast.literal_eval(args.config))

    # Read input sentences from file
    with open(args.filein, 'r', encoding='utf-8') as infile:
        input_sentences = infile.readlines()
    
    print(f"Translating {len(input_sentences)} sentences...")
    
    ip = IndicProcessor(inference=True)

    # Mapping target languages
    lang_tags = ast.literal_eval(args.langtags)
    src_lang = lang_tags.get(args.src, None)
    tgt_lang = lang_tags.get(args.trg, None)

    # Preprocess sentences
    batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    
    print("Tokenizing...")
    # Tokenize the input sentences
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenized_inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
    sentence_ids = list(range(len(input_sentences)))
    
    print("Tokenization done!")

    dataset = TokenizedDataset(tokenized_inputs, sentence_ids)
    
    # Pass the config to the Accelerator
    accelerator = Accelerator(device_placement=True, dataloader_config=dataloader_config)

    # Get the rank of the process (0 to 3 for 4 GPUs)
    rank = accelerator.process_index
    temp_file = f"{args.fileout}.rank{rank}.tmp"  # Create a temporary file for each process

    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)
    dataloader = accelerator.prepare(dataloader)  # Prepare dataset for distributed inference
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
                # Generate output
                translated_batch = model.generate(
                    **batch,
                    num_return_sequences=num_return_sequences,
                    num_beams=num_return_sequences,
                    **config,
                )
                
            # Decode the output
            with tokenizer.as_target_tokenizer():
                translated_batch = tokenizer.batch_decode(translated_batch.detach().cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # Not sure what this line does, takes a long time. Commented it out for the time being.
            #translated_batch = ip.postprocess_batch(translated_batch, lang=tgt_lang)
            
            # Write each translated sentence to the buffer
            for id, sentence in zip(augmented_sentence_ids,translated_batch):

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
        translations_per_second = len(input_sentences) / total_time if total_time > 0 else float('inf')

    # Final progress print
    print(f"Translation complete. Translating {len(input_sentences)} sentences took {total_time:.2f} seconds.")
    print(f"{translations_per_second:.2f} translations/second")

    accelerator.wait_for_everyone()

    # Ensure all processes are done, then merge
    if accelerator.is_main_process:  # Only do the merging from the main process
        merge_temp_files(args.fileout, accelerator.num_processes, num_return_sequences)
        print(f"Merged all files into {args.fileout}")


if __name__ == "__main__":
    main()