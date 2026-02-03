import os
import argparse
import torch
import time
import ast
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, logging
import importlib
import math

# Set verbosity
logging.set_verbosity(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument('token',  type=str, help="HuggingFace token for gated models.")
    parser.add_argument('logfile',  type=str, help="Logfile where prints will be flushed.")
    return parser.parse_args()

def convert_simple_dict(d):
    """Convert numeric strings to integers or floats in a flat dictionary."""
    return {key: ast.literal_eval(value) if isinstance(value, str) and value.isdigit() else value for key, value in d.items()}

class RawDataset(Dataset):
    def __init__(self, text, sentence_ids):
        self.text = text
        self.sentence_ids = sentence_ids

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {"text": self.text[idx], "sentence_id": self.sentence_ids[idx]}

def collate_fn(batch, tokenizer, max_length, logfile, progress_tracker=None):
    """Collate function to tokenize the batch while keeping the sentence IDs."""
    texts = [item['text'] for item in batch]
    sentence_ids = [item['sentence_id'] for item in batch]

    # Tokenize the texts
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Convert sentence_ids to a tensor and add it to tokenized_inputs
    tokenized_inputs['sentence_id'] = torch.tensor(sentence_ids, dtype=torch.long)

    # Update progress
    if progress_tracker is not None:
        progress_tracker['current_batch'] += 1
        current = progress_tracker['current_batch']
        total = progress_tracker['total_batches']
        percentage = (current / total) * 100
        progress_message = f"Tokenization progress: {current}/{total} batches ({percentage:.2f}%)"
        # Print progress to logfile as well
        print(progress_message, file=logfile, flush=True)

    return tokenized_inputs

def preprocess_in_batches(text, batch_size, src_lang, tgt_lang, processor, logfile):
    processed_text = []
    total_batches = math.ceil(len(text) / batch_size)
    
    # Initialize progress tracker
    progress_tracker = {'current_batch': 0, 'total_batches': total_batches}
    
    for i in range(0, len(text), batch_size):
        batch = text[i:i + batch_size]
        
        # Update and log progress
        progress_tracker['current_batch'] += 1
        current = progress_tracker['current_batch']
        total = progress_tracker['total_batches']
        percentage = (current / total) * 100
        progress_message = f"Preprocessing progress: {current}/{total} batches ({percentage:.2f}%)"
        print(progress_message, file=logfile, flush=True)
        
        # Preprocess the batch
        processed_batch = processor.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        processed_text.extend(processed_batch)
    
    return processed_text

def main():
    args = parse_args()
    os.environ['HF_HOME'] = args.modeldir
    logfile = open(args.logfile, "a")
    token = args.token

    print(f"Downloading model to... {os.environ['HF_HOME']}", file=logfile, flush=True)
    
    # Tokenization needs to be done before Accelerator is used
    print(f"Translating {args.filein} from {args.src} to {args.trg} with {args.modelname}...", file=logfile, flush=True)
    if token:
        print(f"HuggingFace Token: {args.token}", file=logfile, flush=True)
    print("PyTorch version:", torch.__version__, file=logfile, flush=True)
    print("CUDA available:", torch.cuda.is_available(), file=logfile, flush=True)
    print("GPUs available:", torch.cuda.device_count(), file=logfile, flush=True)

    model_name = args.modelname
    prompt = args.prompt
    lang_tags = ast.literal_eval(args.langtags)

    # Split the module and class names
    module_name, class_name = args.modelclass.rsplit(".", 1)
    # Import the module
    module = importlib.import_module(module_name)
    # Get the class from the module
    model_class = getattr(module, class_name)
    
    print("Loading model...", file=logfile, flush=True)
    model = model_class.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, token=token) # attn_implementation="flash_attention_2"
    print("Model loaded!", file=logfile, flush=True)
    
    # Mapping target languages
    src_lang = lang_tags.get(args.src, None)
    tgt_lang = lang_tags.get(args.trg, None)

    print(f"Loading tokenizer...", file=logfile, flush=True)
    if args.langinfo in ["True", "true", "1"]:
        print(f"Loading tokenizer with language tags: {src_lang} and {tgt_lang}", file=logfile, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, src_lang=src_lang, tgt_lang=tgt_lang, use_fast=True, token=token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, token=token)
    print(f"Tokenizer loaded!", file=logfile, flush=True)

    num_return_sequences = 8
    batch_size = ast.literal_eval(args.batchsize)

    if args.config == "default":
        config = dict()
    else:
        config = convert_simple_dict(ast.literal_eval(args.config))

    # Read the input text
    print("Reading file...", file=logfile, flush=True)
    with open(args.filein, 'r', encoding='utf-8') as infile:
        text = infile.readlines()
    print("Data read!", file=logfile, flush=True)
    
    if "indictrans" in model_name:
        print("IndicTrans model found! Preprocessing sentences with IndicProcessor...", file=logfile, flush=True)
        from IndicTransToolkit import IndicProcessor
        ip = IndicProcessor(inference=True) 
        text = preprocess_in_batches(text, batch_size, src_lang, tgt_lang, ip, logfile)
    else:
        print("Formatting sentences with prompt...", file=logfile, flush=True)
        # Format sentences with prompt
        text = [prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, source=t) for t in text]
    print("Done!", file=logfile, flush=True)


    # Create a dataset with the raw text and corresponding sentence IDs
    sentence_ids = list(range(len(text)))
    raw_dataset = RawDataset(text, sentence_ids)
    
    print("Initalizing accelerator...",file=logfile, flush=True)
    # Now initialize the Accelerator for distributed processing
    accelerator = Accelerator(device_placement=True)

    # Get the rank of the process (0 to 3 for 4 GPUs)
    rank = accelerator.process_index
    temp_file = f"{args.fileout}.rank{rank}.tmp"  # Create a temporary file for each process
    
    print("Initalizing dataloader and batch tokenization...", file=logfile, flush=True)

    # Prepare the DataLoader with a custom collate function to tokenize in batches
    max_length = 256
    total_batches = math.ceil(len(raw_dataset) / batch_size)

    # Create a progress tracker dictionary to keep track of tokenization progress manually
    tokenization_tracker = {'current_batch': 0, 'total_batches': total_batches}

    dataloader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length, logfile, tokenization_tracker)
    )


    print(f"Total batches: {total_batches}", file=logfile, flush=True)

    dataloader = accelerator.prepare(dataloader)  # Prepare dataloader for distributed inference
    model = accelerator.prepare(model)  # Prepare model for distributed inference

    print("Starting translations...", file=logfile, flush=True)

    # Create a progress tracker for translation progress
    translation_tracker = {'current_batch': 0, 'total_batches': total_batches}

    # Accumulate multiple sentences in memory and write them to the file in larger batches
    buffer_size = 1000000
    buffer = []

    with open(temp_file, 'w', encoding='utf-8') as outfile:
        start_time = time.time()

        for batch in dataloader:
            # Update progress for translation
            translation_tracker['current_batch'] += 1
            current = translation_tracker['current_batch']
            total = translation_tracker['total_batches']
            percentage = (current / total) * 100
            # Print to both console and logfile
            progress_message = f"Translation progress: {current}/{total} batches ({percentage:.2f}%)"
            print(progress_message, file=logfile, flush=True)  # Print to log file

            sentence_ids = batch.pop('sentence_id').tolist()  # Extract sentence IDs from the batch

            augmented_sentence_ids = [x for x in sentence_ids for _ in range(num_return_sequences)]
            
            with torch.no_grad():
                if "indictrans" in model_name:
                    # Generate output
                    translated_batch = model.generate(
                        **batch,
                        num_return_sequences=num_return_sequences,
                        num_beams=num_return_sequences,
                        **config,
                    )
                    with tokenizer.as_target_tokenizer():
                        translated_batch = tokenizer.batch_decode(translated_batch.detach().cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    # We need to perform transliteration if English is not the target language (if any Indic language is)
                    # CURRENTLY NOT WORKING https://github.com/VarunGumma/IndicTransToolkit/issues/18
                    if tgt_lang != "eng_Latn":
                        # Postprocess the translations, including entity replacement
                        translated_batch = ip.postprocess_batch(translated_batch, lang=tgt_lang)
                elif 'nllb' in model_name:
		            # For nllb, we need to add the target language token
                    translated_batch = model.generate(
                        **batch,
                        num_return_sequences=num_return_sequences,
                        num_beams=num_return_sequences,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                        **config,
                    )
                
                    # Decode the output
                    translated_batch = tokenizer.batch_decode(translated_batch, skip_special_tokens=True)
		
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
                
                # Write each translated sentence to the buffer
                for id, sentence in zip(augmented_sentence_ids, translated_batch):
                    curr_prompt = prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, source=text[id])
                    sentence = sentence.replace(curr_prompt, "")
                    buffer.append(f"{id} ||| {sentence}\n")

                    # When buffer is full, write it to file and clear the buffer
                    if len(buffer) >= buffer_size:
                        outfile.writelines(buffer)  # Write buffer to file
                        buffer = []  # Clear the buffer
            torch.cuda.empty_cache()

        if buffer:
            outfile.writelines(buffer)

        end_time = time.time()
        total_time = end_time - start_time
        translations_per_second = len(text) / total_time if total_time > 0 else float('inf')

    print(f"Translation complete. Translating {len(text)} sentences took {total_time:.2f} seconds.", file=logfile, flush=True)
    print(f"{translations_per_second:.2f} translations/second", file=logfile, flush=True)

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
