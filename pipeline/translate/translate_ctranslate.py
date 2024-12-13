import os
#os.environ["CTRANSLATE2_LOG_LEVEL"] = "DEBUG"
#os.environ["CT2_VERBOSE"] = "3"
import ctranslate2
from transformers import AutoTokenizer
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import argparse
import torch
import ast
import queue
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Translate text using Hugging Face pipeline with Ctranslate implementation.")
    parser.add_argument('filein', type=str, help='Input file name')
    parser.add_argument('fileout', type=str, help='Output file name')
    parser.add_argument('modelname', type=str, help='Model name')
    parser.add_argument('modeldir', type=str, help='Model directory')
    parser.add_argument('src', type=str, help='Source language prefix')
    parser.add_argument('trg', type=str, help='Target language prefix')
    parser.add_argument('langinfo', type=str, help="Specify if source and target languages are required")
    parser.add_argument('prompt', type=str, help="Prompt to use for decoding")
    parser.add_argument('langtags', type=str, help="Language tag mapping specific to the model")
    parser.add_argument('config', type=str, help="Specific configuration for decoding")
    parser.add_argument('batchsize', type=str, help="Batch size for decoding. Should be small for large models.")
    parser.add_argument('logfile', type=str, help="Logfile where prints will be flushed.")
    return parser.parse_args()

def convert_simple_dict(d):
    """Convert numeric strings to integers or floats in a flat dictionary."""
    return {key: ast.literal_eval(value) if isinstance(value, str) and value.isdigit() else value for key, value in d.items()}

def split_list_into_sublists(input_list, sublist_size=1):
    return [input_list[i:i + sublist_size] for i in range(0, len(input_list), sublist_size)]

# Define a function to translate a batch using a specific translator
def translate_batch(batch, translator, tokenizer, tgt_lang, model_name, batch_size=64):
    tokenized_batch = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in batch]
    if 'nllb' in model_name:
        tgt_prefix_list = [[tgt_lang]] * len(tokenized_batch)
        results = translator.translate_batch(tokenized_batch, target_prefix=tgt_prefix_list, beam_size=8, num_hypotheses=8, batch_type="tokens", max_batch_size=batch_size)
    else:
        results = translator.translate_batch(tokenized_batch, beam_size=8, num_hypotheses=8, batch_type="tokens", max_batch_size=batch_size)
    hypotheses = []
    for result in results:
        for hyp in result.hypotheses:
            detok_hyp = tokenizer.decode(tokenizer.convert_tokens_to_ids(hyp), skip_special_tokens=True)
            hypotheses.append(detok_hyp)
    return hypotheses

# Worker function to process batches from the task queue
def translate_worker(translator, task_queue, output_lock, output_file, tokenizer, tgt_lang, model_name, logfile, batch_size=64, progress_bar=None):
    while not task_queue.empty():
        try:
            batch = task_queue.get()  # Get the batch (index, sentence) tuples
            indices, sentences = zip(*batch)  # Separate the indices and the sentences

            translated_batches = translate_batch(sentences, translator, tokenizer, tgt_lang, model_name, batch_size=batch_size)
            translated_batches = split_list_into_sublists(translated_batches)
            
            # Write the results to the target file
            with output_lock:
                with open(output_file, 'a', encoding='utf-8') as tgt:
                    for index, translations in zip(indices, translated_batches):
                        for translation in translations:
                            tgt.write(f"{index} ||| {translation}\n")
            
            if progress_bar:
                progress_bar.update(len(batch))
        except queue.Empty:
            break

# Main translation function with GPU or CPU distribution and tracking
def translate_file(source_file, target_file, translators, num_devices, tokenizer, logfile, tgt_lang, model_name, batch_size=64):
    start_time = time.time()

    # Read source file and create task queue
    print("Reading file...", file=logfile, flush=True)
    with open(source_file, 'r', encoding='utf-8') as src:
        lines = src.readlines()
    print("Data read!", file=logfile, flush=True)

    task_queue = Queue()
    for i in range(0, len(lines), batch_size):
        # Create a list of (index, sentence) tuples for the current batch
        batch = [(i + j, lines[i + j]) for j in range(len(lines[i:i + batch_size]))]
        task_queue.put(batch)
    
    # Create a lock for writing to the output file
    output_lock = threading.Lock()

    # Initialize a progress bar
    total_batches = len(lines) // batch_size + (1 if len(lines) % batch_size != 0 else 0)
    with tqdm(total=total_batches, desc="Translating", unit=" lines") as progress_bar:
        # Using ThreadPoolExecutor to manage parallel processing
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for device_index in range(num_devices):
                executor.submit(translate_worker, translators[device_index], task_queue, output_lock, target_file, tokenizer, tgt_lang, model_name, logfile, batch_size, progress_bar)

    end_time = time.time()
    print(f"Translation completed in {end_time - start_time:.2f} seconds.", file=logfile, flush=True)

def main():
    args = parse_args()
    os.environ['HF_HOME'] = args.modeldir
    logfile = open(args.logfile, "w")
    print(f"Translating {args.filein} from {args.src} to {args.trg} with {args.modelname}...", file=logfile, flush=True)

    print("PyTorch version:", torch.__version__, file=logfile, flush=True)
    print("CUDA available:", torch.cuda.is_available(), file=logfile, flush=True)
    print("GPUs available:", torch.cuda.device_count(), file=logfile, flush=True)

    # Paths and settings
    model_path = args.modeldir
    model_name = args.modelname
    batch_size = ast.literal_eval(args.batchsize)

    # Mapping target languages
    lang_tags = ast.literal_eval(args.langtags)
    src_lang = lang_tags.get(args.src, None)
    tgt_lang = lang_tags.get(args.trg, None)
    
    # Determine if CUDA is available and set up the appropriate devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_type = "cuda"
        num_devices = num_gpus
    else:
        num_devices = os.cpu_count()  # Use the number of available CPUs
        device_type = "cpu"

    print("Loading tokenizer...", file=logfile, flush=True)
    # Initialize a tokenizer
    if args.langinfo in ["True", "true", "1"]:
        print(f"Loading tokenizer with language tags: {src_lang} and {tgt_lang}", file=logfile, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, src_lang=src_lang, tgt_lang=tgt_lang, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    print(f"Tokenizer loaded!", file=logfile, flush=True)

    
    print("Loading model...", file=logfile, flush=True)
    # Initialize a translator instance for each device (either GPU or CPU)
    translators = [ctranslate2.Translator(model_path, device=device_type, device_index=i if device_type == "cuda" else 0, compute_type="float16") for i in range(num_devices)]

    print("Model loaded!", file=logfile, flush=True)

    if args.config == "default":
        config = dict()
    else:
        config = convert_simple_dict(ast.literal_eval(args.config))

    source_file = args.filein
    target_file = args.fileout
    translate_file(source_file, target_file, translators, num_devices, tokenizer, logfile, tgt_lang, model_name, batch_size=batch_size)

    print(f"Translation completed. Results are saved in '{target_file}'", file=logfile, flush=True)

if __name__ == "__main__":
    main()
