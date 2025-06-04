import torch
import torch.nn.functional as F
import json
import argparse
from transformers import LogitsProcessor, MarianTokenizer, MarianMTModel, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, AutoConfig, BeamSearchScorer

# This script is used to ensemble models with same vocabulary, with the possibility of defining different inputs for each model.
# The models can be Marian models or LLMs, although note that since they need to share vocabularies, the Marian model vocabulary
# needs to be adapted to the LLM vocab.

def load_model(model_name_or_path, device):
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Check model type by architecture string
    if "Marian" in config.model_type or config.architectures and any("Marian" in arch for arch in config.architectures):
        print(model_name_or_path)
        model = MarianMTModel.from_pretrained(model_name_or_path).to(device).eval()
        model_type = "marian"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16).to(device).eval()
        model_type = "causal_lm"

    return model, model_type
    
def load_tokenizer(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Check model type by architecture string
    if "Marian" in config.model_type or config.architectures and any("Marian" in arch for arch in config.architectures):
        tokenizer = MarianTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return tokenizer
    
def augment_tokenize_batched(texts, terms_list, fuzzy_list, tokenizer, padding_side, device="cuda"):
    """
    Process batched input of texts and terms for augmentation.
    
    Args:
        texts: List of input text strings
        terms_list: List of lists of (source_term, target_term) tuples for each text
        fuzzy_list: List of lists of fuzzies for each text
        tokenizer: Tokenizer to use
        padding_side: "left" or "right" - where to add padding
        device: Device to put tensors on
    """
    vocab = tokenizer.get_vocab()
    all_input_ids = []
    
    for text, terms, fuzzies in zip(texts, terms_list, fuzzy_list):
        text_tokenized = tokenizer(text).input_ids
        if terms:
            for term_source, term_target in terms:
                term_source_tokenized = tokenizer(term_source).input_ids[:-1]  # remove EOS
                term_target_tokenized = tokenizer(text_target=term_target).input_ids[:-1]  # remove EOS
                term_target_tokenized = [vocab["augmentsymbol1"]] + term_target_tokenized + [vocab["augmentsymbol2"]]
                
                current_aug_part_index = 0
                new_text_tokenized = []
                
                for token in text_tokenized:
                    # Check if we're in the middle of matching a term
                    if current_aug_part_index < len(term_source_tokenized) and token == term_source_tokenized[current_aug_part_index]:
                        current_aug_part_index += 1

                        # If we fully matched a term
                        if current_aug_part_index == len(term_source_tokenized):
                            new_text_tokenized.extend([vocab["augmentsymbol0"]] + term_source_tokenized + term_target_tokenized)
                            current_aug_part_index = 0
                        continue
                    
                    # If we partially matched but then failed
                    if current_aug_part_index > 0:
                        new_text_tokenized.extend(term_source_tokenized[:current_aug_part_index])
                        current_aug_part_index = 0
                    
                    # Default case - just add the token
                    new_text_tokenized.append(token)
                
                text_tokenized = new_text_tokenized
        if fuzzies:
            for fuzzy in fuzzies:
                # Tokenize fuzzy with target vocab, removing EOS
                fuzzy_tokenized = tokenizer(text_target=fuzzy).input_ids[:-1]
                # Add augmentsymbol
                fuzzy_tokenized.append(vocab["augmentsymbol1"])
                text_tokenized = fuzzy_tokenized + text_tokenized

        all_input_ids.append(text_tokenized)
    
    # Padding and batching
    max_len = max(len(ids) for ids in all_input_ids)
    padded_input_ids = []
    attention_masks = []
    
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        
        if padding_side == "right":
            padded_ids = ids + [tokenizer.pad_token_id] * pad_len
        else:
            padded_ids = [tokenizer.pad_token_id] * pad_len + ids
            attention_mask = [0] * pad_len + attention_mask[:len(ids)]
        
        padded_input_ids.append(padded_ids)
        attention_masks.append(attention_mask)
    
    # Convert to tensors
    input_ids = torch.tensor(padded_input_ids, device=device)
    attention_mask = torch.tensor(attention_masks, device=device)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
def sum_ignore_inf(tensor):
    # Create a mask for finite values (not -inf or inf)
    finite_mask = torch.isfinite(tensor)
    
    # Apply the mask to get only finite values, setting others to 0
    finite_values = torch.where(finite_mask, tensor, torch.zeros_like(tensor))
    
    # Sum all finite values
    total_sum = torch.sum(finite_values)
    
    return total_sum

def combine_with_guide_softmax(
    softmax_tensor: torch.Tensor,
    tokenizer,
    emphasis_strength: float = 10.0
) -> torch.Tensor:
    """
    Combines model softmaxes using the last model as a guide.
    Emphasizes models whose symbol probabilities differ significantly from the guide.

    Args:
        softmax_tensor: Tensor of shape (num_models, num_positions, num_symbols).
        tokenizer: Tokenizer object with a `decode()` method.
        emphasis_strength: Controls how strongly differences from the guide affect weighting.

    Returns:
        Combined softmax tensor of shape (num_positions, num_symbols).
    """
    guide = softmax_tensor[-1]  # Shape: (num_positions, num_symbols)
    others = softmax_tensor[:-1]  # Shape: (num_models - 1, num_positions, num_symbols)

    # Compute absolute difference from guide
    diff = torch.abs(others - guide.unsqueeze(0))  # Shape: (num_models - 1, num_positions, num_symbols)

    # Higher difference => higher emphasis
    weights = F.softmax(emphasis_strength * diff, dim=0)  # (num_models - 1, num_positions, num_symbols)

    # Weighted sum over other models
    combined = torch.sum(weights * others, dim=0)  # (num_positions, num_symbols)

    # Optionally include the guide in the final combination
    # For example: combine 90% weighted other models + 10% guide
    combined = 0.9 * combined + 0.1 * guide

    # Normalize to ensure valid softmax (numerical safety)
    combined = combined / combined.sum(dim=-1, keepdim=True)
    
    return combined
    
import torch.nn.functional as F

def weighted_softmax_combine(softmax_tensor: torch.Tensor, tokenizer, temperature: float = 1.0) -> torch.Tensor:
    """
    Emphasizes symbols that have significantly higher probabilities in one model compared to others.

    Args:
        softmax_tensor: Tensor of shape (num_models, num_positions, num_symbols),
                        where num_positions = num_beams * batch_size.
        temperature: Controls sharpness of weighting; lower => more emphasis on dominant symbols.

    Returns:
        Combined softmax tensor of shape (num_positions, num_symbols).
    """
    # TODO: check what the probs of "kotelo" and "toimeksianto" are, see if there's a pronounced jump

    # Compute mean and std across models for each symbol
    mean_probs = softmax_tensor.mean(dim=0, keepdim=True)  # Shape: (1, num_positions, num_symbols)
    std_probs = softmax_tensor.std(dim=0, keepdim=True)    # Shape: (1, num_positions, num_symbols)

    # Calculate "dominance score" per model per symbol: how much it exceeds the mean
    dominance_score = (softmax_tensor - mean_probs) / (std_probs + 1e-8)  # Z-score like
    
    # Apply softmax over models (dim=0) for each symbol, using dominance score
    weights = F.softmax(dominance_score / temperature, dim=0)  # Shape: (num_models, num_positions, num_symbols)

    # Weighted sum across models
    combined_softmax = torch.sum(weights * softmax_tensor, dim=0)  # Shape: (num_positions, num_symbols)

    # Top 10 symbols by average weight
    topk = torch.topk(weights, k=10)
    top_indices = topk.indices.tolist()
    top_values = topk.values.tolist()
    """
    print("Top 10 emphasized symbols:")
    for idx, score in zip(top_indices[1], top_values[1]):
        symbol = tokenizer.decode(idx)
        print(f"Token ID {str(idx)}: '{symbol}' (weight: {str(score)})")"""

    for test_word in ["rasia","laatikko","ruutu","kotelo","paketti","pakkaus"]:
        test_index = tokenizer(text_target=test_word).input_ids[0]
        for i in range(0,len(softmax_tensor)):
            test_prob_pos = softmax_tensor[i][0][test_index]
            if test_prob_pos.item() > 0.1:
                print(f"{test_word} {i}: " + str(test_prob_pos))
        test_prob_pos = combined_softmax[0][test_index]
        print(f"{test_word} combined: " + str(test_prob_pos))
    
    return combined_softmax
    
class MultiInputLogitsProcessor(LogitsProcessor):
    def __init__(self, models, model_types, tokenizer, models_info, only_main_model=False, num_beams=1):
        self.models = models
        self.model_types = model_types
        self.tokenizer = tokenizer
        self.models_info = models_info
        self.current_inputs = {}  # Will store prepared inputs for each model
        self.only_main_model = only_main_model
        self.num_beams = num_beams

    def viking_template(self, sentence):
        return f"<|im_start|>user\nTranslate into Finnish: {sentence}<|im_end|>\n<|im_start|>assistant\n"

    def marian_llmvoc_template(self, sentence):
        return sentence + "</s>"

    def prepare_inputs(self, src_and_terms, num_beams):
        """Prepare and store model-specific inputs, expanded for beam search"""
        self.current_inputs = {}
        batch_size = len(src_and_terms[0])
        
        for i, (model, model_type, src_and_terms_for_model) in enumerate(zip(self.models, self.model_types, src_and_terms)):
            print(src_and_terms_for_model)
            src_sentences, terms, fuzzies = zip(*src_and_terms_for_model)
            
            # If this is a Marian model with LLM vocab, apply template to fix LLM tokenizer
            # differences with expected Marian tokenization 
            if model_type == "marian" and not isinstance(self.tokenizer, MarianTokenizer):
                templated_src_sentences = [self.marian_llmvoc_template(x) for x in src_sentences]
                padding_side = "right"
            elif model_type != "marian":
                templated_src_sentences = [self.viking_template(x) for x in src_sentences]
                padding_side = "left"
            else:
                templated_src_sentences = src_sentences
                padding_side = "right"
            if any(terms):
                # inputs = augment_tokenize(
                inputs = augment_tokenize_batched(
                    templated_src_sentences, 
                    terms,
                    fuzzies,
                    self.tokenizer, 
                    padding_side,
                    model.device
                )
            else:
                inputs = self.tokenizer(
                    templated_src_sentences, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    padding_side=padding_side
                ).to(model.device)
            
            # Expand inputs for beam search, except for main model (that already gets
            # expanded by HF)
            if i != 0:
                encoder_input_ids = inputs["input_ids"].unsqueeze(1).expand(-1, num_beams, -1)
                encoder_input_ids = encoder_input_ids.reshape(batch_size * num_beams, -1)
                
                attention_mask = inputs["attention_mask"].unsqueeze(1).expand(-1, num_beams, -1)
                attention_mask = attention_mask.reshape(batch_size * num_beams, -1)
            else:
                encoder_input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
            self.current_inputs[i] = {
                "encoder_input_ids": encoder_input_ids,
                "attention_mask": attention_mask,
                "original_batch_size": batch_size
            }
            
            print(self.tokenizer.batch_decode(encoder_input_ids))

    def __call__(self, input_ids, scores):
        # TODO: test with two aux models, not using the main model at all. Will the terms work there?
        if self.only_main_model:
            # Why does this return non-sense, when averaging identical input sentence outputs
            # does not?
            return scores
        
        avg_probs = self._average_probs(input_ids, scores)
        
        difference = scores-avg_probs
        summed_difference = sum_ignore_inf(difference)
        print(summed_difference)
        top5 = torch.topk(avg_probs,5)
        return avg_probs

    def _average_probs(self, input_ids, scores):
        """Average probabilities from all models (log space)"""
        batch_size_times_beams = input_ids.shape[0]
        vocab_size = scores.shape[-1]
        all_probs = torch.zeros((len(self.models), batch_size_times_beams, vocab_size),
                              device=scores.device)
        
        for i, (model, model_type) in enumerate(zip(self.models, self.model_types)):
            if i == 0:
                all_probs[i] = torch.exp(scores)
            else:
                logits = self._get_model_logits(
                    model,
                    model_type,
                    self.current_inputs[i]["encoder_input_ids"],
                    self.current_inputs[i]["attention_mask"],
                    input_ids,
                    i
                )
                all_probs[i] = torch.nn.functional.softmax(logits, dim=-1)
        
        #remove the main model tensor (it behaves weirdly)
        all_probs = all_probs[1:, :, :]
        #mean_log_probs = torch.log(all_probs.mean(dim=0))
        
        #mean_log_probs = torch.log(weighted_softmax_combine(all_probs, self.tokenizer))
        mean_log_probs = torch.log(combine_with_guide_softmax(all_probs, self.tokenizer))
        
        return mean_log_probs
        #return torch.logsumexp(all_probs, dim=0) - torch.log(torch.tensor(len(self.models), device=scores.device))

    def _get_model_logits(self, model, model_type, encoder_inputs, attention_mask, input_ids, model_idx):
        """Get logits from a single model"""
        with torch.no_grad():
            if model_type == "marian":
                outputs = model(
                    input_ids=encoder_inputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=input_ids,
                )
            else:
                input_ids_full = torch.cat([encoder_inputs, input_ids], dim=-1)
                attention_mask_full = torch.cat([
                    attention_mask,
                    torch.ones_like(input_ids)
                ], dim=-1)
                
                outputs = model(
                    input_ids=input_ids_full,
                    attention_mask=attention_mask_full,
                )
            
            logits = outputs.logits[:, -1, :]
            logits[:, [self.tokenizer.pad_token_id]] = -float('inf')
            return logits
            
class ModelGroup():
    def __init__(self, models_info, tokenizer_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.model_types = []
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        for info in models_info:
            model, model_type = load_model(info["name"], self.device)
            self.models.append(model)
            self.model_types.append(model_type)
            
class ShallowFusion:
    def __init__(self, models_info, model_group, main_model_idx=0):
        self.models_info = models_info
        self.main_model_idx = main_model_idx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = model_group.models
        self.model_types = model_group.model_types
        self.tokenizer = model_group.tokenizer
    
    def translate(self, src_and_terms, num_beams=4, max_length=50, only_main_model=False):
        # Initialize logits processor
        logits_processor = MultiInputLogitsProcessor(
            models=self.models,
            model_types=self.model_types,
            tokenizer=self.tokenizer,
            models_info=self.models_info,
            only_main_model=only_main_model,
            num_beams=num_beams)
        
        # Prepare all model inputs
        logits_processor.prepare_inputs(src_and_terms, num_beams)
        
        # Get main model components
        main_model = self.models[self.main_model_idx]
        main_inputs = logits_processor.current_inputs[self.main_model_idx]
        
        # Generate with ensemble
        if self.model_types[self.main_model_idx] == "marian":
            outputs = main_model.generate(
                input_ids=main_inputs["encoder_input_ids"],
                attention_mask=main_inputs["attention_mask"],
                num_beams=num_beams,
                max_length=max_length,
                logits_processor=[logits_processor],
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            outputs = main_model.generate(
                input_ids=main_inputs["encoder_input_ids"],
                attention_mask=main_inputs["attention_mask"],
                num_beams=num_beams,
                max_length=max_length,
                logits_processor=[logits_processor],
                early_stopping=True,
                eos_token_id=23,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

def create_test_cases(test_suite_path):
    # First expand the test cases to account for all term variants
    expanded_cases = []
    with open(test_suite_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    for entry in json_data:
        source = entry["source"]
        retrieved = entry.get("retrieved", {})
        fuzzy_targets = [f["target"] for f in retrieved.get("fuzzy_matches", [])]
        
        terms = retrieved.get("terms", {})
        if not terms:
            # Case with no terms (just fuzzy matches)
            expanded_cases.append({
                "source": source,
                "fuzzy": fuzzy_targets,
                "terms": None
            })
        else:
            # Create one test case per term variant
            for source_term, target_data in terms.items():
                for variant in target_data.get("target", []):
                    expanded_cases.append({
                        "source": source,
                        "fuzzy": fuzzy_targets,
                        "terms": (source_term, variant["term"])
                    })
    
    # Now build the aligned lists
    guide_input = []
    fuzzy_list = []
    term_list = []
    
    for case in expanded_cases:
        guide_input.append((case["source"], None, None))
        fuzzy_list.append((case["source"], [], case["fuzzy"]) if case["fuzzy"] else None)
        term_list.append((case["source"], [case["terms"]], []) if case["terms"] else None)
    
    return [guide_input, fuzzy_list, term_list, guide_input]

def main(model_dirs, source_lang, target_lang, test_suite_path):
    print("Model directories:", model_dirs)
    print("Source language:", source_lang)
    print("Target language:", target_lang)
    
    # TODO: create a model group consisting of enough models to handle all the test cases in the test suite.
    # Maybe each model should have a config file expressing how many terms, ngrams, full matches they support?
    # That would save having to give the info on the command line. 
    # Then iterate the test cases, and assign retrieved information to each model according to their capabilities.
    
    # First model in the list is the main model, which is actually not used (due to weirdness in logit processor handling),
    # duplicate first model to act as the dummy
    model_dirs = [model_dirs[0]] + model_dirs
    
    # Last model in the list is the guide model, just use the last model as a standin for now (FIX LATER to use base model!)
    model_dirs.append(model_dirs[-1])
    models_info = [{"name": x, "terms": None} for x in model_dirs]
    model_group = ModelGroup(models_info, tokenizer_path=model_dirs[0])
    ensemble = ShallowFusion(models_info, model_group)
    num_beams = 6
    
    test_cases = create_test_cases(test_suite_path)
    translations = ensemble.translate(
        test_cases,
        num_beams=num_beams,
        max_length=100,
        only_main_model=False
    )
    
    print(translations)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate test suite test cases using an ensemble of models accepting different kinds of retrieved information (terms, ngrams, full matches).")
    
    parser.add_argument(
        "model_dirs",
        nargs="+",
        help="List of paths to model directories."
    )
    
    parser.add_argument(
        "--source-lang",
        required=True,
        help="Source language code (e.g., 'en')."
    )
    
    parser.add_argument(
        "--target-lang",
        required=True,
        help="Target language code (e.g., 'fr')."
    )
    
    parser.add_argument(
        "--test_suite_path",
        required=True,
        help="Path to the test suite."
    )
    
    args = parser.parse_args()
    main(args.model_dirs, args.source_lang, args.target_lang, args.test_suite_path)