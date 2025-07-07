import torch
import torch.nn.functional as F
import json
import argparse
from transformers import LogitsProcessor, MarianTokenizer, MarianMTModel, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, AutoConfig, BeamSearchScorer
from itertools import chain, combinations, product
from collections import defaultdict

# This script is used to ensemble models with same vocabulary, with the possibility of defining different inputs for each model.
# The models can be Marian models or LLMs, although note that since they need to share vocabularies, the Marian model vocabulary
# needs to be adapted to the LLM vocab.

def load_model(model_name_or_path, device):
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Check model type by architecture string
    if "Marian" in config.model_type or config.architectures and any("Marian" in arch for arch in config.architectures):
        model = MarianMTModel.from_pretrained(model_name_or_path).to(device).eval()
        model_type = "marian"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16).to(device).eval()
        model_type = "causal_lm"

    return model
    
def load_tokenizer(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Check model type by architecture string
    if "Marian" in config.model_type or config.architectures and any("Marian" in arch for arch in config.architectures):
        tokenizer = MarianTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return tokenizer
    
def augment_tokenize_batched(
        texts,
        terms_list, 
        fuzzy_list, 
        tokenizer, 
        padding_side, 
        device="cuda"):
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
    emphasis_strength: float = 10000.0
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
    def __init__(
            self, 
            models, 
            model_types, 
            tokenizer, 
            only_main_model=False, 
            num_beams=1, 
            single_model=False):
        self.models = models
        self.model_types = model_types
        self.tokenizer = tokenizer
        self.only_main_model = only_main_model
        self.num_beams = num_beams
        self.device = models["base"].device
        self.single_model = single_model

    def viking_template(self, sentence):
        return f"<|im_start|>user\nTranslate into Finnish: {sentence}<|im_end|>\n<|im_start|>assistant\n"

    def marian_llmvoc_template(self, sentence):
        return sentence + "</s>"

    def prepare_inputs(self, src_and_terms, num_beams):
        """Prepare and store model-specific inputs, expanded for beam search"""
        self.current_inputs = {}
        batch_size = len(src_and_terms[0])
        
        for i, src_and_terms_for_model in enumerate(src_and_terms):
            src_sentences, terms, fuzzies = zip(*src_and_terms_for_model)
            
            # If this is a Marian model with LLM vocab, apply template to fix LLM tokenizer
            # differences with expected Marian tokenization 
            # Commented out for now, since we're only using Marian now
            """if model_type == "marian" and not isinstance(self.tokenizer, MarianTokenizer):
                templated_src_sentences = [self.marian_llmvoc_template(x) for x in src_sentences]
                padding_side = "right"
            elif model_type != "marian":
                templated_src_sentences = [self.viking_template(x) for x in src_sentences]
                padding_side = "left"
            else:"""
            
            templated_src_sentences = src_sentences
            padding_side = "right"
            
            if any(terms) or any(fuzzies):
                # inputs = augment_tokenize(
                inputs = augment_tokenize_batched(
                    templated_src_sentences, 
                    terms,
                    fuzzies,
                    self.tokenizer, 
                    padding_side,
                    self.device
                )
            else:
                inputs = self.tokenizer(
                    templated_src_sentences, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    padding_side=padding_side
                ).to(self.device)
            
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
            
            print("Model inputs:\n" + "\n".join(self.tokenizer.batch_decode(encoder_input_ids)))

    def __call__(self, input_ids, scores):
        if self.only_main_model:
            # Why does this return nonsense, when averaging identical input sentence outputs
            # does not?
            return scores

        # TODO: handle baseline and single model output here
        # For some reason, generate produces junk with some converted models,
        # so do the single model generation without ensemble here as well.
        if self.single_model:
            if len(self.model_types) > 3:
                raise Exception('Single model option is only valid if there is only one special model.')
            # The single model will be at position 1
            model_type = self.model_types[1]
            logits = self._get_model_logits(
                    self.models[model_type],
                    model_type,
                    self.current_inputs[1]["encoder_input_ids"],
                    self.current_inputs[1]["attention_mask"],
                    input_ids
                )
            return torch.nn.functional.softmax(logits, dim=-1)
            #return logits

        avg_probs = self._average_probs(input_ids, scores)
        
        #difference = scores-avg_probs
        #summed_difference = sum_ignore_inf(difference)
        #print(summed_difference)
        #top5 = torch.topk(avg_probs,5)
        return avg_probs

    def _average_probs(self, input_ids, scores):
        """Average probabilities from all models (log space)"""
        batch_size_times_beams = input_ids.shape[0]
        vocab_size = scores.shape[-1]
        all_probs = torch.zeros((len(self.model_types), batch_size_times_beams, vocab_size),
                              device=scores.device)
        
        for i, model_type in enumerate(self.model_types):
            if i == 0:
                all_probs[i] = torch.exp(scores)
            else:
                logits = self._get_model_logits(
                    self.models[model_type],
                    model_type,
                    self.current_inputs[i]["encoder_input_ids"],
                    self.current_inputs[i]["attention_mask"],
                    input_ids
                )
                all_probs[i] = torch.nn.functional.softmax(logits, dim=-1)
        
        #remove the main model tensor (it behaves weirdly)
        all_probs = all_probs[1:, :, :]

        #mean_log_probs = torch.log(all_probs.mean(dim=0))
        
        #mean_log_probs = torch.log(weighted_softmax_combine(all_probs, self.tokenizer))
        mean_log_probs = torch.log(combine_with_guide_softmax(all_probs, self.tokenizer))
        
        return mean_log_probs
        #return torch.logsumexp(all_probs, dim=0) - torch.log(torch.tensor(len(self.models), device=scores.device))

    def _get_model_logits(self, model, model_type, encoder_inputs, attention_mask, input_ids):
        """Get logits from a single model"""
        with torch.no_grad():
            outputs = model(
                    input_ids=encoder_inputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=input_ids,
                )
            # Commented, because we only use Marian for now
            """if model_type == "marian":
                
            else:
                input_ids_full = torch.cat([encoder_inputs, input_ids], dim=-1)
                attention_mask_full = torch.cat([
                    attention_mask,
                    torch.ones_like(input_ids)
                ], dim=-1)
                
                outputs = model(
                    input_ids=input_ids_full,
                    attention_mask=attention_mask_full,
                )"""
            
            logits = outputs.logits[:, -1, :]
            # Give no probability to pad token
            logits[:, [self.tokenizer.pad_token_id]] = -float('inf')
            return logits
            
class ModelGroup():
    def __init__(self, base_model, term_model, fuzzy_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.model_types = []
        # Use either fuzzy or term tokenizer, as the base model vocab does not have the
        # special symbols.
        self.tokenizer = load_tokenizer(fuzzy_model)
        
        self.models["base"] = load_model(base_model, self.device)
        self.models["term"] = load_model(term_model, self.device)
        self.models["fuzzy"] = load_model(fuzzy_model, self.device)        
            
class ShallowFusion:
    def __init__(self, model_group):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = model_group.models
        self.tokenizer = model_group.tokenizer
    
    def translate(
            self, 
            src_and_terms, 
            model_types, 
            num_beams=4, 
            max_length=50, 
            only_main_model=False,
            single_model=False):
        # Initialize logits processor
        logits_processor = MultiInputLogitsProcessor(
            models=self.models,
            model_types=model_types,
            tokenizer=self.tokenizer,
            only_main_model=only_main_model,
            num_beams=num_beams,
            single_model=single_model)
        
        # Prepare all model inputs
        # TODO: this does not seem to be set up for batching yet (do we need batching for this experiment)
        logits_processor.prepare_inputs(src_and_terms, num_beams)
        
        # Get main model components
        main_model = self.models["base"]
        main_inputs = logits_processor.current_inputs[0]
        
        # Generate with ensemble
        
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

        #Commented out because we only use Marian for now
        """
        if self.model_types[self.main_model_idx] == "marian":
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
        """
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def powerset(iterable):
    """Returns all subsets (including empty set)"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def powerset_nonempty(iterable):
    """Returns all non-empty subsets"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

def create_test_cases(test_suite_path):
    with open(test_suite_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    grouped = defaultdict(list)

    for entry in json_data:
        source = entry["source"]
        retrieved = entry.get("retrieved", {})
        terms = retrieved.get("terms", {})
        fuzzies = [fuzzy["target"] for fuzzy in retrieved.get("fuzzy_matches", [])]

        fuzzy_subsets = powerset(fuzzies)

        # Group target variants by source term
        term_groups = {
            src_term: [(src_term, v["term"]) for v in variants.get("target", [])]
            for src_term, variants in terms.items()
        }

        # Get all combinations of 1 variant per source term (choose subset of source terms)
        source_term_subsets = powerset(term_groups.keys())

        for subset in source_term_subsets:
            if not subset:
                continue  # skip empty set here, we'll handle fuzzies-only separately

            # For each source term in this subset, pick one variant
            variant_options = [term_groups[term] for term in subset]
            for term_combination in product(*variant_options):
                for fuzzy_subset in fuzzy_subsets:
                    grouped[(len(term_combination), len(fuzzy_subset))].append(
                        (source, list(term_combination), list(fuzzy_subset))
                    )

        # Fuzzies-only subsets (0 terms, ≥1 fuzzies)
        if fuzzies:
            for fuzzy_subset in powerset_nonempty(fuzzies):
                grouped[(0, len(fuzzy_subset))].append(
                    (source, [], list(fuzzy_subset))
                )

    return grouped


def break_down_group(group):
    """
    For a group of test cases with same number of terms/fuzzies,
    break it into a list of model configurations (all same source sentences),
    where each model corresponds to a different term/fuzzy selection across entries.
    """
    base_start = []
    base_end = []
    term_models = []
    fuzzy_models = []

    # Collect all sources, and max number of terms and fuzzies per test case
    sources = [entry[0] for entry in group]
    all_term_variants = []
    all_fuzzy_variants = []

    # Track per test case term and fuzzy variants
    for _, terms, fuzzies in group:
        all_term_variants.append(terms)
        all_fuzzy_variants.append(fuzzies)

    # Base model (start)
    base_start = [(src, None, None) for src in sources]

    # Build term models
    term_model_lists = []
    max_terms = max(len(terms) for terms in all_term_variants)
    for i in range(max_terms):
        model = []
        for terms, src in zip(all_term_variants, sources):
            if i < len(terms):
                model.append((src, [terms[i]], []))
            else:
                model.append((src, [], []))
        term_model_lists.append(model)

    # Build fuzzy models
    fuzzy_model_lists = []
    max_fuzzies = max(len(fz) for fz in all_fuzzy_variants)
    for i in range(max_fuzzies):
        model = []
        for fuzzies, src in zip(all_fuzzy_variants, sources):
            if i < len(fuzzies):
                model.append((src, [], [fuzzies[i]]))
            else:
                model.append((src, [], []))
        fuzzy_model_lists.append(model)

    # Base model (end)
    base_end = [(src, None, None) for src in sources]

    return [base_start] + term_model_lists + fuzzy_model_lists + [base_end]


def generate_unensembled(test_cases_for_model, model, tokenizer):
    src_sentences, terms, fuzzies = zip(*test_cases_for_model[1])

    inputs = augment_tokenize_batched(
        src_sentences, 
        terms,
        fuzzies,
        tokenizer, 
        "right",
        model.device
    )
    print("Model inputs:\n" + "\n".join(tokenizer.batch_decode(inputs["input_ids"])))

    translated = model.generate(**inputs)
    print("Without ensembling:" + str([tokenizer.decode(t, skip_special_tokens=True) for t in translated]))
    return str([tokenizer.decode(t, skip_special_tokens=True) for t in translated])

def main(args):
    print("Base model:", args.base_model)
    print("Term model:", args.term_model)
    print("Fuzzy model:", args.fuzzy_model)
    print("Source language:", args.source_lang)
    print("Target language:", args.target_lang)
    
    # Create a model group containing four types of models:
    # 1. base model, used as the guide model
    # 2. term model, translates source sent annotated with terms
    # 3. fuzzy model, for fuzzies
    # NOT YET IMPLEMENTED: 4. subsegment model, for subsegments
    
    model_group = ModelGroup(args.base_model, args.term_model, args.fuzzy_model)
    ensemble = ShallowFusion(model_group)
    num_beams = 6
    
    test_case_groups = create_test_cases(args.test_suite_path)
    for (term_count,fuzzy_count),test_case_group in test_case_groups.items():
        test_cases = break_down_group(test_case_group)
        model_types = ["base"] + ["term"]*term_count + ["fuzzy"]*fuzzy_count + ["base"]
        translations = ensemble.translate(
            test_cases,
            model_types,
            num_beams=num_beams,
            max_length=100,
            only_main_model=False
        )

        # If only one model is used, also generate unensembled translations for comparison
        if len(model_types) == 3:
            single_model_translations = ensemble.translate(
                test_cases,
                model_types,
                num_beams=num_beams,
                max_length=100,
                only_main_model=False,
                single_model=True
            )

            # This generates the single model translations using the standard
            # generate method instead of the ensemble logit processor. For debugging
            # to see if there's a difference.
            single_model_translations_with_generate = generate_unensembled(
                test_cases,
                model_group.models[model_types[1]],
                model_group.tokenizer)

            print("test")

        # Record terms and fuzzies for print debugging
        terms = []
        fuzzies = []

        for i, model_type in enumerate(model_types):
            if model_type == "term":
                if not terms:
                    terms = [x[1] for x in test_cases[i]]
                else:
                    terms = list(zip(terms,[x[1] for x in test_cases[i]]))
            if model_type == "fuzzy":
                if not fuzzies:
                    fuzzies = [x[2] for x in test_cases[i]]
                else:
                    fuzzies = list(zip(fuzzies,[x[2] for x in test_cases[i]]))

        if not terms:
            terms = [None] * len(fuzzies)
        if not fuzzies:
            fuzzies = [None] * len(terms)

        print("\n".join([f"Fuzzies: {x[0]}, Terms: {x[1]}, Translation: {x[2]}" for x in (zip(fuzzies,terms,translations))]))
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate test suite test cases using an ensemble of models accepting different kinds of retrieved information (terms, ngrams, full matches).")
    
    parser.add_argument(
        "--base_model",
        required=True,
        help="Base model to use as contrast model."
    )

    parser.add_argument(
        "--term_model",
        required=True,
        help="Term model."
    )

    parser.add_argument(
        "--fuzzy_model",
        required=True,
        help="Fuzzy model."
    )
        
    parser.add_argument(
        "--source_lang",
        required=True,
        help="Source language code (e.g., 'en')."
    )
    
    parser.add_argument(
        "--target_lang",
        required=True,
        help="Target language code (e.g., 'fr')."
    )
    
    parser.add_argument(
        "--test_suite_path",
        required=True,
        help="Path to the test suite."
    )
    
    args = parser.parse_args()
    main(args)