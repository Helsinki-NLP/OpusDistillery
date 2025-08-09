import torch
import torch.nn.functional as F
import json
import argparse
import re
import csv

# IMPORTANT: Marian models seem to be broken in later transformers versions, so use 4.28.9 when translating with. That however do not have Gemma3ForCausalLM, so you have to use a newer transformers for that.
from transformers import LogitsProcessor, MarianTokenizer, MarianMTModel, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, AutoConfig, BeamSearchScorer, BitsAndBytesConfig, Gemma3ForCausalLM

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
        self.baseline_tokenizer = load_tokenizer(base_model)
        
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
            pad_token_id=self.tokenizer.pad_token_id
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

def create_test_cases_with_tests(test_suite_path):
    with open(test_suite_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    grouped = defaultdict(list)

    for entry in json_data["examples"]:
        source = entry["main_sentence"]
        domain = entry["domain"]
        terms = entry.get("terms", {})
        fuzzies = [x for x in entry.get("fuzzy_matches", []) if x.get("validated",False) and [y for y in x["translations"] if y.get("validated",False)]]

        # Group target variants by source term
        term_groups = {
            src_term: [(src_term, v["target"], v.get("tests", [])) for v in variants]
            for src_term, variants in terms.items()
        }

        # Get powersets of terms and fuzzies
        source_term_subsets = powerset(term_groups.keys())
        source_fuzzy_subsets = powerset(fuzzies)

        # process all source term subsets, including empty set
        for term_subset in source_term_subsets:
            if not term_subset:
                continue  # skip empty set here, we'll handle fuzzies-only separately

            variant_options = [term_groups[term] for term in term_subset]

            # iterate all possible target variations of the source term subset
            for term_combination in product(*variant_options):

                # process all source fuzzy subsets, including empty set
                for source_fuzzy_subset in source_fuzzy_subsets:
                    
                    # create subset for each combination of fuzzy targets
                    translations_with_tests = [[y for y in x["translations"] if "tests" in y] for x in source_fuzzy_subset]

                    # process all target fuzzy combinations
                    translation_subsets = product(*translations_with_tests)
                    for translation_combination in translation_subsets:
                        all_tests = []

                        # Add term tests
                        for src_term, tgt_term, tests in term_combination:
                            for test in tests:
                                test_with_target = dict(test)
                                test_with_target["target"] = tgt_term
                                test_with_target["source_term"] = src_term
                                all_tests.append(test_with_target)

                        # Add fuzzy tests
                        for translation in translation_combination:
                            test_with_target = \
                            {
                                "target": translation["target"],
                                "type": "fuzzy_tokens",
                                "condition": dict(translation["tests"])
                            }
                            all_tests.append(test_with_target)
                    
                        grouped[(len(term_combination), len(source_fuzzy_subset))].append(
                            (source, [(src_term, tgt_term) for src_term, tgt_term, _ in term_combination],
                            [f["target"] for f in translation_combination], all_tests, domain)
                        )

        # Fuzzies-only subsets (0 terms, ≥1 fuzzies)
        if fuzzies:
            for source_fuzzy_subset in powerset_nonempty(fuzzies):
                # create subset for each combination of fuzzy targets
                translations_with_tests = [[y for y in x["translations"] if "tests" in y] for x in source_fuzzy_subset]

                # process all target fuzzy combinations
                translation_subsets = product(*translations_with_tests)
                for translation_combination in translation_subsets:
                    all_tests = []
                    # Add fuzzy tests
                    for translation in translation_combination:
                        test_with_target = \
                            {
                                "target": translation["target"],
                                "type": "fuzzy_tokens",
                                "condition": dict(translation["tests"])
                            }    
                        all_tests.append(test_with_target)
                    
                    grouped[(0, len(source_fuzzy_subset))].append(
                        (source, [], [f["target"] for f in translation_combination], all_tests, domain)
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
    for _, terms, fuzzies, _, _ in group:
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

def generate_unensembled(test_cases_for_model, model, tokenizer, input_index=1):
    src_sentences, terms, fuzzies = zip(*test_cases_for_model[input_index])

    if input_index==1:
        inputs = augment_tokenize_batched(
            src_sentences, 
            terms,
            fuzzies,
            tokenizer, 
            "right",
            model.device
        )
    else:
        inputs = tokenizer(src_sentences, return_tensors="pt", padding=True).to(model.device)
    print("Model inputs:\n" + "\n".join(tokenizer.batch_decode(inputs["input_ids"])))

    translated = model.generate(**inputs)
    print("Without ensembling:" + str([tokenizer.decode(t, skip_special_tokens=True) for t in translated]))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

class EvaluationResult:
    def __init__(self):
        self.term_failed = 0
        self.term_success = 0
        self.fuzzy_negative_failed = 0
        self.fuzzy_negative_success = 0
        self.fuzzy_positive_failed = 0
        self.fuzzy_positive_success = 0
        self.fuzzy_bigram_positive_success = 0
        self.fuzzy_bigram_positive_failed = 0

def evaluate_translations_2(translations, test_cases):
    """
    translations: list of translation strings (model outputs)
    test_cases: list of tuples (source, terms, fuzzies, tests), aligned with translations
    """
    assert len(translations) == len(test_cases)
    
    results = []
    
    for translation, (source, terms, fuzzies, tests, domain) in zip(translations, test_cases):
        evaluation_result = EvaluationResult()
        
        fuzzy_test_results = [] 

        for test in tests:
            test_type = test["type"]
            
            if test_type == "term_present":
                source_term = test.get("source_term")
                condition = test.get("condition")
                pattern = re.compile(condition)
                target_term = test.get("target")

                if pattern.search(translation):
                    evaluation_result.term_success += 1
                else:
                    evaluation_result.term_failed += 1

            if test.get("type") == "fuzzy_tokens":
                condition = test.get("condition")
                positive_tokens = condition["positive_tokens"]
                negative_tokens = condition["negative_tokens"]
                fuzzy_result = EvaluationResult()

                # also track bigrams to reward correct order            
                bigrams = []
                for i in range(len(positive_tokens) - 1):
                    bigram = positive_tokens[i].lower() + " " + positive_tokens[i+1].lower()
                    if bigram in test["target"].lower():
                        bigrams.append(bigram)

                for pos_token in positive_tokens:
                    if pos_token.lower() in translation.lower():
                        fuzzy_result.fuzzy_positive_success += 1
                    else:
                        fuzzy_result.fuzzy_positive_failed += 1
                
                for pos_bigram in bigrams:
                    if pos_bigram in translation.lower():
                        fuzzy_result.fuzzy_bigram_positive_success += 1
                    else:
                        fuzzy_result.fuzzy_bigram_positive_failed += 1

                for neg_token in negative_tokens:
                    if neg_token.lower() in translation.lower():
                        fuzzy_result.fuzzy_negative_failed += 1
                    else:
                        fuzzy_result.fuzzy_negative_success += 1
                fuzzy_test_results.append(fuzzy_result)
                
            # add the best fuzzy results
            if fuzzy_test_results:
                best_fuzzy = max(fuzzy_test_results,key=lambda x: 
                                x.fuzzy_positive_success+x.fuzzy_negative_success+x.fuzzy_bigram_positive_success-x.fuzzy_positive_failed-x.fuzzy_negative_failed-x.fuzzy_bigram_positive_failed)
                evaluation_result.fuzzy_positive_success = best_fuzzy.fuzzy_positive_success
                evaluation_result.fuzzy_positive_failed = best_fuzzy.fuzzy_positive_failed
                evaluation_result.fuzzy_negative_success = best_fuzzy.fuzzy_negative_success
                evaluation_result.fuzzy_negative_failed = best_fuzzy.fuzzy_negative_failed
                evaluation_result.fuzzy_bigram_positive_success = best_fuzzy.fuzzy_bigram_positive_success
                evaluation_result.fuzzy_bigram_positive_failed = best_fuzzy.fuzzy_bigram_positive_failed

    return results

def evaluate_translations(translations, test_cases, existing_results=None):
    """
    translations: list of translation strings (model outputs)
    test_cases: list of tuples (source, terms, fuzzies, tests), aligned with translations
    """
    assert len(translations) == len(test_cases)
    
    if not existing_results:
        results = {}
        for domain in ["medical","pharmaceutical","public administration","EU texts","IT administration", "IT customer support", "electronics","legal"]:
            results[domain] = {"term_present": {"passed":0,"failed":0}, "fuzzy_tokens": {"negative": {"passed":0,"failed":0},"positive": {"passed":0,"failed":0},"positive_bigram": {"passed":0,"failed":0}}}

    else:
        results = existing_results

    for translation, (source, terms, fuzzies, tests, domain) in zip(translations, test_cases):

        # TODO: This should return the results per sentence, so that they can be put into a csv
        # for fuzzies, collect results separately for all fuzzies and only use the best scoring
        fuzzy_test_results = [] 

        for test in tests:
            test_type = test["type"]
            
            if test_type == "term_present":
                source_term = test.get("source_term")
                condition = test.get("condition")
                pattern = re.compile(condition)
                target_term = test.get("target")

                if pattern.search(translation):
                    results[domain][test_type]["passed"] += 1
                else:
                    results[domain][test_type]["failed"] += 1

            if test.get("type") == "fuzzy_tokens":
                condition = test.get("condition")
                positive_tokens = condition["positive_tokens"]
                negative_tokens = condition["negative_tokens"]
                fuzzy_result = {"negative": {"passed":0,"failed":0},"positive": {"passed":0,"failed":0},"positive_bigram": {"passed":0,"failed":0}}

                #TODO: figure out how to compensate for terms in the bigram reward (terms override fuzzies).

                # also track bigrams to reward correct order            
                bigrams = []
                for i in range(len(positive_tokens) - 1):
                    bigram = positive_tokens[i].lower() + " " + positive_tokens[i+1].lower()
                    if bigram in test["target"].lower():
                        bigrams.append(bigram)

                for pos_token in positive_tokens:
                    if pos_token.lower() in translation.lower():
                        fuzzy_result["positive"]["passed"] += 1
                    else:
                        fuzzy_result["positive"]["failed"] += 1
                
                for pos_bigram in bigrams:
                    if pos_bigram in translation.lower():
                        fuzzy_result["positive_bigram"]["passed"] += 1
                    else:
                        fuzzy_result["positive_bigram"]["failed"] += 1

                for neg_token in negative_tokens:
                    if neg_token.lower() in translation.lower():
                        fuzzy_result["negative"]["failed"] += 1
                    else:
                        fuzzy_result["negative"]["passed"] += 1
                fuzzy_test_results.append(fuzzy_result)
                
            # add the best fuzzy results
            if fuzzy_test_results:
                best_fuzzy = max(fuzzy_test_results,key=lambda x: 
                                x["positive"]["passed"]+x["negative"]["passed"]+x["positive_bigram"]["passed"]-x["positive"]["failed"]-x["positive_bigram"]["failed"]-x["negative"]["failed"])
                results[domain]["fuzzy_tokens"]["positive"]["passed"] += best_fuzzy["positive"]["passed"]
                results[domain]["fuzzy_tokens"]["positive"]["failed"] += best_fuzzy["positive"]["failed"]
                results[domain]["fuzzy_tokens"]["positive_bigram"]["passed"] += best_fuzzy["positive_bigram"]["passed"]
                results[domain]["fuzzy_tokens"]["positive_bigram"]["failed"] += best_fuzzy["positive_bigram"]["failed"]
                results[domain]["fuzzy_tokens"]["negative"]["passed"] += best_fuzzy["negative"]["passed"]
                results[domain]["fuzzy_tokens"]["negative"]["failed"] += best_fuzzy["negative"]["failed"]


    return results

def translate_with_llm(batch,llm_model,llm_tokenizer,device="cuda"):

    templated_prompts = []
    for source, terms, fuzzies, _, _ in batch:
        term_string = "Terms: "
        for term in terms:
            term_string += f"{term[0]}={term[1]}, "
        # remove last comma, add linebreak
        term_string = term_string[0:-1]+"\n"

        fuzzy_string = ""
        for index,fuzzy in enumerate(fuzzies):
            fuzzy_string += f"Fuzzy match {index+1}: {fuzzy}\n"
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a translator translating from English to Finnish."}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": '''Translate the sentence below to Finnish using the specified terms and fuzzy matches. Use the structure of the fuzzy matches in the translation if appropriate, but do not copy parts of the fuzzy match to the translation if they are not semantically present in the source sentence. Using the specified term is more important than using the fuzzy match, so if a term and the fuzzy match conflict, always prefer the term. Output the answer in the following format, and do not output anything else: TRANSLATION: TRANSLATION GOES HERE.
                        {terms}{fuzzies}.
                        Source sentence to translate: {source}
                        '''.format(terms=term_string,fuzzies=fuzzy_string,source=source)
                    }
                ]
            }
        ]

        templated_prompt = llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        templated_prompts.append(templated_prompt)

    inputs = llm_tokenizer(templated_prompts,padding=True,return_tensors="pt").to(device).to(torch.bfloat16)

    with torch.inference_mode():
        outputs = llm_model.generate(**inputs, max_new_tokens=128)

    outputs = llm_tokenizer.batch_decode(outputs)
    only_translations = [x.split("TRANSLATION:")[2].split("<end_of_turn>")[0].strip() for x in outputs]
    
    return only_translations

def main(args):
    print("Base model:", args.base_model)
    print("Term model:", args.term_model)
    print("Fuzzy model:", args.fuzzy_model)
    print("LLM:", args.llm)
    print("Source language:", args.source_lang)
    print("Target language:", args.target_lang)
    
    # Create a model group containing four types of models:
    # 1. base model, used as the guide model
    # 2. term model, translates source sent annotated with terms
    # 3. fuzzy model, for fuzzies
    # NOT YET IMPLEMENTED: 4. subsegment model, for subsegments
    
    if args.base_model and args.term_model and args.fuzzy_model:
        model_group = ModelGroup(args.base_model, args.term_model, args.fuzzy_model)
        ensemble = ShallowFusion(model_group)
        num_beams = 6
    # If only base model specified, generate only baseline translations.
    elif args.base_model:
        model_group = ModelGroup(args.base_model, args.base_model, args.base_model)
        num_beams = 6

    if args.llm:
        # initialize llm model
        model_id = args.llm
        #quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        llm_model = Gemma3ForCausalLM.from_pretrained(
            model_id, device_map="cuda",torch_dtype=torch.bfloat16 #, quantization_config=quantization_config
        ).eval()
        llm_tokenizer = AutoTokenizer.from_pretrained(model_id)

    
    test_case_groups = create_test_cases_with_tests(args.test_suite_path)

    

    # HACK: restrict combos for testing
    # TODO: make a more meaningful selection of a reasonable amount of cases (5000-10000?)
    test_case_groups = {k: v for k,v in test_case_groups.items() if k == (1,1)}
    test_case_groups[(1,1)] = test_case_groups[(1,1)][0:100]

    test_cases_with_translations = []

    print(f"Total test cases: {sum([len(x) for x in test_case_groups.values()])}")

    combined_results = {}
    combined_baseline_results = {}

    for (term_count,fuzzy_count),test_case_group in test_case_groups.items():
        for batch in [test_case_group[i:i + args.batch_size] for i in range(0, len(test_case_group), args.batch_size)]:

            test_cases = break_down_group(batch)
            if args.base_model and args.term_model and args.fuzzy_model:
                model_types = ["base"] + ["term"] * term_count + ["fuzzy"] * fuzzy_count + ["base"]
                translations = ensemble.translate(
                    test_cases,
                    model_types,
                    num_beams=num_beams,
                    max_length=100,
                    only_main_model=False
                )

                if (term_count,fuzzy_count) in combined_results:
                    evaluate_translations(translations,batch,combined_results[(term_count,fuzzy_count)])
                else:
                    eval_results = evaluate_translations(translations,batch)
                    combined_results[(term_count,fuzzy_count)] = eval_results
                
            if args.base_model and not args.term_model and not args.fuzzy_model:    
                # generate baseline translations to see the effect that external info makes. This is kind of wasteful, as it ends up translating the same sentence many times, but no time to unravel that now
                baseline_translations = generate_unensembled(
                    test_cases,
                    model_group.models["base"],
                    model_group.baseline_tokenizer,
                    0)

                if (term_count,fuzzy_count) in combined_baseline_results:
                    evaluate_translations(baseline_translations,batch,combined_baseline_results[(term_count,fuzzy_count)])
                else:
                    eval_results = evaluate_translations(baseline_translations,batch)
                    combined_baseline_results[(term_count,fuzzy_count)] = eval_results

            if args.llm:
                translations = translate_with_llm(batch,llm_model,llm_tokenizer)
                if (term_count,fuzzy_count) in combined_results:
                    evaluate_translations(translations,batch,combined_results[(term_count,fuzzy_count)])
                else:
                    eval_results = evaluate_translations(translations,batch)
                    combined_results[(term_count,fuzzy_count)] = eval_results

                test_cases_with_translations += [(*x[0],x[1]) for x in zip(batch,translations)]

            # If only one model is used, also generate unensembled translations for comparison
            """if len(model_types) == 3:
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
                    model_group.tokenizer)"""
    
    # TODO: incorporarate LLM, implement unified model, implement max 5-term and max 5-fuzzy models,
    # figure out a composite score.

    print(combined_results)
    with open("output.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')

        writer.writerow(["TermFuzzyCounts","Source","Terms","Fuzzies","Tests","Domain","Translation"])
        for sentence_data in test_cases_with_translations:
            writer.writerow(sentence_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate test suite test cases using an ensemble of models accepting different kinds of retrieved information (terms, ngrams, full matches).")
    
    parser.add_argument(
        "--base_model",
        required=False,
        help="Base model to use as contrast model."
    )

    parser.add_argument(
        "--term_model",
        required=False,
        help="Term model."
    )

    parser.add_argument(
        "--fuzzy_model",
        required=False,
        help="Fuzzy model."
    )

    parser.add_argument(
        "--llm",
        required=False,
        help="HF name of LLM to generate translations with."
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
    
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size when translating."
    )

    args = parser.parse_args()
    main(args)