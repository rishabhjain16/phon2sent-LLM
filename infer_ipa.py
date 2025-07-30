import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
import os
import re
import json
import numpy as np
from jiwer import wer
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

# Initialize phonemizer backend for English
backend = EspeakBackend('en-us', preserve_punctuation=True)

def sentence_to_ipa(sentence):
    try:
        # Character-level IPA conversion (always use this approach)
        # Get the IPA with no word boundaries
        phonemes = backend.phonemize([sentence], separator=Separator(word='', phone=''))[0]
        # Add spaces between each IPA character
        phonemes = ' '.join(list(phonemes))
            
        return phonemes
    except Exception as e:
        print(f"Warning: Could not convert sentence to IPA: {e}")
        return None


def load_model(model_dir, base_model_path=None):
    """
    Load a PEFT/LoRA fine-tuned model
    
    Args:
        model_dir: Path to the fine-tuned LoRA adapter
        base_model_path: Path to the base model (needed for tokenizer)
                        If None, will try to extract from the adapter config
    """
    # Determine if this is a LoRA adapter directory
    is_lora = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
    if is_lora:
        # For LoRA models, we need both the adapter and base model
        print(f"Loading LoRA adapter from {model_dir}")
        
        # If base_model_path not provided, try to get it from the adapter config
        if base_model_path is None:
            peft_config = PeftConfig.from_pretrained(model_dir)
            base_model_path = peft_config.base_model_name_or_path
            print(f"Using base model: {base_model_path}")
            
        # If base model path still not available or is a remote path, use the original model path
        if base_model_path is None or base_model_path.startswith("http"):
            base_model_path = "./checkpoints/Llama-2-7b-hf"
            print(f"Using default base model: {base_model_path}")
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load the LoRA adapter on top of the base model
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        # For regular models (not LoRA)
        print(f"Loading regular model from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map="auto"
        )
    
    return model, tokenizer

def ipa_to_sentence(model, tokenizer, ipa_seq):
    if isinstance(ipa_seq, list):
        ipa_str = " ".join(ipa_seq)
    else:
        ipa_str = ipa_seq
    prompt = f"Translate the following IPA phonemes into text: {ipa_str}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # Reduce max tokens to avoid excessive generation
            num_beams=5,
            temperature=0.7,     # Add controlled randomness
            do_sample=True, 
            top_p=0.92,          # Focus sampling on more likely tokens
            top_k=50,            # Limit vocabulary choices
            repetition_penalty=1.3,  # Strengthen repetition penalty
            no_repeat_ngram_size=3,  # Prevent 3-grams from repeating
            eos_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated sentence (removing the prompt)
    result = generated_text[len(prompt):].strip()
    
    # Clean any formatting artifacts
    # Fix contractions by removing spaces around apostrophes
    result = re.sub(r'\s+\'', '\'', result)  # Fix "don 't" → "don't"
    result = re.sub(r'\'\s+', '\'', result)  # Fix "it 's" → "it's"
    
    # Fix spacing around basic punctuation
    result = re.sub(r'\s+([.,?!])', r'\1', result)  # Fix spaces before punctuation
    
    # Fix multiple spaces
    result = re.sub(r'\s{2,}', ' ', result)
    
    return result

def normalize_text(text):
    """
    Normalize text for WER calculation by:
    1. Converting to lowercase
    2. Removing extra whitespace
    3. Standardizing punctuation
    
    Args:
        text: The text to normalize
    
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Optional: Remove or standardize punctuation if needed
    # For now, we'll just keep it simple with lowercase and whitespace normalization
    
    return text

def process_json_data(model, tokenizer, json_file_path):
    """
    Process the JSON data by:
    1. Reading the data from the JSON file
    2. For each sentence in 'hypo', convert it to IPA phonemes
    3. Reconstruct the sentence from the IPA phonemes
    4. Calculate WER between the 'ref' and reconstructed sentence
    
    Args:
        model: The pre-trained model
        tokenizer: The tokenizer for the model
        json_file_path: Path to the JSON file containing the data
    """
    # Read the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the references and hypotheses
    refs = data['ref']
    hypos = data['hypo']
    
    # Lists to store results
    ipa_phonemes_list = []
    reconstructed_sentences = []
    wer_scores = []
    
    # Process each pair of reference and hypothesis
    for i, (ref, hypo) in enumerate(zip(refs, hypos)):
        print(f"Processing sentence {i+1}/{len(refs)}")
        
        # Convert hypothesis to IPA phonemes
        ipa_phonemes = sentence_to_ipa(hypo)
        ipa_phonemes_list.append(ipa_phonemes)
        
        # Reconstruct the sentence from IPA phonemes
        reconstructed = ipa_to_sentence(model, tokenizer, ipa_phonemes)
        reconstructed_sentences.append(reconstructed)
        
        # Normalize both reference and reconstructed text for WER calculation
        normalized_ref = normalize_text(ref)
        normalized_reconstructed = normalize_text(reconstructed)
        
        # Calculate WER on normalized text
        wer_score = wer(normalized_ref, normalized_reconstructed)
        wer_scores.append(wer_score)
        
        # Print results for this pair
        print(f"Reference: {ref}")
        print(f"Hypothesis: {hypo}")
        print(f"IPA Phonemes: {ipa_phonemes}")
        print(f"Reconstructed: {reconstructed}")
        print(f"Normalized Ref: {normalized_ref}")
        print(f"Normalized Reconstructed: {normalized_reconstructed}")
        print(f"WER: {wer_score:.4f}")
        print("-" * 50)
    
    # Calculate and print the average WER
    avg_wer = np.mean(wer_scores)
    print(f"Average WER: {avg_wer:.4f}")
    
    # Save results to a file
    results = {
        'utt_id': data['utt_id'],
        'ref': refs,
        'hypo': hypos,
        'ipa_phonemes': ipa_phonemes_list,
        'reconstructed': reconstructed_sentences,
        'normalized_ref': [normalize_text(ref) for ref in refs],
        'normalized_reconstructed': [normalize_text(rec) for rec in reconstructed_sentences],
        'wer': wer_scores,
        'avg_wer': float(avg_wer)
    }
    
    output_file = os.path.splitext(json_file_path)[0] + "_ipa_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Convert IPA sequences to sentences")
    parser.add_argument("--model_dir", type=str, default="./trained/wiki2_ipa/checkpoint-1000",
                        help="Directory containing the fine-tuned model or LoRA adapter")
    parser.add_argument("--base_model", type=str, 
                        default="./checkpoints/Llama-2-7b-hf",
                        help="Path to the base model (needed for LoRA models)")
    parser.add_argument("--ipa", type=str,
                        default=None,
                        help="IPA phoneme sequence to convert (space-separated)")
    parser.add_argument("--sentence", type=str,
                        default=None,
                        help="English sentence to convert to IPA and back (for validation)")
    parser.add_argument("--json_file", type=str,
                        default=None,
                        help="Path to JSON file containing 'ref' and 'hypo' fields")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_dir, args.base_model)
    
    # Process a JSON file if provided
    if args.json_file is not None:
        print(f"Processing data from JSON file: {args.json_file}")
        process_json_data(model, tokenizer, args.json_file)
    
    # Process an English sentence if provided
    elif args.sentence is not None:
        original_sentence = args.sentence
        print("\nValidating full pipeline:")
        print(f"Original sentence: {original_sentence}")
        
        # Convert sentence to IPA phonemes
        phonemes = sentence_to_ipa(original_sentence)
        print(f"Generated IPA phonemes: {phonemes}")
        
        # Convert IPA phonemes back to text
        reconstructed = ipa_to_sentence(model, tokenizer, phonemes)
        print(f"Reconstructed sentence: {reconstructed}")
        
        # Calculate similarity (just a simple word overlap percentage)
        original_words = set(original_sentence.lower().split())
        reconstructed_words = set(reconstructed.lower().split())
        if original_words:
            overlap = len(original_words.intersection(reconstructed_words)) / len(original_words)
            print(f"Word overlap: {overlap:.2%}")
    
    # Process IPA sequence if provided
    elif args.ipa:
        if " " not in args.ipa:
            # If input doesn't have spaces already, add them between characters
            ipa_seq = " ".join(list(args.ipa))
        else:
            ipa_seq = args.ipa
        
        result = ipa_to_sentence(model, tokenizer, ipa_seq)
        print("\nResults:")
        print(f"IPA Phonemes: {ipa_seq}")
        print(f"Predicted sentence: {result}")
    
    else:
        print("Please provide either --ipa, --sentence, or --json_file")

if __name__ == "__main__":
    main()