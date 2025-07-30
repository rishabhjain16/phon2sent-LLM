import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
import os
import re
import json
from g2p_en import G2p
import numpy as np
from jiwer import wer

# Initialize g2p converter
g2p = G2p()

def sentence_to_phonemes(sentence):
    # g2p returns a list with phonemes and spaces for word boundaries
    phonemes = g2p(sentence)
    # Remove spaces (word boundaries) and join phonemes with space
    phonemes = [ph for ph in phonemes if ph != ' ']
    # Remove stress markers from phonemes (digits after phonemes)
    phonemes = [re.sub(r'(\D+)\d*', r'\1', ph) for ph in phonemes]
    return ' '.join(phonemes)

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

def phonemes_to_sentence(model, tokenizer, phoneme_seq):
    if isinstance(phoneme_seq, list):
        phoneme_str = " ".join(phoneme_seq)
    else:
        phoneme_str = phoneme_seq
    prompt = f"Translate the following phonemes into text: {phoneme_str}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=5,
            temperature=0.2,
            do_sample=True,
            top_p=0.92,
            top_k=40,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated sentence (removing the prompt)
    result = generated_text[len(prompt):].strip()
    
    # Clean any formatting artifacts focusing on apostrophes and basic punctuation
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

def is_valid_reconstruction(ref, reconstructed, max_length_ratio=2.0, min_length_ratio=0.3, max_repetitions=5):
    """
    Check if the reconstructed text is valid by comparing it to the reference.
    Invalid reconstructions include:
    1. Too long compared to reference (potential hallucination or repetition)
    2. Too short compared to reference (potential truncation)
    3. Too many repetitions of the same word (potential degeneration)
    
    Args:
        ref: Reference text
        reconstructed: Reconstructed text
        max_length_ratio: Maximum allowed ratio of reconstructed length to reference length
        min_length_ratio: Minimum allowed ratio of reconstructed length to reference length
        max_repetitions: Maximum number of consecutive repetitions of the same word allowed
    
    Returns:
        Boolean indicating if the reconstruction is valid
    """
    # Normalize texts
    normalized_ref = normalize_text(ref)
    normalized_reconstructed = normalize_text(reconstructed)
    
    # Check length ratio
    ref_words = normalized_ref.split()
    reconstructed_words = normalized_reconstructed.split()
    
    if len(ref_words) == 0:  # Handle empty reference
        return False
    
    length_ratio = len(reconstructed_words) / len(ref_words)
    
    # Check for too long or too short reconstruction
    if length_ratio > max_length_ratio or length_ratio < min_length_ratio:
        return False
    
    # Check for repetitions
    if len(reconstructed_words) >= max_repetitions:
        # Look for repeating patterns
        for i in range(1, len(reconstructed_words) // 2 + 1):
            # Check if there's a repeating pattern of length i
            for start in range(len(reconstructed_words) - i * max_repetitions + 1):
                pattern = reconstructed_words[start:start+i]
                is_repeating = True
                
                # Check the next max_repetitions-1 occurrences
                for rep in range(1, max_repetitions):
                    next_start = start + i * rep
                    next_end = next_start + i
                    
                    if next_end > len(reconstructed_words):
                        is_repeating = False
                        break
                    
                    if reconstructed_words[next_start:next_end] != pattern:
                        is_repeating = False
                        break
                
                if is_repeating:
                    return False
    
    return True

def process_json_data(model, tokenizer, json_file_path):
    """
    Process the JSON data by:
    1. Reading the data from the JSON file
    2. For each sentence in 'hypo', convert it to phonemes
    3. Reconstruct the sentence from the phonemes
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
    phonemes_list = []
    reconstructed_sentences = []
    wer_scores = []
    is_valid_list = []
    
    # Valid reconstructions for calculating average WER
    valid_wer_scores = []
    
    # Process each pair of reference and hypothesis
    for i, (ref, hypo) in enumerate(zip(refs, hypos)):
        print(f"Processing sentence {i+1}/{len(refs)}")
        
        # Convert hypothesis to phonemes
        phonemes = sentence_to_phonemes(hypo)
        phonemes_list.append(phonemes)
        
        # Reconstruct the sentence from phonemes
        reconstructed = phonemes_to_sentence(model, tokenizer, phonemes)
        reconstructed_sentences.append(reconstructed)
        
        # Check if reconstruction is valid
        is_valid = is_valid_reconstruction(ref, reconstructed)
        is_valid_list.append(is_valid)
        
        # Normalize both reference and reconstructed text for WER calculation
        normalized_ref = normalize_text(ref)
        normalized_reconstructed = normalize_text(reconstructed)
        
        # Calculate WER on normalized text if valid
        if is_valid:
            wer_score = wer(normalized_ref, normalized_reconstructed)
            valid_wer_scores.append(wer_score)
        else:
            wer_score = float('nan')  # Use NaN to indicate invalid reconstruction
        
        wer_scores.append(wer_score)
        
        # Print results for this pair
        print(f"Reference: {ref}")
        print(f"Hypothesis: {hypo}")
        print(f"Phonemes: {phonemes}")
        print(f"Reconstructed: {reconstructed}")
        print(f"Normalized Ref: {normalized_ref}")
        print(f"Normalized Reconstructed: {normalized_reconstructed}")
        print(f"Valid: {is_valid}")
        if is_valid:
            print(f"WER: {wer_score:.4f}")
        else:
            print("WER: N/A (Invalid reconstruction)")
        print("-" * 50)
    
    # Calculate and print the average WER for valid reconstructions
    avg_wer = np.mean(valid_wer_scores) if valid_wer_scores else float('nan')
    print(f"Average WER (valid reconstructions only): {avg_wer:.4f}")
    print(f"Valid reconstructions: {len(valid_wer_scores)}/{len(refs)} ({len(valid_wer_scores)/len(refs)*100:.1f}%)")
    
    # Save results to a file
    results = {
        'utt_id': data['utt_id'],
        'ref': refs,
        'hypo': hypos,
        'phonemes': phonemes_list,
        'reconstructed': reconstructed_sentences,
        'normalized_ref': [normalize_text(ref) for ref in refs],
        'normalized_reconstructed': [normalize_text(rec) for rec in reconstructed_sentences],
        'is_valid': is_valid_list,
        'wer': wer_scores,
        'avg_wer': float(avg_wer),
        'valid_count': len(valid_wer_scores),
        'total_count': len(refs),
        'valid_percentage': float(len(valid_wer_scores)/len(refs)*100)
    }
    
    output_file = os.path.splitext(json_file_path)[0] + "_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Convert phoneme sequences to sentences")
    parser.add_argument("--model_dir", type=str, default="./llama2_phoneme2sentence_g2p/checkpoint-1000",
                        help="Directory containing the fine-tuned model or LoRA adapter")
    parser.add_argument("--base_model", type=str, 
                        default="./checkpoints/Llama-2-7b-hf",
                        help="Path to the base model (needed for LoRA models)")
    parser.add_argument("--phonemes", type=str,
                        default=None,
                        help="Phoneme sequence to convert (space-separated)")
    parser.add_argument("--sentence", type=str,
                        default=None,
                        help="English sentence to convert to phonemes and back (for validation)")
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
        
        # Convert sentence to phonemes
        phonemes = sentence_to_phonemes(original_sentence)
        print(f"Generated phonemes: {phonemes}")
        
        # Convert phonemes back to text
        reconstructed = phonemes_to_sentence(model, tokenizer, phonemes)
        print(f"Reconstructed sentence: {reconstructed}")
        
        # Check if reconstruction is valid
        is_valid = is_valid_reconstruction(original_sentence, reconstructed)
        print(f"Valid reconstruction: {is_valid}")
        
        # Calculate similarity (just a simple word overlap percentage)
        original_words = set(original_sentence.lower().split())
        reconstructed_words = set(reconstructed.lower().split())
        if original_words:
            overlap = len(original_words.intersection(reconstructed_words)) / len(original_words)
            print(f"Word overlap: {overlap:.2%}")
    
    # Process phoneme sequence if provided
    elif args.phonemes:
        phoneme_seq = args.phonemes.split()
        result = phonemes_to_sentence(model, tokenizer, phoneme_seq)
        print("\nResults:")
        print(f"Phonemes: {args.phonemes}")
        print(f"Predicted sentence: {result}")
    
    else:
        print("Please provide either --phonemes, --sentence, or --json_file")


if __name__ == "__main__":
    main()
