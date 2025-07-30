# Phoneme-to-Text using LLMS Models

This repository contains scripts for training and using text-to-phoneme and phoneme-to-text models using both Arpabet (CMUdict/G2P) and IPA phonetic representations.

## Setup

### Downloading the Llama-2-7b-hf Checkpoint

You need to download the Llama-2-7b-hf model weights from Hugging Face before training or inference. You must request access to the model on Hugging Face and accept their terms.

1. Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Click "Access repository" and follow the instructions to get permission.
3. Once approved, install the Hugging Face CLI if you haven't already:
   ```bash
   pip install huggingface_hub
   ```
4. Log in to your Hugging Face account:
   ```bash
   huggingface-cli login
   ```
5. Download the model to your desired directory:
   ```bash
   huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./checkpoints/Llama-2-7b-hf --local-dir-use-symlinks False
   ```

Make sure the `model_name` or `--base_model` argument in your scripts points to this directory.

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For IPA phoneme conversion, you need to install espeak-ng (system dependency):
```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# macOS
brew install espeak-ng

# Windows
# Download and install from https://github.com/espeak-ng/espeak-ng/releases
```

3. Make sure you have the Llama-2-7b-hf model downloaded:
```
/home/experiments/checkpoints/Llama-2-7b-hf
```

## Arpabet-based Models
## Major Differences: train_g2p.py vs train2_g2p.py

| Aspect                | train_g2p.py                        | train2_g2p.py                       |
|-----------------------|--------------------------------------|--------------------------------------|
| Allowed punctuation   | Apostrophes only                     | Apostrophes, periods, commas         |
| Phoneme word boundary | Spaces removed (no word boundaries)  | Spaces preserved (word boundaries)   |

**Summary:**
- `train_g2p.py` is stricter (less punctuation, no word boundaries in phoneme input, larger dataset).
- `train2_g2p.py` is more permissive (more punctuation, preserves word boundaries, smaller dataset).

**Dataset Used:**
wikitext-103 (large) 
wikitext-2 (smaller)    

### Training
```bash
python train2.py --output_dir ./trained/wiki2_arpabet --epochs 6 --batch_size 4
```

### Inference
```bash
# Process a phoneme sequence
python infer.py --model_dir ./trained/wiki2_arpabet/checkpoint-1000 --phonemes "DH AH K AE M ER AH IH Z"

# Test a round-trip conversion
python infer.py --model_dir ./trained/wiki2_arpabet/checkpoint-1000 --sentence "The camera is working well."
```

## IPA-based Models

## Choosing Punctuation and Granularity: What Difference Does It Make?

## How to Train with Different Punctuation and Granularity Levels

### Arpabet/CMUdict (Apostrophes only vs. More Punctuation)

- **Apostrophes only (default: train_g2p.py):**
  ```bash
  python train_g2p.py --output_dir ./trained/llama_phon_wiki103 --epochs 10 --batch_size 32
  ```
  - Use this for basic English text, contractions, and possessives only.

- **Apostrophes, periods, commas (default: train2_g2p.py):**
  ```bash
  python train2_g2p.py --output_dir ./trained/wiki2_pron_space --epochs 6 --batch_size 4
  ```
  - Use this if you want the model to learn and generate punctuation in output sentences.

### IPA Models: Character-level vs. Word-level Granularity

- **Character-level IPA:**
  ```bash
  python train_ipa.py --output_dir ./trained/wiki2_ipa_char --epochs 6 --batch_size 4 --granularity char
  ```
  - Each IPA character is a token; no word boundaries are preserved.
  - Use for fine-grained phoneme-to-text mapping or when word boundaries are not needed.

- **Word-level IPA:**
  ```bash
  python train_ipa.py --output_dir ./trained/wiki2_ipa_word --epochs 6 --batch_size 4 --granularity word
  ```
  - Spaces are preserved between words in the IPA sequence.
  - Use if you want the model to reconstruct word boundaries and handle full sentences more naturally.

### Choosing the Right Option
- If your downstream task needs punctuation or word segmentation, train with those features included.
- For speech-to-text or ASR, word-level and punctuation-aware models are usually better.
- For phoneme-level analysis or languages/scripts without clear word boundaries, character-level may suffice.

### Apostrophes and Punctuation
- **With apostrophes only** (e.g., `train_g2p.py`):
  - The model learns to handle contractions (like "don't", "it's") and possessives (like "John's").
  - Output is more normalized, but will not preserve or generate periods, commas, etc.
  - Useful for applications where only basic English text is needed, or for speech tasks where punctuation is not pronounced.
- **With more punctuation (apostrophes, periods, commas, etc.)** (e.g., `train2_g2p.py`):
  - The model can learn to generate and interpret punctuation in sentences.
  - Output will more closely match written English, including pauses and sentence boundaries.
  - Useful for text generation, subtitling, or any use case where punctuation is important.

### Phoneme/IPA Granularity
- **Character-level (char) granularity**:
  - Each phoneme or IPA character is treated as a separate token.
  - No explicit word boundaries are preserved in the input or output.
  - The model focuses on local, fine-grained sound-to-text mapping.
  - Useful for tasks where word boundaries are ambiguous or not needed, or for languages/scripts with no clear word separation.
- **Word-level (word) granularity**:
  - Spaces or special tokens are used to mark word boundaries in the phoneme/IPA sequence.
  - The model can learn to reconstruct word boundaries and handle longer-range dependencies.
  - Output is more likely to match natural word segmentation in English.
  - Useful for applications where word-level alignment or readability is important (e.g., ASR, TTS, or text recovery from phonemes).

### Practical Impact
- **Training with more punctuation or word-level granularity generally makes the model more robust for real-world text, but may require more data and careful cleaning.**
- **Character-level models may be more flexible for noisy or non-standard input, but can struggle with reconstructing proper word boundaries.**
- **If your downstream task needs punctuation or word segmentation, train with those features included.**

### IPA Model Variants

| Model/Script         | Granularity | Description                                 |
|----------------------|-------------|---------------------------------------------|
| train_ipa.py --granularity char | char        | Character-level IPA (spaces between each IPA character) |
| train_ipa.py --granularity word | word        | Word-level IPA (spaces between words, preserves word boundaries) |

**Summary:**
- Character-level IPA models treat each IPA character as a token (no word boundaries).
- Word-level IPA models preserve word boundaries in the IPA sequence.

### Training
```bash
# Character-level IPA (spaces between each phoneme character)
python train_ipa.py --output_dir ./trained/wiki2_ipa_char --epochs 6 --batch_size 4 --granularity char

# Word-level IPA (preserve word boundaries)
python train_ipa.py --output_dir ./trained/wiki2_ipa_word --epochs 6 --batch_size 4 --granularity word
```

### Inference
```bash
# Process an IPA sequence
python infer_ipa.py --model_dir ./trained/wiki2_ipa_char/checkpoint-1000 --ipa "ð ə k æ m ə r ə ɪ z"

# Test a round-trip conversion with character-level IPA
python infer_ipa.py --model_dir ./trained/wiki2_ipa_char/checkpoint-1000 --sentence "The camera is working well." --granularity char

# Test with word-level IPA (if trained that way)
python infer_ipa.py --model_dir ./trained/wiki2_ipa_word/checkpoint-1000 --sentence "The camera is working well." --granularity word
```

## Quick Test

You can quickly test the phonemizer without running the full model:

```bash
python -c "from phonemizer import phonemize; print(phonemize('The camera is working well.', language='en-us'))"
# Output: ðə kæmɹə ɪz wɜːkɪŋ wɛl
```

## Comparison between Arpabet and IPA

Arpabet (from G2P/CMUdict):
- Representation: `DH AH K AE M ER AH IH Z`
- Pros: Simpler ASCII characters, easier to process
- Cons: Less internationally recognized, fewer phonetic distinctions

IPA:
- Representation: `ð ə k æ m ɹ ə ɪ z w ɜː k ɪ ŋ w ɛ l`
- Pros: Universal standard, more precise phonetic representation
- Cons: Uses Unicode characters which can be more complex to handle

### Phoneme Conversion Options

We support two different libraries for phoneme conversion:

1. **G2P-en**: Used for Arpabet phonemes (ASCII-based)
   - Simple to use, no external dependencies
   - Limited to English language only

2. **Phonemizer**: Used for IPA phonemes (Unicode symbols) 
   - Requires espeak-ng system dependency
   - Supports many languages and offers more phonetic detail
   - More actively maintained than alternatives like Epitran

## Tips for Best Results
## Inference Scripts: Usage and Modes

There are two main inference scripts:
- `infer_g2p.py` (for Arpabet/CMUdict phonemes)
- `infer_ipa.py` (for IPA phonemes)

Both scripts support similar usage patterns and command-line arguments:

### 1. Convert a Phoneme Sequence to Text
- For Arpabet:
  ```bash
  python infer_g2p.py --model_dir <checkpoint_dir> --phonemes "DH AH K AE M ER AH IH Z"
  ```
- For IPA:
  ```bash
  python infer_ipa.py --model_dir <checkpoint_dir> --ipa "ð ə k æ m ə r ə ɪ z"
  ```
  (If your IPA string has no spaces, the script will add them automatically.)

### 2. Validate Full Pipeline (Text → Phonemes/IPA → Text)
- For Arpabet:
  ```bash
  python infer_g2p.py --model_dir <checkpoint_dir> --sentence "The camera is working well."
  ```
- For IPA:
  ```bash
  python infer_ipa.py --model_dir <checkpoint_dir> --sentence "The camera is working well."
  ```
  (The script will convert the sentence to phonemes/IPA, then back to text, and print the overlap.)

### 3. Batch Processing from JSON
- For Arpabet:
  ```bash
  python infer_g2p.py --model_dir <checkpoint_dir> --json_file test_json.json
  ```
- For IPA:
  ```bash
  python infer_ipa.py --model_dir <checkpoint_dir> --json_file test_json.json
  ```
  (The JSON file should contain fields like `ref`, `hypo`, etc. The script will process all entries and save results to a new JSON file.)

### 4. Required/Optional Arguments
- `--model_dir`: Path to the trained model or checkpoint.
- `--base_model`: (Optional) Path to the base model, needed for LoRA adapters.
- `--phonemes`/`--ipa`: The phoneme or IPA sequence to convert.
- `--sentence`: An English sentence to run through the full pipeline.
- `--json_file`: Path to a JSON file for batch processing.

---

## Utility: print_json_to_csv.py

## Batch Inference: JSON Structure and Recommendations

For batch inference, the scripts expect a JSON file with the following structure:

```json
{
  "utt_id": ["id1", "id2", ...],
  "ref": ["Reference sentence 1.", "Reference sentence 2.", ...],
  "hypo": ["Hypothesis sentence 1.", "Hypothesis sentence 2.", ...]
}
```

- The `hypo` field should contain sentences (not phonemes or IPA). The script will convert these to phonemes/IPA internally.
- The `ref` field is the ground truth sentence for each example.
- The `utt_id` field is optional but recommended for tracking.

**This format is compatible with outputs from models like AvHubert, which predict sentences.**
If you have a model that outputs sentences (e.g., ASR, speech-to-text, or other sequence models), you can save the results in this JSON format and use it directly for batch evaluation with the inference scripts.

**Note:** If you want to provide phonemes or IPA directly in the JSON, you would need to modify the script. By default, the scripts expect sentences in `hypo` and will perform the phoneme/IPA conversion step themselves.

This script is a utility to convert a JSON results file (from the infer scripts) into a simple CSV for easier analysis.

### What it does:
- Reads a JSON file containing fields like `ref`, `hypo`, `reconstructed`, and optionally `utt_id`.
- Creates a CSV file with columns: `ID`, `Reference`, `Hypothesis`, `Reconstructed`.
- The output CSV can be used for manual inspection, error analysis, or further processing.

### Usage:
```bash
python print_json_to_csv.py results.json
# or specify output file
python print_json_to_csv.py results.json --output my_results.csv
```
If no output file is given, it will create one named like `results_simple.csv`.

1. For complex sentences with punctuation, use a model trained with punctuation included
2. Train longer for better quality (more epochs)
3. Experiment with both character-level and word-level IPA granularity to see which works better for your use case
4. Adjust generation parameters in the inference scripts as needed:
   - Lower temperature (e.g., 0.2) for more conservative outputs
   - Higher repetition_penalty (e.g., 1.2) to avoid repeated words
   - Adjust top_p and top_k for different sampling strategies 