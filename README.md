## Arpabet vs IPA: Quick Example

| Type    | Input (Phonemes/IPA)                  | Output (Text)                |
|---------|---------------------------------------|------------------------------|
| Arpabet | DH AH K AE M ER AH IH Z               | The camera is                |
| IPA     | ð ə k æ m ə r ə ɪ z                   | The camera is                |

**Arpabet** uses ASCII phoneme codes (from CMUdict/G2P), while **IPA** uses international phonetic symbols. Both can be used for phoneme-to-text or text-to-phoneme tasks, but have different vocabularies and tokenization.

### Apostrophes and Punctuation

- **With apostrophes only (e.g., train_g2p.py):**
  - The model learns to handle contractions (like "don't", "it's") and possessives (like "John's").
  - Output is more normalized, but will not preserve or generate periods, commas, etc.
  - Useful for applications where only basic English text is needed, or for speech tasks where punctuation is not pronounced.

- **With more punctuation (apostrophes, periods, commas, etc.) (e.g., train2_g2p.py):**
  - The model can learn to generate and interpret punctuation in sentences.
  - Output will more closely match written English, including pauses and sentence boundaries.
  - Useful for text generation, subtitling, or any use case where punctuation is important.

### Phoneme/IPA Granularity

- **Character-level (char) granularity:**
  - Each phoneme or IPA character is treated as a separate token.
  - No explicit word boundaries are preserved in the input or output.
  - The model focuses on local, fine-grained sound-to-text mapping.
  - Useful for tasks where word boundaries are ambiguous or not needed, or for languages/scripts with no clear word separation.

- **Word-level (word) granularity:**
  - Spaces or special tokens are used to mark word boundaries in the phoneme/IPA sequence.
  - The model can learn to reconstruct word boundaries and handle longer-range dependencies.
  - Output is more likely to match natural word segmentation in English.
  - Useful for applications where word-level alignment or readability is important (e.g., ASR, TTS, or text recovery from phonemes).


# Phoneme-to-Text using LLMs

This repository contains scripts for training and using text-to-phoneme and phoneme-to-text models using both Arpabet (CMUdict/G2P) and IPA phonetic representations.

> **Note:** For best results in reconstructing English sentences from phoneme or IPA sequences, it is recommended to use the G2P (Arpabet/CMUdict) mode. This mode is generally more robust for English and less prone to tokenization or vocabulary issues than direct IPA reconstruction.


wikitext-2 (smaller)    

## 1. Setup

### 1.1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 1.2. Install espeak-ng (for IPA)
```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng
# macOS
brew install espeak-ng
# Windows
# Download and install from https://github.com/espeak-ng/espeak-ng/releases
```

### 1.3. Download Llama-2-7b-hf Model
1. Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Install Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   ```
3. Login:
   ```bash
   huggingface-cli login
   ```
4. Download:
   ```bash
   huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./checkpoints/llm --local-dir-use-symlinks False
   ```
5. Make sure your scripts use `./checkpoints/llm` as the model path.

---

## 2. Model Types & Training

### 2.1. Arpabet-based Models

| Aspect                | train_g2p.py                        | train2_g2p.py                       |
|-----------------------|--------------------------------------|--------------------------------------|
| Allowed punctuation   | Apostrophes only                     | Apostrophes, periods, commas         |
| Phoneme word boundary | Spaces removed (no word boundaries)  | Spaces preserved (word boundaries)   |

- `train_g2p.py`: Stricter, less punctuation, no word boundaries, larger dataset.
- `train2_g2p.py`: More punctuation, preserves word boundaries, smaller dataset.

#### Example Training
```bash
python train_g2p.py --output_dir ./trained/llama_phon_wiki103 --epochs 10 --batch_size 32
python train2_g2p.py --output_dir ./trained/wiki2_pron_space --epochs 6 --batch_size 4
```

### 2.2. IPA-based Models

| Model/Script         | Granularity | Description                                 |
|----------------------|-------------|---------------------------------------------|
| train_ipa.py --granularity char | char        | Character-level IPA (no word boundaries)    |
| train_ipa.py --granularity word | word        | Word-level IPA (preserves word boundaries)  |

#### Example Training
```bash
python train_ipa.py --output_dir ./trained/wiki2_ipa_char --epochs 6 --batch_size 4 --granularity char
python train_ipa.py --output_dir ./trained/wiki2_ipa_word --epochs 6 --batch_size 4 --granularity word
```

---

## 3. Usage & Inference

### 3.1. Inference Scripts

- `infer_g2p.py` (for Arpabet/CMUdict phonemes)
- `infer_ipa.py` (for IPA phonemes)

#### Single Sequence
```bash
python infer_g2p.py --model_dir <checkpoint_dir> --phonemes "DH AH K AE M ER AH IH Z"
python infer_ipa.py --model_dir <checkpoint_dir> --ipa "ð ə k æ m ə r ə ɪ z"
```

#### Full Pipeline
```bash
python infer_g2p.py --model_dir <checkpoint_dir> --sentence "The camera is working well."
python infer_ipa.py --model_dir <checkpoint_dir> --sentence "The camera is working well."
```

#### Batch Processing from JSON
```bash
python infer_g2p.py --model_dir <checkpoint_dir> --json_file test_json.json
python infer_ipa.py --model_dir <checkpoint_dir> --json_file test_json.json
```

##### JSON Structure
```json
{
  "utt_id": ["id1", "id2", ...],
  "ref": ["Reference sentence 1.", "Reference sentence 2.", ...],
  "hypo": ["Hypothesis sentence 1.", "Hypothesis sentence 2.", ...]
}
```
- `hypo` should be sentences (not phonemes/IPA).
- Compatible with outputs from models like AvHubert.

---

## 4. Utility Scripts

### print_json_to_csv.py

Converts a JSON results file (from the infer scripts) into a simple CSV for easier analysis.

```bash
python print_json_to_csv.py results.json
python print_json_to_csv.py results.json --output my_results.csv
```

---

## 5. Tips & Recommendations

- For best results in reconstructing English sentences from phoneme or IPA sequences, use the G2P (Arpabet/CMUdict) mode.
- If your downstream task needs punctuation or word segmentation, train with those features included.
- For speech-to-text or ASR, word-level and punctuation-aware models are usually better.
- For phoneme-level analysis or languages/scripts without clear word boundaries, character-level may suffice.
- Adjust generation parameters in the inference scripts as needed (temperature, repetition_penalty, top_p, top_k).

---

## 6. Comparison: Arpabet vs IPA

- **Arpabet**: Simpler ASCII, easier to process, less phonetic detail.
- **IPA**: Universal, more precise, uses Unicode.