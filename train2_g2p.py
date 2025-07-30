import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import random
import torch
import nltk
import datetime
import logging
import json
from transformers import TrainerCallback, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import re
import argparse

from datasets import Dataset
from g2p_en import G2p

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a phoneme-to-text model")
parser.add_argument("--output_dir", type=str, default="./trained/wiki2_pron_space",
                    help="Directory to save model checkpoints and final model")
parser.add_argument("--epochs", type=int, default=6,
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size per device")
args = parser.parse_args()

# Define output directory (used consistently throughout)
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download CMUdict if not already done
nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.corpus import cmudict

arpabet = cmudict.dict()

g2p = G2p()

def sentence_to_phonemes(sentence):
    try:
        # g2p returns a list with phonemes and spaces for word boundaries
        phonemes = g2p(sentence)
        # Remove spaces (word boundaries) and join phonemes with space
        # phonemes = [ph for ph in phonemes if ph != ' ']
        # Remove stress markers from phonemes (digits after phonemes)
        phonemes = [re.sub(r'(\D+)\d*', r'\1', ph) for ph in phonemes]
        return ' '.join(phonemes)
    except Exception as e:
        # If any error occurs (number too large, etc.), return None
        # This will allow the calling code to skip this sentence
        print(f"Warning: Could not convert sentence to phonemes: {e}")
        return None

# If want to use IPA phonemes instead of CMUdict which is Arpabet
# from epitran import Epitran
# epi = Epitran('eng-Latn')

# def sentence_to_phonemes_ipa(sentence):
#     # Convert to IPA
#     ipa = epi.transliterate(sentence)
#     # Space between phonemes
#     phonemes = ' '.join(list(ipa))
#     return phonemes
########################################################

# def sentence_to_phonemes(sentence):
#     words = sentence.lower().split()
#     phonemes = []
#     for word in words:
#         # Remove punctuation for lookup
#         word_clean = ''.join([c for c in word if c.isalpha()])
#         if word_clean in arpabet:
#             # Use the first pronunciation
#             phonemes.extend(arpabet[word_clean][0])
#         else:
#             # Skip OOV words
#             continue
#     return ' '.join(phonemes)

# 1. Load WikiText dataset
print("Loading WikiText dataset...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

# Clean data to remove WikiText formatting
print("Cleaning WikiText data...")
def clean_wikitext(text):
    # Only remove section headers that are on their own lines
    text = re.sub(r'^=+\s.+\s=+$', '', text, flags=re.MULTILINE)
    
    # Keep the text within links, just remove brackets
    text = re.sub(r'\[\[([^|\]]+)\]\]', r'\1', text)  # Simple links [[link]]
    text = re.sub(r'\[\[[^|]+\|([^\]]+)\]\]', r'\1', text)  # Links with display text [[link|text]]
    
    # Remove templates but keep content where possible
    text = re.sub(r'\{\{[^{}]+\}\}', '', text)  # Simple templates
    
    # Remove HTML tags but keep their content
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs but keep domain for context
    text = re.sub(r'https?://([^/\s]+)[^\s]*', r'\1', text)
    
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    
    # Remove list markers but keep the text
    text = re.sub(r'^\s*[\*#]+\s*', '', text, flags=re.MULTILINE)
    
    # Replace bold/italics with plain text
    text = re.sub(r"'''([^']+)'''", r'\1', text)  # Bold
    text = re.sub(r"''([^']+)''", r'\1', text)   # Italic
    
    # Clean multiple spaces and newlines but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # Keep some paragraph structure
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'=+\s*=+', '', text)
    return text.strip()

# Process all splits
train_texts = [clean_wikitext(x['text']) for x in dataset["train"] if x['text'].strip()]
validation_texts = [clean_wikitext(x['text']) for x in dataset["validation"] if x['text'].strip()]
# Filter out empty texts after cleaning
train_texts = [text for text in train_texts if text.strip()]
validation_texts = [text for text in validation_texts if text.strip()]

print(f"Loaded {len(train_texts)} training texts and {len(validation_texts)} validation texts")

def is_clean_sentence(sent):
    # Reject sentences with remaining wiki formatting
    if re.search(r'=+|\{\{|\}\}|\[\[|\]\]|<|>', sent):
        return False
    
    # Reject sentences with too many non-alphabetic characters
    alpha_chars = sum(c.isalpha() or c.isspace() for c in sent)
    if alpha_chars / len(sent) < 0.75:  # Decrease from 85% to 75% alphabetic content
        return False
        
    # Reject sentences with too many consecutive punctuation marks
    if re.search(r'[^\w\s]{3,}', sent):  # 3+ consecutive non-alphanumeric, non-space chars
        return False
    
    # Reject sentences with too many numeric characters (since phonemes don't represent numbers well)
    num_chars = sum(c.isdigit() for c in sent)
    if num_chars / len(sent) > 0.15:  # Increase from 10% to 15% digits
        return False
    
    # Reject sentences with special characters that don't translate well to phonemes
    # Only allow apostrophes as punctuation
    #re.search(r'[^a-zA-Z0-9\s\'.,?!]', sent) with other punctuations 
    if re.search(r'[^a-zA-Z0-9\s\'.,]', sent):
        return False
    
    # Ensure clean word boundaries by rejecting sentences with odd spacing
    if re.search(r'\s{3,}', sent):  # Only filter out 3+ consecutive spaces
        return False
        
    return True

def clean_sentence(text):
    """Clean sentence text to keep only letters, numbers, spaces and apostrophes"""
    # Keep only letters, numbers, spaces, and apostrophes
    text = re.sub(r'[^a-zA-Z0-9\s\'.,]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Fix spacing issue before apostrophes (e.g., "Ovid 's" -> "Ovid's")
    text = re.sub(r'\s+\'', '\'', text)
    return text.strip()

# 2. Convert sentences to phoneme sequences and create data pairs for each split
def process_texts_to_pairs(texts, split_name):
    data_pairs = []
    print(f"Converting {split_name} sentences to phonemes...")
    for sent in texts:
        # Skip if sentence is too short or too long
        if len(sent.split()) < 3 or len(sent.split()) > 50:
            continue
        
        # Thoroughly clean the sentence before phoneme conversion
        clean_sent = clean_sentence(sent)
        
        # Only process if the cleaned sentence passes our filters
        if clean_sent and is_clean_sentence(clean_sent):
            # Get phonemes from the cleaned sentence
            phonemes = sentence_to_phonemes(clean_sent)
            
            if phonemes and phonemes.strip():
                data_pairs.append({'phonemes': phonemes, 'sentence': clean_sent})
                # Print samples for debugging (every 500 pairs)
                if len(data_pairs) % 500 == 0:
                    print(f"\nSample {split_name} pair #{len(data_pairs)}:")
                    print(f"Original: {sent}")
                    print(f"Cleaned Text: {clean_sent}")
                    print(f"Phonemes: {phonemes}\n")
        
        if len(data_pairs) % 1000 == 0 and len(data_pairs) > 0:
            print(f"Processed {len(data_pairs)} {split_name} sentence-phoneme pairs")
    
    print(f"\n{split_name.capitalize()} filtering summary:")
    print(f"Original texts: {len(texts)}")
    print(f"After filtering: {len(data_pairs)} pairs")
    print(f"Retention rate: {len(data_pairs)/len(texts):.2%}")
    print(f"Created {len(data_pairs)} valid {split_name} phoneme-sentence pairs")
    return data_pairs

# Process train and validation sets
train_pairs = process_texts_to_pairs(train_texts, "training")
val_pairs = process_texts_to_pairs(validation_texts, "validation")

# No need to manually split now, using the dataset's built-in splits
# random.shuffle(data_pairs)
# split_idx = int(len(data_pairs) * 0.95)
# train_pairs = data_pairs[:split_idx]
# val_pairs = data_pairs[split_idx:]

# 4. Load Llama-2-7b with 4-bit quantization
model_name = "/home/rijain@ad.mee.tcd.ie/Experiments/proj/VSR-LLM/checkpoints/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for 4-bit training
model.config.use_cache = False  # Disable KV cache for training
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)

# 5. Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameters info

# 6. Tokenization function with prompt
def preprocess(example):
    # Create instruction prompt format
    prompt = f"Translate the following phonemes into text: {example['phonemes']}"
    # Target with EOS token
    target = f" {example['sentence']}{tokenizer.eos_token}"
    
    # Combine for proper language modeling (complete sequence)
    full_text = prompt + target
    
    # Tokenize the full sequence
    encoded = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Create labels: -100 for prompt tokens (not to be predicted), actual ids for target tokens
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=512)
    prompt_len = len(prompt_tokens["input_ids"])
    
    # Set labels to -100 for the prompt part (we don't want to predict those)
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    
    # Make sure lengths match (padding might have added tokens)
    if len(labels) < len(input_ids):
        labels = labels + [-100] * (len(input_ids) - len(labels))
    elif len(labels) > len(input_ids):
        labels = labels[:len(input_ids)]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_dataset = Dataset.from_list(train_pairs)
val_dataset = Dataset.from_list(val_pairs)

train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

# Setup logging
log_dir = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.txt")

# Configure file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Custom callback to log metrics to file
class MetricsLogger(TrainerCallback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
        # Initialize metrics file
        with open(self.metrics_file, 'w') as f:
            f.write('')
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Add step info if not present
            if 'step' not in logs and state.global_step is not None:
                logs['step'] = state.global_step
            
            # Add timestamp
            logs['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Log to metrics file
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(logs) + '\n')
            
            # Also log to main log file
            if 'loss' in logs:
                logger.info(f"Step {logs.get('step', 'N/A')}: loss = {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                logger.info(f"Evaluation: eval_loss = {logs['eval_loss']:.4f}")

# 7. Training arguments

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding="longest"
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    save_strategy="steps",
    save_steps=500,
    
    # Batch size and gradient accumulation
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    
    # Training length
    num_train_epochs=args.epochs,
    
    # Learning rate and schedule
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # Regularization
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    
    # Performance optimizations
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # More stable
    optim="adamw_torch",
    dataloader_num_workers=4,

    # Logging
    logging_strategy="steps",
    logging_steps=50,
    report_to="none",    
    label_names=["labels"]
)

# Create metrics logger callback
metrics_logger = MetricsLogger(log_dir)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[metrics_logger],
    data_collator=data_collator
)

# Log training start
logger.info(f"Starting training with {len(train_pairs)} training examples, {len(val_pairs)} validation examples")
logger.info(f"Model: {model_name}")
logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

# 9. Train
print("Starting training...")
trainer.train()

# 10. Save the model
print("Saving final model...")
final_model_path = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Training complete. Final model saved to {final_model_path}")

# Save a README with training details
with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(f"# Phoneme-to-Text Model\n\n")
    f.write(f"- Training date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- Epochs: {args.epochs}\n")
    f.write(f"- Batch size: {args.batch_size}\n")
    f.write(f"- Training pairs: {len(train_pairs)}\n")
    f.write(f"- Validation pairs: {len(val_pairs)}\n\n")
    f.write(f"## Usage\n\n")
    f.write(f"For inference, use either:\n")
    f.write(f"- Latest checkpoint: `{OUTPUT_DIR}/checkpoint-XXXX`\n")
    f.write(f"- Final model: `{final_model_path}`\n")
