# --- Import required libraries ---
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from datasets import Dataset
import pandas as pd
from pathlib import Path
import re
import os
import json
import numpy as np
from tqdm.auto import tqdm
import evaluate

# --- 1. Configuration Area ---
# Path where all sequence folders are located
DOWNLOADS_ROOT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Nymeria_Dataset/downloads")
# Model ID for Qwen2.5-VL
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# --- 2. Data Preparation (Scan all participants) ---
print(f"Scanning directory: {DOWNLOADS_ROOT_DIR}")

# We will process every folder found in the directory regardless of participant list
all_sequence_folders = [f for f in DOWNLOADS_ROOT_DIR.iterdir() if f.is_dir()]
eval_data = []
instruction = "This is a first-person view video. Describe the main actions of the person in the third person in one sentence (e.g., 'The person walks down the hall and opens a door.')."

print(f"Found {len(all_sequence_folders)} potential sequences. Extracting metadata...")

for seq_dir in all_sequence_folders:
    video_clips_dir = seq_dir / "extracted_clips"
    csv_file_path = seq_dir / "narration/activity_summarization.csv"

    # Only process if both video clips and annotation CSV exist
    if not video_clips_dir.exists() or not csv_file_path.exists():
        continue

    annotations_df = pd.read_csv(csv_file_path)
    video_files = sorted(list(video_clips_dir.glob("*.mp4")))

    for video_path in video_files:
        # Extract activity index from filename (e.g., activity_001_...)
        match = re.search(r'activity_(\d+)_', video_path.name)
        if not match:
            continue
            
        activity_index = int(match.group(1))
        if activity_index < len(annotations_df):
            summary = annotations_df.iloc[activity_index]['Describe my activity']
            video_uri = f"file://{video_path.resolve()}"
            
            # Format according to Qwen2.5-VL requirements
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "video", "video": video_uri, "fps": 1.0}, 
                        {"type": "text", "text": instruction}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": summary}]
                }
            ]
            eval_data.append({"messages": messages})

if len(eval_data) == 0:
    raise ValueError("No valid evaluation data found. Check your directory structure.")

# Convert to Hugging Face Dataset
eval_dataset = Dataset.from_list(eval_data)
print(f"Successfully prepared evaluation dataset with {len(eval_dataset)} samples.")

# --- 3. Model Loading (Base Model for Evaluation) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Using 4-bit quantization to fit on consumer GPUs during inference
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading model: {model_id}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quantization_config,
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print("Model and Processor loaded successfully.")

# --- 4. Load Evaluation Metrics ---
print("Loading metrics (ROUGE, BERTScore)...")
# Note: make sure to run !pip install evaluate rouge_score bert_score
rouge = evaluate.load('rouge')
bertscore = evaluate.load("bertscore")

# --- 5. Inference and Evaluation Loop ---
predictions = []
references = []

print(f"Starting batch evaluation on {len(eval_dataset)} samples...")

# Iterate through the dataset
for sample in tqdm(eval_dataset):
    # Prepare input messages (Prompt side only)
    user_message = [sample['messages'][0]] # Just the User part
    ground_truth = sample['messages'][1]['content'][0]['text']
    
    # Process for Inference
    text = processor.apply_chat_template(user_message, tokenize=False, add_generation_prompt=True)
    _, video_inputs = process_vision_info(user_message)
    
    inputs = processor(
        text=[text],
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate prediction
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        # Trim the prompt tokens from the output
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    predictions.append(response.strip())
    references.append(ground_truth.strip())

# --- 6. Final Score Calculation ---
print("\nCalculating final scores...")

# Calculate ROUGE
rouge_results = rouge.compute(predictions=predictions, references=references)

# Calculate BERT Score (Average)
bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")

print("\n" + "="*20 + " EVALUATION RESULTS " + "="*20)
print("\n--- ROUGE Scores ---")
for key, value in rouge_results.items():
    print(f"  - {key}: {value:.4f}")

print("\n--- BERT Score (Mean) ---")
print(f"  - Precision: {np.mean(bert_results['precision']):.4f}")
print(f"  - Recall:    {np.mean(bert_results['recall']):.4f}")
print(f"  - F1 Score:  {np.mean(bert_results['f1']):.4f}")
print("="*60)

# Optional: Save results to a CSV
results_df = pd.DataFrame({
    "Ground Truth": references,
    "Model Prediction": predictions
})
results_df.to_csv("evaluation_results.csv", index=False)
print("Detailed results saved to evaluation_results.csv")
