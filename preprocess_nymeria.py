# Cell: [Automation] Process All Sequences, Extract Videos, and Save Metadata (No Split)

import pandas as pd
import cv2
from pathlib import Path
import re
import numpy as np
import time
import shutil
import subprocess
import os
import json

# --- 1. User Configuration ---
# Root directory where all participant data is stored
DOWNLOADS_ROOT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Nymeria_Dataset/downloads")

# Directory to save the final metadata.jsonl
OUTPUT_METADATA_DIR = DOWNLOADS_ROOT_DIR / "full_dataset_metadata"
OUTPUT_METADATA_DIR.mkdir(exist_ok=True)
metadata_file_path = OUTPUT_METADATA_DIR / "metadata.jsonl"

# Instruction for the VL model
INSTRUCTION = "This is a first-person view video. Describe the main actions of the person in the third person in one sentence (e.g., 'The person walks down the hall and opens a door.')."

# --- 2. Import Project Aria Modules ---
try:
    from projectaria_tools.core.data_provider import create_vrs_data_provider
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    print("Successfully imported required Aria modules.")
except ImportError as e:
    print(f"Import failed! Please run '!pip install projectaria-tools' first. Error: {e}")

# --- 3. Scan and Process All Sequences ---
print(f"\nScanning {DOWNLOADS_ROOT_DIR} for all available sequences...")

all_sequence_folders = [f for f in DOWNLOADS_ROOT_DIR.iterdir() if f.is_dir() and f.name != "full_dataset_metadata"]
all_metadata_entries = []

if not all_sequence_folders:
    print(f"No sequence folders found in {DOWNLOADS_ROOT_DIR}.")
else:
    print(f"Found {len(all_sequence_folders)} total sequences to process.")

for i, seq_dir in enumerate(all_sequence_folders):
    print(f"\n======================================================================")
    print(f"Processing Sequence {i+1}/{len(all_sequence_folders)}: {seq_dir.name}")
    print("----------------------------------------------------------------------")

    vrs_file_path = seq_dir / "recording_head/data/data.vrs"
    csv_file_path = seq_dir / "narration/activity_summarization.csv"
    output_video_dir = seq_dir / "extracted_clips"
    output_video_dir.mkdir(parents=True, exist_ok=True)

    if not vrs_file_path.exists() or not csv_file_path.exists():
        print(f"   - Warning: Missing VRS or CSV. Skipping.")
        continue

    try:
        provider = create_vrs_data_provider(str(vrs_file_path))
        rgb_stream_id = provider.get_stream_id_from_label("camera-rgb")

        # Calculate FPS
        num_frames_total = provider.get_num_data(rgb_stream_id)
        if num_frames_total < 2:
            print(f"   - Warning: Not enough frames. Skipping.")
            continue
            
        sample_indices = range(min(100, num_frames_total))
        timestamps = [provider.get_image_data_by_index(rgb_stream_id, j)[1].capture_timestamp_ns for j in sample_indices]
        fps = int(round(1.0 / np.mean(np.diff(timestamps) / 1e9)))
        print(f"   - Detected Video Frame Rate: {fps} FPS.")

        # Load Annotations
        annotations_df = pd.read_csv(csv_file_path)
        print(f"   - Found {len(annotations_df)} activity segments.")

        for index, activity in annotations_df.iterrows():
            start_sec = activity['start_time']
            end_sec = activity['end_time']
            summary = activity['Describe my activity']
            
            start_time_ns, end_time_ns = int(start_sec * 1e9), int(end_sec * 1e9)
            start_index_vid = provider.get_index_by_time_ns(rgb_stream_id, start_time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.AFTER)
            end_index_vid = provider.get_index_by_time_ns(rgb_stream_id, end_time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.BEFORE)

            if start_index_vid >= end_index_vid or start_index_vid == -1:
                continue

            # File Setup
            safe_desc = re.sub(r'[^a-zA-Z0-9_]', '', str(summary).replace(' ', '_'))[:30]
            final_output_path = output_video_dir / f"activity_{index:03d}_{safe_desc}.mp4"

            # 1. Video Extraction (if not already exists)
            if not final_output_path.exists():
                first_frame_data = provider.get_image_data_by_index(rgb_stream_id, start_index_vid)[0].to_numpy_array()
                rotated_sample = cv2.rotate(first_frame_data, cv2.ROTATE_90_CLOCKWISE)
                h, w, _ = rotated_sample.shape
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(final_output_path), fourcc, fps, (w, h))

                for frame_idx in range(start_index_vid, end_index_vid + 1):
                    img_data = provider.get_image_data_by_index(rgb_stream_id, frame_idx)[0].to_numpy_array()
                    frame_bgr = cv2.cvtColor(cv2.rotate(img_data, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                
                video_writer.release()
                print(f"     - Success: Saved {final_output_path.name}")
            else:
                print(f"     - Skip: Clip {index} already exists.")

            # 2. Add to Metadata List
            video_uri = f"file://{final_output_path.resolve()}"
            entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_uri, "fps": 1.0},
                            {"type": "text", "text": INSTRUCTION}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(summary)}]
                    }
                ]
            }
            all_metadata_entries.append(entry)

    except Exception as e:
        print(f"   - Error processing {seq_dir.name}: {e}")

# --- 4. Save Metadata to JSONL ---
print(f"\nSaving metadata to {metadata_file_path}...")
with open(metadata_file_path, 'w') as f:
    for entry in all_metadata_entries:
        f.write(json.dumps(entry) + '\n')

print(f"Successfully saved {len(all_metadata_entries)} records to metadata.jsonl.")
print("\nAll sequences processed and metadata stored.")
