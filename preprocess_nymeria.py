# 单元格: 【自动化】按人名批量处理、提取视频并清理空间

import pandas as pd
import cv2
from pathlib import Path
import re
import numpy as np
import time
import shutil # 用于安全删除文件夹
import wave
import subprocess
import os
from pathlib import Path
import tempfile
import json


# --- 1. 用户配置区 ---
# ‼️ 请在这里输入您刚刚下载并希望处理的人名
PERSON_NAME_TO_PROCESS = "james_johnson"

# 请确认您的总下载目录
DOWNLOADS_ROOT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Nymeria_Dataset/downloads")

# 【安全开关】设置为 True 来启用自动删除功能。
# 建议首次运行时保持 False，确认视频提取无误后再设为 True。
ENABLE_CLEANUP = False

# --- 2. 导入 Project Aria 模块 ---
try:
    from projectaria_tools.core.data_provider import create_vrs_data_provider
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.vrs import extract_audio_track
    print("✅ 成功导入所有必需的模块。")
except ImportError as e:
    print(f"❌ 导入失败! 请先运行 '!pip install projectaria-tools'。错误: {e}")


# @title 单元格 2: 数据准备 (构建 `datasets` 对象)
# --- 1. 用户配置区 ---

# 【关键修正】更新为您实际下载并希望处理的参与者姓名列表
downloaded_participant_names = [
    'james_johnson', 'adriana_gonzalez', 'barbara_norman', 'christopher_martinez', 'frank_hayden',
    'david_hall', 'jacob_webb', 'hannah_brown', 'elizabeth_morgan', 'glenn_richardson', 'christopher_martinez'
]

# 定义训练集和测试集的划分比例（例如，80%训练，20%测试）
TEST_SPLIT_RATIO = 0.3

# 定义文件和目录路径
DOWNLOADS_ROOT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Nymeria_Dataset/downloads")
# 创建一个总的输出目录来存放所有微调相关文件
OUTPUT_DATASET_DIR = DOWNLOADS_ROOT_DIR / "finetune_dataset_split"
OUTPUT_DATASET_DIR.mkdir(exist_ok=True)

# --- 2. 检查并生成元数据文件 ---
train_metadata_file = OUTPUT_DATASET_DIR / "train_metadata.jsonl"
test_metadata_file = OUTPUT_DATASET_DIR / "test_metadata.jsonl"

# ✅ **已修复**: 检查文件是否存在，如果存在则跳过生成步骤
if train_metadata_file.exists() and test_metadata_file.exists():
    print("✅ 训练集和测试集的 metadata.jsonl 文件已存在，跳过生成步骤。")
else:
    print("ℹ️  未找到元数据文件，开始生成...")
    # --- 按参与者姓名划分训练集和测试集 ---
    print("⏳ 正在按参与者姓名划分训练集和测试集...")
    random.seed(42)
    random.shuffle(downloaded_participant_names)
    num_test_participants = int(len(downloaded_participant_names) * TEST_SPLIT_RATIO)
    if num_test_participants == 0 and len(downloaded_participant_names) > 0:
        num_test_participants = 1
    test_participants = set(downloaded_participant_names[:num_test_participants])
    train_participants = set(downloaded_participant_names[num_test_participants:])
    print(f"✅ 划分完成！")
    print(f"   - 训练集参与者 ({len(train_participants)}人): {train_participants}")
    print(f"   - 测试集参与者 ({len(test_participants)}人): {test_participants}")

    # --- 遍历所有下载的序列，准备数据 ---
    print("\n⏳ 开始遍历所有序列文件夹，生成数据集...")
    train_data = []
    test_data = []
    all_sequence_folders = [f for f in DOWNLOADS_ROOT_DIR.iterdir() if f.is_dir()]
    instruction = "This is a first-person view video. Describe the main actions of the person in the third person in one sentence (e.g., 'The person walks down the hall and opens a door.')."

    for seq_dir in all_sequence_folders:
        try:
            parts = seq_dir.name.split('_')
            participant_name = f"{parts[2]}_{parts[3]}"
        except IndexError:
            print(f"⚠️ 无法从文件夹名 {seq_dir.name} 解析参与者，已跳过。")
            continue

        if participant_name in train_participants:
            target_list = train_data
        elif participant_name in test_participants:
            target_list = test_data
        else:
            continue

        video_clips_dir = seq_dir / "extracted_clips"
        csv_file_path = seq_dir / "narration/activity_summarization.csv"

        if not video_clips_dir.exists() or not csv_file_path.exists():
            continue

        annotations_df = pd.read_csv(csv_file_path)
        video_files = sorted(list(video_clips_dir.glob("*.mp4")))

        for video_path in video_files:
            match = re.search(r'activity_(\d+)_', video_path.name)
            if not match:
                continue
            activity_index = int(match.group(1))
            if activity_index < len(annotations_df):
                summary = annotations_df.iloc[activity_index]['Describe my activity']
                video_uri = f"file://{video_path.resolve()}"
                messages = [
                    {"role": "user", "content": [{"type": "video", "video": video_uri, "fps": 1}, {"type": "text", "text": instruction}]},
                    {"role": "assistant", "content": [{"type": "text", "text": summary}]}
                ]
                target_list.append({"messages": messages})

    print("\n✅ 所有序列处理完毕！")

    # --- 将数据写入独立的 JSONL 文件 ---
    with open(train_metadata_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"📝 成功写入训练集文件: {train_metadata_file}，包含 {len(train_data)} 条记录。")

    with open(test_metadata_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    print(f"📝 成功写入测试集文件: {test_metadata_file}，包含 {len(test_data)} 条记录。")

# --- 3. 加载为 Hugging Face Dataset 对象 ---
if train_metadata_file.exists():
    try:
        train_dataset = Dataset.from_json(str(train_metadata_file))
        print("\n✅ 成功将 train_metadata.jsonl 加载为 Dataset 对象。")
        print(f"   - 训练集大小: {len(train_dataset)}")
        print("   - 训练集示例:")
        print(train_dataset[0])
    except Exception as e:
        print(f"❌ 加载训练集时出错: {e}")
else:
    print("\n❌ 训练集元数据文件不存在，无法加载。")

if test_metadata_file.exists():
    try:
        test_dataset = Dataset.from_json(str(test_metadata_file))
        print("\n✅ 成功将 test_metadata.jsonl 加载为 Dataset 对象。")
        print(f"   - 测试集大小: {len(test_dataset)}")
        print("   - 测试集示例:")
        print(test_dataset[0])
    except Exception as e:
        print(f"❌ 加载测试集时出错: {e}")
else:
    print("\nℹ️  测试集元数据文件不存在或为空，未创建Dataset对象。")


# --- 3. 查找属于指定人物的序列文件夹 ---
print(f"\n🔍 正在扫描 {DOWNLOADS_ROOT_DIR} ...")
# 查找所有包含指定人名的文件夹
sequence_folders = [f for f in DOWNLOADS_ROOT_DIR.iterdir() if f.is_dir() and PERSON_NAME_TO_PROCESS in f.name]

if not sequence_folders:
    print(f"❌ 未找到与 '{PERSON_NAME_TO_PROCESS}' 相关的任何序列文件夹。请检查人名是否正确或数据是否已下载。")
else:
    print(f"✅ 找到 {len(sequence_folders)} 个与 '{PERSON_NAME_TO_PROCESS}' 相关的序列，准备开始处理...")


for i, seq_dir in enumerate(sequence_folders):
    print(f"\n======================================================================")
    print(f"🎬 开始处理序列 {i+1}/{len(sequence_folders)}: {seq_dir.name}")
    print("----------------------------------------------------------------------")

    vrs_file_path = seq_dir / "recording_head/data/data.vrs"
    csv_file_path = seq_dir / "narration/activity_summarization.csv"
    output_video_dir = seq_dir / "extracted_clips"
    output_video_dir.mkdir(parents=True, exist_ok=True)

    temp_audio_dir = Path(tempfile.mkdtemp())
    full_audio_path = temp_audio_dir / "full_audio.wav"

    if not vrs_file_path.exists() or not csv_file_path.exists():
        print(f"   - ❌ 错误: 缺少VRS或CSV文件，跳过此序列。")
        continue

    # --- 步骤 1: 使用官方工具 `extract_audio_track` 提取完整音轨 ---
    has_audio = False
    try:
        print(f"   - [步骤1] 正在使用官方工具提取完整音轨...")
        json_output_string = extract_audio_track(str(vrs_file_path), str(full_audio_path))
        json_output = json.loads(json_output_string)
        if json_output and json_output.get("status") == "success":
            has_audio = True
            print(f"   - ✅ 成功提取完整音轨。")
        else:
            print(f"   - ⚠️ 警告: 官方工具无法从此VRS提取音轨。")
            print(f"     - 官方返回信息: {json_output_string}")
    except Exception as e:
        print(f"   - ❌ 调用 extract_audio_track 时发生异常: {e}")

    # --- 步骤 2: 您的原始高效流程，用于提取视频和合并 ---
    try:
        provider = create_vrs_data_provider(str(vrs_file_path))
        rgb_stream_id = provider.get_stream_id_from_label("camera-rgb")

        # 计算FPS
        num_frames_total = provider.get_num_data(rgb_stream_id)
        fps = 15 if num_frames_total < 10 else int(round(1.0 / np.mean(np.diff([provider.get_image_data_by_index(rgb_stream_id, j)[1].capture_timestamp_ns for j in range(min(100, num_frames_total))]) / 1e9)))
        print(f"   - 🖼️  计算视频帧率为 {fps} FPS。")

        annotations_df = pd.read_csv(csv_file_path)
        print(f"   - [步骤2] 加载了 {len(annotations_df)} 个活动注释，开始快速提取和合并...")

        for index, activity in annotations_df.iterrows():
            start_sec = activity['start_time']
            end_sec = activity['end_time']
            duration_sec = end_sec - start_sec
            activity_description = activity['Describe my activity']
            start_time_ns, end_time_ns = int(start_sec * 1e9), int(end_sec * 1e9)

            start_index_vid = provider.get_index_by_time_ns(rgb_stream_id, start_time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.AFTER)
            end_index_vid = provider.get_index_by_time_ns(rgb_stream_id, end_time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.BEFORE)

            if start_index_vid >= end_index_vid:
                continue

            safe_desc = re.sub(r'[^a-zA-Z0-9_]', '', activity_description.replace(' ', '_'))[:30]
            base_filename = f"activity_{index:03d}_{safe_desc}"
            temp_video_path = output_video_dir / f"{base_filename}_temp_video.mp4"
            final_output_path = output_video_dir / f"{base_filename}.mp4"

            # 提取无声视频
            first_frame = provider.get_image_data_by_index(rgb_stream_id, start_index_vid)[0].to_numpy_array()
            h, w, _ = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE).shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (w, h))
            for frame_idx in range(start_index_vid, end_index_vid + 1):
                img_rgb = provider.get_image_data_by_index(rgb_stream_id, frame_idx)[0].to_numpy_array()
                video_writer.write(cv2.cvtColor(cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_RGB2BGR))
            video_writer.release()

            # 使用 FFmpeg 进行最终合并
            if has_audio:
                command = [
                    'ffmpeg', '-y', '-i', str(temp_video_path), '-ss', str(start_sec),
                    '-i', str(full_audio_path), '-t', str(duration_sec),
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', str(final_output_path)
                ]
                subprocess.run(command, capture_output=True, text=True, check=True)
                os.remove(temp_video_path)
                print(f"     - ✅ 成功合并音视频: {final_output_path.name}")
            else:
                os.rename(temp_video_path, final_output_path)
                print(f"     - ✅ 已保存无声视频: {final_output_path.name}")

    except Exception as e:
        print(f"   - ❌ 处理序列 {seq_dir.name} 时发生未知错误: {e}")
    finally:
        if temp_audio_dir.exists():
            shutil.rmtree(temp_audio_dir)

print(f"\n\n🎉🎉🎉 所有与 '{PERSON_NAME_TO_PROCESS}' 相关的序列处理完毕！🎉🎉🎉")
