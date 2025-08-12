# -*- coding: utf-8 -*-

"""
模块: music_recognize.fingerprint_generator
功能: 离线声学指纹生成器 (双重计分卡版)

该脚本负责创建和保存一个包含分离特征向量(Chroma, MFCC)的指纹数据库。
"""

import os
import pickle
import sys
import librosa
import numpy as np
from typing import Dict, List, Tuple

# -- 修复导入问题 --
MODULES_DIR = os.path.dirname(__file__)
if MODULES_DIR not in sys.path:
    sys.path.append(MODULES_DIR)

# 从新的工具模块中导入核心特征提取函数
from audio_utils import extract_feature_from_data

# --- 常量定义 ---
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')
REFERENCE_MUSIC_DIR = os.path.join(BASE_DIR, 'data', 'reference_music')
FINGERPRINT_DIR = os.path.join(BASE_DIR, 'data', 'fingerprints')
FINGERPRINT_FILE = os.path.join(FINGERPRINT_DIR, 'reference_fingerprints.pkl')

# --- 分段参数 ---
SEGMENT_LENGTH_SEC = 10
HOP_LENGTH_SEC = 0.5


def generate_segmented_fingerprints_for_file(audio_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    为单个音频文件生成分段的、分离的指纹列表。

    返回:
        List[Tuple[np.ndarray, np.ndarray]]:
            一个列表，其中每个元素都是一个元组，包含(chroma_vec, mfcc_vec)。
    """
    fingerprint_list = []
    try:
        y, sr = librosa.load(audio_path, sr=None)
        segment_samples = int(SEGMENT_LENGTH_SEC * sr)
        hop_samples = int(HOP_LENGTH_SEC * sr)

        for start_sample in range(0, len(y) - segment_samples + 1, hop_samples):
            end_sample = start_sample + segment_samples
            segment_y = y[start_sample:end_sample]

            # 提取分离的特征
            features = extract_feature_from_data(segment_y, sr)
            if features is not None:
                fingerprint_list.append(features)

    except Exception as e:
        print(f"错误：处理文件 '{audio_path}' 时发生错误: {e}")

    return fingerprint_list


def generate_and_save_fingerprints():
    """
    主函数：遍历参考音乐，生成分段的、分离的指纹库并保存到文件。
    """
    print("开始生成双重计分卡式声学指纹库...")

    if not os.path.exists(FINGERPRINT_DIR):
        os.makedirs(FINGERPRINT_DIR)

    # 新的指纹库结构: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
    fingerprint_database: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    total_fingerprints = 0

    if not os.path.exists(REFERENCE_MUSIC_DIR):
        print(f"错误：参考音乐目录 '{REFERENCE_MUSIC_DIR}' 不存在。")
        return

    for filename in sorted(os.listdir(REFERENCE_MUSIC_DIR)):
        if filename.lower().endswith(('.mp3', '.wav', '.flac')):
            audio_path = os.path.join(REFERENCE_MUSIC_DIR, filename)
            print(f"正在处理: {filename}...")
            
            segment_fingerprints = generate_segmented_fingerprints_for_file(audio_path)

            if segment_fingerprints:
                music_id = os.path.splitext(filename)[0]
                fingerprint_database[music_id] = segment_fingerprints
                print(f"  -> 成功为 '{music_id}' 生成了 {len(segment_fingerprints)} 个分离式指纹。")
                total_fingerprints += len(segment_fingerprints)

    if not fingerprint_database:
        print("未能成功生成任何指纹。")
        return

    try:
        with open(FINGERPRINT_FILE, 'wb') as f:
            pickle.dump(fingerprint_database, f)
        print("-" * 30)
        print(f"成功为 {len(fingerprint_database)} 首歌曲生成了 {total_fingerprints} 个分离式指纹。")
        print(f"指纹库已更新并保存到: {FINGERPRINT_FILE}")
        print("-" * 30)
    except Exception as e:
        print(f"错误：保存指纹文件时发生错误: {e}")


if __name__ == "__main__":
    generate_and_save_fingerprints()
