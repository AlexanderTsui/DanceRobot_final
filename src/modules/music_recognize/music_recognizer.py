# -*- coding: utf-8 -*-

"""
模块: music_recognize.music_recognizer
功能: 实时音乐识别与匹配器 (双重计分卡与几何平均最终版)

该模块采用分离计分和几何平均的策略，以实现高精度和高可靠性的音乐识别。
"""

import os
import pickle
import sys
import math
import numpy as np
from typing import Dict, Optional, Tuple, List

# -- 修复导入问题 --
MODULES_DIR = os.path.dirname(__file__)
if MODULES_DIR not in sys.path:
    sys.path.append(MODULES_DIR)

from audio_recorder import record_audio
from audio_utils import extract_feature

# --- 常量定义 ---
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')
FINGERPRINT_FILE = os.path.join(BASE_DIR, 'data', 'fingerprints', 'reference_fingerprints.pkl')


def load_fingerprints() -> Optional[Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]:
    """
    从文件加载分离式指纹库。
    """
    if not os.path.exists(FINGERPRINT_FILE):
        print(f"错误：指纹库文件 '{FINGERPRINT_FILE}' 不存在。")
        return None
    try:
        with open(FINGERPRINT_FILE, 'rb') as f:
            fingerprints = pickle.load(f)
        # 简单的格式验证
        if fingerprints and not isinstance(list(fingerprints.values())[0][0], tuple):
             print(f"错误：指纹库格式不正确，期望得到分离式指纹库。")
             return None
        return fingerprints
    except Exception as e:
        print(f"错误：加载指纹库时发生错误: {e}")
        return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个Numpy向量之间的余弦相似度。
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    # 防止因浮点数精度问题导致相似度略大于1.0
    return min(1.0, dot_product / (norm_vec1 * norm_vec2))


def find_best_match(
    target_features: Tuple[np.ndarray, np.ndarray],
    fingerprint_db: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    threshold: float = 0.5
) -> Optional[Tuple[str, float]]:
    """
    在分离式指纹库中寻找最佳匹配，使用几何平均合并分数。
    """
    if not fingerprint_db:
        return None

    target_chroma, target_mfcc = target_features
    overall_best_match_id = None
    overall_max_score = -1

    for music_id, segment_fingerprints in fingerprint_db.items():
        max_score_for_song = -1
        
        # 遍历歌曲的每一个分离式指纹片段
        for ref_chroma, ref_mfcc in segment_fingerprints:
            # 1. 分离计分
            score_chroma = cosine_similarity(target_chroma, ref_chroma)
            score_mfcc = cosine_similarity(target_mfcc, ref_mfcc)
            
            # 2. 几何平均合并
            # 确保分数非负
            score_chroma = max(0, score_chroma)
            score_mfcc = max(0, score_mfcc)
            final_score = math.sqrt(score_chroma * score_mfcc)
            
            if final_score > max_score_for_song:
                max_score_for_song = final_score
        
        print(f"  与歌曲 '{music_id}' 的最高几何平均得分: {max_score_for_song:.4f}")

        if max_score_for_song > overall_max_score:
            overall_max_score = max_score_for_song
            overall_best_match_id = music_id

    if overall_best_match_id and overall_max_score >= threshold:
        return overall_best_match_id, overall_max_score
    else:
        print(f"提示：最高分 {overall_max_score:.4f} 未达到阈值 {threshold}。")
        return None


def recognize_music(duration: int = 10) -> Optional[str]:
    """
    执行最终版的音乐识别流程。
    """
    print("--- 开始音乐识别流程 (最终版) ---")

    recorded_audio_path = record_audio(duration=duration)
    if not recorded_audio_path: return None

    print("正在分析录音...")
    target_features = extract_feature(recorded_audio_path)
    if not target_features: return None

    print("正在加载参考指纹库...")
    fingerprint_database = load_fingerprints()
    if not fingerprint_database: return None

    print("开始匹配...")
    match_result = find_best_match(target_features, fingerprint_database)

    if match_result:
        best_id, best_score = match_result
        print("-" * 30)
        print(f"==> 识别完成！最佳匹配是歌曲: 【{best_id}】")
        print(f"    最终得分为: {best_score:.4f}")
        print("-" * 30)
        return best_id
    else:
        print("-" * 30)
        print("==> 识别完成！未能找到匹配的歌曲。")
        print("-" * 30)
        return None


if __name__ == '__main__':
    recognize_music()
