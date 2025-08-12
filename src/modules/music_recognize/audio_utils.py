# -*- coding: utf-8 -*-

"""
模块: music_recognize.audio_utils
功能: 音频处理工具函数 (双重计分卡版)

该脚本的特征提取函数被重构，以返回两个独立的、标准化的特征向量，
分别代表和声(Chroma)和音色(MFCC)，为后续的“双重计分”做准备。
"""

import librosa
import numpy as np
from typing import Optional, Tuple


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    一个内部辅助函数，用于对向量进行中心化和L2归一化。
    """
    centered_vec = vec - np.mean(vec)
    norm = np.linalg.norm(centered_vec)
    if norm > 0:
        return centered_vec / norm
    return centered_vec


def extract_feature_from_data(y: np.ndarray, sr: int, n_mfcc: int = 13) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    【新-双重计分版】从内存中的音频数据中提取分离的特征向量。

    返回:
        Optional[Tuple[np.ndarray, np.ndarray]]: 
            一个元组，包含两个独立的、标准化的向量:
            (normalized_chroma_vector, normalized_mfcc_vector)
    """
    try:
        # 1. 计算并标准化Chroma特征
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        norm_chroma = _normalize_vector(chroma_mean)

        # 2. 计算并标准化MFCC特征 (丢弃第一个系数)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        norm_mfcc = _normalize_vector(mfcc_mean[1:])

        return norm_chroma, norm_mfcc
    except Exception as e:
        print(f"警告：处理音频片段时发生错误: {e}")
        return None


def extract_feature(audio_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    【旧，保留】从单个音频文件中提取分离的声学特征。
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        return extract_feature_from_data(y, sr)
    except Exception as e:
        print(f"错误：在提取特征时处理文件 '{audio_path}' 发生错误: {e}")
        return None
