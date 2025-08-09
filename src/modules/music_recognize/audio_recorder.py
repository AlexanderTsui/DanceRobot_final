# -*- coding: utf-8 -*-

"""
模块: music_recognize.audio_recorder
功能: 实时音频录制模块

该脚本提供了一个函数，用于从设备的默认麦克风录制音频，
并将其保存为 WAV 文件格式。
"""

import os
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
from typing import Optional

# --- 常量定义 ---

# 项目根目录的相对路径基准
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')

# 录制的音频文件临时存放路径
RECORDINGS_DIR = os.path.join(BASE_DIR, 'data', 'recordings')
DEFAULT_OUTPUT_FILENAME = os.path.join(RECORDINGS_DIR, 'temp_recording.wav')

# 定义音频流的参数
DEFAULT_SAMPLING_RATE = 44100  # 采样率 (Hz), CD音质标准
DEFAULT_CHANNELS = 1           #声道数, 1 表示单声道


def record_audio(duration: int = 10,
                 samplerate: int = DEFAULT_SAMPLING_RATE,
                 output_filepath: str = DEFAULT_OUTPUT_FILENAME) -> Optional[str]:
    """
    从默认麦克风录制指定时长的音频并保存为 .wav 文件。

    参数:
        duration (int): 录音时长，单位为秒。默认为10秒。
        samplerate (int): 采样率，单位为赫兹。默认为44100Hz。
        output_filepath (str): 输出的 .wav 文件的完整路径。
                               默认为 data/recordings/temp_recording.wav。

    返回:
        Optional[str]: 如果录音和保存成功，则返回保存文件的路径。
                       如果发生错误，则返回 None。
    """
    try:
        # 1. 确保输出目录存在
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            print(f"创建录音目录: {output_dir}")
            os.makedirs(output_dir)

        # 2. 开始录音
        print("-" * 30)
        print(f"准备开始录音，时长: {duration} 秒...")
        print("请在麦克风附近播放音乐。")

        # sd.rec() 会立即返回并开始在后台录音
        recording = sd.rec(int(duration * samplerate),
                           samplerate=samplerate,
                           channels=DEFAULT_CHANNELS,
                           dtype='float32') # 使用 float32 类型以获得更好的质量

        # sd.wait() 会阻塞程序，直到录音完成
        sd.wait()

        print("录音结束。")
        print("-" * 30)

        # 3. 保存录音文件
        # sounddevice录制的numpy数组是float类型，取值在[-1, 1]
        # wavfile.write 需要一个整数类型的数组，因此需要进行转换
        # 我们乘以32767并转换为16位整数，这是WAV文件的常见格式
        recording_int16 = np.int16(recording * 32767)
        wavfile.write(output_filepath, samplerate, recording_int16)

        print(f"录音已成功保存到: {output_filepath}")
        return output_filepath

    except Exception as e:
        print(f"错误：录音过程中发生错误: {e}")
        return None


if __name__ == '__main__':
    # 当该脚本被直接执行时，运行一个简单的录音测试
    print("--- 音频录制模块测试 ---")
    # 录制5秒的测试音频
    record_audio(duration=5, output_filepath=os.path.join(RECORDINGS_DIR, 'test_recording.wav'))
    print("--- 测试结束 ---")


