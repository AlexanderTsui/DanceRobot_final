#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import time
import glob
import os
import argparse

def list_video_indices_from_dev():
	"""从 /dev/video* 推断索引"""
	paths = sorted(glob.glob("/dev/video*"))
	indices = []
	for p in paths:
		name = os.path.basename(p)
		if name.startswith("video"):
			num = name[5:]
			if num.isdigit():
				indices.append(int(num))
	return sorted(set(indices)), paths

def try_open(index: int, backend=cv2.CAP_V4L2, read_attempts: int = 2):
	"""尝试打开摄像头并读帧，返回(是否能打开, 能否读到帧, 说明)"""
	cap = cv2.VideoCapture(index, backend)
	if not cap.isOpened():
		cap.release()
		return False, False, "open_failed"

	# 轻度预热并尝试读帧
	ok_read = False
	for _ in range(read_attempts):
		_ = cap.read()  # 预热
		time.sleep(0.05)
		ok, _ = cap.read()
		if ok:
			ok_read = True
			break

	cap.release()
	return True, ok_read, ("opened_with_frames" if ok_read else "opened_no_frames")

def main():
	parser = argparse.ArgumentParser(description="扫描可用摄像头索引")
	parser.add_argument("--max-index", type=int, default=10, help="当 /dev/video* 为空时，扫描 0..max-index")
	args = parser.parse_args()

	dev_indices, dev_paths = list_video_indices_from_dev()
	candidates = dev_indices if dev_indices else list(range(0, args.max_index + 1))

	print("候选索引：", candidates, "\n对应设备：", dev_paths if dev_paths else "无 /dev/video*，改用 0..max-index 扫描")
	print("-" * 60)

	available = []
	for idx in candidates:
		opened, has_frame, info = try_open(idx, backend=cv2.CAP_V4L2)
		print(f"index={idx:<2} -> opened={opened}, frames={has_frame}, info={info}")
		if opened:
			available.append((idx, has_frame))

	print("-" * 60)
	if not available:
		print("未发现可用摄像头。若使用的是 libcamera/CSI 摄像头，OpenCV 直接索引可能无效，请改用 GStreamer 管道或 libcamera 工具链测试。")
	else:
		first_with_frame = [i for i, f in available if f]
		recommend = first_with_frame[0] if first_with_frame else available[0][0]
		print(f"可用索引：{[i for i,_ in available]}；优先尝试 index={recommend}（可读帧：{recommend in first_with_frame}）")

if __name__ == "__main__":
	main()