# -*- coding: utf-8 -*-

"""
语音模块模拟器（Voice Simulator）

用途：
- 打开指定串口，支持从命令行手动输入指令并发送给主控；
- 同时异步监听主控回发的数据并打印（例如主控将视觉识别动作透传到语音串口时）。

与主程序的约定：
- 主程序的语音口收到模式切换指令时，期望的是不带 \r\n 的 '1'/'2'/'3'；
- 在“语音模式”下，主程序会将从本端收到的完整动作指令（不带换行）直接转发给电控，并在发送前附加 \r\n 给电控；
- 因此本模拟器发送时，建议：
  - 模式切换请输入：1 或 2 或 3（直接回车发送，不附加换行由串口自动处理）；
  - 其它动作/文本指令：原样输入即可；如果需要显式附加 \r\n，可输入特殊命令 /crlf 切换行为。

运行示例（Windows）：
  python tools/voice_simulator.py --port COM5 --baud 9600
"""

import argparse
import sys
import threading
import time

import serial


def reader_thread(ser: serial.Serial):
    """后台读取线程：打印主控从语音口写回的数据。"""
    try:
        while True:
            try:
                if ser.in_waiting > 0:
                    data = ser.readline()
                    if not data:
                        continue
                    text = data.decode("utf-8", errors="ignore").strip()
                    if text:
                        print(f"\n[VOICE模拟器] <- 收到: '{text}'")
                time.sleep(0.005)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n[VOICE模拟器] 读取异常: {e}")
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="语音模块模拟器 (Voice Simulator)")
    parser.add_argument("--port", "-p", default="COM5", help="串口号（如 COM4）")
    parser.add_argument("--baud", "-b", type=int, default=9600, help="波特率")
    parser.add_argument("--append-crlf", action="store_true", help="发送时附加 \r\n（默认不附加）")
    args = parser.parse_args(argv)

    ser = serial.Serial(port=args.port, baudrate=args.baud, timeout=0.1)
    print(f"[VOICE模拟器] 已打开串口 {args.port} @ {args.baud}")

    stop_flag = False

    t = threading.Thread(target=reader_thread, args=(ser,), daemon=True)
    t.start()

    print("[VOICE模拟器] 输入要发送给主控的内容，回车发送；/quit 退出；/crlf 切换是否附加 \\r\\n。")
    append_crlf = args.append_crlf
    try:
        while True:
            line = input(">> ").strip()
            if not line:
                continue
            if line.lower() in {"/quit", "/exit"}:
                break
            if line.lower() == "/crlf":
                append_crlf = not append_crlf
                print(f"[VOICE模拟器] 发送附加 \\r\\n = {append_crlf}")
                continue

            payload = line + ("\r\n" if append_crlf else "")
            try:
                ser.write(payload.encode("utf-8"))
                shown = line if not append_crlf else f"{line}\\r\\n"
                print(f"[VOICE模拟器] -> 已发送: '{shown}'")
            except Exception as e:
                print(f"[VOICE模拟器] 发送失败: {e}")
    except KeyboardInterrupt:
        print("\n[VOICE模拟器] 中断，准备退出...")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("[VOICE模拟器] 串口已关闭。")


if __name__ == "__main__":
    main()


