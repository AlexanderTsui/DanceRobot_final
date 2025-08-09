# -*- coding: utf-8 -*-

"""
电控模块模拟器（ECU Simulator）

用途：
- 打开指定串口并持续读取主控发送过来的指令；
- 如果收到的内容在“预设可识别指令集合”中，则原样回传同一指令作为回执（不带 \r\n）；
- 否则仅打印日志，不回传。

注意：
- 主控当前通过 send_command_with_retry(...) 发送指令，并 strip 后比对回执；
- 该模拟器按需“无换行”回传即可与主控兼容（主控串口已设置 timeout，会在超时内读到数据）。

运行示例（Windows）：
  python tools/ecu_simulator.py --port COM2 --baud 9600

建议：
- 与主程序联调时，请使用虚拟串口对（例如 com0com）将主程序的 `ECU_SERIAL_PORT` 与此模拟器端口成对相连。
"""

import argparse
import sys
import threading
import time
from typing import Set

import serial


def build_default_allowed_commands() -> Set[str]:
    """
    构建默认的“可识别指令集合”。
    包含：
    - 模式通知：yuyin, shijue, wudao
    - 视觉/动作指令：dazizhan, gongjianbu, jushuangshou, dunxia, biaixin, fuwocheng
    - 音乐ID："1".."9"（可根据需要扩展）
    """
    modes = {"yuyin", "shijue", "wudao"}
    actions = {
        # 视觉/动作指令（原有）
        "dazizhan", "gongjianbu", "jushuangshou", "dunxia", "biaixin", "fuwocheng",
        # 用户补充的动作指令
        "qianjin", "houtui", "zuoyi", "youyi", "juzuoshou", "zuojiaocheng",
        "youjiaocheng", "yewendun", "celi", "xiayaotitui"
    }
    music_ids = {str(i) for i in range(1, 10)}
    return set().union(modes, actions, music_ids)


def run_reader_writer(port: str, baud: int, allowed: Set[str]):
    """
    打开串口并进入主循环：
    - 读取主控发来的行（可能包含 \r\n），去除空白后得到 command
    - 如果 command 在 allowed 内，则回传同样内容（不附加换行）
    - 否则仅打印
    """
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
    print(f"[ECU模拟器] 已打开串口 {port} @ {baud}，开始监听...")

    try:
        while True:
            try:
                # 若缓冲区有数据则读一行；readline 在超时后会返回已有数据
                if ser.in_waiting > 0:
                    raw = ser.readline()
                    if not raw:
                        continue
                    text = raw.decode("utf-8", errors="ignore").strip()
                    if not text:
                        continue
                    print(f"[ECU模拟器] 收到: '{text}'")

                    if text in allowed:
                        # 回执不带 \r\n，符合主控比对逻辑（主控将使用 strip() 比对）
                        ser.write(text.encode("utf-8"))
                        print(f"[ECU模拟器] 已回执: '{text}' (无换行)")
                    else:
                        print(f"[ECU模拟器] 未知指令，忽略: '{text}'")

                # 轻微休眠，降低CPU占用
                time.sleep(0.005)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[ECU模拟器] 警告：循环内异常：{e}")
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[ECU模拟器] 收到中断，准备退出...")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("[ECU模拟器] 串口已关闭。")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="电控模块模拟器 (ECU Simulator)")
    parser.add_argument("--port", "-p", default="COM2", help="串口号（如 COM1）")
    parser.add_argument("--baud", "-b", type=int, default=9600, help="波特率")
    args = parser.parse_args(argv)

    allowed = build_default_allowed_commands()
    print("[ECU模拟器] 允许回执的指令集合：")
    print("  ", sorted(allowed))

    run_reader_writer(args.port, args.baud, allowed)


if __name__ == "__main__":
    main()


