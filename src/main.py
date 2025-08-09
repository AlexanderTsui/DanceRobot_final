# -*- coding: utf-8 -*-

"""
主控程序 (异步舞蹈模式版)

该版本对舞蹈模式进行了异步化改造，以避免阻塞主循环。
"""

import time
import serial
import threading
import queue
from modules.communicate.serial_communication import send_command_with_retry, send_command_fire_and_forget
from modules.vision.vision_recognize import VisionProcessor
# 直接导入核心函数，不再需要实例化类
from modules.music_recognize.music_recognizer import recognize_music

# --- 全局变量与常量定义 ---

# 定义模式常量，使代码更具可读性
MODE_IDLE = 0  # 待机模式
MODE_VOICE = 1  # 语音识别模式
MODE_VISION = 2  # 视觉识别模式
MODE_DANCE = 3  # 舞蹈模式

# 当前工作模式，默认为待机
current_mode = MODE_IDLE

# --- 异步处理相关 ---
music_result_queue = queue.Queue() # 用于线程间通信的队列
music_thread = None # 持有音乐识别线程的引用
stop_music_thread = threading.Event() # 用于通知线程停止的事件

# --- 初始化核心模块 ---

# 串口配置 (请根据实际情况修改端口号)
ECU_SERIAL_PORT = 'COM1'  # 与电控单元通信的串口
VOICE_SERIAL_PORT = 'COM4' # 与语音模块通信的串口
BAUDRATE = 9600

# 全局的模块实例
vision_processor = None
ecu_serial = None
voice_serial = None

try:
    print("正在初始化视觉模块...")
    vision_processor = VisionProcessor()
    print("视觉模块初始化成功。")
except Exception as e:
    print(f"视觉模块初始化失败: {e}")

try:
    print("正在初始化串口...")
    ecu_serial = serial.Serial(ECU_SERIAL_PORT, BAUDRATE, timeout=0.1)
    voice_serial = serial.Serial(VOICE_SERIAL_PORT, BAUDRATE, timeout=0.1)
    print("串口初始化成功。")
except serial.SerialException as e:
    print(f"串口初始化失败: {e}")
    # 清理已初始化的视觉模块
    if vision_processor:
        vision_processor.cleanup()
    exit(1)

# --- 后台任务函数 ---

def background_recognize_task():
    """
    这个函数在独立的后台线程中运行，执行耗时的音乐识别任务。
    """
    print("[后台线程] 开始执行音乐识别...")
    # 调用我们已经完成的、阻塞的识别函数
    music_id = recognize_music(duration=10)
    
    # 如果主线程没有要求停止，则将结果放入队列
    if not stop_music_thread.is_set():
        print(f"[后台线程] 识别完成，结果: {music_id}。已放入队列。")
        music_result_queue.put(music_id)
    else:
        print("[后台线程] 主线程已请求停止，识别结果被丢弃。")


# --- 模式处理函数 ---

def handle_vision_mode():
    """
    处理视觉识别模式的逻辑。
    此函数在主循环中被反复调用，处理单帧视觉识别。
    """
    global current_mode
    if not vision_processor:
        print("[视觉模式] 视觉模块未初始化，跳过处理。")
        time.sleep(1) # 避免在模块不可用时高速空转
        return
    command = vision_processor.process_one_frame()
    if command:
        print(f"视觉模块确认动作: '{command}'")
        
        # 为指令添加 \r\n
        command_to_send = f"{command}\r\n"
        
        # 1. 将指令发送给语音模块（一次性，不重试）
        send_command_fire_and_forget(voice_serial, "视觉->语音", command_to_send)
        
        # 2. 将指令发送给电控模块（带重试机制）
        send_command_with_retry(ecu_serial, "视觉->电控", command_to_send)

def handle_dance_mode():
    """
    在舞蹈模式下，此函数被高频轮询，只做一件事：
    以非阻塞方式检查结果队列。
    """
    global current_mode
    try:
        # 1. 尝试从队列中获取结果，不会阻塞
        music_id = music_result_queue.get_nowait()
        
        print(f"[主线程] 从队列中获取到音乐ID: {music_id}")
        
        # 2. 如果有结果，则发送给ECU
        if music_id:
            # 修正：确保发送给ECU的指令统一添加 \r\n
            send_command_with_retry(ecu_serial, "舞蹈", f"{music_id}\r\n")
        else:
            print("识别结果为None，不发送指令。")
            
        # 3. 完成一次识别流程，自动切回待机模式
        print("单次音乐识别流程结束，自动切换回待机模式。")
        current_mode = MODE_IDLE

    except queue.Empty:
        # 队列为空是正常情况，表示后台线程还在工作中
        # 什么也不做，等待下一次轮询
        pass
    except Exception as e:
        print(f"[舞蹈模式] 处理时发生错误: {e}")
        current_mode = MODE_IDLE


# --- 主逻辑调度函数 ---

def switch_to_mode(new_mode, mode_name, ecu_notification_cmd):
    """
    处理模式切换的通用逻辑。
    无论ECU是否有回执，都会切换模式，但会记录通信状态。

    Args:
        new_mode (int): 要切换到的新模式 (例如, MODE_VOICE)。
        mode_name (str): 新模式的名称，用于日志打印 (例如, "语音识别")。
        ecu_notification_cmd (str): 通知电控模块进入该模式的指令 (例如, "yuyin\r\n")。
    """
    global current_mode
    print(f"请求切换到: {mode_name}模式")

    # 尝试向ECU发送指令，但不再根据其结果来决定是否切换模式
    success = send_command_with_retry(ecu_serial, "电控", ecu_notification_cmd)

    # 无论发送是否成功，都切换模式
    current_mode = new_mode
    print(f"已切换到模式 {new_mode}: {mode_name}模式")

    # 如果ECU通信失败，打印警告信息
    if not success:
        print(f"[警告] 切换到 {mode_name} 模式时，ECU无响应，但程序已强制切换。")

    return True # 始终返回True，以确保依赖此函数的逻辑（如启动线程）能够执行


def process_mode_switch_command(command: str):
    """
    处理并分发所有来自语音模块的模式切换指令。
    这是系统的“事件处理中心”。
    """
    global current_mode, vision_processor, music_thread

    # 在切换出DANCE模式时，确保后台线程被停止
    if command != '3' and current_mode == MODE_DANCE:
        if music_thread and music_thread.is_alive():
            print("正在停止音乐识别线程...")
            stop_music_thread.set() # 发出停止信号
            music_thread.join(timeout=1.0) # 等待线程结束
            
        # 清空队列中可能残留的结果
        while not music_result_queue.empty():
            music_result_queue.get_nowait()

    if command == '1':
        switch_to_mode(MODE_VOICE, "语音识别", "yuyin\r\n")

    elif command == '2':
        if vision_processor:
            switch_to_mode(MODE_VISION, "视觉识别", "shijue\r\n")
        else:
            print("[警告] 视觉模块未初始化，无法进入视觉模式。")

    elif command == '3':
        # 只有在成功切换模式后才启动后台任务
        if switch_to_mode(MODE_DANCE, "舞蹈", "wudao\r\n"):
            # 舞蹈模式的特殊逻辑：启动后台识别线程
            if not (music_thread and music_thread.is_alive()):
                print("正在启动后台音乐识别线程...")
                stop_music_thread.clear() # 重置停止信号
                music_thread = threading.Thread(target=background_recognize_task)
                music_thread.start()
            else:
                print("音乐识别线程已在运行中。")

    # 注意：这里的else分支被移除了，因为非模式切换指令在主循环中处理

def run_polling_tasks():
    """
    执行所有需要主动轮询外部输入的任务。
    这是系统的“轮询任务中心”。
    """
    # 视觉模式现在也需要轮询
    if current_mode == MODE_VISION:
        handle_vision_mode()
    
    # 舞蹈模式需要持续轮询麦克风进行音乐识别
    elif current_mode == MODE_DANCE:
        handle_dance_mode()


# --- 主函数 ---

def main():
    """
    程序主入口
    """
    global current_mode
    print("舞蹈机器人主控程序已启动，等待指令...")

    while True:
        try:
            # 1. 优先处理来自语音模块的指令（包括模式切换）
            voice_data = voice_serial.readline().decode('utf-8').strip()
            if voice_data:
                # 检查是否为模式切换指令 (不带\r\n)
                if voice_data in ['1', '2', '3']:
                    print(f"收到来自[语音模块]的模式切换指令: '{voice_data}'")
                    process_mode_switch_command(voice_data)
                # 如果是语音模式，则作为待转发的动作指令处理 (自带\r\n)
                elif current_mode == MODE_VOICE:
                    print(f"收到来自[语音模块]的待转发指令: '{voice_data}'")
                    # 直接将带有 \r\n 的指令通过重试机制发送给电控
                    send_command_with_retry(ecu_serial, "语音->电控", voice_data + '\r\n')
                else:
                    print(f"在模式 {current_mode} 下收到来自[语音模块]的未知指令: '{voice_data}' (已忽略)")

            # 2. 检查来自ECU的数据 (当前仅用于日志记录，不用于模式切换)
            ecu_data = ecu_serial.readline().decode('utf-8').strip()
            if ecu_data:
                # 注意：ECU的回执在send_command_with_retry中处理，这里收到的是预期外的异步消息
                print(f"收到来自[ECU]的异步数据: '{ecu_data}' (当前配置下，该数据被忽略)")

            # 3. 执行轮询任务
            run_polling_tasks()

            # 短暂延时，避免CPU占用过高
            time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n程序被用户中断。")
            break
        except Exception as e:
            print(f"主循环发生未知错误: {e}")
            break

    # --- 清理资源 ---
    print("正在清理资源...")
    if music_thread and music_thread.is_alive():
        stop_music_thread.set()
        music_thread.join(timeout=1.0)
    if vision_processor:
        vision_processor.cleanup()
    if ecu_serial and ecu_serial.is_open:
        ecu_serial.close()
    if voice_serial and voice_serial.is_open:
        voice_serial.close()
    print("资源清理完毕，程序结束。")

if __name__ == '__main__':
    main()
