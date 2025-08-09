import serial
import time

# 定义重试相关的常量
RETRY_ATTEMPTS = 3  # 最大尝试次数
RETRY_TIMEOUT_SECONDS = 1.5  # 每次尝试的超时时间（秒）

def send_command_with_retry(ecu_serial: serial.Serial, module_name: str, command: str) -> bool:
    """
    向电控单元发送指令，并包含通用的重试和确认机制。

    这个函数会发送一个指令，然后等待电控返回与发送的指令相同的回执。
    如果在指定时间内未收到回执，它会重新发送指令，最多尝试 RETRY_ATTEMPTS 次。

    Args:
        ecu_serial: 与电控单元通信的、已经打开的串口对象。
        module_name: 调用此函数的模块名称（如 "语音" 或 "视觉"），用于日志打印。
        command: 要发送的指令字符串。

    Returns:
        bool: 如果成功发送并收到与command相同的确认回执，则返回 True。
              如果所有尝试都失败，则返回 False。
    """
    for attempt in range(RETRY_ATTEMPTS):
        print(f"[{module_name}模块] 尝试发送指令 '{command}'... (第 {attempt + 1}/{RETRY_ATTEMPTS} 次)")

        # 1. 发送指令
        try:
            ecu_serial.write(command.encode('utf-8'))
        except serial.SerialException as e:
            print(f"[{module_name}模块] 错误：串口发送失败，无法写入。原因: {e}")
            return False

        # 2. 等待与发送指令相同的回执
        start_time = time.time()
        while time.time() - start_time < RETRY_TIMEOUT_SECONDS:
            if ecu_serial.in_waiting > 0:
                response = ecu_serial.readline().decode('utf-8').strip()
                # 核心修正：比对时，将待发送的指令也进行strip，以匹配不带`\r\n`的回执
                if response and response == command.strip():
                    print(f"[{module_name}模块] 成功：收到来自电控的 '{response}' 确认回执。")
                    return True
                elif response: # 如果收到了任何非预期的回执
                     print(f"[{module_name}模块] 警告：在等待'{command.strip()}'回执期间，收到了异常指令'{response}'，本次将忽略。")

        if attempt < RETRY_ATTEMPTS - 1:
            print(f"[{module_name}模块] 警告：发送成功，但在 {RETRY_TIMEOUT_SECONDS} 秒内未收到 '{command.strip()}' 回执。即将重试...")
        
    print(f"[{module_name}模块] 错误：指令 '{command.strip()}' 发送 {RETRY_ATTEMPTS} 次后仍未收到回执确认。")
    return False

def send_command_fire_and_forget(target_serial: serial.Serial, module_name: str, command: str):
    """
    向指定串口发送一个“即发即弃”的指令。

    这个函数只负责发送指令，不等待任何回执，也没有重试机制。
    适用于那些不需要确认的通知性指令。

    Args:
        target_serial: 目标通信串口对象。
        module_name: 调用此函数的模块名称，用于日志打印。
        command: 要发送的指令字符串。
    """
    print(f"[{module_name}模块] 正在向目标设备发送一次性指令: '{command}'")
    try:
        target_serial.write(command.encode('utf-8'))
    except serial.SerialException as e:
        print(f"[{module_name}模块] 错误：串口发送失败，无法写入。原因: {e}") 