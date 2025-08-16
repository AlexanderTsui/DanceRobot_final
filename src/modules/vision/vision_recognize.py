import cv2
import mediapipe as mp
import numpy as np

# --- MediaPipe 初始化 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class VisionProcessor:
    """
    一个封装了 MediaPipe 姿态识别和状态管理的类。
    它被设计为非阻塞的，由主循环驱动。
    """
    def __init__(self, use_flexible_logic: bool = True):
        """
        初始化处理器，打开摄像头并准备 MediaPipe。
        """
        print("[视觉模块] 初始化 VisionProcessor...")
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        self.landmarks = None
        self.use_flexible_logic = use_flexible_logic

        # --- 连续帧检测所需的状态变量 ---
        self.last_action = None
        self.consecutive_frames = 0
        self.REQUIRED_CONSECUTIVE_FRAMES = 3

        if not self.cap.isOpened():
            print("[视觉模块] 错误: 无法打开摄像头。")
            raise IOError("无法打开摄像头")

        print("[视觉模块] 摄像头已成功启动。")
        print(f"[视觉模块] 稳定保持一个动作 {self.REQUIRED_CONSECUTIVE_FRAMES} 帧后将被识别。")

    def _judge_pose(self, landmarks) -> str | None:
        """
        根据姿态关键点判断具体动作。
        
        它会按预设顺序调用一系列独立的检查函数，每个函数负责一种姿态。
        一旦检测到某个姿态，就会立即返回该姿态的名称，不再继续检查。
        这种设计使得添加新姿态或修改现有姿态变得简单且隔离。

        Args:
            landmarks: MediaPipe检测到的姿态关键点。

        Returns:
            如果识别到预设的动作，则返回对应的字符串指令 (例如 "dazizhan")。
            如果没有识别到任何特定动作，则返回 None。
        """
        self.landmarks = landmarks # 将关键点保存为实例变量，方便后续函数访问

        if self.use_flexible_logic:
            # 使用灵活的姿态检查函数 (推荐)
            pose_checks = [
                ("biaixin", self._check_biaixin_flexible),
                ("gongjianbu", self._check_gongjianbu_flexible),
                ("dazizhan", self._check_dazhan_flexible),
                ("jushuangshou", self._check_jushuangshou_flexible),
                ("dunxia", self._check_dunxia)
                # ("fuwocheng", self._check_fuwocheng_flexible),
            ]
        else:
            # 使用原始的姿态检查函数
            pose_checks = [
                ("dazizhan", self._check_dazhan),
                ("gongjianbu", self._check_gongjianbu),
                ("jushuangshou", self._check_jushuangshou),
                ("dunxia", self._check_dunxia),
                ("biaixin", self._check_biaixin)
                # ("fuwocheng", self._check_fuwocheng),
            ]

        for pose_name, check_func in pose_checks:
            if check_func():
                return pose_name
            
        return None

    def _check_dazhan(self):
        """检查“大字站”姿势。
        
        核心逻辑:
        1. 四肢完全伸直。
        2. 手臂与肩同高，水平打开。
        3. 双腿分开，宽度大于肩宽。
        """
        required = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        if not self._check_visibility(required): return False

        # 1. 双臂伸直
        left_arm_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW), self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST))
        right_arm_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW), self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST))
        if not (left_arm_angle > 160 and right_arm_angle > 160): return False

        # 2. 双腿伸直
        left_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP), self._get_coords(mp_pose.PoseLandmark.LEFT_KNEE), self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE))
        right_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP), self._get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE))
        if not (left_leg_angle > 160 and right_leg_angle > 160): return False

        # 3. 手臂打开 (y坐标接近)
        if not (abs(self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] - self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1]) < 0.25 and \
                abs(self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]) < 0.25): return False
        
        # 4. 双腿分开
        shoulder_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])
        ankle_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)[0])
        if not ankle_dist > shoulder_dist * 1.2: return False
        
        return True

    def _check_gongjianbu(self):
        """检查“弓箭步”姿势。
        
        核心逻辑:
        1. 一条腿在前弯曲，另一条腿在后伸直。
        2. 前腿同侧的手臂上举，后腿同侧的手臂下放。
        3. 需要覆盖左弓步和右弓步两种情况。
        """
        required = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
        if not self._check_visibility(required): return False

        # 左右弓步判断
        # Case 1: 左弓步 (左腿在前弯曲)
        left_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP), self._get_coords(mp_pose.PoseLandmark.LEFT_KNEE), self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE))
        right_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP), self._get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE))
        left_arm_up = self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1]
        right_arm_down = self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] > self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]
        arms_straight = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW), self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)) > 150 and self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW), self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)) > 150
        
        is_left_lunge = 90 < left_leg_angle < 165 and right_leg_angle > 160 and left_arm_up and right_arm_down and arms_straight

        # Case 2: 右弓步 (右腿在前弯曲)
        right_arm_up = self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]
        left_arm_down = self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] > self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1]

        is_right_lunge = 90 < right_leg_angle < 165 and left_leg_angle > 160 and right_arm_up and left_arm_down and arms_straight

        return is_left_lunge or is_right_lunge

    def _check_jushuangshou(self):
        """检查“举双手”姿势。
        
        核心逻辑:
        1. 双腿伸直站立且基本并拢。
        2. 双手高举过头。
        """
        required = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
        if not self._check_visibility(required): return False

        left_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP), self._get_coords(mp_pose.PoseLandmark.LEFT_KNEE), self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE))
        right_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP), self._get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE))
        if not (left_leg_angle > 160 and right_leg_angle > 160): return False

        shoulder_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])
        ankle_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)[0])
        if not ankle_dist < shoulder_dist * 1.1: return False

        if not (self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.NOSE)[1] and self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.NOSE)[1]): return False

        if not (self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)[1] < self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1] and self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)[1] < self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]): return False

        return True

    def _check_dunxia(self):
        """检查“蹲下”姿势（侧向蹲防）。
        
        核心逻辑:
        1. 身体半蹲，双腿基本并拢。
        2. 双臂伸直，水平举向身体的一侧。
        3. 通过动态判断可见的身体部位，来兼容正面和侧面视角。
        """
        # 1. 检查躯干可见性，确定后续判断的基准
        left_torso_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP])
        right_torso_visible = self._check_visibility([mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP])

        if not left_torso_visible and not right_torso_visible: return False

        # 2. 动态计算身体中线x坐标、肩膀平均y坐标和肩宽，用于后续容差判断
        center_x, shoulder_y_avg, shoulder_width = 0, 0, 0
        shoulders_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER])

        if shoulders_visible:
            left_shoulder_coord = self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder_coord = self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_hip_coord = self._get_coords(mp_pose.PoseLandmark.LEFT_HIP)
            right_hip_coord = self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
            
            center_x = (left_hip_coord[0] + right_hip_coord[0]) / 2
            shoulder_y_avg = (left_shoulder_coord[1] + right_shoulder_coord[1]) / 2
            shoulder_width = abs(left_shoulder_coord[0] - right_shoulder_coord[0])
        elif left_torso_visible:
            center_x = self._get_coords(mp_pose.PoseLandmark.LEFT_HIP)[0]
            shoulder_y_avg = self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1]
        else: # right_torso_visible
            center_x = self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP)[0]
            shoulder_y_avg = self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]

        # 3. 腿部检查
        visible_legs_count = 0
        legs_are_bent = True
        if self._check_visibility([mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE]):
            visible_legs_count += 1
            left_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP), self._get_coords(mp_pose.PoseLandmark.LEFT_KNEE), self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE))
            if not (70 < left_leg_angle < 160): legs_are_bent = False
        if self._check_visibility([mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]):
            visible_legs_count += 1
            right_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP), self._get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE))
            if not (70 < right_leg_angle < 160): legs_are_bent = False

        if visible_legs_count == 0 or not legs_are_bent: return False
        
        if visible_legs_count == 2 and shoulder_width > 0:
            left_ankle_coord = self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)
            right_ankle_coord = self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)
            if abs(left_ankle_coord[0] - right_ankle_coord[0]) > shoulder_width: return False

        # 4. 手臂检查
        visible_arm_parts = []
        if self._check_visibility([mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]):
            if self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW), self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)) > 150:
                visible_arm_parts.append({"wrist": self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST), "elbow": self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)})
        if self._check_visibility([mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]):
            if self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW), self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)) > 150:
                visible_arm_parts.append({"wrist": self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST), "elbow": self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)})

        if len(visible_arm_parts) == 0: return False

        arms_to_left = all(part["wrist"][0] < center_x and part["elbow"][0] < center_x for part in visible_arm_parts)
        arms_to_right = all(part["wrist"][0] > center_x and part["elbow"][0] > center_x for part in visible_arm_parts)

        if not (arms_to_left or arms_to_right): return False

        if shoulder_width > 0:
            height_tolerance = shoulder_width * 0.5
            for part in visible_arm_parts:
                if abs(part["wrist"][1] - shoulder_y_avg) > height_tolerance: return False

        return True

    def _check_biaixin(self):
        """检查“比爱心”姿势（侧弓步爱心）。
        
        核心逻辑:
        1. 身体重心下移，一条腿弯曲支撑，另一条腿向侧方伸直。
        2. 双手在头顶靠近，形成爱心状。
        3. 覆盖左腿支撑和右腿支撑两种情况。
        """
        required = [mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
        if not self._check_visibility(required): return False

        # 腿部姿态
        left_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP), self._get_coords(mp_pose.PoseLandmark.LEFT_KNEE), self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE))
        right_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP), self._get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE))
        left_leg_bent = 90 < left_leg_angle < 165 and right_leg_angle > 160
        right_leg_bent = 90 < right_leg_angle < 165 and left_leg_angle > 160
        if not (left_leg_bent or right_leg_bent): return False

        # 手臂姿态
        if not (self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.LEFT_EYE)[1] and self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.RIGHT_EYE)[1]): return False

        shoulder_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])
        wrist_dist = np.linalg.norm(np.array(self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)) - np.array(self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)))
        if not wrist_dist < shoulder_dist: return False

        left_elbow_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW), self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST))
        right_elbow_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW), self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST))
        if not (70 < left_elbow_angle < 130 and 70 < right_elbow_angle < 130): return False

        return True

    def _check_fuwocheng(self):
        """检查“俯卧撑”姿势。
        
        核心逻辑:
        1. 身体从肩到脚踝呈一条直线。
        2. 身体姿态基本与地面平行。
        3. 由手臂在身前支撑。
        4. 此检测强依赖于用户侧对摄像头。
        """
        required = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]
        if not self._check_visibility(required): return False

        # 计算中点
        shoulder_mid = [(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] + self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0]) / 2, (self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1] + self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]) / 2]
        hip_mid = [(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP)[0] + self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP)[0]) / 2, (self._get_coords(mp_pose.PoseLandmark.LEFT_HIP)[1] + self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP)[1]) / 2]
        ankle_mid = [(self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)[0] + self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)[0]) / 2, (self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)[1] + self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)[1]) / 2]

        body_angle = self._calculate_angle(shoulder_mid, hip_mid, ankle_mid)
        if not body_angle > 160: return False

        # 身体水平
        if not (abs(shoulder_mid[1] - hip_mid[1]) < 0.15 and abs(hip_mid[1] - ankle_mid[1]) < 0.15): return False

        # 手臂支撑
        if not (self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] > self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1] and self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] > self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]): return False

        return True


    # =================================================================================
    # V2 - 灵活识别逻辑 (Flexible Recognition Logic)
    # =================================================================================

    def _check_biaixin_upper_body(self):
        """检查“比爱心”姿势的上半身组件。"""
        # 核心逻辑: 双手在头顶靠近，形成爱心状。
        required = [
            mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE, 
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, 
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW, 
            mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        if not self._check_visibility(required): return False

        # 1. 双手高举过眼
        if not (self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.LEFT_EYE)[1] and \
                self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.RIGHT_EYE)[1]): 
            return False

        # 2. 双手靠近
        shoulder_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])
        wrist_dist = np.linalg.norm(np.array(self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)) - np.array(self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)))
        if not wrist_dist < shoulder_dist: 
            return False

        # 3. 手肘弯曲形成爱心弧度
        left_elbow_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW), self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST))
        right_elbow_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW), self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST))
        if not (70 < left_elbow_angle < 130 and 70 < right_elbow_angle < 130): 
            return False

        return True

    def _check_biaixin_flexible(self):
        """[灵活版] 检查“比爱心”姿势。
        决策逻辑：上半身的“比心”手势是决定性特征，只要满足就判定成功。
        """
        if self._check_biaixin_upper_body():
            return True
        return False

    def _check_gongjianbu_lower_body(self):
        """检查“弓箭步”姿势的下半身组件。"""
        # 核心逻辑: 标准弓步，前腿弯后腿直
        required = [
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        if not self._check_visibility(required): return False

        left_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP), self._get_coords(mp_pose.PoseLandmark.LEFT_KNEE), self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE))
        right_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP), self._get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE))

        is_left_lunge = 30 < left_leg_angle < 150 and right_leg_angle > 160 # 左弓步 (左腿在前弯曲)
        is_right_lunge = 30 < right_leg_angle < 150 and left_leg_angle > 160 # 右弓步 (右腿在前弯曲)

        return is_left_lunge or is_right_lunge

    def _check_gongjianbu_flexible(self):
        """[灵活版] 检查“弓箭步”姿势。
        决策逻辑：下半身的“弓步”姿态是决定性特征，只要满足就判定成功。
        """
        if self._check_gongjianbu_lower_body():
            return True
        return False

    def _check_dazhan_upper_body(self):
        """检查“大字站”姿势的上半身组件。"""
        # 核心逻辑: 双臂水平伸直打开
        required = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        if not self._check_visibility(required): return False

        # 1. 双臂伸直
        left_arm_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW), self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST))
        right_arm_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW), self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST))
        if not (left_arm_angle > 140 and right_arm_angle > 140): return False

        # 2. 手臂水平打开 (y坐标接近)
        if not (abs(self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] - self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1]) < 0.35 and \
                abs(self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]) < 0.35): return False
        
        return True

    def _check_dazhan_lower_body(self):
        """检查“大字站”姿势的下半身组件。"""
        # 核心逻辑: 双腿伸直，分开宽度大于肩宽
        required = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        if not self._check_visibility(required): return False

        # 1. 双腿伸直
        left_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP), self._get_coords(mp_pose.PoseLandmark.LEFT_KNEE), self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE))
        right_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP), self._get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE))
        if not (left_leg_angle > 140 and right_leg_angle > 140): return False

        # 2. 双腿分开
        shoulder_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])
        ankle_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)[0])
        if not ankle_dist > shoulder_dist * 1.1: return False

        return True

    def _check_dazhan_flexible(self):
        """[灵活版] 检查“大字站”姿势。
        决策逻辑：上下半身特征都较强，只要可见的部分匹配，就判定成功。
        """
        upper_body_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST], mode='any')
        lower_body_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE], mode='any')

        if not upper_body_visible and not lower_body_visible:
            return False

        # 只要可见的部分不匹配，就判定失败
        if upper_body_visible and not self._check_dazhan_upper_body():
            return False
        if lower_body_visible and not self._check_dazhan_lower_body():
            return False

        return True

    def _check_jushuangshou_upper_body(self):
        """检查“举双手”姿势的上半身组件。"""
        # 核心逻辑: 双手高举过头，手臂基本伸直
        required = [
            mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        if not self._check_visibility(required): return False

        # 1. 手臂高举过头
        if not (self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.NOSE)[1] and \
                self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] < self._get_coords(mp_pose.PoseLandmark.NOSE)[1]): 
            return False

        # 2. 肘部也要抬起
        if not (self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)[1] < self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1] and \
                self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)[1] < self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]): 
            return False

        # 3. 为了和比爱心区分，手臂要比较直
        left_arm_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW), self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST))
        right_arm_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW), self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST))
        if not (left_arm_angle > 140 and right_arm_angle > 140): return False

        return True

    def _check_jushuangshou_lower_body(self):
        """检查“举双手”姿势的下半身组件。"""
        # 核心逻辑: 双腿站直并拢
        required = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        if not self._check_visibility(required): return False
        
        # 1. 双腿伸直
        left_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP), self._get_coords(mp_pose.PoseLandmark.LEFT_KNEE), self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE))
        right_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP), self._get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE))
        if not (left_leg_angle > 160 and right_leg_angle > 160): return False

        # 2. 双腿并拢
        shoulder_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])
        ankle_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)[0])
        if not ankle_dist < shoulder_dist * 1.1: return False
        
        return True

    def _check_jushuangshou_flexible(self):
        """[灵活版] 检查“举双手”姿势。
        决策逻辑：上半身的“举手”姿态是决定性特征。为提高准确性，当下半身可见但不匹配时，判定失败。
        """
        if self._check_jushuangshou_upper_body():
            # 如果下半身可见，则必须匹配站直姿态，否则可能误判（例如蹲下时举手）
            lower_body_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE], mode='any')
            if lower_body_visible and not self._check_jushuangshou_lower_body():
                return False
            return True
        return False

    def _check_dunxia_upper_body(self):
        """检查“蹲下”姿势的上半身组件。"""
        # 核心逻辑: 双臂伸直，水平伸向身体同一侧
        
        # 1. 确定身体中心线
        left_torso_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP])
        right_torso_visible = self._check_visibility([mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP])
        if not left_torso_visible and not right_torso_visible: return False

        center_x, shoulder_y_avg, shoulder_width = 0, 0, 0
        shoulders_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER])

        if shoulders_visible:
            center_x = (self._get_coords(mp_pose.PoseLandmark.LEFT_HIP)[0] + self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP)[0]) / 2
            shoulder_y_avg = (self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1] + self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]) / 2
            shoulder_width = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])
        elif left_torso_visible:
            center_x = self._get_coords(mp_pose.PoseLandmark.LEFT_HIP)[0]
            shoulder_y_avg = self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1]
        else: # right_torso_visible
            center_x = self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP)[0]
            shoulder_y_avg = self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]
        
        # 2. 手臂检查
        visible_arm_parts = []
        if self._check_visibility([mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]):
            if self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW), self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)) > 150:
                visible_arm_parts.append({"wrist": self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST), "elbow": self._get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)})
        if self._check_visibility([mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]):
            if self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW), self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)) > 150:
                visible_arm_parts.append({"wrist": self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST), "elbow": self._get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)})

        if len(visible_arm_parts) == 0: return False # 至少一条手臂可见且伸直

        # 3. 手臂同向
        arms_to_left = all(part["wrist"][0] < center_x and part["elbow"][0] < center_x for part in visible_arm_parts)
        arms_to_right = all(part["wrist"][0] > center_x and part["elbow"][0] > center_x for part in visible_arm_parts)
        if not (arms_to_left or arms_to_right): return False

        # 4. 手臂同高 (仅当双肩可见时检查)
        if shoulder_width > 0:
            height_tolerance = shoulder_width * 0.5
            for part in visible_arm_parts:
                if abs(part["wrist"][1] - shoulder_y_avg) > height_tolerance: return False
        
        return True

    def _check_dunxia_lower_body(self):
        """检查“蹲下”姿势的下半身组件。"""
        # 核心逻辑: 身体半蹲，双腿基本并拢
        
        # 1. 腿部检查
        visible_legs_count = 0
        legs_are_bent = True
        if self._check_visibility([mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE]):
            visible_legs_count += 1
            left_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP), self._get_coords(mp_pose.PoseLandmark.LEFT_KNEE), self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE))
            if not (70 < left_leg_angle < 160): legs_are_bent = False
        if self._check_visibility([mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]):
            visible_legs_count += 1
            right_leg_angle = self._calculate_angle(self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP), self._get_coords(mp_pose.PoseLandmark.RIGHT_KNEE), self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE))
            if not (70 < right_leg_angle < 160): legs_are_bent = False

        if visible_legs_count == 0 or not legs_are_bent: return False
        
        # 2. 双腿并拢 (仅当双腿和双肩均可见时检查)
        if visible_legs_count == 2 and self._check_visibility([mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]):
            shoulder_width = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0])
            if shoulder_width > 0:
                ankle_dist = abs(self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)[0] - self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)[0])
                if ankle_dist > shoulder_width: return False
        
        return True

    def _check_dunxia_flexible(self):
        """[灵活版] 检查“蹲下”姿势。
        决策逻辑：这是一个关联性很强的动作，所有可见的部分都必须匹配才判定成功。
        """
        # 检查基本的躯干可见性
        torso_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                                                 mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                                 mode='any')
        if not torso_visible: return False

        upper_body_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST], mode='any')
        lower_body_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE], mode='any')

        if not upper_body_visible and not lower_body_visible:
            return False

        # 只要有一个部分可见但不匹配，就失败
        if (upper_body_visible and not self._check_dunxia_upper_body()) or \
           (lower_body_visible and not self._check_dunxia_lower_body()):
            return False
            
        return True

    def _check_fuwocheng_flexible(self):
        """[灵活版] 检查“俯卧撑”姿势。
        决策逻辑：这是一个整体性动作，无法简单拆分。逻辑会根据可见关键点动态判断。
        """
        # 1. 检查核心躯干和手臂支撑的可见性
        torso_required = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, 
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
        ]
        arms_required = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]
        if not self._check_visibility(torso_required) or not self._check_visibility(arms_required):
            return False

        # 2. 手臂必须在身前支撑
        if not (self._get_coords(mp_pose.PoseLandmark.LEFT_WRIST)[1] > self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1] and \
                self._get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)[1] > self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]): 
            return False

        # 3. 计算身体中点
        shoulder_mid = [(self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[0] + self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[0]) / 2, 
                        (self._get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)[1] + self._get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)[1]) / 2]
        hip_mid = [(self._get_coords(mp_pose.PoseLandmark.LEFT_HIP)[0] + self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP)[0]) / 2, 
                   (self._get_coords(mp_pose.PoseLandmark.LEFT_HIP)[1] + self._get_coords(mp_pose.PoseLandmark.RIGHT_HIP)[1]) / 2]
        
        # 4. 身体必须基本水平
        if not (abs(shoulder_mid[1] - hip_mid[1]) < 0.25): return False

        # 5. (可选) 如果腿部可见，则检查身体是否成一条直线
        legs_visible = self._check_visibility([mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
                                               mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE])
        if legs_visible:
            ankle_mid = [(self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)[0] + self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)[0]) / 2, 
                         (self._get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)[1] + self._get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)[1]) / 2]
            
            body_angle = self._calculate_angle(shoulder_mid, hip_mid, ankle_mid)
            if not body_angle > 160: return False
            # 进一步检查腿部是否也水平
            if not abs(hip_mid[1] - ankle_mid[1]) < 0.15: return False
        
        return True


    def _get_coords(self, landmark_name) -> tuple[float, float, float] | None:
        """安全地获取指定 landmark 的坐标。"""
        if self.landmarks is None:
            return None
        
        landmark = self.landmarks.landmark[landmark_name]
        # 增加一个基础的可见性检查，如果点几乎不可见，则不返回坐标
        if landmark.visibility < 0.1:
            return None
            
        return (landmark.x, landmark.y, landmark.z)

    def _check_visibility(self, landmarks_to_check: list, mode: str = 'all') -> bool:
        """
        检查一组关键点是否都可见。

        Args:
            landmarks_to_check: 需要检查的 MediaPipe PoseLandmark 列表。
            mode (str): 'all' 表示所有点都必须可见, 'any' 表示至少一个点可见。

        Returns:
            如果满足可见性要求，则返回 True，否则返回 False。
        """
        if self.landmarks is None:
            return False
            
        visible_landmarks = [
            landmark for landmark in landmarks_to_check
            if self.landmarks.landmark[landmark].visibility > 0.6
        ]
        
        if mode == 'all':
            return len(visible_landmarks) == len(landmarks_to_check)
        elif mode == 'any':
            return len(visible_landmarks) > 0
        else:
            return False

    def _calculate_angle(self, a, b, c) -> float:
        """
        计算由三点构成的角度（以b为顶点）。
        
        Args:
            a, b, c: 每个点都是包含x, y, (可选z)坐标的元组或列表。
        
        Returns:
            返回0到180之间的角度值。
        """
        if a is None or b is None or c is None:
            return 0.0

        # 将输入转换为numpy数组以进行向量运算
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        # 计算从b到a和从b到c的向量
        ba = a - b
        bc = c - b

        # 计算两个向量的点积和模
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # 确保cosine_angle在[-1, 1]范围内，防止浮点数误差
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        # 计算角度的弧度值，然后转换为度
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def process_one_frame(self) -> str | None:
        """
        处理单帧图像：进行姿态识别、更新状态、显示画面。
        如果识别成功（满足连续帧数），则返回指令字符串。
        否则，返回 None。
        """
        command = None
        
        success, image = self.cap.read()
        if not success:
            print("[视觉模块] 警告: 无法从摄像头读取画面。")
            return None

        # 图像处理与姿态检测
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 颜色空间转换
        image_rgb.flags.writeable = False # 为提高mediapipe的性能，设置为不可写
        results = self.pose.process(image_rgb) # 姿态检测
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # 颜色空间转换
        image_bgr.flags.writeable = True # 恢复可写

        # 核心逻辑：连续帧判断
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            current_action = self._judge_pose(results.pose_landmarks)

            # 如果识别到动作，则更新计数器
            if current_action:
                if current_action == self.last_action: # 如果当前动作与上一帧相同，则计数器加1
                    self.consecutive_frames += 1
                else:
                    self.last_action = current_action # 如果当前动作与上一帧不同，则重置计数器并更新上一帧动作
                    self.consecutive_frames = 1
            
            # 如果未识别到动作，则重置计数器并更新上一帧动作
            else: 
                self.last_action = None
                self.consecutive_frames = 0

            # 如果连续帧数达到要求，则返回指令字符串
            if self.consecutive_frames >= self.REQUIRED_CONSECUTIVE_FRAMES:
                command = self.last_action
                print(f"[视觉模块] 动作确认: {command} (连续 {self.REQUIRED_CONSECUTIVE_FRAMES} 帧)")
                # 识别成功后重置，防止重复触发
                self.last_action = None
                self.consecutive_frames = 0
        else:
            self.last_action = None
            self.consecutive_frames = 0

        # 在画面上显示调试信息
        status_text = f"Action: {self.last_action} ({self.consecutive_frames}/{self.REQUIRED_CONSECUTIVE_FRAMES})"
        cv2.putText(image_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Pose - Dance Robot', image_bgr)
        
        # 必须有 waitKey 才能让 imshow 正常工作
        cv2.waitKey(1)

        return command
        
    def hide_window(self):
        """
        仅关闭OpenCV的显示窗口，而不释放摄像头等核心资源。
        """
        try:
            print("[视觉模块] 隐藏显示窗口。")
            cv2.destroyWindow('MediaPipe Pose - Dance Robot')
            # 在某些GUI后端，需要一个waitKey来处理窗口关闭事件
            cv2.waitKey(1)
        except Exception as e:
            # 如果窗口已经被关闭，destroyWindow会引发异常，这里可以安全地忽略它
            print(f"[视觉模块] 关闭窗口时发生异常 (可能是窗口已关闭): {e}")
        
    def cleanup(self):
        """
        释放摄像头资源并关闭所有OpenCV窗口。
        """
        print("[视觉模块] 关闭摄像头并清理资源...")
        if self.cap.isOpened():
            self.cap.release()
        self.pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 此代码块用于独立测试 VisionProcessor 模块

    processor = None
    try:
        processor = VisionProcessor()
        print("\n--- 开始持续姿态检测 (按 'q' 键或 Ctrl+C 退出) ---")
        while True:
            # process_one_frame 会自动显示画面
            action = processor.process_one_frame()
            if action:
                print(f"检测到动作: {action}")

            # 为了能够通过 'q' 键退出，需要在主循环中检查
            # 虽然 process_one_frame 内部有 waitKey(1)，但如果想外部控制，可以在此添加逻辑
            # 例如: if cv2.waitKey(1) & 0xFF == ord('q'): break
            # 但通常情况下，直接运行此文件时，通过关闭窗口或Ctrl+C退出更方便

    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if processor:
            processor.cleanup()
        print("程序结束。") 
