import json
import os
import time
import argparse
import ctypes
import configparser
import subprocess
import sys
import threading

import cv2
import dxcam
import numpy as np
import onnxruntime as ort

from aim_core import Candidate
from box_tracker import BoxTracker
from mouse_transport import LatestMouseSender
from target_policy import class_boxes, StandingBodyFilter
from person_aim import (StickyBodyTracker, DampedController, TargetLead,
                        AimSanityGate, match_head, person_target)
from win_serial import WinSerial
from status_overlay_client import StatusOverlay
from yellow_proposals import find_yellow_proposals, recheck_yellow, zoom_locked_target
from mixed_aim import MixedAimPolicy
from crop_policy import choose_color_candidate, should_zoom_lock
from motion_predictor import BackgroundFlow, PredictedBridge


parser = argparse.ArgumentParser(description="640px per-person locked aim experiment")
parser.add_argument("--model", default="models/瓦黄v11s640.onnx")
parser.add_argument("--serial-port", default=None, help="例如 COM4；不传时只预览计算结果")
parser.add_argument("--baud", type=int, default=115200)
parser.add_argument("--window-title", default="VALORANT", help="仅在此窗口处于前台时允许输出")
parser.add_argument("--kp", type=float, default=0.06, help="ROI 像素到鼠标步数的比例")
parser.add_argument("--ki", type=float, default=0.0)
parser.add_argument("--kd", type=float, default=0.0)
parser.add_argument("--max-step", type=int, default=12, help="单帧每轴步数上限")
parser.add_argument("--headless", action="store_true", help="不显示预览窗口，供限帧验证")
parser.add_argument("--interactive-preview", action="store_true",
                    help="预览窗口可接收鼠标，用于手动校准；正常游戏时不要启用")
parser.add_argument("--max-frames", type=int, default=0, help="处理指定帧数后退出；0 为持续运行")
parser.add_argument("--image", default=None, help="使用本地截图回放检测流程，不访问桌面采集")
parser.add_argument("--video", default=None, help="逐帧回放本地视频，不访问桌面采集")
parser.add_argument("--simulate-control", action="store_true",
                    help="仅视频回放：运行瞄准决策但不连接串口")
parser.add_argument("--config", default=None, help="INI 参数文件；其中的设置覆盖命令行默认值")
args = parser.parse_args()
if args.image and args.video:
    parser.error("--image 与 --video 只能选择一个")
if args.simulate_control and not args.video:
    parser.error("--simulate-control 仅用于 --video 离线回放")

args.deadzone = 2.0
args.lock_radius = 42.0
args.lock_misses = 3
args.switch_misses = 2
args.far_boost = 1.0
args.far_threshold = 120.0
args.near_boost = 1.0
args.near_radius = 90.0
args.lead_ms = 35.0
args.confidence = 0.35
args.nms_threshold = 0.45
args.capture_fps = 240
args.split_merged = True
args.secondary_inference = False  # Fast crouch test: one full-frame YOLO pass only.
args.color_sentinel = True  # Replace, never add, one full-frame pass on color probe frames.
args.motion_compensation = True
args.prediction_bridge = True
args.trigger_cooldown_ms = 35.0
args.trigger_radius_ratio = 0.45
if args.config:
    config = configparser.ConfigParser()
    if not config.read(args.config, encoding="utf-8-sig"):
        parser.error(f"找不到参数文件: {args.config}")
    args.kp = config.getfloat("control", "kp", fallback=args.kp)
    args.ki = config.getfloat("control", "ki", fallback=args.ki)
    args.kd = config.getfloat("control", "kd", fallback=args.kd)
    args.max_step = config.getint("control", "max_step", fallback=args.max_step)
    args.deadzone = config.getfloat("control", "deadzone", fallback=args.deadzone)
    args.far_boost = config.getfloat("control", "far_boost", fallback=args.far_boost)
    args.far_threshold = config.getfloat("control", "far_threshold", fallback=args.far_threshold)
    args.near_boost = config.getfloat("control", "near_boost", fallback=args.near_boost)
    args.near_radius = config.getfloat("control", "near_radius", fallback=args.near_radius)
    args.lead_ms = config.getfloat("control", "lead_ms", fallback=args.lead_ms)
    args.lock_radius = config.getfloat("tracking", "lock_radius", fallback=args.lock_radius)
    args.lock_misses = config.getint("tracking", "lock_misses", fallback=args.lock_misses)
    args.switch_misses = config.getint("tracking", "switch_misses", fallback=args.switch_misses)
    args.confidence = config.getfloat("detector", "confidence", fallback=args.confidence)
    args.nms_threshold = config.getfloat("detector", "nms_threshold", fallback=args.nms_threshold)
    args.capture_fps = config.getint("detector", "capture_fps", fallback=args.capture_fps)
    args.split_merged = config.getboolean("detector", "split_merged", fallback=args.split_merged)
    args.secondary_inference = config.getboolean("detector", "secondary_inference", fallback=args.secondary_inference)
    args.color_sentinel = config.getboolean("detector", "color_sentinel", fallback=args.color_sentinel)
    args.motion_compensation = config.getboolean("detector", "motion_compensation",
                                                  fallback=args.motion_compensation)
    args.prediction_bridge = config.getboolean("detector", "prediction_bridge",
                                                fallback=args.prediction_bridge)
    args.trigger_cooldown_ms = config.getfloat("trigger", "cooldown_ms", fallback=args.trigger_cooldown_ms)
    args.trigger_radius_ratio = config.getfloat("trigger", "radius_ratio", fallback=args.trigger_radius_ratio)
    args.serial_port = (config.get("device", "port", fallback=args.serial_port) or "").strip() or None
    args.baud = config.getint("device", "baud", fallback=args.baud)
    args.window_title = config.get("device", "window_title", fallback=args.window_title).strip()

if not (0 < args.kp <= 1 and 0 <= args.ki <= 1 and 0 <= args.kd <= 1
        and 1 <= args.max_step <= 240 and 0 <= args.deadzone <= 50
        and 1 <= args.far_boost <= 3 and 10 <= args.far_threshold <= 500
        and 1 <= args.near_boost <= 2.5 and 20 <= args.near_radius <= 300
        and 0 <= args.lead_ms <= 80
        and 1 <= args.lock_radius <= 160 and 1 <= args.lock_misses <= 20
        and 1 <= args.switch_misses <= 20
        and 0 < args.confidence < 1 and 0 < args.nms_threshold < 1
        and 1 <= args.capture_fps <= 500
        and 10 <= args.trigger_cooldown_ms <= 250
        and 0.10 <= args.trigger_radius_ratio <= 0.45):
    parser.error("参数文件存在超出范围的设置")
print(f"参数: kp={args.kp:g} ki={args.ki:g} kd={args.kd:g} "
      f"max_step={args.max_step} deadzone={args.deadzone:g} "
      f"lock_radius={args.lock_radius:g} lock_misses={args.lock_misses}")


def reload_live_config():
    """Apply tunable values after the UI atomically saves the INI file."""
    global CONF, NMS_THRESHOLD, SPLIT_MERGED, tracker, box_tracker, controller, lead_tracker, safety_gate, armed
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config, encoding="utf-8-sig"):
        raise FileNotFoundError(args.config)
    values = {
        "kp": cfg.getfloat("control", "kp", fallback=args.kp),
        "ki": cfg.getfloat("control", "ki", fallback=args.ki),
        "kd": cfg.getfloat("control", "kd", fallback=args.kd),
        "max_step": cfg.getint("control", "max_step", fallback=args.max_step),
        "deadzone": cfg.getfloat("control", "deadzone", fallback=args.deadzone),
        "far_boost": cfg.getfloat("control", "far_boost", fallback=args.far_boost),
        "far_threshold": cfg.getfloat("control", "far_threshold", fallback=args.far_threshold),
        "near_boost": cfg.getfloat("control", "near_boost", fallback=args.near_boost),
        "near_radius": cfg.getfloat("control", "near_radius", fallback=args.near_radius),
        "lead_ms": cfg.getfloat("control", "lead_ms", fallback=args.lead_ms),
        "lock_radius": cfg.getfloat("tracking", "lock_radius", fallback=args.lock_radius),
        "lock_misses": cfg.getint("tracking", "lock_misses", fallback=args.lock_misses),
        "switch_misses": cfg.getint("tracking", "switch_misses", fallback=args.switch_misses),
        "confidence": cfg.getfloat("detector", "confidence", fallback=args.confidence),
        "nms_threshold": cfg.getfloat("detector", "nms_threshold", fallback=args.nms_threshold),
        "split_merged": cfg.getboolean("detector", "split_merged", fallback=args.split_merged),
        "trigger_cooldown_ms": cfg.getfloat("trigger", "cooldown_ms", fallback=args.trigger_cooldown_ms),
        "trigger_radius_ratio": cfg.getfloat("trigger", "radius_ratio", fallback=args.trigger_radius_ratio),
    }
    if not (0 < values["kp"] <= 1 and 0 <= values["ki"] <= 1
            and 0 <= values["kd"] <= 1 and 1 <= values["max_step"] <= 240
            and 0 <= values["deadzone"] <= 50
            and 1 <= values["far_boost"] <= 3
            and 10 <= values["far_threshold"] <= 500
            and 1 <= values["near_boost"] <= 2.5
            and 20 <= values["near_radius"] <= 300
            and 0 <= values["lead_ms"] <= 80
            and 1 <= values["lock_radius"] <= 160
            and 1 <= values["lock_misses"] <= 20
            and 1 <= values["switch_misses"] <= 20
            and 0 < values["confidence"] < 1
            and 0 < values["nms_threshold"] < 1
            and 10 <= values["trigger_cooldown_ms"] <= 250
            and 0.10 <= values["trigger_radius_ratio"] <= 0.45):
        raise ValueError("参数超出范围")
    for key, value in values.items():
        setattr(args, key, value)
    CONF = args.confidence
    NMS_THRESHOLD = args.nms_threshold
    SPLIT_MERGED = args.split_merged
    tracker = StickyBodyTracker()
    box_tracker = BoxTracker(match_radius=args.lock_radius, max_prediction_frames=3)
    lead_tracker = TargetLead()
    safety_gate = AimSanityGate()
    background_flow.reset()
    prediction_bridge.reset()
    controller = DampedController(kp=args.kp, ki=args.ki, deadzone=args.deadzone,
                                  max_step=args.max_step, far_boost=args.far_boost,
                                  far_threshold=args.far_threshold,
                                  near_boost=args.near_boost, near_radius=args.near_radius)
    print(f"参数已应用: kp={args.kp:g}, max_step={args.max_step}, "
          f"deadzone={args.deadzone:g}；瞄准状态保持不变")


def foreground_title():
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    title = ctypes.create_unicode_buffer(512)
    ctypes.windll.user32.GetWindowTextW(hwnd, title, len(title))
    return title.value


def is_game_title(title):
    if not args.window_title:
        return True
    return title.strip().casefold() == args.window_title.strip().casefold()


def game_focused():
    return is_game_title(foreground_title())


def make_preview_click_through():
    """Keep the small topmost preview visible without stealing game clicks."""
    user32 = ctypes.windll.user32
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype = ctypes.c_void_p
    hwnd = user32.FindWindowW(None, WINDOW_NAME)
    if not hwnd:
        return False
    user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    user32.SetWindowLongW.restype = ctypes.c_long
    style = user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
    style |= 0x00000020 | 0x00080000 | 0x08000000  # TRANSPARENT | LAYERED | NOACTIVATE
    user32.SetWindowLongW(hwnd, -20, style)
    user32.SetLayeredWindowAttributes(hwnd, 0, 255, 0x02)  # LWA_ALPHA, opaque
    user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), 20, 40, 0, 0,
                        0x0001 | 0x0010 | 0x0020)  # NOSIZE | NOACTIVATE | FRAMECHANGED
    return bool(user32.GetWindowLongW(hwnd, -20) & 0x00000020)


_key_down = {}


def hotkey_pressed(vk):
    state = ctypes.windll.user32.GetAsyncKeyState(vk)
    down = bool(state & 0x8000)
    # 低位补捉两次轮询之间已经按下又松开的短按。
    pressed = bool(state & 0x0001) or (down and not _key_down.get(vk, False))
    _key_down[vk] = down
    return pressed


class MouseTransport:
    def __init__(self, port):
        self.serial = None
        self.sender = None
        if port:
            # Native USB CDC may briefly remain busy after a monitor closes or
            # the board re-enumerates. Retry only transient port errors.
            for attempt in range(6):
                try:
                    self.serial = WinSerial(port, args.baud)
                    break
                except OSError as exc:
                    if getattr(exc, "winerror", None) not in (2, 5) or attempt == 5:
                        raise
                    print(f"{port} 暂不可用（{exc}），1 秒后重试 {attempt + 1}/5", flush=True)
                    time.sleep(1.0)
            time.sleep(2.0)  # ESP32-S3 USB 串口打开后可能重新启动
            self.sender = LatestMouseSender(self.serial.write)

    def send(self, dx, dy):
        if self.sender is None:
            return 0
        return self.sender.send(dx, dy)

    def click(self):
        if self.sender is None:
            raise RuntimeError("F1 自动扳机需要 ESP32 串口")
        self.sender.click()

    def cancel(self):
        if self.sender is not None:
            self.sender.cancel()

    def close(self):
        if self.sender is not None:
            self.sender.close()
        if self.serial is not None:
            self.serial.close()


class ImageCamera:
    def __init__(self, path):
        self.frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.frame is None:
            raise FileNotFoundError(path)
        self.region = None

    def grab(self):
        return self.frame

    def start(self, region, target_fps):
        self.region = region

    def get_latest_frame(self):
        left, top, right, bottom = self.region
        return self.frame[top:bottom, left:right].copy()

    def stop(self):
        pass


class VideoCamera:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise FileNotFoundError(path)
        self.region = None

    def grab(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def start(self, region, target_fps):
        self.region = region

    def get_latest_frame(self):
        frame = self.grab()
        if frame is None:
            return None
        left, top, right, bottom = self.region
        return frame[top:bottom, left:right].copy()

    def stop(self):
        self.capture.release()

# ================= 配置 =================

MODEL_PATH = args.model

INPUT_SIZE = 640
ROI_SIZE = 960
PREVIEW_SIZE = 360  # Native window + borders remain left of the 960px capture ROI.

CONF = args.confidence
NMS_THRESHOLD = args.nms_threshold

CAPTURE_FPS = args.capture_fps
SHOW_WINDOW = not args.headless

WINDOW_NAME = "YOLO GPU"

# ================= 粘连目标拆分配置 =================

SPLIT_MERGED = args.split_merged
SPLIT_RATIO = 1.6        # 框的长边/短边 超过这个值才认为是多个球粘在一起

# ================= 准星校准配置 =================

CALIB_FILE = "aim_calib.json"   # Per-user calibration; excluded from source archive.

AUTO_CAL_DELAY = 6.0     # 按下 A 后的准备时间（秒）：用来切回游戏，这段时间不采样
AUTO_CAL_SECONDS = 5.0   # 之后的采样时长（秒）：期间在游戏里晃动鼠标

EDGE_THR = 18            # 边缘阈值：准星对比度低、找不到时调小（如 12）
PERSIST_RATIO = 0.80     # 某个边缘像素在多少比例的帧里都存在，才算"固定在屏幕上的东西"
MIN_BLOB = 6             # 准星候选区域最小尺寸（原图像素）
MAX_BLOB = 60            # 最大尺寸，用来排除枪、HUD 大块区域

aim_x = INPUT_SIZE / 2.0    # 320x320 预览图坐标系下的准星位置
aim_y = INPUT_SIZE / 2.0

calibration_mode = False    # True：手动校准模式（隐藏标记，点击准星）
calibrated = False


def save_calib():
    if not args.interactive_preview:
        return  # Gameplay preview must never overwrite the known-good crosshair.
    try:
        with open(CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"region": list(region), "aim_x": aim_x, "aim_y": aim_y},
                f,
            )
        print(f"校准结果已保存到 {CALIB_FILE}")
    except Exception as e:
        print("保存校准结果失败:", e)


def load_calib():
    global aim_x, aim_y, calibrated
    if not os.path.exists(CALIB_FILE):
        calibrated = True
        print("未找到校准文件，先使用检测区域中心作为准星；需要时再单独校准")
        return
    try:
        with open(CALIB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("region") == list(region):
            loaded_x = float(data["aim_x"])
            loaded_y = float(data["aim_y"])
            if (not np.isfinite(loaded_x) or not np.isfinite(loaded_y)
                    or abs(loaded_x - INPUT_SIZE / 2) > 80
                    or abs(loaded_y - INPUT_SIZE / 2) > 80):
                print("校准坐标偏离屏幕中心，已使用中心准星")
                loaded_x = loaded_y = INPUT_SIZE / 2
            aim_x, aim_y = loaded_x, loaded_y
            calibrated = True
            print(f"已读取上次的校准结果: ({aim_x:.1f}, {aim_y:.1f})")
        else:
            print("检测区域与上次不同，忽略旧校准结果，请按 A 重新校准")
    except Exception as e:
        print("读取校准文件失败:", e)


def on_mouse(event, x, y, flags, userdata):
    global aim_x, aim_y, calibration_mode, calibrated

    if event == cv2.EVENT_LBUTTONDOWN and calibration_mode:
        aim_x = float(np.clip(x * INPUT_SIZE / PREVIEW_SIZE, 0, INPUT_SIZE - 1))
        aim_y = float(np.clip(y * INPUT_SIZE / PREVIEW_SIZE, 0, INPUT_SIZE - 1))
        calibrated = True
        calibration_mode = False
        print(f"\n手动校准成功！准星坐标 = ({aim_x:.0f}, {aim_y:.0f})")
        save_calib()


# ================= 自动准星校准 =================

class AutoCalibrator:
    """
    原理：游戏准星固定在屏幕同一位置，而背景/靶子在晃动鼠标时会移动。
    采样期间统计"每个像素位置上出现高对比度边缘的帧占比"，
    在所有帧里都是边缘的位置就是固定在屏幕上的元素（准星、枪、HUD）。
    再按尺寸筛掉枪和 HUD 的大块区域，取最靠近画面中心的小块，其质心就是准星。

    流程：按 A -> 准备倒计时(AUTO_CAL_DELAY，不采样) -> 采样(AUTO_CAL_SECONDS) -> 出结果
    """

    def __init__(self):
        self.active = False

    def start(self):
        self.active = True
        self.t_begin = time.perf_counter()
        self.t_sample = None
        self.acc = None
        self.n = 0
        self.first = None
        self.last = None

    def status(self):
        """返回给预览窗口显示的状态文字。"""
        now = time.perf_counter()
        wait = AUTO_CAL_DELAY - (now - self.t_begin)
        if wait > 0:
            return f"AutoCal: go back to game {wait:.0f}s"
        p = 0.0 if self.t_sample is None else min(
            1.0, (now - self.t_sample) / AUTO_CAL_SECONDS
        )
        return f"AutoCal: sampling {p * 100:.0f}% move mouse"

    def update(self, frame):
        """喂入一帧原图(BGR)，采样结束返回 True。"""
        now = time.perf_counter()

        # 准备阶段：不采样，给你切回游戏的时间
        if now - self.t_begin < AUTO_CAL_DELAY:
            return False

        if self.t_sample is None:
            self.t_sample = now
            print("开始采样：请在游戏里晃动鼠标")

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(g, (0, 0), 2.0)
        edge = (cv2.absdiff(g, blur) > EDGE_THR).astype(np.uint16)

        if self.acc is None:
            self.acc = np.zeros(g.shape, np.uint16)
            self.first = g
        self.acc += edge
        self.last = g
        self.n += 1

        return (now - self.t_sample) >= AUTO_CAL_SECONDS

    def finish(self):
        """返回 (准星在原图中的坐标或 None, 画面运动量)。"""
        self.active = False
        h, w = self.acc.shape

        motion = float(np.mean(cv2.absdiff(self.first, self.last)))

        persist = (self.acc >= PERSIST_RATIO * self.n).astype(np.uint8) * 255
        persist = cv2.dilate(persist, np.ones((3, 3), np.uint8), iterations=2)

        num, labels, stats, cents = cv2.connectedComponentsWithStats(
            persist, connectivity=8
        )

        best = None
        best_d = float("inf")

        for k in range(1, num):
            x, y, bw, bh, area = stats[k]

            if bw > MAX_BLOB or bh > MAX_BLOB:
                continue                      # 枪、HUD 之类的大块
            if bw < MIN_BLOB and bh < MIN_BLOB:
                continue                      # 噪点
            aspect = bw / max(bh, 1)
            if aspect < 0.5 or aspect > 2.0:
                continue                      # 细长的 HUD 线条/文字笔画

            cx, cy = cents[k]
            d = np.hypot(cx - w / 2, cy - h / 2)
            if d < best_d:
                best_d = d
                best = (float(cx), float(cy))

        return best, motion


autocal = AutoCalibrator()


# ================= 粘连目标拆分 =================

def split_merged(boxes, scores, keep):
    """
    球是圆的，框应该接近正方形。
    如果某个框明显偏长（长边/短边 >= SPLIT_RATIO），
    认为是 2 个或多个球并排粘在一起，沿长边拆成多个正方形框。
    只对横向/纵向并排有效，斜着叠的拆不了。
    """
    out_b, out_s = [], []

    for i in keep:
        x, y, bw, bh = boxes[i]
        s = scores[i]

        long_ = max(bw, bh)
        short_ = min(bw, bh)
        ratio = long_ / max(short_, 1e-6)

        if ratio < SPLIT_RATIO:
            out_b.append([x, y, bw, bh])
            out_s.append(s)
            continue

        n = max(2, int(round(ratio)))
        step = (long_ - short_) / (n - 1)

        for k in range(n):
            if bh > bw:      # 竖向并排
                out_b.append([x, y + k * step, bw, bw])
            else:            # 横向并排
                out_b.append([x + k * step, y, bh, bh])
            out_s.append(s)

    boxes = np.array(out_b, np.float32).reshape(-1, 4)
    scores = np.array(out_s, np.float32)
    return boxes, scores, np.arange(len(boxes))


def save_miss_snapshot(image, details):
    """Persist a raw ROI and detector scores without blocking the aim loop."""
    try:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "miss_samples")
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
        stem = os.path.join(folder, "miss_" + stamp)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        encoded.tofile(stem + ".jpg")  # Unicode Windows paths work with tofile.
        with open(stem + ".json", "w", encoding="utf-8") as out:
            json.dump(details, out, ensure_ascii=False, indent=2)
        print("漏检样本已保存:", stem + ".jpg", flush=True)
    except Exception as exc:
        print("保存漏检样本失败:", exc, flush=True)


# ================= 1. GPU 初始化 =================

print("正在加载 YOLOv8 ONNX...")

ort.preload_dlls(directory="")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=[
        (
            "CUDAExecutionProvider",
            {"device_id": 0, "cudnn_conv_algo_search": "EXHAUSTIVE"},
        ),
        "CPUExecutionProvider",
    ],
)

providers = session.get_providers()
print("推理后端:", providers)

if "CUDAExecutionProvider" not in providers:
    raise RuntimeError("CUDA 未成功启用")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
if session.get_inputs()[0].shape != [1, 3, 640, 640] or session.get_outputs()[0].shape[1] != 9:
    raise ValueError("实验模型应为 [1,3,640,640] 输入、[1,9,N] 输出")
target_class_id = 1  # 1=head-first, 0=body-center
mixed_mode = False
mixed_aim = MixedAimPolicy()
print("当前模式: 锁定身体 ID，优先瞄该人的头部；按 F6 切换身体中心")

# ================= 2. GPU IOBinding =================

x_gpu = ort.OrtValue.ortvalue_from_numpy(
    np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32), "cuda", 0
)

io = session.io_binding()
io.bind_ortvalue_input(input_name, x_gpu)
io.bind_output(output_name, "cpu")


def infer(blob):
    x_gpu.update_inplace(blob)
    session.run_with_iobinding(io)
    return io.copy_outputs_to_cpu()[0]


dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
for _ in range(10):
    infer(dummy)

print("GPU 预热完成！")

# ================= 3. DXCam 初始化 =================

camera = (ImageCamera(args.image) if args.image else
          VideoCamera(args.video) if args.video else dxcam.create(output_color="BGR"))

frame = camera.grab()
while frame is None:
    time.sleep(0.01)
    frame = camera.grab()

H, W = frame.shape[:2]

size = min(ROI_SIZE, H, W)
left = (W - size) // 2
top = (H - size) // 2

region = (left, top, left + size, top + size)

print("屏幕:", W, H)
print("检测区域:", region)

load_calib()

tracker = StickyBodyTracker()
box_tracker = BoxTracker(match_radius=args.lock_radius, max_prediction_frames=3)
lead_tracker = TargetLead()
safety_gate = AimSanityGate()
background_flow = BackgroundFlow()
prediction_bridge = PredictedBridge()
standing_filter = StandingBodyFilter()
controller = DampedController(kp=args.kp, ki=args.ki, deadzone=args.deadzone,
                              max_step=args.max_step, far_boost=args.far_boost,
                              far_threshold=args.far_threshold,
                              near_boost=args.near_boost, near_radius=args.near_radius)
transport = MouseTransport(None if args.image or args.video else args.serial_port)
EVENT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4_events.log")


def log_event(message):
    with open(EVENT_LOG, "a", encoding="utf-8") as out:
        out.write(f"{time.strftime('%H:%M:%S')} {message}\n")


log_event(f"START pid={os.getpid()} port={args.serial_port} model={MODEL_PATH}")
last_event_state_at = float("-inf")
last_event_tx_at = float("-inf")
last_pred_log_at = float("-inf")
last_event_gate_at = float("-inf")
last_color_log_at = float("-inf")
last_zoom_log_at = float("-inf")
camera.start(region=region, target_fps=CAPTURE_FPS)
armed = True  # 默认瞄准开启；仍受游戏前台与校准状态约束。
overlay = StatusOverlay() if not args.image else None


def mode_code():
    return "MIX" if mixed_mode else "HEAD" if target_class_id == 1 else "BODY"


def publish_overlay():
    if overlay is not None:
        overlay.send(("ON" if armed else "OFF") + "," + mode_code())


publish_overlay()
trigger_enabled = False
tuner_process = None
last_config_check = 0.0
last_config_stamp = os.stat(args.config).st_mtime_ns if args.config else None
print("控制输出:", args.serial_port if args.serial_port and not args.image else "仅预览，不发送")
print("启动后默认持续瞄准已选目标；无需点击左键；此实验版不自动点击")
print("F5: 头身混合开关；F6: 头部/身体；F9: 关闭；F10: 退出；F11: 自瞄开关；F12: 调参")

# ================= 4. 预览窗口 =================

if SHOW_WINDOW:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WINDOW_NAME, 20, 40)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
preview_clickthrough_ready = False

print("\n====== 操作说明 ======")
print(f"A：自动校准准星（按下后有 {AUTO_CAL_DELAY:.0f} 秒准备时间切回游戏，"
      f"之后 {AUTO_CAL_SECONDS:.0f} 秒内晃动鼠标）")
print("H：手动校准（隐藏标记，左键点击游戏准星）")
print("R：重新手动校准")
print("Q：退出")
print("注意：按键要在预览窗口获得焦点时按")
print("======================\n")

# ================= 5. 性能统计 =================

last_time = time.perf_counter()
last_print = last_time

fps_smooth = 0.0
t_cap = t_pre = t_inf = t_post = t_show = 0.0
last_multi_at = float("-inf")
last_motion_step = 0
last_motion_xy = (0, 0)
last_miss_saved_at = float("-inf")
miss_save_count = 0


def ema(old, new, a=0.9):
    if old == 0:
        return new
    return old * a + new * (1 - a)


# ================= 6. 主循环 =================

try:
    frame_count = 0
    last_full_miss = False
    color_mode = False
    color_mode_frames = 0
    replay_counts = {"real": 0, "bridge": 0, "gated": 0}
    while True:

        t0 = time.perf_counter()

        # 游戏失焦时可能停止出帧；全局退出必须在取帧之前轮询。
        if hotkey_pressed(0x79):  # F10: always exit, even when the preview has no focus.
            print("F10: 退出程序", flush=True)
            log_event("EXIT_F10")
            break

        # F12 must also be checked before grabbing a new frame.
        if hotkey_pressed(0x7B):  # F12
            log_event("F12_PRESSED")
            if not args.config:
                continue
            trigger_enabled = False
            controller.reset()
            transport.cancel()
            if tuner_process is None or tuner_process.poll() is not None:
                ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "v4_tuning_ui.py")
                try:
                    tuner_process = subprocess.Popen(
                        [sys.executable, ui_path, args.config],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    log_event(f"F12_MENU_STARTED pid={tuner_process.pid}")
                    print("参数窗口已打开；关闭后切回游戏会继续瞄准")
                except Exception as exc:
                    log_event(f"F12_MENU_ERROR {exc}")
                    print("打开参数窗口失败:", exc, flush=True)
            else:
                hwnd = ctypes.windll.user32.FindWindowW(None, "V4 参数调节")
                if hwnd:
                    ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                print("参数窗口仍在运行，已尝试切到前台")

        if hotkey_pressed(0x75):  # F6: head / body
            mixed_mode = False
            target_class_id = 0 if target_class_id == 1 else 1
            controller.reset()
            transport.cancel()
            lead_tracker.reset()
            prediction_bridge.reset()
            publish_overlay()
            print("当前瞄点:", "锁定人的头部" if target_class_id == 1 else "锁定人的身体中心", flush=True)
        if hotkey_pressed(0x74):  # F5: optional stable head/body mixed mode
            mixed_mode = not mixed_mode
            controller.reset()
            transport.cancel()
            lead_tracker.reset()
            prediction_bridge.reset()
            publish_overlay()
            log_event(f"AIM_MODE {mode_code()}")
            print("头身混合:", "开启（每 3 个目标约 1 个头部）" if mixed_mode else "关闭", flush=True)

        # ---------- 取最新帧 ----------

        frame = camera.get_latest_frame()

        if frame is None:
            if args.video:
                print(f"VIDEO_END frames={frame_count} counts={replay_counts}")
                break
            time.sleep(0.001)
            continue
        frame_count += 1

        if args.config and t0 - last_config_check >= 0.25:
            last_config_check = t0
            try:
                stamp = os.stat(args.config).st_mtime_ns
                if stamp != last_config_stamp:
                    last_config_stamp = stamp
                    reload_live_config()
            except Exception as exc:
                print("重新读取参数失败:", exc)

        t1 = time.perf_counter()

        h, w = frame.shape[:2]

        # ---------- 自动校准（使用原始画面，不含任何 YOLO 标记） ----------

        if autocal.active:
            if autocal.update(frame):
                res, motion = autocal.finish()

                if motion < 1.0:
                    print("警告：采样期间画面几乎没有变化，请在游戏里晃动鼠标后再按 A 重试")

                if res is None:
                    print("自动校准失败：没找到准星。")
                    print("可尝试：1) 采样时多晃动鼠标；2) 调小 EDGE_THR；3) 用 H 手动校准")
                else:
                    cxf, cyf = res
                    aim_x = cxf * INPUT_SIZE / w
                    aim_y = cyf * INPUT_SIZE / h
                    calibrated = True
                    print(
                        f"\n自动校准成功！原图坐标 = ({cxf:.1f}, {cyf:.1f})，"
                        f"预览坐标 = ({aim_x:.1f}, {aim_y:.1f})"
                    )
                    save_calib()

        # ---------- 预处理 ----------

        # Recheck a recent real small lock at higher scale, refreshing the full
        # scene every fifth frame. A miss falls back to the full/color path.
        tracked_previous = next((track for track in box_tracker.tracks
                                 if track.id == tracker.track_id), None)
        locked_previous = (tracked_previous if tracked_previous is not None
                           and tracked_previous.observed else None)
        lock_box = locked_previous.box if locked_previous is not None else None
        flow_box = (tracked_previous.box if tracked_previous is not None
                    and tracked_previous.box[3] <= 80 else None)
        flow_shift = (background_flow.update(frame, flow_box)
                      if args.motion_compensation else None)
        zoom_lock_used = should_zoom_lock(lock_box, frame_count)
        # Search yellow candidates only when the locked crop is not scheduled.
        probe_due = args.color_sentinel and not zoom_lock_used and (
            (color_mode and color_mode_frames < 24)
            or (not color_mode and last_full_miss and frame_count % 2 == 0))
        proposals = (find_yellow_proposals(frame, max_candidates=3)
                     if probe_due else [])
        lock_center = None
        if color_mode and lock_box is not None:
            lock_center = ((lock_box[0] + lock_box[2] / 2) * frame.shape[1] / INPUT_SIZE,
                           (lock_box[1] + lock_box[3] / 2) * frame.shape[0] / INPUT_SIZE)
        probe_candidates = choose_color_candidate(proposals, frame_count // 2,
                                                  lock_center)
        color_probe_used = bool(probe_candidates)
        yellow_candidates = (find_yellow_proposals(frame, max_candidates=2)
                             if args.secondary_inference and not color_probe_used
                             and not zoom_lock_used else [])

        if not color_probe_used and not zoom_lock_used:
            small = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE),
                               interpolation=cv2.INTER_LINEAR)
            blob = small[:, :, ::-1].transpose(2, 0, 1)
            blob = np.ascontiguousarray(blob[None], dtype=np.float32) / 255.0

        t2 = time.perf_counter()

        # ---------- GPU 推理 ----------

        color_count = 0
        if zoom_lock_used:
            body_boxes, body_scores, head_boxes, head_scores = zoom_locked_target(
                frame, lock_box, infer, class_boxes, INPUT_SIZE, CONF, NMS_THRESHOLD)
            output = None
        elif color_probe_used:
            body_boxes, body_scores, head_boxes, head_scores, _ = recheck_yellow(
                frame, infer, class_boxes, INPUT_SIZE, CONF, NMS_THRESHOLD,
                max_candidates=1, proposals=probe_candidates)
            output = None
        else:
            output = infer(blob)

        t3 = time.perf_counter()

        # ---------- YOLO 后处理 ----------

        if color_probe_used or zoom_lock_used:
            head_keep = np.arange(len(head_boxes), dtype=int)
            body_keep = np.arange(len(body_boxes), dtype=int)
        else:
            head_boxes, head_scores, head_keep = class_boxes(output, 1, CONF, NMS_THRESHOLD)
            body_boxes, body_scores, body_keep = class_boxes(output, 0, CONF, NMS_THRESHOLD)
        body_keep = standing_filter.filter(body_boxes, body_keep, t3,
                                           head_boxes[head_keep],
                                           allow_tiny_confirmed=True)
        if zoom_lock_used and t3 - last_zoom_log_at >= 1.0:
            log_event(f"ZOOM_REPLACE confirmed={len(body_keep)} heads={len(head_keep)}")
            last_zoom_log_at = t3
        if color_probe_used:
            color_count = len(body_keep)
            if t3 - last_color_log_at >= 1.0:
                log_event(f"COLOR_CROP proposals=1 confirmed={color_count}")
                last_color_log_at = t3
        if (args.secondary_inference and not color_probe_used
                and not zoom_lock_used and locked_previous is not None):
            old_box = locked_previous.box
            old_cx = old_box[0] + old_box[2] / 2
            old_cy = old_box[1] + old_box[3] / 2
            near_current = [int(i) for i in body_keep
                            if np.hypot(body_boxes[i, 0] + body_boxes[i, 2] / 2 - old_cx,
                                        body_boxes[i, 1] + body_boxes[i, 3] / 2 - old_cy) < 55]
            head_missing = (near_current and
                            match_head(body_boxes[near_current[0]], head_boxes[head_keep]) is None)
            if not near_current or (head_missing and frame_count % 2 == 0):
                zoom_bodies, zoom_scores, zoom_heads, zoom_head_scores = zoom_locked_target(
                    frame, old_box, infer, class_boxes, INPUT_SIZE, CONF, NMS_THRESHOLD)
                if time.perf_counter() - last_zoom_log_at >= 1.0:
                    log_event(f"ZOOM_LOCK confirmed={len(zoom_bodies)} heads={len(zoom_heads)}")
                    last_zoom_log_at = time.perf_counter()
                if len(zoom_bodies):
                    if not near_current:
                        body_boxes, body_scores = zoom_bodies, zoom_scores
                        body_keep = standing_filter.filter(body_boxes, [0], t3, zoom_heads)
                    if len(zoom_heads):
                        head_boxes = np.concatenate((head_boxes[head_keep], zoom_heads))
                        head_scores = np.concatenate((head_scores[head_keep], zoom_head_scores))
                        head_keep = np.arange(len(head_boxes), dtype=int)
        if len(body_keep) == 0 and yellow_candidates:
            recovered, recovered_scores, recovered_heads, recovered_head_scores, _ = recheck_yellow(
                frame, infer, class_boxes, INPUT_SIZE, CONF, NMS_THRESHOLD,
                proposals=yellow_candidates)
            if time.perf_counter() - last_color_log_at >= 1.0:
                log_event(f"YELLOW_SENTINEL proposals={len(yellow_candidates)} confirmed={len(recovered)}")
                last_color_log_at = time.perf_counter()
            if len(recovered):
                body_boxes, body_scores = recovered, recovered_scores
                body_keep = np.arange(len(recovered), dtype=int)
                head_boxes = np.concatenate((head_boxes[head_keep], recovered_heads))
                head_scores = np.concatenate((head_scores[head_keep], recovered_head_scores))
                head_keep = np.arange(len(head_boxes), dtype=int)
                color_count = len(recovered)
                body_keep = standing_filter.filter(body_boxes, body_keep, t3,
                                                   head_boxes[head_keep])
        t3 = time.perf_counter()
        observed_heads = head_boxes[head_keep]
        boxes, scores, keep = body_boxes, body_scores, body_keep
        last_full_miss = len(body_keep) == 0
        if color_probe_used:
            color_mode = color_count > 0
            color_mode_frames = color_mode_frames + 1 if color_mode else 0
        elif color_mode:
            color_mode = not probe_due and last_full_miss
            color_mode_frames = 0
        raw_scores = output[0, 4] if output is not None else np.empty(0, dtype=np.float32)
        finite_scores = raw_scores[np.isfinite(raw_scores)]
        raw_max = float(np.max(finite_scores)) if len(finite_scores) else 0.0
        raw_above_020 = int(np.count_nonzero(finite_scores >= 0.20))
        best = None

        yolo_targets = len(keep)
        focused = game_focused() or args.simulate_control
        menu_open = (not focused and tuner_process is not None
                     and tuner_process.poll() is None)
        if t3 - last_event_state_at >= 1.0:
            log_event(f"STATE focus={int(focused)} calibrated={int(calibrated)} "
                      f"armed={int(armed)} bodies={yolo_targets} "
                      f"heads={len(head_keep)} aim={aim_x:.1f},{aim_y:.1f}")
            last_event_state_at = t3

        # The matcher sees only this frame's actual detections. On a hit it
        # publishes that raw box; only a miss publishes a predicted box.
        detections = [(boxes[i], scores[i]) for i in keep]
        tracked = box_tracker.update(detections, time.perf_counter())
        boxes = np.asarray([track.box for track in tracked], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray([track.score for track in tracked], dtype=np.float32)
        keep = np.arange(len(tracked), dtype=int)
        real_keep = np.asarray([i for i, track in enumerate(tracked) if track.observed], dtype=int)
        candidates = [Candidate(i, float(boxes[i, 0] + boxes[i, 2] / 2),
                                float(boxes[i, 1] + boxes[i, 3] / 2), float(scores[i]),
                                tracked[i].track_id, tracked[i].observed)
                      for i in keep]
        best, acquired = tracker.update(candidates, aim_x, aim_y, t3)

        if len(real_keep) >= 2 and focused and armed:
            last_multi_at = t3
        elif (yolo_targets == 0 and focused and armed and not (args.image or args.video)
              and not autocal.active
              and t3 - last_multi_at < 1.0 and t3 - last_miss_saved_at >= 5.0
              and miss_save_count < 5):
            details = {
                "screen_region": list(region),
                "model_input": INPUT_SIZE,
                "confidence_threshold": CONF,
                "raw_max_confidence": raw_max,
                "raw_count_above_020": raw_above_020,
                "raw_count_above_threshold": int(np.count_nonzero(finite_scores >= CONF)),
                "detected_targets": yolo_targets,
                "color_recovered_targets": color_count,
                "frame_shape": list(frame.shape),
            }
            threading.Thread(target=save_miss_snapshot,
                             args=(frame.copy(), details), daemon=True).start()
            last_miss_saved_at = t3
            miss_save_count += 1
        if hotkey_pressed(0x7A):  # F11: global auto-aim toggle
            armed = not armed
            if not armed:
                trigger_enabled = False
                transport.cancel()
                prediction_bridge.reset()
                background_flow.reset()
                tracker.reset()
                box_tracker.reset()
            controller.reset()
            log_event(f"F11_AUTO_AIM={int(armed)}")
            publish_overlay()
            print("F11 自动瞄准:", "开启" if armed else "关闭", flush=True)
        if hotkey_pressed(0x78):  # F9: explicit off; F11 can enable again
            if armed:
                print("自瞄控制: 关闭")
            armed = False
            publish_overlay()
            trigger_enabled = False
            controller.reset()
            transport.cancel()
            prediction_bridge.reset()
            background_flow.reset()
            tracker.reset()
            box_tracker.reset()
        elif armed and not focused and not (args.image or args.video):
            # 失焦时立即取消待发鼠标命令；保留瞄准请求，回到游戏自动恢复。
            trigger_enabled = False
            controller.reset()
            transport.cancel()
            prediction_bridge.reset()
            background_flow.reset()
            tracker.reset()
            box_tracker.reset()

        step_x = step_y = tx_packets = 0
        if acquired:
            controller.reset()
            transport.cancel()
            lead_tracker.reset()
            prediction_bridge.reset()
        can_drive = (armed and focused and (calibrated or args.simulate_control)
                     and not calibration_mode
                     and not autocal.active and not menu_open)
        gate_reason = ""
        predicted_point = None
        if (args.prediction_bridge and best is not None and can_drive
                and not tracked[best].observed
                and safety_gate.confirmed(tracked[best].track_id)):
            selected = tracked[best]
            predicted_point = prediction_bridge.predict(
                selected.track_id, selected.box, selected.misses, flow_shift, t3)
        if best is None or not can_drive or (not tracked[best].observed
                                               and predicted_point is None):
            if args.simulate_control:
                replay_counts["gated"] += 1
            controller.reset()
            transport.cancel()
            last_motion_step = 0
            last_motion_xy = (0, 0)
            prediction_bridge.reset()
            if best is None or not can_drive:
                lead_tracker.reset()
                safety_gate.reset()
            else:
                can_drive = False
                gate_reason = "PREDICTED BODY"
                if t3 - last_event_gate_at >= 1.0:
                    log_event("TARGET_GATE PREDICTED BODY DISPLAY ONLY")
                    last_event_gate_at = t3
        elif predicted_point is not None:
            if args.simulate_control:
                replay_counts["bridge"] += 1
            tx, ty, _, _, _ = predicted_point
            error_x = (tx - aim_x) * w / INPUT_SIZE
            error_y = (ty - aim_y) * h / INPUT_SIZE
            step_x, step_y = controller.update(error_x, error_y, time.perf_counter())
            # Bridge only a brief gap; no full-scale pull or click on a ghost.
            step_x = max(-8, min(8, step_x))
            step_y = max(-8, min(8, step_y))
            try:
                tx_packets = transport.send(step_x, step_y)
                last_motion_step = max(abs(step_x), abs(step_y))
                last_motion_xy = (step_x, step_y)
                if (tx_packets or args.simulate_control) and t3 - last_pred_log_at >= 1.0:
                    log_event(f"PRED_BRIDGE step={step_x},{step_y} "
                              f"misses={selected.misses} error={error_x:.1f},{error_y:.1f}")
                    last_pred_log_at = t3
            except Exception as exc:
                print("串口发送失败:", exc)
                log_event(f"SERIAL_ERROR {exc}")
                armed = False
                publish_overlay()
                controller.reset()
                prediction_bridge.reset()
        else:
            if args.simulate_control:
                replay_counts["real"] += 1
            selected = tracked[best]
            use_head = (mixed_aim.head_for(selected.track_id)
                        if mixed_mode else target_class_id == 1)
            head_point = None
            if use_head and selected.observed:
                head_box = match_head(boxes[best], observed_heads)
                if head_box is not None:
                    hx, hy, hw, hh = map(float, head_box)
                    head_point = (hx + hw / 2, hy + hh / 2)
            body_centers = {
                track.track_id: (float(boxes[i, 0] + boxes[i, 2] / 2),
                                 float(boxes[i, 1] + boxes[i, 3] / 2))
                for i, track in enumerate(tracked) if track.observed
            }
            point = lead_tracker.update(
                selected.track_id, boxes[best], head_point, body_centers, t3,
                head_mode=use_head, observed=selected.observed,
                lead_s=args.lead_ms / 1000.0, last_step=last_motion_xy,
                camera_shift=prediction_bridge.camera_since_real(
                    selected.track_id, flow_shift))
            if point is not None:
                allowed, reason = safety_gate.allow(selected.track_id,
                                                    point[1] - point[4])
                if not allowed:
                    gate_reason = reason
                    point = None
            if point is None:
                transport.cancel()
                can_drive = False
                gate_reason = gate_reason or "NO CURRENT HEAD"
                last_motion_step = 0
                last_motion_xy = (0, 0)
                if t3 - last_event_gate_at >= 1.0:
                    log_event(f"TARGET_GATE {gate_reason}")
                    last_event_gate_at = t3
            else:
                tx, ty, aim_source, lead_x, lead_y = point
                prediction_bridge.observe(selected.track_id, point, boxes[best],
                                          lead_tracker.velocity, t3)
                error_x = (tx - aim_x) * w / INPUT_SIZE
                error_y = (ty - aim_y) * h / INPUT_SIZE
                step_x, step_y = controller.update(error_x, error_y, time.perf_counter())
                try:
                    tx_packets = transport.send(step_x, step_y)
                    last_motion_step = max(abs(step_x), abs(step_y))
                    last_motion_xy = (step_x, step_y)
                    if (tx_packets or args.simulate_control) and t3 - last_event_tx_at >= 0.2:
                        log_event(f"TX step={step_x},{step_y} packets={tx_packets} "
                                  f"error={error_x:.1f},{error_y:.1f} "
                                  f"source={aim_source} lead={lead_x:.1f},{lead_y:.1f}")
                        last_event_tx_at = t3
                except Exception as exc:
                    print("串口发送失败:", exc)
                    log_event(f"SERIAL_ERROR {exc}")
                    armed = False
                    publish_overlay()
                    can_drive = False
                    controller.reset()

        t4 = time.perf_counter()

        # ================= 7. 画面绘制 =================

        ax = int(np.clip(round(aim_x), 0, INPUT_SIZE - 1))
        ay = int(np.clip(round(aim_y), 0, INPUT_SIZE - 1))

        if SHOW_WINDOW:

            if not calibration_mode:

                for i in real_keep:
                    x, y, bw_, bh_ = boxes[i]

                    # 红框只表示此帧实际允许控制；黄色表示已选中但未开启输出。
                    color = ((0, 0, 255) if can_drive else (0, 215, 255)) if i == best else (0, 255, 0)
                    p1 = (int(x), int(y))
                    p2 = (int(x + bw_), int(y + bh_))

                    cv2.rectangle(small, p1, p2, color, 1)

                    cv2.putText(
                        small,
                        ("C" if color_count else f"{scores[i]:.2f}"),
                        (p1[0], max(p1[1] - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

                if best is not None and tracked[best].observed:
                    x, y, bw_, bh_ = boxes[best]

                    heads_for_person = observed_heads if tracked[best].observed else []
                    preview_head = (mixed_aim.head_for(tracked[best].track_id)
                                    if mixed_mode else target_class_id == 1)
                    tx, ty, used_real_head = person_target(boxes[best], heads_for_person,
                                                            head_mode=preview_head)

                    ex = (tx - aim_x) * w / INPUT_SIZE
                    ey = (ty - aim_y) * h / INPUT_SIZE

                    if calibrated:
                        cv2.line(
                            small, (ax, ay), (int(tx), int(ty)),
                            (0, 0, 255) if can_drive else (0, 215, 255), 1
                        )

                    cv2.putText(
                        small,
                        f"Err: {ex:.0f},{ey:.0f}",
                        (6, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )

                # 紫色准星（单像素）
                small[ay, ax] = (255, 0, 255)

                now = time.perf_counter()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now
                fps_smooth = ema(fps_smooth, fps)

                cv2.putText(small, f"FPS: {fps_smooth:.1f}", (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(small, f"GPU: {(t3 - t2) * 1000:.1f} ms", (6, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.putText(small, f"Aim: {aim_x:.1f},{aim_y:.1f}", (6, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 255), 1)
                cv2.putText(small, f"Targets: {len(real_keep)} real / {len(keep)-len(real_keep)} pred", (6, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                if yolo_targets == 0:
                    cv2.putText(small, f"YOLO:0 max:{raw_max:.2f}",
                                (6, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 255, 255), 1)
                status = "ARMED" if can_drive else "OFF"
                cv2.putText(small, f"Control: {status} ({step_x},{step_y})", (6, 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255) if can_drive else (0, 215, 255), 1)
                if can_drive and tx_packets:
                    cv2.putText(small, f"TX: {tx_packets} x <=20", (6, 128),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(small, "AUTO AIM ON / NO AUTO FIRE",
                            (6, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (180, 180, 180), 1)
                target_label = ("Target: MIX 1/3" if mixed_mode else
                                "Target: HEAD" if target_class_id == 1 else "Target: BODY")
                cv2.putText(small, target_label,
                            (6, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1)
                if best is not None and not can_drive:
                    reason = ("CALIBRATE WITH A" if not calibrated else
                              "FOCUS GAME" if not focused else
                              "PRESS F11 TO ENABLE" if not armed else
                              "F12 MENU OPEN" if menu_open else gate_reason or "NO TARGET")
                    cv2.putText(small, reason, (6, 128),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

                if autocal.active:
                    cv2.putText(small, autocal.status(), (6, 96),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            preview = cv2.resize(small, (PREVIEW_SIZE, PREVIEW_SIZE),
                                 interpolation=cv2.INTER_AREA)
            cv2.rectangle(preview, (0, 0), (PREVIEW_SIZE - 1, 42), (0, 0, 0), -1)
            preview_status = ("CALIB" if calibration_mode else
                              "OFF" if not armed else "AIM" if can_drive else "READY")
            cv2.putText(preview, f"FPS {fps_smooth:.0f} | {len(real_keep)} targets | {preview_status}",
                        (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            cv2.putText(preview, "F11: AIM ON/OFF | F12: MENU | F10: EXIT",
                        (5, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1)
            cv2.imshow(WINDOW_NAME, preview)
            if not args.interactive_preview and not preview_clickthrough_ready:
                preview_clickthrough_ready = make_preview_click_through()
                if preview_clickthrough_ready:
                    log_event("PREVIEW_CLICKTHROUGH=1")

            # Win32 waitKey(1) often yields one full timer quantum (~14 ms).
            # pollKey pumps HighGUI messages without delaying inference/control.
            key = cv2.pollKey() & 0xFF

            if key == ord('q'):
                break

            elif key == ord('a') and args.interactive_preview:
                if not autocal.active:
                    calibration_mode = False
                    autocal.start()
                    print(
                        f"\n自动校准已启动：请在 {AUTO_CAL_DELAY:.0f} 秒内切回游戏，"
                        f"随后 {AUTO_CAL_SECONDS:.0f} 秒内晃动鼠标"
                    )

            elif key == ord('h') and args.interactive_preview:
                calibration_mode = not calibration_mode
                if calibration_mode:
                    print("\n进入手动校准：请点击画面里游戏自带的准星")
                else:
                    print("\n返回正常检测模式")

            elif key == ord('r') and args.interactive_preview:
                calibration_mode = True
                print("\n重新手动校准：请点击真实准星")

        else:
            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps_smooth = ema(fps_smooth, fps)

        t5 = time.perf_counter()
        now = t5

        if SHOW_WINDOW and calibration_mode:
            last_time = t5

        # ================= 8. 耗时统计 =================

        t_cap = ema(t_cap, (t1 - t0) * 1000)
        t_pre = ema(t_pre, (t2 - t1) * 1000)
        t_inf = ema(t_inf, (t3 - t2) * 1000)
        t_post = ema(t_post, (t4 - t3) * 1000)
        t_show = ema(t_show, (t5 - t4) * 1000)

        if now - last_print > 1.0:
            last_print = now
            print(
                f"FPS {fps_smooth:6.1f} | 取帧 {t_cap:5.2f} | 预处理 {t_pre:5.2f} | "
                f"推理 {t_inf:5.2f} | 后处理 {t_post:5.2f} | 显示 {t_show:5.2f} (ms)"
            )
        if args.max_frames and frame_count >= args.max_frames:
            print(f"REPLAY_OK frames={frame_count} targets={len(keep)} best={best} "
                  f"counts={replay_counts if args.simulate_control else 'n/a'}")
            break

finally:
    if overlay is not None:
        overlay.close()
    if tuner_process is not None and tuner_process.poll() is None:
        tuner_process.terminate()
    transport.close()
    camera.stop()
    del camera
    cv2.destroyAllWindows()
