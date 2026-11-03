"""F12 parameter panel for the V4 640px person-locked experiment."""

import argparse
import configparser
import os
from pathlib import Path
import re
import tkinter as tk
from tkinter import messagebox, ttk


FIELDS = (
    ("control", "kp", "追人速度", 0.001, 0.50, 0.001,
     "跟不上就加一点；冲过头就减一点。"),
    ("control", "max_step", "最大跨步", 1, 240, 1,
     "只管远距离赶路；一般先保持不动。"),
    ("control", "deadzone", "中心停止范围", 0, 10, 0.5,
     "调大更稳，但可能停得离中心稍远。"),
    ("control", "far_boost", "远处冲刺力度", 1.0, 3.0, 0.1,
     "只影响远处目标；近处发晃不用调它。"),
    ("control", "far_threshold", "远处冲刺起点", 50, 300, 10,
     "距离超过这个值后，才开始额外冲刺。"),
    ("control", "near_boost", "贴近吸附力", 1.0, 2.5, 0.05,
     "贴近还跟不上就加；左右摆就减。"),
    ("control", "near_radius", "吸附生效范围", 30, 250, 5,
     "调大后，吸附会更早开始。"),
    ("control", "lead_ms", "移动提前量", 0, 80, 5,
     "追移动 bot 落后就加；左右摆就减。"),
    ("tracking", "lock_radius", "同一目标范围", 10, 100, 1,
     "太大容易换错人；太小容易跟丢。"),
    ("detector", "confidence", "识别门槛", 0.15, 0.85, 0.01,
     "误认底座就加；识别不到人就减。"),
)


def load_config(path):
    cfg = configparser.ConfigParser()
    if not cfg.read(path, encoding="utf-8-sig"):
        raise FileNotFoundError(path)
    return cfg


def replace_values(path, updates):
    """Replace only known key-value lines; keep the human-readable comments."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    section = None
    found = set()
    out = []
    for line in lines:
        newline = line[len(line.rstrip("\r\n")):]
        body = line[:-len(newline)] if newline else line
        section_match = re.match(r"\s*\[([^]]+)\]", line)
        if section_match:
            section = section_match.group(1).strip().lower()
        key_match = re.match(r"(\s*[^;#\s][^=]*?\s*=)\s*([^;#]*)(.*)", body)
        if key_match:
            key = key_match.group(1).split("=", 1)[0].strip().lower()
            pair = (section, key)
            if pair in updates:
                line = f"{key_match.group(1)} {updates[pair]}{key_match.group(3)}{newline}"
                found.add(pair)
        out.append(line)
    missing = set(updates) - found
    if missing:
        raise ValueError(f"参数文件缺少字段: {sorted(missing)}")
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(out), encoding="utf-8")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check:
        print("CONFIG_OK", [(section, key, cfg.get(section, key))
                            for section, key, *_ in FIELDS])
        return

    root = tk.Tk()
    root.title("V4 参数调节")
    root.geometry("650x610")
    root.attributes("-topmost", True)
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="先调常用项；一次只改一个，保存后切回游戏观察。",
              font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
    ttk.Label(frame, text="发晃：先减移动提前量 5；仍晃再减贴近吸附力 0.1。追不上：追人速度加 0.01。",
              foreground="#555555").pack(anchor="w", pady=(0, 8))

    notebook = ttk.Notebook(frame)
    common = ttk.Frame(notebook, padding=8)
    advanced = ttk.Frame(notebook, padding=8)
    notebook.add(common, text="常用")
    notebook.add(advanced, text="高级")
    notebook.pack(fill="both", expand=True)
    common_keys = {"kp", "deadzone", "near_boost", "lead_ms", "confidence"}

    widgets = {}
    for section, key, title, low, high, resolution, hint in FIELDS:
        parent = common if key in common_keys else advanced
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=title, width=15).pack(side="left")
        scale = tk.Scale(row, from_=low, to=high, resolution=resolution,
                         orient="horizontal", length=330, showvalue=True)
        scale.set(cfg.getfloat(section, key))
        scale.pack(side="left", fill="x", expand=True)
        ttk.Label(parent, text=hint, foreground="#666666").pack(anchor="w", padx=130)
        widgets[(section, key)] = scale

    split_var = tk.BooleanVar(value=cfg.getboolean("detector", "split_merged"))
    ttk.Checkbutton(advanced, text="拆分粘连框（人形 bot 保持关闭）", variable=split_var).pack(anchor="w", pady=10)
    ttk.Label(advanced, text="串口、帧率等设置在参数文件中；这里只显示影响手感的项目。",
              foreground="#666666").pack(anchor="w", pady=4)

    def save():
        updates = {}
        for section, key, _title, _low, _high, resolution, _hint in FIELDS:
            value = widgets[(section, key)].get()
            updates[(section, key)] = str(int(value)) if resolution == 1 else f"{value:g}"
        updates[("detector", "split_merged")] = "yes" if split_var.get() else "no"
        try:
            replace_values(args.config, updates)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=root)
            return
        root.destroy()

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=12)
    ttk.Button(buttons, text="保存并应用", command=save).pack(side="right", padx=4)
    ttk.Button(buttons, text="取消", command=root.destroy).pack(side="right", padx=4)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    main()
