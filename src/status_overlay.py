"""Small translucent, click-through status panel for the V4 experiment."""

import ctypes
from ctypes import wintypes
import queue
import sys
import threading
import tkinter as tk


KEY_COLOR = "#ff00ff"
WS_EX_TRANSPARENT = 0x20
WS_EX_TOOLWINDOW = 0x80
WS_EX_NOACTIVATE = 0x08000000
GWL_EXSTYLE = -20
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
HWND_TOPMOST = wintypes.HWND(-1)


def make_click_through(root):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_long)
    user32.SetWindowLongW.restype = ctypes.c_long
    user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    wintypes.UINT)
    hwnd = user32.GetAncestor(root.winfo_id(), 2) or root.winfo_id()
    exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                          exstyle | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                        SWP_NOACTIVATE | SWP_SHOWWINDOW | 0x0001 | 0x0002)


def main():
    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg=KEY_COLOR)
    try:
        root.wm_attributes("-transparentcolor", KEY_COLOR)
        root.wm_attributes("-alpha", 0.74)
    except tk.TclError:
        root.configure(bg="#111b25")

    width, height = 238, 82
    root.geometry(f"{width}x{height}+{max(12, root.winfo_screenwidth() - width - 18)}+16")
    canvas = tk.Canvas(root, width=width, height=height, bg=KEY_COLOR,
                       highlightthickness=0, bd=0)
    canvas.pack()
    canvas.create_polygon(14, 1, width - 14, 1, width - 2, 11,
                          width - 2, height - 11, width - 14, height - 1,
                          14, height - 1, 2, height - 11, 2, 11,
                          smooth=True, splinesteps=12,
                          fill="#111b25", outline="#415463", width=1)
    accent = canvas.create_line(13, 30, 13, 65, fill="#f28b82", width=3)
    canvas.create_text(23, 16, text="V4  /  AIM", anchor="w", fill="#90a5b0",
                       font=("Segoe UI", 9, "bold"))
    canvas.create_text(23, 39, text="自瞄状态", anchor="w", fill="#a9bac3",
                       font=("Microsoft YaHei UI", 10))
    canvas.create_text(23, 63, text="瞄准部位", anchor="w", fill="#a9bac3",
                       font=("Microsoft YaHei UI", 10))
    state_text = canvas.create_text(103, 39, text="关闭", anchor="w", fill="#f28b82",
                                    font=("Microsoft YaHei UI", 11, "bold"))
    mode_text = canvas.create_text(103, 63, text="头部", anchor="w", fill="#dfe9ef",
                                   font=("Microsoft YaHei UI", 10, "bold"))

    root.update_idletasks()
    make_click_through(root)
    root.deiconify()
    make_click_through(root)

    commands = queue.SimpleQueue()

    def read_commands():
        try:
            for line in sys.stdin:
                commands.put(line.strip().upper())
        finally:
            commands.put(None)

    def refresh():
        while not commands.empty():
            command = commands.get_nowait()
            if command is None:
                root.destroy()
                return
            fields = command.split(",")
            if len(fields) != 2 or fields[0] not in ("ON", "OFF"):
                continue
            if fields[1] not in ("HEAD", "BODY", "MIX"):
                continue
            on = fields[0] == "ON"
            color = "#69dfa9" if on else "#f28b82"
            canvas.itemconfigure(accent, fill=color)
            canvas.itemconfigure(state_text, text="开启" if on else "关闭", fill=color)
            canvas.itemconfigure(mode_text, text={"HEAD": "头部", "BODY": "身体",
                                                  "MIX": "头身混合 · 1/3头"}[fields[1]])
        root.after(40, refresh)

    threading.Thread(target=read_commands, daemon=True).start()
    root.after(40, refresh)
    root.mainloop()


if __name__ == "__main__":
    main()
