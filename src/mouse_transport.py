"""Batch ESP32 mouse commands without Windows timer-quantum stalls."""

import threading

from aim_core import click_command, mouse_command, split_mouse_steps


class LatestMouseSender:
    """One writer thread; replace queued motion, batch each frame's HID chunks."""

    def __init__(self, write, interval_s=0.001, chunk=20):
        self._write = write
        self._interval_s = float(interval_s)
        self._chunk = int(chunk)
        self._condition = threading.Condition()
        self._pending_move = None
        self._click_pending = False
        self._epoch = 0
        self._closed = False
        self._error = None
        self._thread = threading.Thread(target=self._run, name="esp32-mouse-writer",
                                        daemon=True)
        self._thread.start()

    def _check(self):
        if self._error is not None:
            raise OSError("ESP32 串口后台发送失败") from self._error
        if self._closed:
            raise RuntimeError("ESP32 串口发送器已关闭")

    def send(self, dx, dy):
        dx, dy = int(dx), int(dy)
        if not dx and not dy:
            self.cancel()
            return 0
        with self._condition:
            self._check()
            self._epoch += 1
            self._pending_move = (dx, dy)
            self._condition.notify()
        return (max(abs(dx), abs(dy)) + self._chunk - 1) // self._chunk

    def click(self):
        with self._condition:
            self._check()
            self._epoch += 1
            self._pending_move = None
            self._click_pending = True
            self._condition.notify()

    def cancel(self):
        with self._condition:
            if self._closed:
                return
            self._epoch += 1
            self._pending_move = None
            self._condition.notify()

    def _run(self):
        try:
            while True:
                with self._condition:
                    while (not self._closed and not self._click_pending
                           and self._pending_move is None):
                        self._condition.wait()
                    if self._closed:
                        return
                    if self._click_pending:
                        self._click_pending = False
                        move = None
                    else:
                        move = self._pending_move
                        self._pending_move = None
                        epoch = self._epoch
                if move is None:
                    self._write(click_command())
                    continue
                payload = b"".join(mouse_command(step_x, step_y)
                                   for step_x, step_y in
                                   split_mouse_steps(*move, chunk=self._chunk))
                with self._condition:
                    if self._closed or self._click_pending or self._epoch != epoch:
                        continue
                # Firmware queues <=240 counts/axis and emits 20-count reports
                # at 1 ms. A single COM write avoids Windows ~16 ms sleeps per chunk.
                self._write(payload)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._closed = True
                self._pending_move = None
                self._click_pending = False
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._pending_move = None
            self._click_pending = False
            self._epoch += 1
            self._condition.notify()
        self._thread.join(timeout=1.0)
