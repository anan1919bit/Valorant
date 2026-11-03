"""Launch and update the F9 indicator without drawing inside captured frames."""

from pathlib import Path
import subprocess
import sys


class StatusOverlay:
    def __init__(self):
        script = Path(__file__).with_name("status_overlay.py")
        error_log = script.with_name("status_overlay_error.log")
        with error_log.open("a", encoding="utf-8") as error:
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(script)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=error, creationflags=subprocess.CREATE_NO_WINDOW,
            )
        self.send("OFF,HEAD")

    def send(self, state):
        if self.process.poll() is not None:
            return False
        try:
            self.process.stdin.write((state + "\n").encode("ascii"))
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def close(self):
        if self.process.poll() is not None:
            return
        try:
            self.process.stdin.close()
            self.process.wait(timeout=1)
        except Exception:
            self.process.terminate()
