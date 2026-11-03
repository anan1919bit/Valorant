"""Small Windows COM-port writer using the same Win32 API as the existing R probe."""

import ctypes
from ctypes import wintypes


class DCB(ctypes.Structure):
    _fields_ = [
        ("DCBlength", wintypes.DWORD),
        ("BaudRate", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("wReserved", wintypes.WORD),
        ("XonLim", wintypes.WORD),
        ("XoffLim", wintypes.WORD),
        ("ByteSize", wintypes.BYTE),
        ("Parity", wintypes.BYTE),
        ("StopBits", wintypes.BYTE),
        ("XonChar", ctypes.c_char),
        ("XoffChar", ctypes.c_char),
        ("ErrorChar", ctypes.c_char),
        ("EofChar", ctypes.c_char),
        ("EvtChar", ctypes.c_char),
        ("wReserved1", wintypes.WORD),
    ]


class COMMTIMEOUTS(ctypes.Structure):
    _fields_ = [(name, wintypes.DWORD) for name in (
        "ReadIntervalTimeout", "ReadTotalTimeoutMultiplier",
        "ReadTotalTimeoutConstant", "WriteTotalTimeoutMultiplier",
        "WriteTotalTimeoutConstant",
    )]


class WinSerial:
    def __init__(self, port, baud=115200):
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.HANDLE]
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.GetCommState.argtypes = [wintypes.HANDLE, ctypes.POINTER(DCB)]
        k32.SetCommState.argtypes = [wintypes.HANDLE, ctypes.POINTER(DCB)]
        k32.SetCommTimeouts.argtypes = [wintypes.HANDLE, ctypes.POINTER(COMMTIMEOUTS)]
        k32.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                                  ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.k32 = k32
        self.handle = k32.CreateFileW(f"\\\\.\\{port}", 0xC0000000, 0, None, 3, 0, None)
        if self.handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            dcb = DCB()
            dcb.DCBlength = ctypes.sizeof(DCB)
            if not k32.GetCommState(self.handle, ctypes.byref(dcb)):
                raise ctypes.WinError(ctypes.get_last_error())
            dcb.BaudRate = int(baud)
            dcb.ByteSize = 8
            dcb.Parity = 0
            dcb.StopBits = 0
            if not k32.SetCommState(self.handle, ctypes.byref(dcb)):
                raise ctypes.WinError(ctypes.get_last_error())
            timeouts = COMMTIMEOUTS(0, 0, 0, 0, 10)
            if not k32.SetCommTimeouts(self.handle, ctypes.byref(timeouts)):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            self.close()
            raise

    def write(self, payload):
        payload = bytes(payload)
        written = wintypes.DWORD(0)
        buf = ctypes.create_string_buffer(payload)
        if not self.k32.WriteFile(self.handle, buf, len(payload),
                                  ctypes.byref(written), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if written.value != len(payload):
            raise OSError(f"short COM write: {written.value}/{len(payload)}")

    def close(self):
        if self.handle is not None:
            self.k32.CloseHandle(self.handle)
            self.handle = None
