// ESP32-S3 USB HID bridge for Aim Lab.
// ASCII serial protocol, one command per line:
//   M,dx,dy\n  relative move (x right, y down)
//   C\n        one left-button pulse (12 ms)
//   P\n        reply PONG
// No automatic firing in the firmware: C is sent only by the host F1 gate.
// Arduino-ESP32 3.x: ESP32S3 Dev Module, USB Mode = USB-OTG (TinyUSB).
// USB CDC On Boot = Enabled for native USB CDC + HID on one cable;
// or Disabled for UART COM port + native USB HID on two cables.

#include <stdlib.h>
#include <string.h>
#include "USB.h"
#include "USBHIDMouse.h"

USBHIDMouse Mouse;

static const int CMD_LIMIT = 20;            // Host sends chunks <= 20 (verified movement).
static const int PENDING_LIMIT = 400;
static const uint32_t SEND_INTERVAL_US = 1000;
static const uint32_t CLICK_HOLD_MS = 12;
static const uint32_t STALE_MS = 150;

static char lineBuf[32];
static uint8_t lineLen = 0;
static bool overflow = false;
static int pendingX = 0;
static int pendingY = 0;
static bool clickPending = false;
static bool clickHeld = false;
static uint32_t clickDownMs = 0;
static uint32_t lastCommandMs = 0;
static uint32_t lastReportUs = 0;

static int clampInt(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

static void handleLine(const char *line) {
  if (strcmp(line, "P") == 0) {
    Serial.println("PONG");
    return;
  }
  if (strcmp(line, "C") == 0) {
    // Give the click priority and discard stale motion near the target.
    pendingX = 0;
    pendingY = 0;
    clickPending = true;
    lastCommandMs = millis();
    return;
  }
  if (line[0] != 'M' || line[1] != ',') {
    return;
  }
  char *endX;
  char *endY;
  long dx = strtol(line + 2, &endX, 10);
  if (endX == line + 2 || *endX != ',') {
    return;
  }
  long dy = strtol(endX + 1, &endY, 10);
  if (endY == endX + 1 || *endY != '\0') {
    return;
  }
  pendingX = clampInt(pendingX + clampInt((int)dx, -CMD_LIMIT, CMD_LIMIT),
                      -PENDING_LIMIT, PENDING_LIMIT);
  pendingY = clampInt(pendingY + clampInt((int)dy, -CMD_LIMIT, CMD_LIMIT),
                      -PENDING_LIMIT, PENDING_LIMIT);
  lastCommandMs = millis();
}

void setup() {
  Serial.begin(115200);
  Mouse.begin();
  USB.begin();
}

void loop() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (!overflow && lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
      }
      lineLen = 0;
      overflow = false;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = ch;
    } else {
      overflow = true;
    }
  }

  uint32_t nowUs = micros();
  uint32_t nowMs = millis();
  bool reportReady = nowUs - lastReportUs >= SEND_INTERVAL_US;

  if (clickHeld) {
    if (nowMs - clickDownMs >= CLICK_HOLD_MS && reportReady) {
      Mouse.release(MOUSE_LEFT);
      clickHeld = false;
      lastReportUs = nowUs;
    }
    return;  // Do not move while the left button is held.
  }
  if (clickPending && reportReady) {
    Mouse.press(MOUSE_LEFT);
    clickPending = false;
    clickHeld = true;
    clickDownMs = nowMs;
    lastReportUs = nowUs;
    return;
  }

  if ((pendingX || pendingY) && nowMs - lastCommandMs > STALE_MS) {
    pendingX = 0;
    pendingY = 0;
  }
  if ((pendingX || pendingY) && reportReady) {
    int sx = clampInt(pendingX, -CMD_LIMIT, CMD_LIMIT);
    int sy = clampInt(pendingY, -CMD_LIMIT, CMD_LIMIT);
    Mouse.move((int8_t)sx, (int8_t)sy);
    pendingX -= sx;
    pendingY -= sy;
    lastReportUs = nowUs;
  }
}
