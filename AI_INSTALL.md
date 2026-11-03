# 让自己的 AI 工具协助安装

把下面这段完整贴给你使用的本地 AI 编程工具，并让它在本仓库目录工作：

> 请先阅读 README.md、MODEL.md 和 install.ps1。只在这个项目目录内操作，不改动其他 VisionHID 版本或系统驱动。确认 Windows 与 Python 3.12 可用后运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1`。脚本应创建 `.venv`、安装 requirements.txt、从 v0.4.2 发布页下载 ONNX 模型并核对 SHA-256、生成 config.ini。再运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -VerifyOnly` 和 `py -3.12 -m unittest discover -s src -p "test_*.py"`。请报告每一步的真实输出。不要自动烧录 ESP32 或猜测 COM 端口；请列出我需要手动确认的串口、固件和 CUDA/cuDNN 步骤。不要启动实时控制程序替我做硬件实测。

不使用 AI 工具也可以直接双击 `一键安装.cmd`。安装脚本只处理 Python 环境、依赖、模型和配置模板；板子固件、串口与 GPU 运行库因设备不同，需在本机另行确认。
