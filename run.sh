#!/bin/bash
source "$(dirname "$0")/venv/bin/activate"
echo "✅ 虚拟环境已激活：$VIRTUAL_ENV"

echo "🪐 程序启动"
python main.py