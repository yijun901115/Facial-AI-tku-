# -*- coding: utf-8 -*-
"""
setup_and_run.py

功能：
1. 檢查 Python 是否可用
2. 如果 Poetry 尚未安裝，才下載並執行 Poetry 安裝腳本；若已安裝則跳過
3. 切換到此腳本所在目錄
4. 執行：poetry lock → poetry install → poetry run python run.py
   （如果有需要，可自行修改最後一行的執行命令）
"""

import sys
import os
import subprocess
import urllib.request
import tempfile
import shutil

def pause(message="按 Enter 鍵繼續..."):
    try:
        input(message)
    except KeyboardInterrupt:
        pass

def check_python():
    # sys.executable 本身就是 Python 可執行路徑，代表 Python 可呼叫
    if not sys.executable:
        print("[錯誤] 無法找到 Python，可執行檢查。")
        pause()
        sys.exit(1)

def is_poetry_installed():
    """
    檢查系統 PATH 是否能找到 'poetry' 指令。
    如果找到則回傳 True，否則 False。
    """
    return shutil.which("poetry") is not None

def download_poetry_installer(dest_path):
    url = "https://install.python-poetry.org"
    try:
        print("── 下載 Poetry 安裝腳本中……")
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        print(f"[錯誤] 下載 Poetry 安裝腳本失敗：{e}")
        pause()
        sys.exit(1)

def run_poetry_installer(script_path):
    try:
        print("── 開始執行 Poetry 安裝腳本……")
        result = subprocess.run([sys.executable, script_path], check=False)
        if result.returncode != 0:
            print(f"[錯誤] Poetry 安裝失敗 (exit code {result.returncode})。")
            print(f"請手動用 `{sys.executable} {script_path}` 嘗試。")
            pause()
            sys.exit(1)
    except Exception as e:
        print(f"[例外] 執行安裝腳本時出錯：{e}")
        pause()
        sys.exit(1)

def run_shell_command(cmd_args, error_message):
    """
    cmd_args: list，比如 ["poetry", "lock"]
    error_message: 若執行失敗要印出的訊息
    """
    try:
        completed = subprocess.run(cmd_args, check=False)
        if completed.returncode != 0:
            print(f"[錯誤] {error_message} (exit code {completed.returncode})")
            pause()
            sys.exit(1)
    except FileNotFoundError:
        print(f"[錯誤] 無法找到指令：{cmd_args[0]}，請確認已安裝並加入 PATH。")
        pause()
        sys.exit(1)
    except Exception as e:
        print(f"[例外] 執行 `{cmd_args}` 時發生錯誤：{e}")
        pause()
        sys.exit(1)

def main():
    # 1. 檢查 Python
    check_python()
    print("── 已找到 Python，可執行環境檢查。")

    # 2. 如果沒有安裝 Poetry，就下載並安裝
    if is_poetry_installed():
        print("── 發現系統已安裝 Poetry，跳過安裝步驟。")
    else:
        # 下載並安裝 Poetry
        print("\n── 執行 `pip install poetry`……")
        run_shell_command(["pip", "install", "poetry"], "pip install poetry 執行失敗")
        pause()

    # 3. 切換到此 Python 腳本所在資料夾
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(script_dir)

    # 4. 執行 poetry lock / install / run
    print("\n── 執行 `poetry lock`……")
    run_shell_command(["poetry", "lock"], "poetry lock 執行失敗，請檢查 pyproject.toml 格式")

    print("\n── 執行 `poetry install`……")
    run_shell_command(["poetry", "install"], "poetry install 執行失敗，請確認 Poetry 與 pyproject.toml 是否正常")

    print("\n── 相依套件安裝完成，開始執行 `poetry run python run.py`……")
    # 如果你的專案不是要執行 run.py，請自行把下面這行改成正確的指令
    run_shell_command(["poetry", "run", "python", "run.py"], "poetry run 執行指令失敗，請確認你填的指令是否正確")

    print("\n>>> 全部流程完成！")
    pause()

if __name__ == "__main__":
    main()
