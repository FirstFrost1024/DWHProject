#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将 Excel 中 “评论内容” 列翻译为中文，写入新列 “评论翻译”
特点：
  - 进度条显示（tqdm）
  - 一次性写出到新文件，避免 Windows 文件占用导致 PermissionError
  - 指数退避重试，稳健处理429/502/503/504等
  - 支持本地/自定义 DeepSeek 网关（DEEPSEEK_API_BASE）
依赖：
  pip install pandas openpyxl requests tqdm python-dotenv(可选)
环境变量：
  - DEEPSEEK_API_KEY   必填
  - DEEPSEEK_API_BASE  选填，默认 https://api.deepseek.com
"""

import os
import time
import json
import requests
import pandas as pd
from tqdm import tqdm
from typing import Optional

# ===== 基本配置 =====
EXCEL_PATH = "Instagram—评论表.xlsx"
SOURCE_COL = "评论内容"
TARGET_COL = "评论翻译"
MODEL = "gpt-oss:latest"

API_KEY = "ollama"
API_BASE = "http://127.0.0.1:11434"
API_URL = f"{API_BASE.rstrip('/')}/api/chat"

# 请求与重试
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # 秒
TIMEOUT = 60           # 单次请求超时

SYSTEM_PROMPT = (
    "你是专业的翻译助手。将输入文本准确流畅地翻译成简体中文，"
    "保持原意与口吻；若含表情、话题、@、链接等，按中文表达习惯处理；"
    "仅输出译文，不要添加解释。"
)

def ensure_api_key():
    if not API_KEY:
        raise RuntimeError("未检测到环境变量 DEEPSEEK_API_KEY，请先设置。")

def call_deepseek_translate(text: str) -> str:
    """
    调用 DeepSeek Chat Completions 接口，把 text 翻译为中文。
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
            # 常见需要重试的状态码
            if resp.status_code in (429, 502, 503, 504):
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt >= MAX_RETRIES:
                # 最终失败，返回占位文本以不中断全局流程
                return f"[翻译失败:{type(e).__name__}] {str(e)[:120]}"
            time.sleep(backoff)
            backoff *= 1.8

def safe_output_path(base_path: str) -> str:
    """
    根据输入 Excel 路径生成不冲突的输出文件名：
      Instagram—评论表.xlsx -> Instagram—评论表_已翻译.xlsx
      若已存在，则 _已翻译(1).xlsx, _已翻译(2).xlsx ...
    """
    stem, ext = os.path.splitext(base_path)
    candidate = f"{stem}_已翻译{ext}"
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate = f"{stem}_已翻译({i}){ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

def main():
    ensure_api_key()

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"未找到文件：{EXCEL_PATH}")

    # 读取 Excel
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    if SOURCE_COL not in df.columns:
        raise KeyError(f"Excel 中未找到列：{SOURCE_COL}。现有列为：{list(df.columns)}")

    # 准备目标列
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = None

    total = len(df)
    print(f"共 {total} 行，将翻译《{SOURCE_COL}》写入《{TARGET_COL}》。")

    # 仅处理需要翻译的行（原来已有译文的跳过可节省额度）
    # 注意：保持原行顺序，使用索引列表遍历可避免 iterrows 的开销与副作用
    indices_to_process = []
    for idx in range(total):
        existing = df.at[idx, TARGET_COL] if TARGET_COL in df.columns else None
        if isinstance(existing, str) and existing.strip():
            continue
        src = df.at[idx, SOURCE_COL] if SOURCE_COL in df.columns else None
        if pd.isna(src) or str(src).strip() == "":
            # 空源文本直接置空译文
            df.at[idx, TARGET_COL] = ""
            continue
        indices_to_process.append(idx)

    # 翻译主循环（带进度条）
    for idx in tqdm(indices_to_process, desc="翻译进度", total=len(indices_to_process)):
        src_text = str(df.at[idx, SOURCE_COL]).strip()
        zh_text = call_deepseek_translate(src_text)
        df.at[idx, TARGET_COL] = zh_text

    # 一次性写出到新文件，避免文件锁冲突
    out_path = safe_output_path(EXCEL_PATH)
    # 此处只写一次，Windows 下不会与 Excel 文件锁冲突（确保 Excel 未占用原文件即可）
    df.to_excel(out_path, index=False, engine="openpyxl")

    print(f"完成。已将结果保存到：{out_path}")

if __name__ == "__main__":
    main()
