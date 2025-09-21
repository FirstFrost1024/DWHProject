#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并发版：将 Excel 中 “评论内容” 列翻译为中文，写入新列 “评论翻译”
- asyncio + aiohttp 并发
- 连接复用、指数退避重试
- 去重缓存（相同源文本只请求一次）
- 速率限制（每秒最多多少请求）
- 进度条显示（tqdm.asyncio）
"""

import os
import time
import json
import math
import asyncio
import hashlib
import pandas as pd
from typing import Optional, Dict
from collections import defaultdict

import aiohttp
from aiohttp import ClientSession, TCPConnector
from tqdm.asyncio import tqdm_asyncio as tqdm

# ===== 基本配置 =====
EXCEL_PATH = "Instagram—评论表.xlsx"
SOURCE_COL = "评论内容"
TARGET_COL = "评论翻译"

# 模型与接口（本地）
MODEL = "gemma3:27b"
API_KEY = "ollama"
API_BASE = "http://frp-day.com:11434"
API_URL  = f"{API_BASE.rstrip('/')}/api/chat"   # Ollama /api/chat

# 请求与并发
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.5              # s
TIMEOUT = 60                        # 单次请求超时（秒）
MAX_CONCURRENCY = 4                 # 最大并发任务数（按你的GPU/CPU调）
RATE_LIMIT_PER_SEC = 6              # 每秒请求上限（设 0 或 None 关闭）

# 性能小优化：简化 system 提示，降低 token 占用
SYSTEM_PROMPT = (
    "你是翻译助手。仅把用户输入翻译成简体中文；保持原意与语气；"
    "对表情/话题/链接按中文习惯处理；只输出译文，不要解释。"
)

def ensure_api_key():
    if not API_KEY:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY（或本地API密钥）。")

def is_chinese_text(s: str) -> bool:
    # 轻量判断：如果中文字符占比 > 20% 就视为中文
    if not s: return False
    total = len(s)
    zh = sum(1 for ch in s if '\u4e00' <= ch <= '\u9fff')
    return (zh / max(1, total)) > 0.2

def safe_output_path(base_path: str) -> str:
    stem, ext = os.path.splitext(base_path)
    candidate = f"{stem}_已翻译{ext}"
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        c = f"{stem}_已翻译({i}){ext}"
        if not os.path.exists(c):
            return c
        i += 1

class RateLimiter:
    """简单令牌桶速率限制：每秒最多 N 个许可。"""
    def __init__(self, rate_per_sec: Optional[int]):
        self.rate = rate_per_sec or 0
        self._tokens = self.rate
        self._last = time.perf_counter()
        self._lock = asyncio.Lock()

    async def acquire(self):
        if self.rate <= 0:
            return
        async with self._lock:
            now = time.perf_counter()
            elapsed = now - self._last
            # 补充令牌
            self._tokens = min(self.rate, self._tokens + elapsed * self.rate)
            if self._tokens < 1:
                # 需等待
                need = 1 - self._tokens
                await asyncio.sleep(need / self.rate)
                now = time.perf_counter()
                elapsed = now - self._last
                self._tokens = min(self.rate, self._tokens + elapsed * self.rate)
            # 消耗1个令牌
            self._tokens -= 1
            self._last = now

async def call_translate(session: ClientSession, text: str) -> str:
    """
    调用本地 /api/chat 翻译为中文。
    - 适配 Ollama：messages + stream:false
    - 如换 vLLM(OpenAI兼容)：改 URL 与 payload / 取值即可
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "stream": False,
        "options": {
            # 可按需要传递一些推理参数（不同后端名称可能不同）
            # "num_ctx": 2048,
            # "temperature": 0.0,
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT) as resp:
                # 常见需要重试的状态码
                if resp.status in (429, 502, 503, 504):
                    text_snip = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text_snip[:200]}")
                resp.raise_for_status()
                data = await resp.json()
                # Ollama /api/chat 返回结构：{"message":{"content": "..."}}
                # 兼容部分实现也会返回 choices 结构，这里先取 message.content
                if "message" in data:
                    return (data["message"]["content"] or "").strip()
                elif "choices" in data:
                    return (data["choices"][0]["message"]["content"] or "").strip()
                else:
                    return ""
        except Exception as e:
            if attempt >= MAX_RETRIES:
                return f"[翻译失败:{type(e).__name__}] {str(e)[:160]}"
            await asyncio.sleep(backoff)
            backoff *= 1.8

async def worker(idx: int,
                 src_text: str,
                 session: ClientSession,
                 sem: asyncio.Semaphore,
                 limiter: RateLimiter,
                 cache: Dict[str, str]) -> (int, str):
    """
    单个任务：受限并发 + 速率限制 + 去重缓存
    """
    if not src_text:
        return idx, ""

    key = hashlib.md5(src_text.encode("utf-8")).hexdigest()
    if key in cache:
        return idx, cache[key]

    async with sem:
        await limiter.acquire()
        zh = await call_translate(session, src_text)
        cache[key] = zh
        return idx, zh

async def run_concurrent(df: pd.DataFrame) -> pd.DataFrame:
    # 需要处理的行（跳过已译、空、明显中文）
    indices = []
    for i in range(len(df)):
        existing = df.at[i, TARGET_COL] if TARGET_COL in df.columns else None
        if isinstance(existing, str) and existing.strip():
            continue
        src = df.at[i, SOURCE_COL] if SOURCE_COL in df.columns else None
        if pd.isna(src) or str(src).strip() == "":
            df.at[i, TARGET_COL] = ""
            continue
        s = str(src).strip()
        if is_chinese_text(s):
            df.at[i, TARGET_COL] = s  # 已是中文，直接复用
            continue
        indices.append(i)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    limiter = RateLimiter(RATE_LIMIT_PER_SEC)
    cache: Dict[str, str] = {}

    connector = TCPConnector(limit=MAX_CONCURRENCY * 2, enable_cleanup_closed=True)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=TIMEOUT)

    tasks = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for idx in indices:
            src_text = str(df.at[idx, SOURCE_COL]).strip()
            tasks.append(worker(idx, src_text, session, sem, limiter, cache))

        # 带进度条并发执行
        for coro in tqdm.as_completed(tasks, desc="翻译进度", total=len(tasks)):
            i, zh = await coro
            df.at[i, TARGET_COL] = zh

    return df

def main():
    ensure_api_key()
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"未找到文件：{EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    if SOURCE_COL not in df.columns:
        raise KeyError(f"Excel 中未找到列：{SOURCE_COL}，现有列：{list(df.columns)}")
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = None

    print(f"共 {len(df)} 行，将翻译《{SOURCE_COL}》写入《{TARGET_COL}》。")
    # 事件循环并发执行
    df = asyncio.run(run_concurrent(df))

    out_path = safe_output_path(EXCEL_PATH)
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"完成。已将结果保存到：{out_path}")

if __name__ == "__main__":
    main()
