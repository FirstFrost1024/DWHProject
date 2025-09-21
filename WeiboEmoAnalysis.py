# -*- coding: utf-8 -*-
"""
weibo_sentiment_deepseek_async.py
功能：
1) 使用 DeepSeek（OpenAI 兼容接口）做情感分析
2) 可调并发（CONCURRENCY）
3) 仅写入两列：情感结果、情感得分
4) 显示进度条
5) 输入为 ./weibo.xlsx（Excel），默认读取第一张表，文本列为“微博正文”
"""

import os
import math
import asyncio
import re
import pandas as pd
from typing import Tuple, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# ========== 可配置区域 ==========
API_KEY = "sk-72b2489cb5714766842065b7f2aeef7d"   # 建议用环境变量
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # 如用本地兼容端点，改成 http://172.31.10.61:8000/v1
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")  # 或你的本地模型名（OpenAI兼容）

INPUT_XLSX = "./weibo_cleaned.xlsx"
SHEET_NAME = 0              # 也可改成工作表名
TEXT_COL = "微博正文"
OUTPUT_XLSX = "./weibo_sentiment.xlsx"

# 并发大小：数值越大速度越快，但更易触发限流/超时。局域网/本地可适当开大。
CONCURRENCY = 20

# 单条重试设置
MAX_RETRIES = 4
RETRY_BASE_DELAY = 1.2   # 秒，指数退避基数

# ========== 提示词（仅返回 情感结果+得分） ==========
SYSTEM_PROMPT = (
    "你需要对输入的评论文本进行情感分析，只输出两部分：情感结果 和 情感得分。\n"
    "情感结果仅为三选一：'positive'、'negative'、'neutral'。\n"
    "情感得分为 -1 到 1 的数，保留两位小数，越接近 1 越积极，越接近 -1 越消极。\n"
    "严格只输出 'positive,0.80' 或 'negative,-0.37' 这样的格式，不要输出其它任何内容。"
)

# 正则去除思维链等包裹标签
THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _parse_result(text: str) -> Tuple[str, float]:
    """
    将模型返回解析为 (sentiment, score)。
    返回非法时，给出 ('neutral', 0.0) 兜底。
    """
    if not isinstance(text, str):
        return "neutral", 0.0
    cleaned = THINK_TAG_RE.sub("", text).strip().lower()
    try:
        sentiment_str, score_str = [t.strip() for t in cleaned.split(",", 1)]
        if sentiment_str not in ("positive", "negative", "neutral"):
            return "neutral", 0.0
        score = float(score_str)
        # 限幅 & 保留两位
        score = max(-1.0, min(1.0, score))
        score = float(f"{score:.2f}")
        return sentiment_str, score
    except Exception:
        return "neutral", 0.0


async def analyze_one(
    client: AsyncOpenAI, text: Optional[str], sem: asyncio.Semaphore, idx: int
) -> Tuple[str, float]:
    """
    处理单条文本，带重试与限流信号量。
    """
    if text is None or (isinstance(text, float) and math.isnan(text)) or str(text).strip() == "":
        return "neutral", 0.0

    prompt_user = str(text).strip()
    attempt = 0
    while True:
        attempt += 1
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_user},
                    ],
                    temperature=0.0,
                    stream=False,
                    timeout=60,  # 秒
                )
            content = resp.choices[0].message.content
            return _parse_result(content)
        except Exception as e:
            if attempt >= MAX_RETRIES:
                # 最终失败兜底
                return "neutral", 0.0
            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))


async def main_async():
    # 1) 读取Excel
    try:
        df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"读取 Excel 失败：{INPUT_XLSX}\n错误：{e}")

    if TEXT_COL not in df.columns:
        raise RuntimeError(f"找不到文本列“{TEXT_COL}”，请检查表头。已有列：{list(df.columns)}")

    texts = df[TEXT_COL].tolist()

    # 2) 初始化客户端（OpenAI 兼容；DeepSeek 官方/本地都可）
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 3) 并发执行 + 进度条
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [analyze_one(client, t, sem, i) for i, t in enumerate(texts)]
    results: list[Tuple[str, float]] = []
    async for r in atqdm.as_completed(tasks, total=len(tasks), desc="情感分析中"):
        results.append(await r)

    # 注意：as_completed返回的顺序与输入不同，需要按原顺序再跑一次更简单：
    # 上面写法是展示进度用的。为了结果不乱序，我们再来一次严格按顺序收集：
    results_ordered: list[Tuple[str, float]] = []
    # 重新生成任务（因为前面已经await完毕的 r 无法复用），这里只是简单遍历逐个再跑一次会重复请求。
    # 更优解：在第一次创建 tasks 时就保存其 Future 位置映射。
    # 为避免重复请求，改为保存第一轮的 future 列表，然后逐个 await（同时还能显示进度）。
    # 这里做一次小调整：

async def main():
    # 读取Excel
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME, engine="openpyxl")
    if TEXT_COL not in df.columns:
        raise RuntimeError(f"找不到文本列“{TEXT_COL}”，请检查表头。已有列：{list(df.columns)}")

    texts = df[TEXT_COL].tolist()

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY)

    # 先创建所有任务（保持原始顺序）
    tasks = [asyncio.create_task(analyze_one(client, t, sem, i)) for i, t in enumerate(texts)]

    # 用进度条按顺序等待（不打乱顺序）
    sentiments = []
    scores = []
    for coro in atqdm(tasks, desc="情感分析中", total=len(tasks)):
        s, sc = await coro
        sentiments.append(s)
        scores.append(sc)

    # 写入两列
    df["情感结果"] = sentiments
    df["情感得分"] = scores

    # 输出到新的 Excel
    df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print(f"处理完成：共 {len(df)} 行。结果已保存到 {OUTPUT_XLSX}")


if __name__ == "__main__":
    # Windows/Conda 下运行 asyncio 的推荐入口
    try:
        asyncio.run(main())
    except RuntimeError:
        # 某些环境下（如嵌套事件循环）退回旧写法
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
