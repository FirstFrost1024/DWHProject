import os
import re
import time
import math
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from threading import Lock
from typing import Tuple, Optional
from tqdm import tqdm

# ===== 基本配置 =====
API_KEY = "ollama"  # 本地部署模型不需要 key，随便写
API_ENDPOINT = "http://127.0.0.1:11434/v1/"  # 你的 OpenAI 兼容网关
MODEL_NAME = "gemma3:27b"  # 你的本地模型名称（按实际调整）

INPUT_CSV = "Instagram—评论表_已翻译.csv"
TEXT_COL = "评论翻译"
OUT_PATH = os.path.join("others", "ins", "deepseek_ins_output_local.csv")
OS_MAKEDIRS = True                 # 若上级目录不存在则自动创建
ENCODINGS_TO_TRY = [None, "utf-8", "gbk", "gb2312"]  # 自动编码探测顺序(None=默认)

# ===== 并发/速率/重试 =====
MAX_WORKERS = 10  # 线程数：先按 CPU 开，外网/FRP 建议 4~16 测
QPS = 5               # 每秒最多请求数（外网/FRP建议保守；内网可更高）
TIMEOUT = 30          # 每次请求超时秒
MAX_RETRIES = 5       # 最大重试次数
RETRY_BASE = 0.8      # 指数退避基数
RETRY_CAP = 8.0       # 最大退避秒（上限）

# ===== 断点续跑（可选）=====
ENABLE_RESUME = True  # 开启后，如已有输出文件，会尽量跳过已完成的行
RESUME_KEY_COLS = [TEXT_COL]  # 用于匹配的主键列（简单做法：用原文本匹配）

client = OpenAI(api_key=API_KEY, base_url=API_ENDPOINT)

SYSTEM_PROMPT = (
    "你需要对用户输入的文本进行情感分析，并反馈给我情感结果和情感得分，"
    "情感结果仅有三种'positive', 'negative', 'neutral'。"
    "情感得分是0~1之间的两位小数，越接近1越积极。"
    "输出结果格式严格按照 '情感结果,情感得分'，不要有任何其它内容。"
    "示例：positive,0.80"
)
RESULT_PATTERN = re.compile(r"\b(positive|negative|neutral)\s*,\s*(0?\.\d{2}|1\.00)\b", re.I)

# 简单的滑动窗口速率限制（线程安全）
class RateLimiter:
    def __init__(self, qps: int, window: float = 1.0):
        self.qps = max(1, int(qps))
        self.window = window
        self.times = deque()
        self.lock = Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                # 清理窗口外时间戳
                while self.times and (now - self.times[0]) > self.window:
                    self.times.popleft()
                if len(self.times) < self.qps:
                    self.times.append(now)
                    return
                # 需要等待窗口滚动
                wait = self.window - (now - self.times[0])
            time.sleep(max(0.001, wait))

rate_limiter = RateLimiter(QPS)

def call_model(text: str) -> str:
    """
    调用本地模型，返回原始字符串（可能带 think 标签、杂质）。
    """
    # 速率限制
    rate_limiter.acquire()

    # 带重试请求
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                stream=False,
                timeout=TIMEOUT,  # OpenAI SDK 支持 timeout
            )
            content = (resp.choices[0].message.content or "").strip()
            return content
        except Exception as e:
            last_err = e
            # 指数退避
            backoff = min(RETRY_CAP, (RETRY_BASE ** attempt) * 2.0)
            time.sleep(backoff)
    # 全部失败
    raise RuntimeError(f"调用模型失败（已重试 {MAX_RETRIES} 次）: {last_err}")

def parse_result(raw: str) -> Tuple[str, float]:
    """
    解析模型返回，严格提取 sentiment 和 score。
    不合规返回 ('unknown', 0.0)。
    """
    # 去除 <think>…</think>
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    print(raw)
    m = RESULT_PATTERN.search(raw)
    if not m:
        # 兜底：尝试 split 解析
        try:
            s, v = raw.split(",", 1)
            s = s.strip().lower()
            v = float(v.strip())
            if s in {"positive", "negative", "neutral"} and 0.0 <= v <= 1.0:
                # 统一两位小数
                return s, float(f"{v:.2f}")
        except Exception:
            pass
        return "unknown", 0.0
    s = m.group(1).lower()
    v = float(m.group(2))
    # 保险起见裁剪到 2 位
    return s, float(f"{v:.2f}")

def analyze_one(idx: int, text: str) -> Tuple[int, str, float]:
    """
    处理单条，返回 (idx, sentiment, score)
    """
    try:
        raw = call_model(text)
        sentiment, score = parse_result(raw)
        return idx, sentiment, score
    except Exception as e:
        # 最终失败
        return idx, "unknown", 0.0

def smart_read_csv(path: str, columns: Optional[list] = None) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
            if columns:
                # 确保需要的列存在
                missing = [c for c in columns if c not in df.columns]
                if missing:
                    raise ValueError(f"缺少必要列: {missing}")
            return df
        except Exception as e:
            last_err = e
    raise last_err

def maybe_resume(df: pd.DataFrame) -> pd.DataFrame:
    """
    如果启用断点续跑且已存在输出文件，则尝试合并已完成结果，避免重复计算。
    简单对齐逻辑：按 RESUME_KEY_COLS 做左连接。
    """
    if not ENABLE_RESUME or not os.path.exists(OUT_PATH):
        return df

    try:
        done = pd.read_csv(OUT_PATH)
        if not set(RESUME_KEY_COLS).issubset(done.columns):
            return df
        # 仅保留我们需要的结果列
        keep_cols = RESUME_KEY_COLS + ["情感结果", "情感得分"]
        done = done[[c for c in keep_cols if c in done.columns]].dropna(subset=RESUME_KEY_COLS)
        # 合并
        merged = df.merge(done, on=RESUME_KEY_COLS, how="left", suffixes=("", "_done"))
        # 若已有结果则保留
        merged["情感结果"] = merged["情感结果"].combine_first(merged.get("情感结果_done"))
        merged["情感得分"] = merged["情感得分"].combine_first(merged.get("情感得分_done"))
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_done")], errors="ignore")
        return merged
    except Exception:
        return df

def main():
    if OS_MAKEDIRS:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print(f"开始读取文件: {INPUT_CSV}")
    df = smart_read_csv(INPUT_CSV, columns=[TEXT_COL])
    df = maybe_resume(df)

    total = len(df)
    print(f"总共需要处理 {total} 条微博数据")
    if "情感结果" not in df.columns:
        df["情感结果"] = pd.NA
    if "情感得分" not in df.columns:
        df["情感得分"] = pd.NA

    # 需要计算的索引列表（跳过已有结果的行）
    to_do_idxs = [i for i, (s, v) in enumerate(zip(df["情感结果"], df["情感得分"])) if pd.isna(s) or pd.isna(v)]
    print(f"本次需要新计算 {len(to_do_idxs)} 条（其余行已存在结果将跳过）")

    # 并发执行
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for i in to_do_idxs:
            text = str(df.at[i, TEXT_COL]) if not pd.isna(df.at[i, TEXT_COL]) else ""
            futures.append(ex.submit(analyze_one, i, text))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="情感分析并发处理中", unit="条"):
            idx, sentiment, score = fut.result()
            results[idx] = (sentiment, score)

    # 回填结果
    for idx, (sentiment, score) in results.items():
        df.at[idx, "情感结果"] = sentiment
        df.at[idx, "情感得分"] = score

    # 保存
    print("所有微博数据处理完成，开始保存结果到新文件")
    # 为了 Excel 兼容性和中文 BOM，可选择 utf-8-sig；纯 CSV 用 utf-8 即可
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"情感分析结果已保存到 {OUT_PATH} 文件中。")

if __name__ == "__main__":
    main()
