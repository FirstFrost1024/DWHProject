import os
import re
import sys
import math
import pandas as pd
import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import emoji

CSV_PATH = "positive.csv"            # 输入CSV
TEXT_COL = "微博正文"                   # 文本列名（必须存在）
OUT_FREQ_CSV = "positive_word_freq.csv"   # 词频表输出
OUT_WC_PNG = "positive_wordcloud.png"     # 词云图输出
STOPWORDS_FILE = "stopwords.txt"          # 可选停用词

# ------- 工具函数 -------
def try_read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 最后尝试不指定编码
    return pd.read_csv(path)

def find_chinese_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",                         # Windows 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",                       # Windows 黑体
        r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",    # Linux 文泉驿
        r"/System/Library/Fonts/PingFang.ttc",                # macOS
        "SimHei.ttf", "simhei.ttf", "msyh.ttc", "NotoSansCJK-Regular.ttc"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def load_stopwords():
    base = {
        "的","了","和","是","在","就","都","而","及","与","着","或","一个","没有",
        "我们","你们","他们","她们","它们","以及","因此","但是","如果","因为",
        "这","那","呢","啊","吗","吧","呀","哦","嗯","哦哦","哈哈","哈哈哈",
        "一个","自己","还有","而且","这个","那个","这些","那些","什么","怎么",
        "不是","就是","以及","并且","非常","真的","然后","但是","还有","以及",
        "可以","不会","不能","可能","已经","还是","就是","的话","真的","一下",
        "一下子","一下儿","比如","比如说","比如讲","以及","同时","目前",
        "今天","昨天","明天","现在","正在","已经","还是","因为","所以"
    }
    if os.path.exists(STOPWORDS_FILE):
        try:
            with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
                user = {line.strip() for line in f if line.strip()}
                base |= user
        except Exception:
            pass
    return base

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # 去除 @、话题、URL、表情、转义空白
    s = re.sub(r"http[s]?://\S+", " ", s)              # URL
    s = re.sub(r"@\S+", " ", s)                        # 提及
    s = re.sub(r"#([^#]+)#", r" \1 ", s)               # 话题保留中间词
    s = emoji.replace_emoji(s, replace=" ")            # 移除emoji
    # 只保留中英文、数字
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", s)
    # 合并空白
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(text: str):
    # jieba 分词，中文保留 >=2 长度，英文统一小写并过滤过短
    for tok in jieba.cut(text, cut_all=False):
        t = tok.strip()
        if not t:
            continue
        # 过滤纯数字或长度为1的中文符号
        if re.fullmatch(r"[0-9]+", t):
            continue
        # 英文转小写
        if re.fullmatch(r"[A-Za-z]+", t):
            t = t.lower()
            if len(t) <= 2:   # 过滤过短英文词
                continue
        # 过滤单字（常见噪音）
        if len(t) == 1:
            continue
        yield t

# ------- 主流程 -------
def main():
    if not os.path.exists(CSV_PATH):
        print(f"[错误] 找不到文件：{CSV_PATH}")
        sys.exit(1)

    df = try_read_csv(CSV_PATH)
    if TEXT_COL not in df.columns:
        print(f"[错误] CSV中不存在列：{TEXT_COL}，现有列：{list(df.columns)}")
        sys.exit(1)

    texts = df[TEXT_COL].dropna().astype(str).tolist()
    if not texts:
        print("[警告] 文本为空，退出。")
        sys.exit(0)

    stopwords = load_stopwords()
    counter = Counter()

    for line in texts:
        c = clean_text(line)
        if not c:
            continue
        for tok in tokenize(c):
            if tok in stopwords:
                continue
            counter[tok] += 1

    if not counter:
        print("[警告] 分词结果为空，可能是清洗过度或列内容为空。")
        sys.exit(0)

    # 导出词频表（按频次降序）
    freq_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    freq_df = pd.DataFrame(freq_items, columns=["word", "freq"])
    freq_df.to_csv(OUT_FREQ_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] 已导出词频表：{OUT_FREQ_CSV}（共 {len(freq_df)} 个词）")

    # 生成词云
    font_path = find_chinese_font()
    if not font_path:
        print("[警告] 未找到中文字体，将使用默认字体（可能中文无法正常显示）。建议安装或放置中文字体到脚本目录。")

    # 取前 N 个词生成词云（避免过大）
    TOP_N = 500
    wc_freq = dict(freq_items[:TOP_N])

    wc = WordCloud(
        font_path=font_path,
        width=1600,
        height=900,
        background_color="white",
        max_words=TOP_N,
        prefer_horizontal=0.9,
        collocations=False,  # 避免将两个词强拼为一个
    ).generate_from_frequencies(wc_freq)

    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_WC_PNG, dpi=200)
    plt.close()
    print(f"[OK] 已生成词云图：{OUT_WC_PNG}")

    # 终端预览Top 50
    print("\nTop 50 高频词：")
    print(freq_df.head(50).to_string(index=False))

if __name__ == "__main__":
    main()
