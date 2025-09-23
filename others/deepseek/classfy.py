import pandas as pd
import os

# 输入文件路径
input_file = r"others\deepseek\deepseek_weibo_output_local.csv"

# 输出文件路径
neutral_file = "neutral.csv"
positive_file = "positive.csv"
negative_file = "negative.csv"   # 注意用户写的是 "negitive"，保持一致

# 读取CSV
df = pd.read_csv(input_file)

# 确认列名
if "情感结果" not in df.columns:
    raise ValueError("CSV 文件中没有找到 '情感结果' 列，请检查文件格式。")

# 分类写出
df[df["情感结果"] == "neutral"].to_csv(neutral_file, index=False, encoding="utf-8-sig")
df[df["情感结果"] == "positive"].to_csv(positive_file, index=False, encoding="utf-8-sig")
df[df["情感结果"] == "negative"].to_csv(negative_file, index=False, encoding="utf-8-sig")

print("分类完成，已生成 neutral.csv, positeive.csv, negative.csv")
