import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取 CSV 数据
df = pd.read_csv("others\deepseek\deepseek_weibo_output_local.csv")

# 确保“发布时间”为时间类型
df["发布时间"] = pd.to_datetime(df["发布时间"], errors="coerce")

# 设置中文字体（适配 Windows/WSL/Linux）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体，防止中文乱码
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# === 以日期为单位聚合 ===
posts_per_day = df.groupby(df["发布时间"].dt.date).size()
sentiment_per_day = df.groupby([df["发布时间"].dt.date, "情感结果"]).size().unstack(fill_value=0)
avg_sentiment = df.groupby(df["发布时间"].dt.date)["情感得分"].mean()

# ===== 图表1：发帖量随时间变化 =====
fig, ax = plt.subplots(figsize=(10, 5))
posts_per_day.plot(ax=ax, marker="o")
ax.set_title("微博发帖量随时间变化")
ax.set_xlabel("日期")
ax.set_ylabel("发帖数量")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # 每 5 天一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===== 图表2：情感结果随时间变化 =====
fig, ax = plt.subplots(figsize=(10, 6))
sentiment_per_day.plot(ax=ax, marker="o")
ax.set_title("情感结果随时间变化")
ax.set_xlabel("日期")
ax.set_ylabel("数量")
ax.legend(title="情感结果")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===== 图表3：平均情感得分随时间变化 =====
fig, ax = plt.subplots(figsize=(10, 5))
avg_sentiment.plot(ax=ax, marker="o", color="purple")
ax.set_title("平均情感得分随时间变化")
ax.set_xlabel("日期")
ax.set_ylabel("平均情感得分")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
