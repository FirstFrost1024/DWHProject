# 微博 & Instagram 数据分析项目

本项目包含微博与 Instagram 数据的预处理、情感分析、LDA 主题建模以及词云可视化。

---

## 📂 项目结构

- **LDA/**  
  存放 LDA 分析结果文件（如 `football_weibo--result.txt`, `football_ins--result.txt`）。

- **OriginalData/**  
  原始数据。

- **WeiboClassfied/**  
  微博情感分类后的文件。

- **others/**  
  情感分析后的文件。

- **.bak 文件**  
  临时文件，可忽略。

- **csv 文件**  
  常见的数据文件格式，本项目多数数据分析代码基于 CSV 格式。

---

## 📝 主要代码说明

- **Preprocess.py**  
  数据预处理脚本。

- **DeepseekEmotionalAnalysis.py** / **WeiboEmoAnalysis.py**  
  用于微博情感分析的代码。

- **LDA_Process.py**  
  LDA 主题建模代码。

- **translate.py** / **translate_excel_deepseek.py**  
  用于 Instagram 数据翻译。

- **wc.py**  
  生成词云图的代码。

---

## 📊 数据说明

- **weibo.xlsx**  
  整合后的微博数据。

- **weibo_cleaned.xlsx**  
  清洗后的微博数据。

- **myword.txt**  
  用户词典，用于词云和 LDA 分析。

- **stopword.txt**  
  停用词表，用于词云和 LDA 分析。

---

## 🚀 使用方法

1. **数据预处理**  
   ```bash
   python Preprocess.py
   ```

2. **情感分析**  
   ```bash
   python DeepseekEmotionalAnalysis.py
   ```
   或  
   ```bash
   python WeiboEmoAnalysis.py
   ```

3. **LDA 主题建模**  
   ```bash
   python LDA_Process.py
   ```

4. **词云生成**  
   ```bash
   python wc.py
   ```

5. **数据翻译（Instagram 数据）**  
   ```bash
   python translate_excel_deepseek.py
   ```

---

## ⚙️ 环境依赖

- Python 3.11+
- Pandas
- Jieba
- Matplotlib / WordCloud
- scikit-learn
- openai / requests (用于本地或 API 情感分析、翻译)

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 📌 注意事项

- `.bak` 文件为临时文件，可忽略。
- CSV 是类似 Excel 的常用数据格式，推荐在 Python 中直接使用 `pandas` 处理。
- 运行脚本前请确认数据路径是否正确。

---

## 🛠️ 结果展示

- **LDA 结果文件：** `football_weibo--result.txt`, `football_ins--result.txt`
- **词云图：** `wc.py` 生成
- **情感分析结果：** `WeiboClassfied/`, `others/`


