# -*- coding:utf-8 -*-
import pandas as pd
import re
import jieba
import os
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
import pyLDAvis.gensim_models
from gensim.models.ldamulticore import LdaMulticore
import pickle
from multiprocessing import Pool, cpu_count

def parallel_cut_texts(texts, stopwords):
    # 用 partial 将 stopwords 固定传参
    from functools import partial
    with Pool(processes=cpu_count()) as pool:
        return pool.map(partial(pre_process_chinese, stopwords=stopwords), texts)

# 加载停用词表

def load_stopwords(filepath='stop_word_list.txt'):
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(["的", "了", "是", "在", "我", "有", "和", "就", "不", "都", "一个", "上", "看", "好", "这"]))  # 可补充
        print("已自动创建基础 stop_word_list.txt")
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


# 中文文本预处理
def pre_process_chinese(text, stopwords):
    text = str(text).strip().lower()  # 增加 lower()，统一大小写
    text = re.sub(r"http\S+|www.\S+", "", text)  # 去除 URL
    jieba.load_userdict("myword.txt")  # 加载用户词典
    words = jieba.lcut(text)  # 中文分词
    return [w for w in words if w not in stopwords and len(w.strip()) > 1]


# 主题建模函数
def do_LDA(data, words_set, filename):
    dictionary = corpora.Dictionary(words_set)
    corpus = [dictionary.doc2bow(text) for text in words_set]

    perplexity = []
    coherences = []

    for num_topics in range(3, 21):
        #lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, random_state=42)
        lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        workers=min(32,os.cpu_count()),  # 启用最大线程数
        random_state=42,
        passes=50,                  # 迭代次数
        chunksize=800,
        eval_every=None
        )
        with open(f"LDA主题模型-主题数为{num_topics}.pickle", 'wb') as f:
            pickle.dump(lda_model, f)

        lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(lda_vis, f"模型可视化结果-主题数为{num_topics}.html")

        # 输出每个主题关键词
        with open(f'每个主题的词汇-主题数为{num_topics}.txt', 'w', encoding='utf-8') as f:
            for topic in lda_model.print_topics(num_words=15):
                f.write(f"主题 {topic[0]}: {topic[1]}\n\n")

        # 计算困惑度与一致性得分
        perplexity_val = lda_model.log_perplexity(corpus)
        coherence_model = CoherenceModel(model=lda_model, texts=words_set, dictionary=dictionary, coherence='c_v')
        coherence_val = coherence_model.get_coherence()

        perplexity.append(perplexity_val)
        coherences.append(coherence_val)

        print(f"主题数={num_topics}，Perplexity={perplexity_val:.4f}，Coherence={coherence_val:.4f}")
        with open(filename + '--result.txt', 'a', encoding='utf-8') as f:
            f.write(f"主题数为{num_topics}时，Perplexity为:{perplexity_val},Coherence Score为:{coherence_val}\n")

    draw(perplexity, coherences, filename)

# 可视化评价指标
def draw(perplexity, coherences, filename):
    x = range(3, 21)

    plt.figure()
    plt.plot(x, perplexity, marker='o')
    plt.xlabel("Num Topics")
    plt.title("Perplexity")
    plt.grid(True)
    plt.savefig(filename + "--perplexity.png", dpi=300)

    plt.figure()
    plt.plot(x, coherences, marker='o')
    plt.xlabel("Num Topics")
    plt.title("Coherence Score")
    plt.grid(True)
    plt.savefig(filename + "--coherence.png", dpi=300)

# 主程序入口
if __name__ == '__main__':
    # Step 1: 加载数据
    # data = pd.read_csv(r'others\deepseek\deepseek_weibo_output_local.csv', encoding='utf-8')
    # data = data.dropna(subset=['微博正文']).drop_duplicates(subset='微博正文')
    data = pd.read_csv(r'Instagram—评论表_已翻译.csv', encoding='utf-8')
    data = data.dropna(subset=['评论翻译']).drop_duplicates(subset='评论翻译')
    

    # Step 2: 加载停用词
    stopwords = load_stopwords('stop_word_list.txt')

    # Step 3: 并行分词
    cut_results = parallel_cut_texts(data['评论翻译'].tolist(), stopwords)
    data['cut_text'] = cut_results

    # Step 4: 执行LDA主题建模
    word_set = data['cut_text'].tolist()
    dataset_name = 'football_Ins'
    do_LDA(data, word_set, dataset_name)
