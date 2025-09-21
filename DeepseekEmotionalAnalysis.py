import pandas as pd
from openai import OpenAI
import re
import os

API_KEY = "ollama"  #本地部署模型不需要api key，这里随便填写一个内容即可
API_ENDPOINT = "http://127.0.0.1:11434/v1/"   #这里写部署的本地deepseek模型地址

client = OpenAI(api_key=API_KEY, base_url=API_ENDPOINT)

def analyze_sentiment(text, index):
    """
    使用 DeepSeek API 对输入的文本进行情感分析。

    :param text: 待分析的文本
    :param index: 当前处理数据的索引，用于输出进度信息
    :return: 情感分析结果（例如：positive, negative, neutral）和情感得分
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-r1:14b",     #这里写本地部署的模型名称
            messages=[
                {"role": "system", "content": "你需要对用户输入的文本进行情感分析，并反馈给我情感结果和情感得分，情感结果仅有三种'positive', 'negative', 'neutral'。情感得分是0~1之间的两位小数，情感得分反映情感的状态，得分越接近1，表明情感越积极。输出结果格式严格按照 '情感结果,情感得分'，不要有任何与'情感结果,情感得分'无关的输出内容，输出结果例如 'positive,0.8',不要带上引号。"},
                {"role": "user", "content": text}
            ],
            stream=False
        )
        result = response.choices[0].message.content.strip().lower()
        pattern = r"(positive|negative|neutral),\d\.\d{2}"
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        #result = re.findall(pattern, result)
        #match = re.search(pattern,result)
        
        try:
            '''
            if match:
                #sentiment,score_str = match.groups()
                sentiment, score_str = result.split(',')
                score = float(score_str)
                valid_results = ['positive', 'negative', 'neutral']
            '''
            sentiment, score_str = result.split(',')
            score = float(score_str)
            valid_results = ['positive', 'negative', 'neutral']
            if sentiment in valid_results:
                print(f"第 {index + 1} 条微博数据的 DeepSeek API 请求成功，情感分析结果为: {sentiment}，得分: {score}")
                return sentiment, score
            else:
                print(f"第 {index + 1} 条微博数据的 DeepSeek API 返回了无效情感结果: {sentiment}，将标记为 'unknown'，得分标记为 0")
                #print(response)
                #print(filtered_result)
                return 'unknown', 0
        except ValueError:
            print(f"第 {index + 1} 条微博数据的 DeepSeek API 返回结果格式错误: {result}，将标记为 'unknown'，得分标记为 0")
            return 'unknown', 0
    except Exception as e:
        print(f"第 {index + 1} 条微博数据的请求出错: {e}")
        return 'unknown', 0

def main():
    # 读取 CSV 文件
    csv_file_path = 'others\deepseek\deepseek.csv'
    print(f"开始读取文件: {csv_file_path}")
    try:
        # 尝试使用 utf-8 编码读取文件
        df = pd.read_csv(csv_file_path)
        print("使用 utf-8 编码成功读取文件")
    except UnicodeDecodeError:
        try:
            # 若 utf-8 失败，尝试使用 gbk 编码
            df = pd.read_csv(csv_file_path, encoding='gbk')
            print("使用 gbk 编码成功读取文件")
        except UnicodeDecodeError:
            # 若 gbk 也失败，尝试使用 gb2312 编码
            df = pd.read_csv(csv_file_path, encoding='gb2312')
            print("使用 gb2312 编码成功读取文件")

    total_rows = len(df)
    print(f"总共需要处理 {total_rows} 条微博数据")

    results = [analyze_sentiment(text, i) for i, text in enumerate(df['微博正文'])]
    df['情感结果'] = [res[0] for res in results]
    df['情感得分'] = [res[1] for res in results]

    # 定义新文件路径
    new_file_path = 'others\deepseek\deepseek_weibo_output_local.csv'
    print("所有微博数据处理完成，开始保存结果到新文件")
    # 将结果保存到新的 CSV 文件
    df.to_csv(new_file_path, index=False)
    print(f"情感分析结果已保存到 {new_file_path} 文件中。")

if __name__ == "__main__":
    main()