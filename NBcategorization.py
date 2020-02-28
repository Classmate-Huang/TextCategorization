import jieba
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 读取五个.csv文件
df_technology = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df_technology = df_technology.dropna()

df_car = pd.read_csv("./data/car_news.csv", encoding='utf-8')
df_car = df_car.dropna()

df_entertainment = pd.read_csv("./data/entertainment_news.csv", encoding='utf-8')
df_entertainment = df_entertainment.dropna()

df_military = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df_military = df_military.dropna()

df_sports = pd.read_csv("./data/sports_news.csv", encoding='utf-8')
df_sports = df_sports.dropna()

# 每个类别抽取20000个数据
technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainment.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]

# 进行分词和去除停用词
stopwords = pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values


# 处理文件数据 去除停用词，加入标签信息
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
        except Exception:
            print(line)
            continue


# 生成训练数据
sentences = []

preprocess_text(technology, sentences, 'technology')
preprocess_text(car, sentences, 'car')
preprocess_text(entertainment, sentences, 'entertainment')
preprocess_text(military, sentences, 'military')
preprocess_text(sports, sentences, 'sports')
# 打乱数据
random.shuffle(sentences)
# 利用sklearn划分数据集
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)


# 定义朴素贝叶斯分类模型
class NBClassifier():
    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        # 词袋模型
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 4), max_features=2000)

    def features(self, X):
        # 将X转换为词向量
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        # 向量化X
        self.vectorizer.fit(X)
        # 训练模型
        self.classifier.fit(self.features(X), y)

    def predict(self, x): # 预测
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):  # 打分
        return self.classifier.score(self.features(X), y)


# 应用模型
text_classifier = NBClassifier()
text_classifier.fit(x_train, y_train)
# 测试
print(text_classifier.predict('中国 男篮 什么 时候 才能 进军 奥运会 8强'))
# 测试集评分
print(text_classifier.score(x_test, y_test))
