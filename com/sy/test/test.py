#!/user/bin/python
#coding:utf-8
__author__ = 'yan.shi'

from com.sy.similaritywords.words_coccurrence import SimWordsCoccurrence
from com.sy.similaritywords.word2_vec import SimWordsWord2Vec

class Test():
    def wordsSimWord2vec(self):
        sim = SimWordsWord2Vec()
        # 加载停词
        # sim.loadStopWords('G:\淘宝项目\stopwords.txt')
        # 将原始文本进行分词存入一个文本中
        # sim.segment('G:\\python workspace\\语料\\红楼梦.txt','G:\\python workspace\\语料\\wordssegment.txt')
        # 利用分词后的文本进行训练word2vec模型，并保存模型
        # sim.word2vec('G:\\python workspace\\语料\\wordssegment.txt','G:\\python workspace\\语料\\modeldata.model')
        keyWords = ['薛宝钗', '贾琏', '巧姐', '贾雨村', '凤姐', '贾宝玉', '林黛玉', '贾母', '邢夫人', '史湘云']
        sim.wordSimilarity('G:\\model\\modeldata.model', keyWords)  # 模型文件路径不能有中文

    def wordsSimCoccurrence(self):
        cocc = SimWordsCoccurrence()
        # cocc.sentSentences('合同的重要性不是一时半会能説清楚的.合同的重要性不是一时半会能説清楚的。合同的重要性不是一时半会能説清楚的!合同的重要性不是一时半会能説清楚的！合同的重要性不是一时半会能説清楚的?合同的重要性不是一时半会能説清楚的？合同的重要性不是一时半会能説清楚的;合同的重要性不是一时半会能説清楚的；')
        # cocc.loadStopWords('G:\淘宝项目\stopwords.txt')
        keyWords = ['薛宝钗', '贾琏', '巧姐', '贾雨村', '凤姐', '贾宝玉', '林黛玉', '贾母', '邢夫人', '史湘云']
        cocc.wordCocc('G:\淘宝项目\分句后.txt', keyWords)

if __name__=='__main__':
    test=Test()
    test.wordsSimWord2vec()
    test.wordsSimCoccurrence()