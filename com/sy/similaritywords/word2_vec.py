#!/user/bin/python
#coding:utf-8
__author__='yanshi'

import jieba
from jieba import posseg
from jieba import analyse
import codecs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

'''
利用word2vec，先将文本分词(词间用空格)，
再利用word2vec训练并保存模型（注意使用模型计算词的相似度时，模型路径不能有中文，否则出错）
分词使用是jieba，这个可以提取词性
最后计算的相关词可以用gephi作网络图显示
'''
class SimWordsWord2Vec:

    #加载停词
    def loadStopWords(self,stopwordsPath):
        print("加载停词...")
        stopWordsList=[line.strip() for line in codecs.open(stopwordsPath,'r','utf-8').readlines()]
        self.stopWords={}.fromkeys(stopWordsList)

    # 是否是英文
    def isEnglishWord(self, word):
        try:
            return word.encode('ascii').isalpha()
        except UnicodeEncodeError:
            return False

    # 对文本分词，并存入dataToWrite文件中
    def segment(self, dataToRead, dataToWrite):
        with codecs.open(dataToWrite, 'w', 'utf-8') as wopen:
            print('开始分词...')
            with codecs.open(dataToRead, 'r', 'utf-8') as ropen:
                linesList = ropen.readlines()
                lines = ''.join(linesList)
                words = posseg.cut(lines)
                seg = ''
                for word in words:
                    if word.word in self.stopWords or word.flag!='nr':#只留下词性为nr人名的
                        continue
                    if len(word.word)<=1 or len(word.word)>=4:
                        continue
                    if word.word!= '\t':
                        # 去除长度为1的词以及英文
                        if self.isEnglishWord(word.word) == False:
                            seg +=word.word+ ' '
                wopen.write(seg)
        print('分词结束!')

    #利用jieba只获取人名的词
    def namedEntityRecognition(self,dataToRead):
            with codecs.open(dataToRead,'r','utf-8') as ropen:
                #创建set集合，去重
                wordsSet=set()
                linesList = ropen.readlines()
                lines = ''.join(linesList)
                words = posseg.cut(lines)
                for word in words:
                    if(word.flag=='nr'):#人名
                        wordsSet.add(word.word)
                for word in wordsSet:
                    if len(word) ==1 or len(word)>=4:#去除长度小于1的以及长度大于等于4的词
                        wordsSet.remove(word)
                    if word in self.stopWords:
                        wordsSet.remove(word)
            return wordsSet

    # 获取文本中的关键词
    def getKeyWords(self, stopwords, rawData,wordsList):
        with codecs.open(rawData, 'r', 'utf-8') as ropen:
            linesList = ropen.readlines()
            text = ''.join(linesList)
        extractKeyWords = analyse.extract_tags
        analyse.set_stop_words(stopwords)
        #抽取topK个关键词，withWeight=True返回词的权重
        keyWords = extractKeyWords(text, topK=1000,withWeight=True)
        key_words={} #key是word，value是tf-idf值
        for word,value in keyWords:
            if word in wordsList:
                key_words[word]=value
        return key_words

    #训练模型
    def word2vec(self,segmentPath,modelPath):
        print('训练word2vec...')
        model=Word2Vec(LineSentence(segmentPath),size=400,window=5,min_count=5,workers=multiprocessing.cpu_count())
        model.save(modelPath)
        print('训练结束')

    #词的相似计算
    def wordSimilarity(self,modelPath,keyWords):
        model=Word2Vec.load(modelPath)
        print('Source,Target,Weight,Type')
        for word in keyWords:
            try:
                simwords=model.most_similar(word,topn=10)
            except KeyError:
                continue
            for simword in simwords:
                print('%s,%s,%s,%s' %(word,simword[0],simword[1],'undirected'))#使用gephi格式画图
