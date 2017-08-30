#!/user/bin/python
#coding:utf-8
__author__ = 'yan.shi'

from jieba import posseg
import jieba
import codecs
import re
from jieba import analyse

'''
词共现，先对文本分句，这里利用的是一个句子中的词共现方式，也可以设置一个共现窗口
分词使用是jieba，这个可以提取词性
最后计算的相关词可以用gephi作网络图显示
'''
class SimWordsCoccurrence():

    # 加载停词
    def loadStopWords(self, stopwordsPath):
        print("加载停词...")
        stopWordsList = [line.strip() for line in codecs.open(stopwordsPath, 'r', 'utf-8').readlines()]
        self.stopWords = {}.fromkeys(stopWordsList)

    #词共现，这里使用出现在一句话中的词共现，也可以使用一个窗口，比如两个词相隔多少词出现作为共现条件
    def wordCocc(self,dataPath,keywords):
        self.words_coccurrence={}
        with codecs.open(dataPath,'r','utf-8') as file:
            for line in file:
                wordPosition={}
                tokens=jieba.tokenize(line)#位置，词在文本中的位置，可以作为共现窗口
                for token in tokens:
                    wordPosition[token[0]]=token[1]
                for word,position in wordPosition.items():
                    if (word in self.stopWords) or (word=='\t'):
                        continue
                    # 去除长度为1的词以及英文和数字
                    if len(word) <= 1 or self.isEnglishWordORDigit(word):
                        continue
                    self.words_coccurrence.setdefault(word,{})
                    for word2,position2 in wordPosition.items():
                        if word==word2:
                            continue
                        if (word2 in self.stopWords) or (word2 == '\t'):
                            continue
                        # 去除长度为1的词以及英文和数字
                        if len(word2) <= 1 or self.isEnglishWordORDigit(word2):
                            continue
                        if word2 in self.words_coccurrence[word]:
                            self.words_coccurrence[word][word2]+=1
                        else:
                            self.words_coccurrence[word][word2]=0
        i=0
        for key,value in self.words_coccurrence.items():
            if key in keywords:
                sort=sorted(value.items(),key=lambda item:item[1],reverse=True)
                j=0
                #print('%s,%f:' %(key,keywords[key]))
                for w,v in sort:
                    if v==0:
                        continue
                    j+=1
                    print('%s,%s,%d' %(key,w,v))
                    if j==10:
                        break


    # 是否是英文或数字
    def isEnglishWordORDigit(self, word):
        try:
            return word.encode('ascii').isalpha() or word.isdigit()
        except UnicodeEncodeError:
            return False

    #对文本进行分句
    def sentSentences(self,text):
        sentSeg=re.split('[!?。！？.;；]',text)
        return sentSeg

    # 获取关键词
    def getKeyWords(self, stopwords, rawData):
        with codecs.open(rawData, 'r', 'utf-8') as ropen:
            linesList = ropen.readlines()
            text = ''.join(linesList)
        extractKeyWords = analyse.extract_tags
        analyse.set_stop_words(stopwords)
        keyWords = extractKeyWords(text, topK=1000)
        return keyWords
