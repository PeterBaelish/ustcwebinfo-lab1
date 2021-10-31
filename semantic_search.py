import json 
import nltk
import os
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import math

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

Range_Number = 2
wnl = WordNetLemmatizer()

#######文件名处理#########

path = ['2018_01','2018_02','2018_03','2018_04','2018_05']
Path = ['','','','','']
for i in range(Range_Number):
    Path[i] = './'+ path[i]


string.punctuation#标点符号表
punctuation_string = string.punctuation

#####构建停用词表##########
stop_words = set(stopwords.words('english'))#停用词处理
for w in ['!',',','.','?','-s','-ly','</s>','s',"''","'s",'``','$','|',"'"]:
    stop_words.add(w)


N = 0


##########遍历所有文件############
text_dft = {}
text_doc_tftd = {}

for i in range(Range_Number):
    for info in os.listdir(Path[i]):
        print('2018_0',i+1,':',info,'is runing')
        info = Path[i]+'/'+info
        fp = open(info,'rb')
        data = json.load(fp)['text']
        
        for p in string.punctuation:#去除所有的标点并转变为空格
            data = data.replace(p ," ")
            
        data = data.lower()#转化成小写
        
        word_split = nltk.word_tokenize(data)#分词
        
        word_split = [w for w in word_split if((w not in stop_words) and (not w.isdigit()))]#去除停用词
        
        word_tagged = nltk.pos_tag(word_split)    #获取单词词性
        lemmas_sent = []
        for tag in word_tagged:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        
        print(lemmas_sent)

        N = N + 1

        fp.close()

        for text in set(lemmas_sent): #计算dft
            if text not in text_dft:
                text_dft[text] = 1
            else:
                text_dft[text] += 1

        text_doc_tftd[info]={}

        for text in lemmas_sent: #计算tftd
            if text not in text_doc_tftd[info]:
                text_doc_tftd[info][text] = 1
            else:
                text_doc_tftd[info][text] += 1
        

wtd = {}

fp = open('./tfidf','w')

for doc in text_doc_tftd:#计算tfidf表
    for text in text_doc_tftd[doc]:
        if doc not in wtd:
            wtd[doc] = {}
            wtd[doc][text] = (1+math.log(text_doc_tftd[doc][text]))*math.log(N/text_dft[text])
        else:
            wtd[doc][text] = (1+math.log(text_doc_tftd[doc][text]))*math.log(N/text_dft[text])

json.dump(wtd,fp)

fp.close()


