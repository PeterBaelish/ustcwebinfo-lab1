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

wnl = WordNetLemmatizer()
string.punctuation#标点符号表
punctuation_string = string.punctuation

#####构建停用词表##########
stop_words = set(stopwords.words('english'))#停用词处理
for w in ['!',',','.','?','-s','-ly','</s>','s',"''","'s",'``','$','|',"'"]:
    stop_words.add(w)


fp = open('tfidf','r')
data = json.load(fp)

while 1:
    query = input("insert some words\n")
    if query == '$quit':
        break

    word_split = query.split()

    word_split = [w for w in word_split if((w not in stop_words) and (not w.isdigit()))]#去除停用词
        
    word_tagged = nltk.pos_tag(word_split)    #获取单词词性
    lemmas_sent = []
    for tag in word_tagged:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    simliar_dict = {}

    for doc in data:

        s = 0
        for text in data[doc]:
            s += (data[doc][text])*(data[doc][text])
        s = math.sqrt(s)

        m = 0
        for word in lemmas_sent:
            if word in data[doc]:
                m += data[doc][word]
        m /= s
        if m!=0:
            simliar_dict[doc] = m
    sorted(simliar_dict.items(),key=lambda x:x[1])

    for i in range(10):
        print(list(simliar_dict.keys())[i])