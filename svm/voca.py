#su ly cac tu quan trong
import numpy as np
import pandas as pd
import math
#tao ra tu dien

df = pd.DataFrame()


def flatten(list):
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list


def sentence_line(line):
    words = line[0:len(line)].strip().split("<###>")
    return words


def sentence_word(words):
  #  print(words)
    words = words[0:len(words)].strip().split(" ")
    words = [word for word in words if len(word) > 1]
    words = [str for str in words if str]
    return words


words = []


def readFile(path):
    print("processing...")
    docnum = 0
    words = []
    tudien = {}
    file = open(path, "r")
    lines = file.readlines()
    for line in lines:
       # word = sentence_line(line)
        word = sentence_word(line)
        wor = np.asarray(word)
        w, c = np.unique(wor, return_counts=True)
        tudien[docnum] = {}
        for i in range(len(w)):
            tudien[docnum][w[i]] = c[i]
        docnum = docnum + 1
        words.append(word)
    file.close()
    return tudien, words


def compu_TF(dictionaryy):
    tudien = {}
    for k in dictionaryy.keys():
      #  print(k)
        tudien[k] = {}
        n = 0
        for i in dictionaryy[k].keys():
            n = n + dictionaryy[k][i]
     #   print(n)
        for i in dictionaryy[k].keys():
            tudien[k][i] = dictionaryy[k][i]/float(n)
    return tudien


def compu_IDF(list_words, dictionaryy):
    n = len(dictionaryy)
    tudien = {}
    for word in list_words:
        fre = 0
        for k in dictionaryy.keys():
            if word in dictionaryy[k]:
                fre = fre + 1
        tudien[word] = math.log10(n/float(fre))
    return tudien


def compu_TFIDF(dictionaryy, listIDF):
    tudien = {}
    for k in dictionaryy.keys():
        tudien[k] = {}
        for w in dictionaryy[k].keys():
            if w in listIDF:
                tudien[k][w] = dictionaryy[k][w]*listIDF[w]
    return tudien


dictionary, words = readFile("D:\\python\\preprocess_train.txt")
df['tu'] =  flatten(words)
#print(df['tu'].values)
mang, so = np.unique(df['tu'].values, return_counts=True)
print(mang[2])
dictionary = compu_TF(dictionary)
tudienIDF = compu_IDF(mang, dictionary)
dictionary = compu_TFIDF(dictionary, tudienIDF)

list_of_words = []
list_of_freq = []
for k in dictionary.keys():
    for w in dictionary[k].keys():
        if(dictionary[k][w] < 0.5):
            list_of_words.append(w)
            list_of_freq.append(dictionary[k][w])
#print(list_of_words)

freq, wrds = (list(i) for i in zip(*(sorted(zip(list_of_freq, list_of_words), reverse=True))))

wrds, freq = np.unique(wrds, return_counts=True)

features = wrds[0:10000]

f = open("features.txt", "w+")
for i in features:
    f.writelines(i+'\n')
f.close()
fo = open("x_train.txt", "w+")
for k in dictionary.keys():
    index = 0
    row = []
    for wor in features:
        if wor in dictionary[k].keys():
            charr = []
            charr.append(str(index))
            charr.append(str(float(dictionary[k][wor])))
            charr = ':'.join(charr)
            row.append(charr)
        index = index + 1
    row = ' '.join(row)
    fo.writelines(row+'\n')
fo.close()
