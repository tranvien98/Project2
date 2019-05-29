import pandas as pd
import numpy as np
import string
def flatten(list):
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list


def sentence_word(words):
  #  print(words)
    words = words[0:len(words)].strip().split(" ")
    words = [word for word in words if len(word) > 1]
    words = [str for str in words if str]
    return words

def readFile(path):
    words = []
    f = open(path,"r")
    lines = f.readlines()
    for line in lines:
        word = sentence_word(line)
        words.append(word)

    return words
list_of_words = []
list_of_words = readFile("D:\\python\\naive_bayes\\preprocess_test.txt")
df = pd.DataFrame()
df['word'] = flatten(list_of_words)
words, counts  = np.unique(df['word'].values, return_counts=True)

freq, wrs = (list(i) for i in zip(*sorted(zip(counts, words), reverse = True)))

dicti = {}
doc_num = 1
for doc_words in list_of_words:
    np_doc_words = np.asarray(doc_words)
    w, c = np.unique(np_doc_words, return_counts=True)
    dicti[doc_num] = {}
    for i in range(len(w)):
        dicti[doc_num][w[i]] = c[i]
    doc_num = doc_num + 1
print("processing...")
n = 10000
out = {}
index = 1
features = wrs[0:n]

for k in dicti.keys():
    row = []
    cc = 0
    out[index] = {}
    for f in features:
        if(f in dicti[k].keys()):
           out[index][cc] = dicti[k][f]
           row.append(dicti[k][f])
        else:
            row.append(0)
        cc = cc + 1
    index = index + 1
fo = open("D:\\python\\naive_bayes\\x_test.txt", "w+")

for k in dicti.keys():
    index = 0
    row = []
    for wor in features:
        if wor in dicti[k].keys():
            charr = []
            charr.append(str(index))
            charr.append(str(float(dicti[k][wor])))
            charr = ':'.join(charr)
            row.append(charr)
        index = index + 1
    row = ' '.join(row)
    fo.writelines(row+'\n')
fo.close()
