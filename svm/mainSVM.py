from scipy import sparse, io
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

def sentence_line(line):
    
    words = line[0:len(line)].strip().split("<###>")
    return words


def sentence_word(words):
    words = words[0:len(words)].strip().split(" ")
    return words

def readFile(path):
    f = open(path,"r")
    doc = f.readlines()

    for line in doc :

        list_of_words.append(line)
       
    f.close()
    
la = []
fio = open("D:\\python\\svm\\cate_test.txt", "r")
line = fio.readlines()
for ll in line:
    la.append(ll.strip())
df = pd.DataFrame()
df['labels'] = la
print(df['labels'].values)



X_train = io.mmread('D:\\python\\svm\\x_test.mtx')


Y_train = df['labels'].values
clf = svm.SVC(gamma='scale')
clf.fit(X_train, Y_train)
Y_predict_tr = clf.predict(X_train)
print(clf.score(X_train, Y_train))
print(classification_report(Y_train, Y_predict_tr))
