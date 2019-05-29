
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from scipy import sparse, io
from sklearn import svm
def sentence(line):
    words = line[0:len(line)].strip().split(" ")
    return words



def readFile(path):
    row = []
    col = []
    data = []
    index = 0
    fo = open(path,"r")
    lines = fo.readlines()
    for line in lines:
        words = sentence(line)
        for wo in words:
            if len(wo) != 0:
                wrs = wo[0:len(wo)].strip().split(":")
                row.append(index)
                col.append(int(wrs[0]))
                data.append(float(wrs[1]))
        index = index + 1
    return row, col, data
df = pd.DataFrame()

df['row'], df['col'], df['data'] = readFile("D:\\python\\naive_bayes\\x_test.txt")

X_train = sparse.coo_matrix((df['data'].values, (df['row'].values, df['col'].values)), shape=(5000, 10000))


fo = open("D:\\python\\naive_bayes\\cate_test.txt", "r")
doc = fo.readlines()
cate = []
for line in doc:
    cate.append(line.strip())
fo.close()
Y_train = np.asarray(cate)


clf = MultinomialNB()
clf.fit(X_train, Y_train)
Y_predict_tr = clf.predict(X_train)
print(clf.score(X_train, Y_train))
print(classification_report(Y_train, Y_predict_tr))
