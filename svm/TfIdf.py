#tao tfidf
from scipy import sparse,io
from sklearn.feature_extraction.text import TfidfVectorizer
import _pickle as cPickle

def readFile(path):
    list_of_word = []
    file = open(path,"r")
    lines = file.readlines()
    for line in lines:
        #cprint(line)
        list_of_word.append(line)
    return list_of_word


words = readFile("D:\\python\\svm\\preprocess_test.txt")
#print(words)
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000)
X = vectorizer.fit_transform(words)
terms = vectorizer.get_feature_names()
f = open("D:\\python\\svm\\features.txt", "w+")
for i in terms:
    f.writelines(i+'\n')
f.close()

io.mmwrite('D:\\python\\svm\\x_test.mtx', X)
