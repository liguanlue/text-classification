# -*- coding: utf-8 -*-

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from MultinomialNB import My_MultinomialNB
import time

class Corpus(object):
    def __init__(self):
        self.word2idx = {}
        self.tags = defaultdict(int)
        self.docs = []
        self.total = 0 # Total number of articles
        self.docidx = 0
        self.catemun = 100

    def MyPrint(self,count,location):
        location = "./result/"+location+".txt"
        with open(location,'w',encoding = 'UTF-8') as result:
            result.write(count)

    def save_sparse_csr(self, array, location):
        location = "./result/" + location + ".txt"
        np.savez(location, data=array.data, indices=array.indices,
                 indptr=array.indptr, shape=array.shape)

    def Read_data(self,location):
        with open(location, 'r', encoding='UTF-8') as result:
            return result.read().strip().split('\n')

    def process_data(self):
        start = time.clock()
        # Word vector
        train_texts = []
        test_texts = []
        train_label = []
        test_label = []
        #locations = ['0-2','2-3,'3-4','4-6','8-10','10-12']
        locations = ['2-3']
        for i in locations:
            train_texts += self.Read_data('./data1/test'+i+'/TrainData.csv')
            train_label += self.Read_data('./data1/test' + i + '/TrainLable.csv')
            test_texts += self.Read_data('./data1/test' + i + '/TestData.csv')
            test_label += self.Read_data('./data1/test' + i + '/TestLable.csv')

        all_text = train_texts + test_texts
        count_v0 = CountVectorizer();
        counts_all = count_v0.fit_transform(all_text);
        count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_);
        counts_train = count_v1.fit_transform(train_texts);
        count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_);
        counts_test = count_v2.fit_transform(test_texts);

        #feature_name = counts_test.get_feature_names()
        #print(feature_name)
        print("the shape of train is " + repr(counts_train.shape))
        #self.save_sparse_csr(counts_train,"counts_train")
        #self.save_sparse_csr(counts_test, "counts_test")
        #print(counts_train)
        tfidftransformer = TfidfTransformer();
        train_data = tfidftransformer.fit(counts_train).transform(counts_train);
        test_data = tfidftransformer.fit(counts_test).transform(counts_test);
        #print(type(test_data))
        print("the shape of test is " + repr(counts_test.shape))

        #self.save_sparse_csr(train_data,"train_data")
        #self.save_sparse_csr(test_data, "test_data")

        x_train = train_data
        y_train = train_label
        x_test = test_data
        y_test = test_label

        x_train=x_train.todense()
        x_test=x_test.todense()

        print('(3) Naive Bayes...')

        clf = My_MultinomialNB(alpha=0.01)
        clf.fit(x_train, y_train);

        train_over = time.clock()
        preds = clf.predict(x_test);
        tags_name = ["baby", "discovery", "ent", "finance", "game", "history", "military", "sports", "tech", "travel"]
        tags = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]
        #ConfuMatri = np.zeros([10,10],dtype= [("0",int),("1",int),("2",int),("3",int),("4",int),("5",int),("6",int),("7",int),("8",int),("9",int),("10",int),])
        ConfuMatri = np.zeros([10,10],dtype= int)
        preds = preds.tolist()
        for i, pred in enumerate(preds):
            if pred == y_test[i]:
                ConfuMatri[int(pred),int(pred)] += 1
            else: ConfuMatri[int(y_test[i]),int(pred)] += 1
        test_over = time.clock()
        ##Output result
        print("Confusion matrix：")
        print(ConfuMatri)
        for i,tag in enumerate(tags):
            print("%d %10s Accuracy：%4f  Recall rate：%4f" %(i,tags_name[i],ConfuMatri[i,i]/sum(ConfuMatri[i]),ConfuMatri[i,i]/sum(ConfuMatri[:,i])))
        print("train time: {}  test time: {}".format(train_over-start,test_over-train_over))



if __name__ == '__main__':
    a = Corpus()
    a.process_data()