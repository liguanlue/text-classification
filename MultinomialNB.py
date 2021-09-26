import numpy as np


class My_MultinomialNB(object):
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = None
        self.classes = None
        self.conditional_prob = None
        self.predict_prob = None
        '''
        fit_class: Whether to learn the prior probability of the class, False uses a unified prior
        class_prior: The prior probability of the class, if specified, the prior cannot be adjusted according to the data

    '''

    def fit(self, x, y):
        # Calculate the prior probability of category y
        self.classes = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]
        class_num = len(self.classes)
        self.class_prior = {}
        for d in self.classes:
            #c_num = np.sum(np.equal(y, d))
            #c_num = np.sum(d.count(d)
            c_num = np.sum(y == d)
            print(c_num)
            self.class_prior[d] = (c_num + self.alpha) / (float(len(y) + class_num * self.alpha))
            print('c_num:{}  len(y):{}')
        print("Category prior probability")
        print(self.class_prior)
        # Calculate conditional probability ------ polynomial
        self.conditional_prob = {}  # {x1|y1:p1,x2|y1:p2,.....,x1|y2:p3,x2|y2:p4,.....}
        for yy in self.class_prior.keys():
            y_index = [i for i, label in enumerate(y) if label == yy]
            print(y_index)# Prior Probability of Label
            for i in range(len(x)):
                x_class = np.unique(x[i])
                for c in list(x_class):
                    x_index = [x_i for x_i, value1 in enumerate(list(x[i])) if value1 == c]
                    xy_count = len(set(x_index) & set(y_index))
                    pkey = str(c) + '|' + str(yy)
                    self.conditional_prob[pkey] = (xy_count + self.alpha) / (
                                float(len(y_index)) + len(list(np.unique(x[i]))))
        return self

    def predict(self, X_test):  # Here, only one sample can be input to test, and multiple samples can be tested by adding loops. Gaussian model can be realized
        self.predict_prob = {}
        for i in self.classes:
            self.predict_prob[i] = self.class_prior[i]

            for d in X_test:
                tkey = str(d) + '|' + str(i)
                self.predict_prob[i] = self.predict_prob[i] * self.conditional_prob[tkey]
        label = max(self.predict_prob, key=self.predict_prob.get)
        return label
