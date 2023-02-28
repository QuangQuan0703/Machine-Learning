from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np
#train data
d1= [2,1,1,0,0,0,0,0,0]
d2= [1,1,0,1,1,0,0,0,0]
d3= [0,1,0,0,1,1,0,0,0]
d4= [0,1,0,0,0,0,1,1,1]

train_data = np.array([d1,d2,d3,d4])
label = np.array(['B','B','B','N'])

d5 = np.array([[2,0,0,1,0,0,0,1,0]])
clf = MultinomialNB ()
clf.fit(train_data, label)
print('Predicting class of d5:', str(clf.predict(d5)[0]))
print('Probability of d5 in each class:', clf.predict_proba(d5))

d6 = np.array([[0,1,0,0,0,0,0,1,1]])
asd = BernoulliNB()
asd.fit(train_data, label)
print('Predicting class of d6:', str(asd.predict(d6)))
print('Probability of d5 in each class:', asd.predict_proba(d6))

