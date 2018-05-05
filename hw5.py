# -*- coding: utf-8 -*-
"""
Created on Wed May  2 12:08:25 2018

@author: yumen
"""

import pandas as pd
from sklearn import preprocessing, cross_validation, svm, neighbors
from sklearn.linear_model import LinearRegression
import numpy as np
import math
from textblob import TextBlob

''' 
important quantities from train_frame:
    independent:
        
    HelpfulnessNumerator: Number of users who find the review helpfull
    HelpfulnessDenominator: Number of users who indicated they found the review helpfull
    
    dependent:
    Score: rating between 1 and 5 


    Id: Unique identifier of the review

Extracting important information and storing it in a new csv file (SO we don't need to to it everytime we want to process)
'''
train_frame = pd.DataFrame.from_csv("train.csv")  
file = open("trained_with_summary.csv", "w")

file.write("ProductId,HelpfulnessDenominator,HelpfulnessNumerator,Text,Score\n")
for i in range(250):
    try:
        for row in train_frame.loc[i].iterrows():
            text = row[1]["Text"]
            blob = TextBlob(text)
            Id = str(row[1]["ProductId"])
            info = Id + "," + str(i)+ "," +str(row[1]["HelpfulnessNumerator"]) +"," + str(blob.sentiment.polarity)+","+str(row[1]["Score"])+"\n"
            file.write(info)
    except Exception:
            try:
                text = train_frame["Text"]
                blob = TextBlob(text)
                Id = str(row["ProductId"])
                info = Id + "," + str(i)+ "," +str(row["HelpfulnessNumerator"]) +"," + str(blob.sentiment.polarity)+"," +str(row["Score"])+"\n"
                file.write(info)
            except Exception:
                print(i," is throwing an exception")
file.close()
'''
Training algorithm and obatining confidence interval and Cross-validation

'''
trained_frame = pd.DataFrame.from_csv("trained.csv")
train_frame = trained_frame.dropna()
Y = np.array([train_frame.loc[:,"Score"].values])
X = np.array([train_frame.loc[:, "HelpfulnessDenominator"].values, train_frame.loc[:,"HelpfulnessNumerator"].values, train_frame.loc[:,"Text"]])
Y = Y.T
X = X.T
X_train, X_test,Y_train, Y_test = cross_validation.train_test_split(X,Y, test_size = 0.8)
Y_test = np.ravel(Y_test)
Y_train = np.ravel(Y_train)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)
confidence =  clf.score(X_test, Y_test)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_test, Y_test)
print(confidence, scores)


#Obatining review approximations and writing to CSV 
results = open("results.csv", "w")
results.write("ID,Score\n")
for i in trained_frame.iterrows():
    if math.isnan(i[1]["Score"]):
        predict = np.array([i[1]["HelpfulnessDenominator"], i[1]["HelpfulnessNumerator"], i[1]["Text"]])
        predict = predict.reshape(1,-1)
        prediction = clf.predict(predict)
        prediction = int(prediction[0])
        print(i[0], prediction)
        info = str(i[0])+"," + str(prediction) +"\n"
        results.write(info)
results.close()
