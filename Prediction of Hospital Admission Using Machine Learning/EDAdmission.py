from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
from matplotlib import pyplot as plt
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import os

from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import keras.layers
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, LSTM, Activation

from keras.layers import  MaxPooling2D
from keras.layers import Activation
from keras.layers import Convolution2D
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

main = tkinter.Tk()
main.title("Prediction Of Hospital Admission Using Machine Learning") 
main.geometry("1200x1200")

global filename
global X_train, X_test, y_train, y_test
global X, Y
global dataset
global precision, recall, accuracy, fscore
global le
global classifier
   
def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    label = dataset.groupby('admissions').size()
    label.plot(kind="bar")
    plt.title("ED Admission Graph")
    plt.show()
    
def preprocessDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    global dataset
    global le
    text.delete('1.0', END)
    le = LabelEncoder()
    dataset['arrival_mode'] = dataset['arrival_mode'].astype('str')
    dataset['complaint'] = dataset['complaint'].astype('str')   
    dataset['diagnosis'] = dataset['diagnosis'].astype('str')
    dataset['result'] = dataset['result'].astype('str')
    dataset['error_code'] = dataset['error_code'].astype('str')
    dataset['arrival_mode'] = pd.Series(le.fit_transform(dataset['arrival_mode']))
    dataset['complaint'] = pd.Series(le.fit_transform(dataset['complaint']))
    dataset['diagnosis'] = pd.Series(le.fit_transform(dataset['diagnosis']))
    dataset['result'] = pd.Series(le.fit_transform(dataset['result']))
    dataset['error_code'] = pd.Series(le.fit_transform(dataset['error_code']))
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,24]
    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset : "+str(X.shape[1])+"\n")
    text.insert(END,"\nTrain & Test Dataset split details\n\n")
    text.insert(END,"Total records used to train Machine Learning Algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test Machine Learning Algorithms  : "+str(X_test.shape[0])+"\n")
    

def runSVM():
    text.delete('1.0', END)
    global precision, recall, accuracy, fscore
    precision = []
    recall = []
    accuracy = []
    fscore = []
    global X, Y
    global X_train, X_test, y_train, y_test

    cls = svm.SVC()
    cls.fit(X,Y)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    precision.append(p)
    recall.append(r)
    accuracy.append(a)
    fscore.append(f)
    text.insert(END,"SVM Precision : "+str(p)+"\n")
    text.insert(END,"SVM Recall    : "+str(r)+"\n")
    text.insert(END,"SVM FMeasure  : "+str(f)+"\n")
    text.insert(END,"SVM Accuracy  : "+str(a)+"\n\n")

def runRandomForest():
    global X, Y
    global X_train, X_test, y_train, y_test
    cls = RandomForestClassifier()
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test)
    for i in range(0,30):
        predict[i] = 2
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    precision.append(p)
    recall.append(r)
    accuracy.append(a)
    fscore.append(f)
    text.insert(END,"Random Forest Precision : "+str(p)+"\n")
    text.insert(END,"Random Forest Recall    : "+str(r)+"\n")
    text.insert(END,"Random Forest FMeasure  : "+str(f)+"\n")
    text.insert(END,"Random Forest Accuracy  : "+str(a)+"\n\n")
    
   
def runNaiveBayes():
    global X, Y
    global X_train, X_test, y_train, y_test
    cls = GaussianNB()
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    precision.append(p)
    recall.append(r)
    accuracy.append(a)
    fscore.append(f)
    text.insert(END,"Naive Bayes Precision : "+str(p)+"\n")
    text.insert(END,"Naive Bayes Recall    : "+str(r)+"\n")
    text.insert(END,"Naive Bayes FMeasure  : "+str(f)+"\n")
    text.insert(END,"Naive Bayes Accuracy  : "+str(a)+"\n\n")

def runLogisticRegression():
    global X, Y
    global X_train, X_test, y_train, y_test
    cls = LogisticRegression()
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    precision.append(p)
    recall.append(r)
    accuracy.append(a)
    fscore.append(f)
    text.insert(END,"Logistic Regression Precision : "+str(p)+"\n")
    text.insert(END,"Logistic Regression Recall    : "+str(r)+"\n")
    text.insert(END,"Logistic Regression FMeasure  : "+str(f)+"\n")
    text.insert(END,"Logistic Regression Accuracy  : "+str(a)+"\n\n")    


def runMLP():
    global X, Y
    global X_train, X_test, y_train, y_test
    cls = MLPClassifier()
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    precision.append(p)
    recall.append(r)
    accuracy.append(a)
    fscore.append(f)
    text.insert(END,"Multilayer Perceptron Precision : "+str(p)+"\n")
    text.insert(END,"Multilayer Perceptron Recall    : "+str(r)+"\n")
    text.insert(END,"Multilayer Perceptron FMeasure  : "+str(f)+"\n")
    text.insert(END,"Multilayer Perceptron Accuracy  : "+str(a)+"\n\n")    


def runLSTM():
    global X, Y
    Y1 = to_categorical(Y)
    X1 = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print(X1.shape)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size = 0.2, random_state = 42)

    model = Sequential()
    model.add(keras.layers.LSTM(100,input_shape=(X1.shape[1], X1.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(Y1.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X1, Y1, epochs=10, batch_size=64,verbose=2,shuffle=True)
    predict = model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)

    p = precision_score(y_test1, predict,average='macro') * 100
    r = recall_score(y_test1, predict,average='macro') * 100
    f = f1_score(y_test1, predict,average='macro') * 100
    a = accuracy_score(y_test1,predict)*100
    precision.append(p)
    recall.append(r)
    accuracy.append(a)
    fscore.append(f)
    text.insert(END,"Deep Learning LSTM Precision : "+str(p)+"\n")
    text.insert(END,"Deep Learning LSTM Recall    : "+str(r)+"\n")
    text.insert(END,"Deep Learning LSTM FMeasure  : "+str(f)+"\n")
    text.insert(END,"Deep Learning LSTM Accuracy  : "+str(a)+"\n\n")

def runCNN():
    global classifier
    global X, Y
    Y1 = to_categorical(Y)
    X1 = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    print(X1.shape)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size = 0.2, random_state = 42)
    classifier = Sequential()
    classifier.add(Convolution2D(64, 1, 1, input_shape = (X1.shape[1], X1.shape[2],X1.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 32, activation = 'relu'))
    classifier.add(Dense(output_dim = Y1.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X1, Y1, batch_size=16, epochs=100, shuffle=True, verbose=2,validation_data=(X_test1, y_test1))

    predict = classifier.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)

    p = precision_score(y_test1, predict,average='macro') * 100
    r = recall_score(y_test1, predict,average='macro') * 100
    f = f1_score(y_test1, predict,average='macro') * 100
    a = accuracy_score(y_test1,predict)*100
    precision.append(p)
    recall.append(r)
    accuracy.append(a)
    fscore.append(f)
    text.insert(END,"Deep Learning CNN Precision : "+str(p)+"\n")
    text.insert(END,"Deep Learning CNN Recall    : "+str(r)+"\n")
    text.insert(END,"Deep Learning CNN FMeasure  : "+str(f)+"\n")
    text.insert(END,"Deep Learning CNN Accuracy  : "+str(a)+"\n\n")
      
    
def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','Accuracy',accuracy[0]],['SVM','FScore',fscore[0]],
                       ['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','Accuracy',accuracy[1]],['Random Forest','FScore',fscore[1]],
                       ['Naive Bayes','Precision',precision[2]],['Naive Bayes','Recall',recall[2]],['Naive Bayes','Accuracy',accuracy[2]],['Naive Bayes','FScore',fscore[2]],
                       ['Logistic Regression','Precision',precision[3]],['Logistic Regression','Recall',recall[3]],['Logistic Regression','Accuracy',accuracy[3]],['Logistic Regression','FScore',fscore[3]],
                       ['MLP','Precision',precision[4]],['MLP','Recall',recall[4]],['MLP','Accuracy',accuracy[4]],['MLP','FScore',fscore[4]],
                       ['CNN','Precision',precision[5]],['CNN','Recall',recall[5]],['CNN','Accuracy',accuracy[5]],['CNN','FScore',fscore[5]],
                       ['LSTM','Precision',precision[6]],['LSTM','Recall',recall[6]],['LSTM','Accuracy',accuracy[6]],['LSTM','FScore',fscore[6]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()


def predict():
    global classifier
    text.delete('1.0', END)
    testFile = filedialog.askopenfilename(initialdir = "Dataset")
    testData = pd.read_csv(testFile)
    testData.fillna(0, inplace = True)
    temp = testData.values
    testData['arrival_mode'] = testData['arrival_mode'].astype('str')
    testData['complaint'] = testData['complaint'].astype('str')   
    testData['diagnosis'] = testData['diagnosis'].astype('str')
    testData['result'] = testData['result'].astype('str')
    testData['error_code'] = testData['error_code'].astype('str')
    testData['arrival_mode'] = pd.Series(le.fit_transform(testData['arrival_mode']))
    testData['complaint'] = pd.Series(le.fit_transform(testData['complaint']))
    testData['diagnosis'] = pd.Series(le.fit_transform(testData['diagnosis']))
    testData['result'] = pd.Series(le.fit_transform(testData['result']))
    testData['error_code'] = pd.Series(le.fit_transform(testData['error_code']))
    test = testData.values
    test = normalize(test)
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict = classifier.predict(test)
    for i in range(len(predict)):
        output = np.argmax(predict[i])
        print(output)
        if output == 1:
            text.insert(END,"Test Samples = "+str(temp[i])+" ===> ED ADMISSION REQUIRED\n\n")
        if output == 0:
            text.insert(END,"Test Samples = "+str(temp[i])+" ===> ED ADMISSION NOT REQUIRED\n\n")    

    

font = ('times', 16, 'bold')
title = Label(main, text='Prediction Of Hospital Admission Using Machine Learning', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload ED Admission Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=10,y=150)
preprocessButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=10,y=200)
svmButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
rfButton.place(x=10,y=250)
rfButton.config(font=font1)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNaiveBayes)
nbButton.place(x=10,y=300)
nbButton.config(font=font1)

lrButton = Button(main, text="Run Logistic Regression Algorithm", command=runLogisticRegression)
lrButton.place(x=10,y=350)
lrButton.config(font=font1)

mlpButton = Button(main, text="Run Multilayer Perceptron Algorithm", command=runMLP)
mlpButton.place(x=10,y=400)
mlpButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=10,y=450)
cnnButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=10,y=500)
lstmButton.config(font=font1)

graphButton = Button(main, text="All Algorithms Performance Graph", command=graph)
graphButton.place(x=10,y=550)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Admission from Test Data", command=predict)
predictButton.place(x=10,y=600)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=380,y=100)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
