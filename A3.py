import pandas as pd 
import csv
import sys
import numpy as np
from sklearn.svm import SVC
#pd.set_option('display.max_colwidth', -1)

def main():
    for args in sys.argv:
        print(str(args))
    # reading test and train dataset from the names supplied on command line
    test_dataset = pd.read_csv(sys.argv[2],sep='\t')
    train_dataset = pd.read_csv(sys.argv[1],sep='\t')
    
    traindata_targets = train_dataset['group']
    traindata_features = train_dataset.loc[:,train_dataset.columns!='group']

    X = traindata_features
    y = traindata_targets
    print("Fitting the training data on SVC Method...")
    est = SVC(C=1000,kernel='linear',probability=True)
    est.fit(X, y)
    print("Predicting confidence score for different groups...")
    y_score = est.predict_proba(test_dataset)
    print("Writing the confidence predictions ")
    y_score.to_csv("g13_predictions.txt", sep="\t", index=False,float_format='%.4f')

if __name__== '__main__':
    main()
