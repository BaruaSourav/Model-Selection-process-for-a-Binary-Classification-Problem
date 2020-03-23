import pandas as pd 
import csv
import sys
import numpy as np
import decimal 
import scipy.stats as stats
import sklearn.datasets as datasets
from sklearn.svm import SVC
#pd.set_option('display.max_colwidth', -1)

def main():
    for args in sys.argv:
        print(str(args))
    # reading test and train dataset from the names supplied on command line
    test_dataset = pd.read_csv(sys.argv[1],sep='\t')
    train_dataset = pd.read_csv(sys.argv[2],sep='\t')
    
    traindata_targets = train_dataset['group']
    traindata_features = train_dataset.loc[:,train_dataset.columns!='group']

    X = traindata_features
    y = traindata_targets

    est = SVC(C=1000,kernel='linear',probability=True)
    est.fit(X, y)
    y_score = est.predict_proba(test_dataset)

    
if __name__== '__main__':
    main()
