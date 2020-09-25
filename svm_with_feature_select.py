import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

data = pd.read_csv('C:/Users/Matt/Documents/max_relevancy_min_redundancy/breast_cancer_data.csv')
y = data.pop('diagnosis')
x = data
x_selected = x[['area_se','texture_worst','concavity_worst','fractal_dimension_worst','compactness_mean']]
x_train, x_test, y_train, y_test = train_test_split(x_selected,y, test_size=0.33)

clf = svm.SVC()
clf.fit(x_train, y_train)




accuracy_array = []
for i in range(100):
    x_test_sample, x_test_leftover, y_test_sample, y_test_leftover = train_test_split(x_test, y_test, test_size = .25)
    sample_accuracy = clf.score(x_test_sample, y_test_sample)
    accuracy_array.append(sample_accuracy)
    
print(np.mean(accuracy_array))
print(np.std(accuracy_array))