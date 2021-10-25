# TODO - only if we have time - I agree with that

import h5py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, f1_score



stratified = True
data = np.load('C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data\\out_dataset_spec.npy')
flattened_data =np.array([data_matrix.flatten() for data_matrix in data])

labels = np.load('C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data\\out_labels_spec.npy')

if stratified:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(flattened_data, labels):
        X_train, X_test = flattened_data[train_idx], flattened_data[test_idx]
        Y_train, Y_test = labels[train_idx], labels[test_idx]
else:
    X_train, X_test, Y_train, Y_test =train_test_split(flattened_data, labels, test_size = 0.2, random_state = 7)
model = LinearSVC(C=0.0000005, dual=False, verbose= 1, class_weight='balanced')
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)



print(precision_score(Y_test, y_pred, average = None))

print(f1_score(Y_test, y_pred, average = 'weighted'))
print(accuracy_score(Y_test, y_pred))


print('finished')



