# TODO - only if we have time - I agree with that

import h5py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score




data = np.load('C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data\\out_dataset_norm_subset.npy')
flattened_data =np.array([data_matrix.flatten() for data_matrix in data])

labels = np.load('C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data\\out_labels_norm_subset.npy')

X_train, X_test, Y_train, Y_test =train_test_split(flattened_data, labels, test_size = 0.2, random_state = 7)
model = LinearSVC(C=0.0000005, dual=False, verbose= 1, class_weight='balanced')
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print(precision_score(Y_test, y_pred, average = None))
print(accuracy_score(Y_test, y_pred))


print('finished')



