import h5py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, f1_score
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from utils.plot import save_confusion_matrix

def train_and_val(test_size, data_path='C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data', from_model=True, stratified=True, save_model=False):
    data = np.load(
        os.path.join(data_path,'out_dataset_def.npy'))
    flattened_data = np.array([data_matrix.flatten() for data_matrix in data])

    labels = np.load(
        os.path.join(data_path, 'out_labels_def.npy'))
    if from_model:
        filename = 'trained_svm.sav'
        # load the model from disk
        model = pickle.load(open(os.path.join(data_path,filename), 'rb'))
    else:
        model = LinearSVC(C=0.0000005, dual=False, verbose=1, class_weight='balanced')

    if stratified:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        for train_idx, test_idx in sss.split(flattened_data, labels):
            X_train, X_test = flattened_data[train_idx], flattened_data[test_idx]
            Y_train, Y_test = labels[train_idx], labels[test_idx]
    else:
        X_train, X_test, Y_train, Y_test =train_test_split(flattened_data, labels, test_size = 0.2, random_state = 7)

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    if save_model:
        #save the model to disk
        filename = 'trained_svm.sav'
        pickle.dump(model, open(os.path.join(
            'C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data',
            filename), 'wb'))

        print('f1 score')
        print(f1_score(Y_test, y_pred, average='weighted'))
        print('accuracy score')
        print(accuracy_score(Y_test, y_pred))


def train_and_test(data_path='C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data',
                  save_model=True,
                from_model = True):

    ''''
        train the model on the whole dataset and test it
        on test dataset, taking only test segment with one instrument in it
    '''

    train_data = np.load(
        os.path.join(data_path, 'out_dataset_def.npy'))

    X_train = np.array([data_matrix.flatten() for data_matrix in train_data])

    Y_train = np.load(
        os.path.join(data_path, 'out_labels_def.npy'))

    test_data = np.load(
        os.path.join(data_path, 'out_dataset_test.npy'))
    X_test = np.array([data_matrix.flatten() for data_matrix in test_data])

    Y_test = np.load(
        os.path.join(data_path, 'out_labels_test.npy'))

    print('number of test sample: '+str(len(Y_test)))


    if from_model:
        print('loading model...')
        filename = 'trained_svm_on_all_data.sav'
        # load the model from disk
        model = pickle.load(open(os.path.join(data_path, filename), 'rb'))
    else:
        model = LinearSVC(C=0.0000005, dual=False, verbose=1, class_weight='balanced')
        print('training model on whole training set...')
        model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    if save_model:
        # save the model to disk
        filename = 'trained_svm_on_all_data.sav'
        pickle.dump(model, open(os.path.join(
            'C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data',
            filename), 'wb'))

        print('f1 score')
        print(f1_score(Y_test, y_pred, average='weighted'))
        print('accuracy score')
        print(accuracy_score(Y_test, y_pred))

        class_list = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        save_confusion_matrix(Y_test, y_pred, class_list, 'svm')



def print_features_importance(data_path='C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\music_informatics\\data'):
    # we retrieve the feat importance for each features
    features_importance = [[], [], [], [], [], [], [], [], [], [], []]
    class_list = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

    filename = 'trained_svm.sav'
    # load the model from disk
    model = pickle.load(open(os.path.join(data_path, filename), 'rb'))
    coefficient = model.coef_

    for i in range(11):
        features_importance[i] = []
        splits = np.array_split(coefficient[i], 65)
        features_importance[i].append(np.array(splits).prod(axis=0))


    image_path = 'C:\\Users\\Prestige\\Desktop\\Paolo\\UNi\\ERASMUS\\KTH\\P1\\Music Informatics\\fp_musinfo\\images'
    plt.tight_layout()
    for i in range(11):
        plt.bar(range(0,25), features_importance[i][0])
        plt.title('features importance of '+ class_list[i] )
        plt.savefig(os.path.join(image_path, 'feat_imp_mul_'+class_list[i]+'.pdf'), format='pdf')
        plt.show()





if __name__ == '__main__':
    print_features_importance()