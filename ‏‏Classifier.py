import numpy as np
import os
import ntpath
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction import DictVectorizer
from Android_Permissions import load_permissions, get_permissions
import csv

clf_names = [
            "Linear Support Vector",
            "Random Forest",
            "Naive Bayes"
]

methods_names = [
            "Source Code"
#            "Combination"
#            "Permissions"
]

classifiers = [
    OneVsRestClassifier(LinearSVC(random_state=0)),
    RandomForestClassifier(n_estimators=50),
    MultinomialNB(alpha=0.01)
]

feature_selection_dim_names = ['100', '200', '500', '1000']
feature_selection_dim = [100, 200, 500, 1000]

CLASS_IDMAP_PATH = ".\\idMap.txt"
with open(CLASS_IDMAP_PATH) as f:
    Id2Apk = {int(line.strip().split(",")[0]) : (line.strip().split(",")[1], line.strip().split(",")[2], line.strip().split("\\")[-1]) for line in f}
f.close()

CLASS_MAP_PATH = ".\\classMap.txt"
with open(CLASS_MAP_PATH, 'r ') as f:
    Id2Family = {int(line.rstrip().split(";")[0]): line.rstrip().split(";")[1] for line in f.readlines()}
Family2Id = {v:k for k, v in Id2Family.iteritems()}
print "Family 2 id: ", Family2Id
f.close()

file = open('measures_families.csv', 'a')
fields = ('File Type','Method', 'True Family', 'Classifier', 'Max Features','Max FF', 'BoW Method', "Selected Features Dimensions", 'N-Token', 'Generalization','Primary Max FF','Accuracy','MisClassification')
writer = csv.writer(file, lineterminator='\n')
writer.writerow(fields)

apk2perm = load_permissions()

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

results_dir = ".\\matrix_run\\"

import pandas as pd
for src_dir, dirs, files in os.walk(results_dir):
    for file_ in files:
        file_ext = ntpath.basename(file_).split('.')[1]
        if file_ext == "csv":
            src_file = os.path.join(src_dir, file_)
            print (src_file)
            df = pd.read_csv(src_file, skiprows=1, header=0)
            print "Finished Loading"

            index = np.array(df.index.tolist())
            labels = index[:,1]
            feature_names = df.iloc[0, 1:].index.tolist()
            data_sc = df.iloc[:, 1:].values

            data_permissions_vectorized, labels = get_permissions(index[:,0], labels, apk2perm)

            for method in methods_names:
                if method == "Source Code":
                    data = data_sc
                elif method == "Permissions":
                    data = data_permissions_vectorized
                else: # Combination
                    data = np.concatenate((data_sc, data_permissions_vectorized), axis=1)

                # feature extraction
                for fs_dim_name, dim in zip(feature_selection_dim_names, feature_selection_dim):
                    BoW_method_name = ntpath.basename(file_).split('.')[0]
                    max_df_name = os.path.basename(src_dir).split('_')[0]
                    max_features_name = os.path.basename(src_dir).split('_')[1]

                    print "Len Data: ", len(data)
                    print "Len Labels: ", len(labels)

                    for clf_name, clf in zip(clf_names, classifiers):
                        total_cm = 0.0
                        fold_i = 0
                        print(datetime.datetime.now().time())
                        for train_index, test_index in skf.split(data, labels):
                            fold_i += 1
                            print "Running Fold", fold_i, "/", n_folds
                            X_train, X_test = data[train_index], data[test_index]
                            y_train, y_test = labels[train_index], labels[test_index]
                            # Feature Selection
                            fs = SelectKBest(score_func=chi2, k=dim)
                            X_train = fs.fit_transform(X_train, y_train)
                            X_test = fs.transform(X_test)
                            #Classification
                            clf.fit(X_train, y_train)
                            predicted = clf.predict(X_test)
                            total_cm += confusion_matrix(y_test, predicted, labels=range(len(Id2Family.keys())))

                        avg_acc = np.trace(total_cm) / float(np.sum(total_cm))
                        row_measures = ['APK', method, 'Total', clf_name, max_features_name, max_df_name, BoW_method_name, fs_dim_name, '3','ST','0.6', avg_acc, '-']
                        print(row_measures)
#                            writer.writerow(row_measures)

#                            cm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]

#                            thres = 0.1
#                            for i in range(cm.shape[0]):
#                                row_list = []
#                                for j in range(cm.shape[1]):
#                                    if cm[i, j] > thres:
#                                        row_list.append((str(round(cm[i, j], 2)), Id2Family[j]))
#                                row_list.sort(reverse=True)
#                                row_measures = ['APK', method, Id2Family[i], clf_name, max_features_name, max_df_name, BoW_method_name, fs_dim_name, '1', '-','-',cm[i, i], row_list]
#                                writer.writerow(row_measures)

file.close()
