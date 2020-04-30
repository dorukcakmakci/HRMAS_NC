import pickle 
import pdb
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,\
                             fbeta_score, roc_auc_score, precision_recall_curve, auc,\
                             brier_score_loss)

sys.path.insert(1, '../')

# import data
# uncomment corresponding line to process the corresponding problem
dataset_name = "benign_aggressive" 

fold_1_data = f"../data/folds/{dataset_name}_hydrogens_fold_1"
fold_1_labels = f"../data/folds/{dataset_name}_labels_fold_1"
fold_2_data = f"../data/folds/{dataset_name}_hydrogens_fold_2"
fold_2_labels = f"../data/folds/{dataset_name}_labels_fold_2"
fold_3_data = f"../data/folds/{dataset_name}_hydrogens_fold_3"
fold_3_labels = f"../data/folds/{dataset_name}_labels_fold_3"
fold_4_data = f"../data/folds/{dataset_name}_hydrogens_fold_4"
fold_4_labels = f"../data/folds/{dataset_name}_labels_fold_4"
fold_5_data = f"../data/folds/{dataset_name}_hydrogens_fold_5"
fold_5_labels = f"../data/folds/{dataset_name}_labels_fold_5"
fold_6_data = f"../data/folds/{dataset_name}_hydrogens_fold_6"
fold_6_labels = f"../data/folds/{dataset_name}_labels_fold_6"
fold_7_data = f"../data/folds/{dataset_name}_hydrogens_fold_7"
fold_7_labels = f"../data/folds/{dataset_name}_labels_fold_7"
fold_8_data = f"../data/folds/{dataset_name}_hydrogens_fold_8"
fold_8_labels = f"../data/folds/{dataset_name}_labels_fold_8"

# load spectra
with open(fold_1_data, 'rb') as f:
    fold_1_hydrogens = pickle.load(f, encoding='bytes')
with open(fold_2_data, 'rb') as f:
    fold_2_hydrogens = pickle.load(f, encoding='bytes')
with open(fold_3_data, 'rb') as f:
    fold_3_hydrogens = pickle.load(f, encoding='bytes')
with open(fold_4_data, 'rb') as f:
    fold_4_hydrogens = pickle.load(f, encoding='bytes')
with open(fold_5_data, 'rb') as f:
    fold_5_hydrogens = pickle.load(f, encoding='bytes')
with open(fold_6_data, 'rb') as f:
    fold_6_hydrogens = pickle.load(f, encoding='bytes')
with open(fold_7_data, 'rb') as f:
    fold_7_hydrogens = pickle.load(f, encoding='bytes')
with open(fold_8_data, 'rb') as f:
    fold_8_hydrogens = pickle.load(f, encoding='bytes')

# load labels
with open(fold_1_labels, 'rb') as f:
    fold_1_labels = pickle.load(f, encoding='bytes')
with open(fold_2_labels, 'rb') as f:
    fold_2_labels = pickle.load(f, encoding='bytes')
with open(fold_3_labels, 'rb') as f:
    fold_3_labels = pickle.load(f, encoding='bytes')
with open(fold_4_labels, 'rb') as f:
    fold_4_labels = pickle.load(f, encoding='bytes')
with open(fold_5_labels, 'rb') as f:
    fold_5_labels = pickle.load(f, encoding='bytes')
with open(fold_6_labels, 'rb') as f:
    fold_6_labels = pickle.load(f, encoding='bytes')
with open(fold_7_labels, 'rb') as f:
    fold_7_labels = pickle.load(f, encoding='bytes')
with open(fold_8_labels, 'rb') as f:
    fold_8_labels = pickle.load(f, encoding='bytes')

fold_1_hydrogens = fold_1_hydrogens.T
fold_2_hydrogens = fold_2_hydrogens.T
fold_3_hydrogens = fold_3_hydrogens.T
fold_4_hydrogens = fold_4_hydrogens.T
fold_5_hydrogens = fold_5_hydrogens.T
fold_6_hydrogens = fold_6_hydrogens.T
fold_7_hydrogens = fold_7_hydrogens.T
fold_8_hydrogens = fold_8_hydrogens.T

print(dataset_name)
print("Fold 1 Hydrogens: ", fold_1_hydrogens.shape, "\t Fold 1 Labels: ", fold_1_labels.shape)

bin_size = 2

temp = []
for i in range(fold_1_hydrogens.shape[0]):
    temp.append(np.maximum.reduceat(fold_1_hydrogens[i], np.r_[:fold_1_hydrogens[i].shape[0]:bin_size]))
fold_1_hydrogens = np.array(temp)

temp = []
for i in range(fold_2_hydrogens.shape[0]):
    temp.append(np.maximum.reduceat(fold_2_hydrogens[i], np.r_[:fold_2_hydrogens[i].shape[0]:bin_size]))
fold_2_hydrogens = np.array(temp)

temp = []
for i in range(fold_3_hydrogens.shape[0]):
    temp.append(np.maximum.reduceat(fold_3_hydrogens[i], np.r_[:fold_3_hydrogens[i].shape[0]:bin_size]))
fold_3_hydrogens = np.array(temp)

temp = []
for i in range(fold_4_hydrogens.shape[0]):
    temp.append(np.maximum.reduceat(fold_4_hydrogens[i], np.r_[:fold_4_hydrogens[i].shape[0]:bin_size]))
fold_4_hydrogens = np.array(temp)

temp = []
for i in range(fold_5_hydrogens.shape[0]):
    temp.append(np.maximum.reduceat(fold_5_hydrogens[i], np.r_[:fold_5_hydrogens[i].shape[0]:bin_size]))
fold_5_hydrogens = np.array(temp)

temp = []
for i in range(fold_6_hydrogens.shape[0]):
    temp.append(np.maximum.reduceat(fold_6_hydrogens[i], np.r_[:fold_6_hydrogens[i].shape[0]:bin_size]))
fold_6_hydrogens = np.array(temp)

temp = []
for i in range(fold_7_hydrogens.shape[0]):
    temp.append(np.maximum.reduceat(fold_7_hydrogens[i], np.r_[:fold_7_hydrogens[i].shape[0]:bin_size]))
fold_7_hydrogens = np.array(temp)

temp = []
for i in range(fold_8_hydrogens.shape[0]):
    temp.append(np.maximum.reduceat(fold_8_hydrogens[i], np.r_[:fold_8_hydrogens[i].shape[0]:bin_size]))
fold_8_hydrogens = np.array(temp)


spectrum_length = fold_1_hydrogens.shape[1]

# 8 fold cross validation
auroc_folds = []
aupr_folds = []
precision_folds = []
recall_folds= []
f1_folds = []
train_time_folds = []
test_time_folds = []
acc_folds = []

for test_index in [1, 2, 3, 4, 5, 6, 7, 8]:
    print()
    print("************ Test Fold " + str(test_index) + " ************")
    print()

    train_indices = list(range(1,9))
    train_indices.remove(test_index)
    print(train_indices)

    test_hydrogens = eval("fold_" + str(test_index) + "_hydrogens")
    test_labels = eval("fold_" + str(test_index) + "_labels")
    train_hydrogens_tuple = tuple([eval('fold_' + str(x) + '_hydrogens') for x in train_indices])
    train_labels_tuple = tuple([eval('fold_' + str(x) + '_labels') for x in train_indices])
    train_hydrogens = np.vstack(train_hydrogens_tuple)
    train_labels = np.vstack(train_labels_tuple)
    # print("Train hydrogens shape: ", train_hydrogens.shape, "\tTrain labels shape: ", train_labels.shape)
    # print("Test hydrogens shape: ", test_hydrogens.shape, "\tTrain labels shape: ", test_labels.shape)

    # scale all samples according to training set
    scaler = preprocessing.MinMaxScaler().fit(train_hydrogens)
    train_hydrogens_normalized = scaler.transform(train_hydrogens)
    test_hydrogens_normalized = scaler.transform(test_hydrogens)

    train_start = time.time()
    clf = SVC(C=100, gamma='scale', kernel='rbf', class_weight='balanced', probability=True)
    clf.fit(train_hydrogens_normalized, np.ravel(train_labels))
    train_end = time.time()

    test_start = time.time()
    test_pred = clf.predict(test_hydrogens_normalized)
    test_end = time.time()
    test_pred_prob = clf.predict_proba(test_hydrogens_normalized)
    
    cm = confusion_matrix(test_labels, test_pred)

    auroc = roc_auc_score(test_labels, test_pred_prob[:,1])
    auroc_folds.append(auroc)

    precision, recall, thresh = precision_recall_curve(test_labels, test_pred_prob[:,1])
    aupr = auc(recall, precision)
    aupr_folds.append(aupr)

    prec = precision_score(test_labels, test_pred, average='binary')
    precision_folds.append(prec)

    rec = recall_score(test_labels, test_pred, average='binary')
    recall_folds.append(rec)

    f1 = f1_score(test_labels, test_pred, average='binary')
    f1_folds.append(f1)

    acc_folds.append((cm[1,1] + cm[0,0]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]))
    train_time_folds.append(train_end - train_start)
    test_time_folds.append(test_end - test_start)

    accuracy = (cm[1,1] + cm[0,0]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])


    print("Class based accuracies: ")
    for i in range(2):
        print("Class % d: % f" %(i, cm[i,i]/np.sum(cm[i,:])))
    print("Confusion Matrix")
    print(cm)
    print("Accuracy: ",  accuracy)
    print("Precision: ", prec)
    print("Recall: ", rec)
    print("F1 Score: ", f1)
    print("AUROC: ", auroc)
    print("AUPR: ", aupr)

# # append metrics per fold to a run file
# with open("./logs/SVM/auroc_scores.txt", "a") as f:
#     for auroc in auroc_folds:
#         f.write("%f\n" % (auroc))
# with open("./logs/SVM/aupr_scores.txt", "a") as f:
#     for aupr in aupr_folds:
#         f.write("%f\n" % (aupr))
# with open("./logs/SVM/f1_scores.txt", "a") as f:
#     for f1 in f1_folds:
#         f.write("%f\n" % (f1))  
# with open("./logs/SVM/precision_scores.txt", "a") as f:
#     for prec in precision_folds:
#         f.write("%f\n" % (prec))
# with open("./logs/SVM/recall_scores.txt", "a") as f:
#     for rec in recall_folds:
#         f.write("%f\n" % (rec)) 
# with open("./logs/SVM/acc_scores.txt", "a") as f:
#     for acc in acc_folds:
#         f.write("%f\n" % (acc))
# with open("./logs/SVM/test_timing.txt", "a") as f:
#     for test in test_time_folds:
#         f.write("%f\n" % (test))
# with open("./logs/SVM/train_timing.txt", "a") as f:
#     for train in train_time_folds:
#         f.write("%f\n" % (train))    