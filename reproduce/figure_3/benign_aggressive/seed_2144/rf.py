import pickle 
import pdb
import sys
import time
import shap

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,\
                             fbeta_score, roc_auc_score, precision_recall_curve, auc,\
                             brier_score_loss)


sys.path.insert(1, '../')
from plot_shap import find_ppm_value, plot_all_shap_values, plot_top_k_shap_values
from load_data import *

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

    # scale all samples according to training set
    scaler = preprocessing.MinMaxScaler().fit(train_hydrogens)
    train_hydrogens_normalized = scaler.transform(train_hydrogens)
    test_hydrogens_normalized = scaler.transform(test_hydrogens)

    # parameters are found by grid search
    clf = RandomForestClassifier(max_depth=15, min_samples_leaf=10, min_samples_split=15, n_estimators=300, class_weight='balanced')
    clf.fit(train_hydrogens_normalized, np.ravel(train_labels))

    # shap value calculation and explanation
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(test_hydrogens_normalized)

    # for aggressive class
    if test_index == 6:
        plot_all_shap_values(shap_values[1],test_hydrogens_normalized,"./plots/fig_3_panel_b")