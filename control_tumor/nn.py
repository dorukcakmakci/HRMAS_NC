import torch
import pickle 
import copy
import sys
import pdb
import time

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,\
                             fbeta_score, roc_auc_score, precision_recall_curve, auc,\
                             brier_score_loss)
from collections import Counter

sys.path.insert(1, '../')
from focal_loss import FocalLoss

# network related constants 
ITERATION = 140
NUM_CLASSES = 2
ETA = 10**-4

# two running environment options below:
# device = torch.device("cpu")
device = torch.device("cuda:0")
dtype = torch.float

# import data
# uncomment corresponding line to process the corresponding problem
dataset_name = "control_tumor" 

fold_1_data = f"../data/folds/{dataset_name}_hydrogens_fold_1"
fold_1_labels = f"../data/folds/{dataset_name}_labels_fold_1"
fold_1_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_1"
fold_1_nrmn = f"../data/folds/{dataset_name}_nrmn_1"

fold_2_data = f"../data/folds/{dataset_name}_hydrogens_fold_2"
fold_2_labels = f"../data/folds/{dataset_name}_labels_fold_2"
fold_2_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_2"
fold_2_nrmn = f"../data/folds/{dataset_name}_nrmn_2"

fold_3_data = f"../data/folds/{dataset_name}_hydrogens_fold_3"
fold_3_labels = f"../data/folds/{dataset_name}_labels_fold_3"
fold_3_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_3"
fold_3_nrmn = f"../data/folds/{dataset_name}_nrmn_3"

fold_4_data = f"../data/folds/{dataset_name}_hydrogens_fold_4"
fold_4_labels = f"../data/folds/{dataset_name}_labels_fold_4"
fold_4_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_4"
fold_4_nrmn = f"../data/folds/{dataset_name}_nrmn_4"

fold_5_data = f"../data/folds/{dataset_name}_hydrogens_fold_5"
fold_5_labels = f"../data/folds/{dataset_name}_labels_fold_5"
fold_5_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_5"
fold_5_nrmn = f"../data/folds/{dataset_name}_nrmn_5"

fold_6_data = f"../data/folds/{dataset_name}_hydrogens_fold_6"
fold_6_labels = f"../data/folds/{dataset_name}_labels_fold_6"
fold_6_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_6"
fold_6_nrmn = f"../data/folds/{dataset_name}_nrmn_6"

fold_7_data = f"../data/folds/{dataset_name}_hydrogens_fold_7"
fold_7_labels = f"../data/folds/{dataset_name}_labels_fold_7"
fold_7_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_7"
fold_7_nrmn = f"../data/folds/{dataset_name}_nrmn_7"

fold_8_data = f"../data/folds/{dataset_name}_hydrogens_fold_8"
fold_8_labels = f"../data/folds/{dataset_name}_labels_fold_8"
fold_8_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_8"
fold_8_nrmn = f"../data/folds/{dataset_name}_nrmn_8"

fold_9_data = f"../data/folds/{dataset_name}_hydrogens_fold_9"
fold_9_labels = f"../data/folds/{dataset_name}_labels_fold_9"
fold_9_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_9"
fold_9_nrmn = f"../data/folds/{dataset_name}_nrmn_9"

fold_10_data = f"../data/folds/{dataset_name}_hydrogens_fold_10"
fold_10_labels = f"../data/folds/{dataset_name}_labels_fold_10"
fold_10_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_10"
fold_10_nrmn = f"../data/folds/{dataset_name}_nrmn_10"

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
# with open(fold_9_data, 'rb') as f:
#     fold_9_hydrogens = pickle.load(f, encoding='bytes')
# with open(fold_10_data, 'rb') as f:
#     fold_10_hydrogens = pickle.load(f, encoding='bytes')

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
# with open(fold_9_labels, 'rb') as f:
#     fold_9_labels = pickle.load(f, encoding='bytes')
# with open(fold_10_labels, 'rb') as f:
#     fold_10_labels = pickle.load(f, encoding='bytes')

# load pathology classes
with open(fold_1_pathology_classes, 'rb') as f:
    fold_1_pathology_classes = pickle.load(f, encoding='bytes')
with open(fold_2_pathology_classes, 'rb') as f:
    fold_2_pathology_classes = pickle.load(f, encoding='bytes')
with open(fold_3_pathology_classes, 'rb') as f:
    fold_3_pathology_classes = pickle.load(f, encoding='bytes')
with open(fold_4_pathology_classes, 'rb') as f:
    fold_4_pathology_classes = pickle.load(f, encoding='bytes')
with open(fold_5_pathology_classes, 'rb') as f:
    fold_5_pathology_classes = pickle.load(f, encoding='bytes')
with open(fold_6_pathology_classes, 'rb') as f:
    fold_6_pathology_classes = pickle.load(f, encoding='bytes')
with open(fold_7_pathology_classes, 'rb') as f:
    fold_7_pathology_classes = pickle.load(f, encoding='bytes')
with open(fold_8_pathology_classes, 'rb') as f:
    fold_8_pathology_classes = pickle.load(f, encoding='bytes')
# with open(fold_9_pathology_classes, 'rb') as f:
#     fold_9_pathology_classes = pickle.load(f, encoding='bytes')
# with open(fold_10_pathology_classes, 'rb') as f:
#     fold_10_pathology_classes = pickle.load(f, encoding='bytes')

# load nrmn
with open(fold_1_nrmn, 'rb') as f:
    fold_1_nrmn = pickle.load(f, encoding='bytes')
with open(fold_2_nrmn, 'rb') as f:
    fold_2_nrmn = pickle.load(f, encoding='bytes')
with open(fold_3_nrmn, 'rb') as f:
    fold_3_nrmn = pickle.load(f, encoding='bytes')
with open(fold_4_nrmn, 'rb') as f:
    fold_4_nrmn = pickle.load(f, encoding='bytes')
with open(fold_5_nrmn, 'rb') as f:
    fold_5_nrmn = pickle.load(f, encoding='bytes')
with open(fold_6_nrmn, 'rb') as f:
    fold_6_nrmn = pickle.load(f, encoding='bytes')
with open(fold_7_nrmn, 'rb') as f:
    fold_7_nrmn = pickle.load(f, encoding='bytes')
with open(fold_8_nrmn, 'rb') as f:
    fold_8_nrmn = pickle.load(f, encoding='bytes')
# with open(fold_9_nrmn, 'rb') as f:
#     fold_9_nrmn = pickle.load(f, encoding='bytes')
# with open(fold_10_nrmn, 'rb') as f:
#     fold_10_nrmn = pickle.load(f, encoding='bytes')

fold_1_hydrogens = fold_1_hydrogens.T
fold_2_hydrogens = fold_2_hydrogens.T
fold_3_hydrogens = fold_3_hydrogens.T
fold_4_hydrogens = fold_4_hydrogens.T
fold_5_hydrogens = fold_5_hydrogens.T
fold_6_hydrogens = fold_6_hydrogens.T
fold_7_hydrogens = fold_7_hydrogens.T
fold_8_hydrogens = fold_8_hydrogens.T
# fold_9_hydrogens = fold_9_hydrogens.T
# fold_10_hydrogens = fold_10_hydrogens.T

bin_size = 1

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

print(dataset_name)
print("Fold 1 Hydrogens: ", fold_1_hydrogens.shape, "\t Fold 1 Labels: ", fold_1_labels.shape)

spectrum_length = fold_1_hydrogens.shape[1]

# control vs aggressive
class model_ca(nn.Module):
    def __init__(self):
        super(model_ca, self).__init__()
        self.class_count = NUM_CLASSES
        # self.hidden = [1977, 1000, 200]
        self.hidden = [spectrum_length, 4000, 1000]
        self.attn_layer_1 = nn.Linear(self.hidden[0], 10000)
        self.attn_layer_2 = nn.Linear(10000, self.hidden[0])
        self.fc1 = nn.Linear(self.hidden[0], self.hidden[1])
        self.fc2 = nn.Linear(self.hidden[1], self.hidden[2])
        self.fc3 = nn.Linear(self.hidden[2], self.class_count)

        self.dropout = nn.Dropout(0)
        

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

# custom weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


print()
print("********** Classifier Model Training **********")
print()



auroc_folds = []
aupr_folds = []
f1_folds = []
precision_folds = []
recall_folds = []
train_time_folds = []
test_time_folds = []
acc_folds = []

idx = 0
seed = np.random.randint(low=0, high=3000)
print("Seed: ", seed)

# valid_index = np.random.randint(low=1, high=9)
# print("Validation fold is: ", valid_index)
test_indices = [1,2,3,4,5,6,7,8]
# test_indices.remove(valid_index)

for test_index in test_indices:

    torch.cuda.manual_seed_all(seed)

    train_indices = copy.deepcopy(test_indices)
    train_indices.remove(test_index)

    print()
    print("************ Test Fold " + str(test_index) + " ************")
    print()

    test_hydrogens = eval("fold_" + str(test_index) + "_hydrogens")
    test_labels = eval("fold_" + str(test_index) + "_labels")

    # valid_hydrogens = eval("fold_" + str(valid_index) + "_hydrogens")
    # valid_labels = eval("fold_" + str(valid_index) + "_labels")

    train_hydrogens_tuple = tuple([eval('fold_' + str(x) + '_hydrogens') for x in train_indices])
    train_labels_tuple = tuple([eval('fold_' + str(x) + '_labels') for x in train_indices])
    train_hydrogens = np.vstack(train_hydrogens_tuple)
    train_labels = np.vstack(train_labels_tuple)

    test_fold_data = torch.from_numpy(test_hydrogens).float()
    train_fold_data = torch.from_numpy(train_hydrogens).float()
    # valid_fold_data = torch.from_numpy(valid_hydrogens).float()
    test_fold_labels = torch.from_numpy(test_labels).long()
    train_fold_labels = torch.from_numpy(train_labels).long()
    # valid_fold_labels = torch.from_numpy(valid_labels).long()

    model = model_ca()
    model.apply(init_weights)
    model = model.to(device)

    # optimizers
    adam = torch.optim.Adam(model.parameters(), lr=ETA, weight_decay=0.001)
    sgd = torch.optim.SGD(model.parameters(), lr=ETA, weight_decay=0.001, momentum=0.85, nesterov=True)
    optimizer = adam # selected optimizer

    # learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=20, verbose=True,min_lr=1e-5)

    # weighted cross entropy loss function for training 
    counter = Counter(train_fold_labels.numpy().T.reshape(1,-1)[0,:].tolist())
    mw =  max([counter[x] for x in range(NUM_CLASSES)]) 
    weight = torch.tensor([mw/counter[x] for x in range(NUM_CLASSES)]).to(device)
    # print ("Weights: ", [mw/counter[x] for x in range(NUM_CLASSES)])
    # loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    loss_fn = FocalLoss(class_num=2, gamma=2, alpha=F.softmax(weight,dim=0))

    # # weighted cross entropy loss for validation dataset
    # counter = Counter(valid_fold_labels.numpy().T.reshape(1,-1)[0,:].tolist())
    # mw =  max([counter[x] for x in range(NUM_CLASSES)]) 
    # weight = torch.tensor([mw/counter[x] for x in range(NUM_CLASSES)]).to(device)
    # # valid_loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    # valid_loss_fn = FocalLoss(class_num=2, gamma=1, alpha=weight)

    # scale all samples according to training set
    scaler = preprocessing.MinMaxScaler().fit(train_fold_data.numpy())
    train_fold_data_normalized = torch.from_numpy(scaler.transform(train_fold_data.numpy())).float().to(device)
    test_fold_data_normalized = torch.from_numpy(scaler.transform(test_fold_data.numpy())).float().to(device)
    # valid_fold_data_normalized = torch.from_numpy(scaler.transform(valid_fold_data.numpy())).float().to(device)

    # convert to test set to torch variables
    test_data_torch = test_fold_data_normalized
    test_labels_torch = test_fold_labels
    test_labels_torch = torch.transpose(test_labels_torch, 0, 1)
    test_labels = test_labels_torch.to(device)

    # convert to test set to torch variables
    # valid_data_torch = valid_fold_data_normalized
    # valid_labels_torch = valid_fold_labels
    # valid_labels_torch = torch.transpose(valid_labels_torch, 0, 1)
    # valid_labels = valid_labels_torch.to(device)

    train_labels = train_fold_labels.to(device)

    # training and validation log related
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    aupr_history = []
    auc_history = []
    max_accuracy = 0

    # in order to find the best model based on validation loss
    best_model = model
    min_validation_loss = 1e1

    model.train()
    train_start = time.time()
    for epoch in range(ITERATION):
        # Forward pass
        train_fold_pred = model(train_fold_data_normalized)
        # Compute and save loss.
        loss = loss_fn(train_fold_pred, train_labels.squeeze())
        train_loss_history.append(loss.item())
        # compute training accuracy
        train_acc = (torch.transpose(train_labels, 0, 1) == torch.max(train_fold_pred,1)[1]).sum().cpu().numpy()/float(len(train_fold_labels))
        train_acc_history.append(train_acc)
        # try model on test set
        with torch.no_grad():
            model.eval()

            # predict test dataset
            test_pred =  model(test_data_torch)
            test_labels_pred = torch.max(test_pred,1)[1]
            test_labels_pred = test_labels_pred.cpu().numpy()
            # calculate auroc
            auc_ = roc_auc_score(test_labels.cpu()[0], test_pred.cpu().numpy()[:,1])
            auc_history.append(auc_)
            # calculate aupr
            precision, recall, thresh = precision_recall_curve(test_labels.cpu().numpy().T, test_pred.cpu().numpy()[:,1])
            aupr = auc(recall, precision)
            aupr_history.append(aupr)
            # calculate validation set loss
            # valid_fold_pred = model(valid_fold_data_normalized)
            # valid_pred =  model(valid_data_torch)
            # valid_labels_pred = torch.max(valid_pred,1)[1]
            # valid_labels_pred = valid_labels_pred.cpu().numpy()
            # valid_loss = valid_loss_fn(valid_fold_pred, valid_labels.squeeze())

            model.train()

        # print("Epoch: ", epoch, "\tTraining Loss: ", loss.item())
        # print("Epoch: ", epoch, "\tTraining Loss: ", loss.item(), "Validation Loss: ", valid_loss.item())
        # clear gradient history
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # parameter update
        optimizer.step()
        # scheduler.step(valid_loss)
        # find best model based on validation loss
        # if epoch == 0:
        #     min_validation_loss = valid_loss.item()
        # elif valid_loss.item() < min_validation_loss:
        #     min_validation_loss = valid_loss.item()
        #     best_model = copy.deepcopy(model)

    train_end = time.time()

    # clear gradient history
    optimizer.zero_grad()

    # use best model
    model = best_model.eval()

    # predict test set 
    test_start = time.time()
    test_labels_pred_ = model(test_data_torch)
    test_end = time.time()
    test_labels_pred = torch.max(test_labels_pred_,1)[1].detach().cpu()
    test_labels_pred = test_labels_pred.numpy()

    test_labels = test_labels.cpu().numpy()

    # find the ids of falsely predicted examples
    check = np.equal(test_labels.T, test_labels_pred.reshape(-1,1))
    wrong_predictions = np.where(check == False)[0]

    print("Micclassified Data: ")
    for nrmn, label in zip(eval("fold_" + str(test_index) + "_nrmn")[wrong_predictions], eval("fold_" + str(test_index) + "_pathology_classes")[wrong_predictions]):
        print(nrmn)
        print(label)
        print()

    # calculate various metrics
    # all metrics are calculated by taking aggressive as positive class
    cm = confusion_matrix(test_labels.T, test_labels_pred.reshape(-1,1))
    prec = precision_score(test_labels.T, test_labels_pred.reshape(-1,1), average='binary')
    rec = recall_score(test_labels.T, test_labels_pred.reshape(-1,1), average='binary')
    auroc = auc_history[len(auc_history) - 1]
    aupr = aupr_history[len(aupr_history) - 1]
    f1 = f1_score(test_labels.T, test_labels_pred.reshape(-1,1), average='binary')

    # record the calculated metrics 
    auroc_folds.append(auroc)
    aupr_folds.append(aupr)
    f1_folds.append(f1)
    precision_folds.append(prec)
    recall_folds.append(rec)
    acc_folds.append((cm[1,1] + cm[0,0]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]))
    train_time_folds.append(train_end - train_start)
    test_time_folds.append(test_end - test_start)


    # class classification rate from confusion matrix
    print("Confusion Matrix")
    print(cm)
    print("Class based accuracies: ")
    for i in range(NUM_CLASSES):
        print("Class % d: % f" %(i, cm[i,i]/np.sum(cm[i,:])))
    print("Accuracy: ",  (cm[1,1] + cm[0,0]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]))
    print("Precision: ", prec)
    print("Recall: ", rec)
    print("AUROC: ", auroc)
    print("AUPR: ", aupr)
    print("F1 Score: ", f1)

    model = None
    optimizer = None
    loss_fn = None
    loss_fn_2 = None
    valid_fold_data = None
    valid_fold_labels = None
    valid_labels_pred = None
    valid_pred = None
    train_fold_pred = None
    adam = None
    rmsprop = None
    scaler = None

# with open("./logs/NN/auroc_scores.txt", "a") as f:
#     for auroc in auroc_folds:
#         f.write("%f\n" % (auroc))
# with open("./logs/NN/aupr_scores.txt", "a") as f:
#     for aupr in aupr_folds:
#         f.write("%f\n" % (aupr))
# with open("./logs/NN/f1_scores.txt", "a") as f:
#     for f1 in f1_folds:
#         f.write("%f\n" % (f1))  
# with open("./logs/NN/precision_scores.txt", "a") as f:
#     for prec in precision_folds:
#         f.write("%f\n" % (prec))
# with open("./logs/NN/recall_scores.txt", "a") as f:
#     for rec in recall_folds:
#         f.write("%f\n" % (rec)) 
# with open("./logs/NN/acc_scores.txt", "a") as f:
#     for acc in acc_folds:
#         f.write("%f\n" % (acc))
# with open("./logs/NN/test_timing.txt", "a") as f:
#     for test in test_time_folds:
#         f.write("%f\n" % (test))
# with open("./logs/NN/train_timing.txt", "a") as f:
#     for train in train_time_folds:
#         f.write("%f\n" % (train)) 
         
        