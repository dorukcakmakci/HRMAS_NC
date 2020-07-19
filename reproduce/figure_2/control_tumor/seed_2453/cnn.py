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

from load_data import *

# network related constants 
ITERATION = 200
NUM_CLASSES = 2
ETA = 1e-4

# two running environment options below:
# device = torch.device("cpu")
device = torch.device("cuda:0")
dtype = torch.float

class model_ca(nn.Module):
    def __init__(self):
        super(model_ca, self).__init__()
        self.class_count = NUM_CLASSES
        # self.hidden = [1977, 1000, 200]
        self.hidden = [8112,4000, 800] # previously more previously 799 

        self.mp4 = nn.MaxPool1d(4)
        self.mp2 = nn.MaxPool1d(2)

        # self.conv1 = nn.Conv1d(1,1,256, stride=1, dilation=1)
        # self.conv2 = nn.Conv1d(1,1,8, stride=1, dilation=1)
        self.conv3 = nn.Conv1d(1,1,16, stride=1, dilation=1)
        self.conv4 = nn.Conv1d(1,1,32, stride=1, dilation=1)
        self.conv5 = nn.Conv1d(1,1,64, stride=1, dilation=1)
        self.conv6 = nn.Conv1d(1,1,128, stride=1, dilation=1)

        

        self.fc1 = nn.Linear(self.hidden[0], self.hidden[1])
        self.fc2 = nn.Linear(self.hidden[1], self.hidden[2])
        self.fc3 = nn.Linear(self.hidden[2], self.class_count)

    def forward(self, x_):
        x = x_.data.unsqueeze(1)

        # x1 = self.mp4(F.relu(self.conv1(x))).squeeze()
        # x2 = self.mp4(F.relu(self.conv2(x))).squeeze()
        x3 = self.mp4(F.relu(self.conv3(x))).squeeze()
        x4 = self.mp4(F.relu(self.conv4(x))).squeeze()
        x5 = self.mp4(F.relu(self.conv5(x))).squeeze()
        x6 = self.mp4(F.relu(self.conv6(x))).squeeze()
        
        x = torch.cat((x3,x4,x5,x6), dim= 1).data.unsqueeze(1)

        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x.squeeze(), dim=1)

# custom weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


print()
print("********** Classifier Model Training **********")
print()

auroc_folds = []
aupr_folds = []

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
    loss_fn = FocalLoss(class_num=2, gamma=1.2, alpha=weight)

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
    


    # clear gradient history
    optimizer.zero_grad()

    # use best model
    model = best_model.eval()

    # predict test set 
    test_labels_pred_ = model(test_data_torch)
    test_labels_pred = torch.max(test_labels_pred_,1)[1].detach().cpu()
    test_labels_pred = test_labels_pred.numpy()

    test_labels = test_labels.cpu().numpy()

    # calculate various metrics
    # all metrics are calculated by taking aggressive as positive class
    auroc = auc_history[len(auc_history) - 1]
    aupr = aupr_history[len(aupr_history) - 1]

    # record the calculated metrics 
    auroc_folds.append(auroc)
    aupr_folds.append(aupr)


    # class classification rate from confusion matrix
    print("AUROC: ", auroc)
    print("AUPR: ", aupr)

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

with open("./logs/CNN/auroc_scores.txt", "w") as f:
    for auroc in auroc_folds:
        f.write("%f\n" % (auroc))
with open("./logs/CNN/aupr_scores.txt", "w") as f:
    for aupr in aupr_folds:
        f.write("%f\n" % (aupr))