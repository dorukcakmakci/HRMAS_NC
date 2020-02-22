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
ITERATION = 200
NUM_CLASSES = 2
ETA = 1e-4

# since numerous executions will be made define the following variable
# to store logs and plots reasonably 
execution = "run_1"

# two running environment options below:
# device = torch.device("cpu")
device = torch.device("cuda:7")
dtype = torch.float

# import data
# uncomment corresponding line to process the corresponding problem
dataset_name = "benign_aggressive" 

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

# fold_9_data = f"../data/folds/{dataset_name}_hydrogens_fold_9"
# fold_9_labels = f"../data/folds/{dataset_name}_labels_fold_9"
# fold_9_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_9"
# fold_9_nrmn = f"../data/folds/{dataset_name}_nrmn_9"

# fold_10_data = f"../data/folds/{dataset_name}_hydrogens_fold_10"
# fold_10_labels = f"../data/folds/{dataset_name}_labels_fold_10"
# fold_10_pathology_classes = f"../data/folds/{dataset_name}_pathology_classes_10"
# fold_10_nrmn = f"../data/folds/{dataset_name}_nrmn_10"

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

print(dataset_name)
print("Fold 1 Hydrogens: ", fold_1_hydrogens.shape, "\t Fold 1 Labels: ", fold_1_labels.shape)

spectrum_length = fold_1_hydrogens.shape[1]

# class model_ca(nn.Module):
#     def __init__(self):
#         super(model_ca, self).__init__()
#         self.class_count = NUM_CLASSES
#         # self.hidden = [1977, 1000, 200]
#         self.hidden = [7104, 800] # 5034

#         self.conv1 = nn.Conv1d(1,3,16, stride=2)
#         self.conv2 = nn.Conv1d(1,3,32, stride=4)
#         self.conv3 = nn.Conv1d(1,3,64, stride=8, dilation=4)

#         self.conv4 = nn.Conv1d(1,1,100, stride=4)


#         self.conv_1_1x1 = nn.Conv1d(3,2,1)
#         self.conv_2_1x1 = nn.Conv1d(3,2,1)
#         self.conv_3_1x1 = nn.Conv1d(3,2,1)
#         self.conv_4_1x1 = nn.Conv1d(4,1,1)

#         self.mp2 = nn.MaxPool1d(2)
#         self.mp3 = nn.MaxPool1d(3)
#         self.mp4 = nn.MaxPool1d(4)

#         self.dropout = nn.Dropout(0.2)

#         self.fc1 = nn.Linear(self.hidden[0], self.hidden[1])
#         self.fc2 = nn.Linear(self.hidden[1], self.class_count)

#     def forward(self, x_):
#         x = x_.data.unsqueeze(1)

#         x1 = self.conv_1_1x1(self.mp2(F.relu(self.conv1(x))))
#         x2 = self.conv_2_1x1(self.mp2(F.relu(self.conv2(x))))
#         x3 = self.conv_3_1x1(self.mp2(F.relu(self.conv3(x))))

#         # x1_4 = F.relu(self.conv4(x1))
#         # x2_4 = F.relu(self.conv4(x2))

#         x1 = x1.view(x1.size(0), -1)
#         x2 = x2.view(x2.size(0), -1)
#         x3 = x3.view(x3.size(0), -1)

#         x = torch.cat((x1,x2,x3), dim=1).data.squeeze()

#         # x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.softmax(x, dim=1)

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
        

# class model_ca(nn.Module):
#     def __init__(self):
#         super(model_ca, self).__init__()
#         self.class_count = NUM_CLASSES
#         # self.hidden = [1977, 1000, 200]
#         self.hidden = [4166]

#         self.conv1 = nn.Conv1d(1,10,20, stride=2, dilation=4)
#         self.conv2 = nn.Conv1d(10,20,60, stride=2, dilation=2)
#         self.conv3 = nn.Conv1d(1,8,100, stride=1,dilation=8)

#         self.conv_1_1x1 = nn.Conv1d(20,1,1)
#         self.conv_2_1x1 = nn.Conv1d(8,1,1)

#         self.mp2 = nn.MaxPool1d(2)
#         self.mp3 = nn.MaxPool1d(3)
#         self.mp4 = nn.MaxPool1d(4)

#         self.dropout = nn.Dropout(0.1)

#         self.fc1 = nn.Linear(self.hidden[0], self.class_count)

#     def forward(self, x_):
#         x = x_.data.unsqueeze(1)

#         x1 = self.mp2(F.relu(self.conv1(x)))
#         x1 = self.mp2(self.conv_1_1x1(F.relu(self.conv2(x1))))

#         x2 = self.mp2(self.conv_2_1x1(F.relu(self.conv3(x))))

#         x = torch.cat((x1,x2), dim=2).data.squeeze()
#         x = self.dropout(x)
#         x = self.fc1(x)
#         return F.softmax(x, dim=1)

# class model_ca(nn.Module):
#     def __init__(self):
#         super(model_ca, self).__init__()
#         self.class_count = NUM_CLASSES
#         # self.hidden = [1977, 1000, 200]
#         self.hidden = [532]

#         self.mp4 = nn.MaxPool1d(4)
#         self.mp2 = nn.MaxPool1d(2)

#         self.conv1 = nn.Conv1d(1,2,4, stride=1, dilation=4)
#         self.conv2 = nn.Conv1d(1,2,8, stride=1, dilation=4)
#         self.conv3 = nn.Conv1d(1,2,16, stride=2, dilation=4)
#         self.conv4 = nn.Conv1d(1,2,32, stride=2, dilation=4)
#         self.conv5 = nn.Conv1d(1,2,64, stride=3, dilation=1)
#         self.conv6 = nn.Conv1d(1,2,128, stride=3, dilation=1)

        # self.conv_1_1x1 = nn.Conv1d(2,1,1)
        # self.conv_2_1x1 = nn.Conv1d(2,1,1)
        # self.conv_3_1x1 = nn.Conv1d(2,1,1)
        # self.conv_4_1x1 = nn.Conv1d(2,1,1)
        # self.conv_5_1x1 = nn.onv1d(2,1,1)
        # self.conv_6_1x1 = nn.Conv1d(2,1,1)

#         self.conv_comb = nn.Conv1d(1,3,32, stride=5)

#         self.dropout = nn.Dropout(0.3)

#         self.conv_comb_1x1 = nn.Conv1d(3,1,1)

#         self.fc1 = nn.Linear(self.hidden[0], self.class_count)

#     def forward(self, x_):
#         x = x_.data.unsqueeze(1)

#         x2 = self.conv_2_1x1(self.mp4(F.relu(self.conv2(x)))).squeeze()
#         x3 = self.conv_3_1x1(self.mp4(F.relu(self.conv3(x)))).squeeze()
#         x4 = self.conv_4_1x1(self.mp4(F.relu(self.conv4(x)))).squeeze()
#         x6 = self.conv_6_1x1(self.mp4(F.relu(self.conv6(x)))).squeeze()
        
#         x = torch.cat((x3,x4,x6), dim= 1).data.unsqueeze(1)

#         x = F.relu(self.conv_comb(x))
#         x = self.conv_comb_1x1(x)
#         x = self.dropout(x)
#         x = self.fc1(x)

#         return F.softmax(x.squeeze(), dim=1)

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
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    # loss_fn = FocalLoss(class_num=2, gamma=1.2, alpha=weight)

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

    # print("F2 Score: ", fbeta_score(test_labels.T, test_labels_pred.reshape(-1,1), beta=2, average='binary'))
    # print("Confusion Matrix: ", cm)
    # print("Brier Score:", brier_score)
    # print("NPV: ", cm[1,1] / (cm[1,1] + cm[1,0]))
    # print("FPR: ", cm[1,0] / (cm[1,1] + cm[1,0]))
    # print("FDR: ", cm[1,0] / (cm[0,0] + cm[1,0]))

    # generate perfomance metrics logs
    # title = "test_fold_" + str(test_index) + "_" + "valid_fold_" + str(valid_index) + "_"
    # for layer in model.hidden:
    #     title += str(layer) + "_"
    # title += "epoch_" + str(ITERATION) + "_performance_metrics.txt"
    # path = "./logs/CNN/" + execution + "/" + title
    # with open(path, 'w') as f:
    #     f.write("Confusion Matrix\n")
    #     for line in cm:
    #         f.write(" ".join(str(line)) + "\n")
    #     f.write("Class based accuracies:\n")
    #     for i in range(NUM_CLASSES):
    #         temp = "Class % d: % f\n" %(i, cm[i,i]/np.sum(cm[i,:]))
    #         f.write(temp)
    #     f.write("Accuracy: %f\n" %  ((cm[1,1] + cm[0,0]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])))
    #     f.write("Precision: %f\n" % precision_score(test_labels.T, test_labels_pred.reshape(-1,1), average='binary'))
    #     f.write("Recall: %f\n" % recall_score(test_labels.T, test_labels_pred.reshape(-1,1), average='binary'))
    #     f.write("F1 Score: %f\n" % f1_score(test_labels.T, test_labels_pred.reshape(-1,1), average='binary'))
    #     f.write("F2 Score: %f\n" % fbeta_score(test_labels.T, test_labels_pred.reshape(-1,1), beta=2, average='binary'))
    #     f.write("NPV: %f\n" % (cm[1,1] / (cm[1,1] + cm[1,0])))
    #     f.write("FPR: %f\n" % (cm[1,0] / (cm[1,1] + cm[1,0])))
    #     f.write("FDR: %f\n" % (cm[1,0] / (cm[0,0] + cm[1,0])))
    #     for line in  eval("fold_" + str(test_index) + "_pathology_classes")[wrong_predictions]:
    #         f.write(" ".join(str(line)) + "\n")
    #     f.write("NRMN of Wrongly Predicted Data:\n")
    #     for line in  eval("fold_" + str(test_index) + "_nrmn")[wrong_predictions]:
    #         f.write(" ".join(str(line)) + "\n")

    # # plot training loss
    # plt.figure()
    # plt.plot(range(len(train_loss_history)), train_loss_history)
    # plt.xlabel("Epoch")
    # plt.ylabel("Training Loss")
    # title = "test_fold_" + str(test_index) + "_" + "validation_fold_" + str(valid_index) + "_"
    # for layer in model.hidden:
    #     title += str(layer) + "_"
    # title += "epoch_" + str(ITERATION) + "_training_loss"
    # plt.title(title)
    # plt.savefig("./plots/CNN/" + execution + "/" + title + ".png")
    # plt.close()

    # # plot training accuracy
    # plt.figure()
    # plt.plot(range(len(train_acc_history)), train_acc_history)
    # plt.xlabel("Epoch")
    # plt.ylabel("Training Accuracy")
    # title = "test_fold_" + str(test_index) + "_" + "validation_fold_" + str(valid_index) + "_"
    # for layer in model.hidden:
    #     title += str(layer) + "_"
    # title += "epoch_" + str(ITERATION) + "_training_accuracy"
    # plt.title(title)
    # plt.savefig("./plots/CNN/" + execution + "/" + title + ".png")
    # plt.close()


    # # plot test set loss
    # plt.figure()
    # plt.plot(range(len(valid_loss_history)), valid_loss_history)
    # plt.xlabel("Epoch")
    # plt.ylabel("Test Set Loss")
    # title = "test_fold_" + str(test_index) + "_" + "validation_fold_" + str(valid_index) + "_"
    # for layer in model.hidden:
    #     title += str(layer) + "_"
    # title += "epoch_" + str(ITERATION) + "_validation_set_loss"
    # plt.title(title)
    # plt.savefig("./plots/CNN/" + execution + "/" + title + ".png")
    # plt.close()

    # # plot roc auc of test set during training 
    # plt.figure()
    # plt.plot(range(len(auc_history)), auc_history)
    # plt.xlabel("Epoch")
    # plt.ylabel("AUC")
    # title = "test_fold_" + str(test_index) + "_" + "validation_fold_" + str(valid_index) + "_"
    # for layer in model.hidden:
    #     title += str(layer) + "_"
    # title += "epoch_" + str(ITERATION) + "_auc"
    # plt.title(title)
    # plt.savefig("./plots/CNN/" + execution + "/" + title + ".png")
    # plt.close()

    # save model weights
    # title = "test_fold_" + str(test_index) + "_" + "validation_fold_" + str(valid_index) + "_"
    # for layer in model.hidden:
    #     title += str(layer) + "_"
    # title += "epoch_" + str(ITERATION) + "_weights.pt"
    # weights_path = "./weights/CNN/" + execution + "/" + title
    # torch.save(model.state_dict(), weights_path)

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

with open("./logs/CNN/auroc_scores.txt", "a") as f:
    for auroc in auroc_folds:
        f.write("%f\n" % (auroc))
with open("./logs/CNN/aupr_scores.txt", "a") as f:
    for aupr in aupr_folds:
        f.write("%f\n" % (aupr))
with open("./logs/CNN/f1_scores.txt", "a") as f:
    for f1 in f1_folds:
        f.write("%f\n" % (f1))  
with open("./logs/CNN/precision_scores.txt", "a") as f:
    for prec in precision_folds:
        f.write("%f\n" % (prec))
with open("./logs/CNN/recall_scores.txt", "a") as f:
    for rec in recall_folds:
        f.write("%f\n" % (rec))  
with open("./logs/CNN/acc_scores.txt", "a") as f:
    for acc in acc_folds:
        f.write("%f\n" % (acc))
with open("./logs/CNN/test_timing.txt", "a") as f:
    for test in test_time_folds:
        f.write("%f\n" % (test))
with open("./logs/CNN/train_timing.txt", "a") as f:
    for train in train_time_folds:
        f.write("%f\n" % (train)) 
        