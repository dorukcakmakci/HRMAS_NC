import matplotlib.pyplot as plt
import pandas as pd

import pdb
import sys
import os 

plsda_prefix = "./logs/PLSDA/"
svm_prefix = "./logs/SVM/"
rf_prefix = "./logs/RF/"
nn_prefix = "./logs/NN/"
cnn_prefix = "./logs/CNN/"
bilstm_prefix = "./logs/BILSTM/"
cnn_bilstm_prefix = "./logs/CNN_BILSTM/"


auroc = []
aupr = []
f1 = []
precision = []
recall = []

auroc.append(pd.read_csv(os.path.join(plsda_prefix, "auroc_scores.txt")).values[:,0])
aupr.append(pd.read_csv(os.path.join(plsda_prefix, "aupr_scores.txt")).values[:,0])
precision.append(pd.read_csv(os.path.join(plsda_prefix, "precision_scores.txt")).values[:,0])
recall.append(pd.read_csv(os.path.join(plsda_prefix, "recall_scores.txt")).values[:,0])
f1.append(pd.read_csv(os.path.join(plsda_prefix, "f1_scores.txt")).values[:,0])

auroc.append(pd.read_csv(os.path.join(svm_prefix, "auroc_scores.txt")).values[:,0])
aupr.append(pd.read_csv(os.path.join(svm_prefix, "aupr_scores.txt")).values[:,0])
precision.append(pd.read_csv(os.path.join(svm_prefix, "precision_scores.txt")).values[:,0])
recall.append(pd.read_csv(os.path.join(svm_prefix, "recall_scores.txt")).values[:,0])
f1.append(pd.read_csv(os.path.join(svm_prefix, "f1_scores.txt")).values[:,0])

auroc.append(pd.read_csv(os.path.join(rf_prefix, "auroc_scores.txt")).values[:,0])
aupr.append(pd.read_csv(os.path.join(rf_prefix, "aupr_scores.txt")).values[:,0])
precision.append(pd.read_csv(os.path.join(rf_prefix, "precision_scores.txt")).values[:,0])
recall.append(pd.read_csv(os.path.join(rf_prefix, "recall_scores.txt")).values[:,0])
f1.append(pd.read_csv(os.path.join(rf_prefix, "f1_scores.txt")).values[:,0])

auroc.append(pd.read_csv(os.path.join(nn_prefix, "auroc_scores.txt")).values[:,0])
aupr.append(pd.read_csv(os.path.join(nn_prefix, "aupr_scores.txt")).values[:,0])
precision.append(pd.read_csv(os.path.join(nn_prefix, "precision_scores.txt")).values[:,0])
recall.append(pd.read_csv(os.path.join(nn_prefix, "recall_scores.txt")).values[:,0])
f1.append(pd.read_csv(os.path.join(nn_prefix, "f1_scores.txt")).values[:,0])

auroc.append(pd.read_csv(os.path.join(cnn_prefix, "auroc_scores.txt")).values[:,0])
aupr.append(pd.read_csv(os.path.join(cnn_prefix, "aupr_scores.txt")).values[:,0])
precision.append(pd.read_csv(os.path.join(cnn_prefix, "precision_scores.txt")).values[:,0])
recall.append(pd.read_csv(os.path.join(cnn_prefix, "recall_scores.txt")).values[:,0])
f1.append(pd.read_csv(os.path.join(cnn_prefix, "f1_scores.txt")).values[:,0])



figure, axes = plt.subplots(1,2)
plt.subplots_adjust(wspace=0.4, hspace=0.3, top=0.8)
# figure.suptitle("Distinguishing Aggressive Labelled Samples From Control Labelled Samples")
# figure.set_figheight(3)
# figure.set_figwidth(20)
# plots from left to right: AUROC, AUPR, F1, Precision, Recall

axes[0].get_xaxis().tick_bottom()
axes[0].get_yaxis().tick_left()
axes[0].set_ylabel("AUC")
bp = axes[0].boxplot(auroc, patch_artist=True, showfliers=False)
axes[0].set_xticklabels([])
axes[0].set_xticks([])
axes[0].set_ylim(0.6, 1)
# outline color 
bp['boxes'][0].set(color='#b0bf1a') # PLSDA
bp['boxes'][1].set(color='#b284be') # SVM
bp['boxes'][2].set(color='#72a0c1') # RF
bp['boxes'][3].set(color='#d3212d') # NN
bp['boxes'][4].set(color='#d58044') # CNN
# face color 
bp['boxes'][0].set(facecolor='#b0bf1a') # PLSDA
bp['boxes'][1].set(facecolor='#b284be') # SVM
bp['boxes'][2].set(facecolor='#72a0c1') # RF
bp['boxes'][3].set(facecolor='#d3212d') # NN
bp['boxes'][4].set(facecolor='#d58044') # CNN

axes[1].get_xaxis().tick_bottom()
axes[1].get_yaxis().tick_left()
axes[1].set_ylabel("AUPR")
bp = axes[1].boxplot(aupr, patch_artist=True, showfliers=False)
axes[1].set_xticklabels([])
axes[1].set_xticks([])
axes[1].set_ylim(0.6, 1)
# outline color 
bp['boxes'][0].set(color='#b0bf1a') # PLSDA
bp['boxes'][1].set(color='#b284be') # SVM
bp['boxes'][2].set(color='#72a0c1') # RF
bp['boxes'][3].set(color='#d3212d') # NN
bp['boxes'][4].set(color='#d58044') # CNN
# face color 
bp['boxes'][0].set(facecolor='#b0bf1a') # PLSDA
bp['boxes'][1].set(facecolor='#b284be') # SVM
bp['boxes'][2].set(facecolor='#72a0c1') # RF
bp['boxes'][3].set(facecolor='#d3212d') # NN
bp['boxes'][4].set(facecolor='#d58044') # CNN
axes[1].legend(handles=bp["boxes"], labels=['PLS-DA', "SVM", "RF", "NN", "CNN"], loc="lower right" )

# axes[2].get_xaxis().tick_bottom()
# axes[2].get_yaxis().tick_left()
# axes[2].set_ylabel("Precision")
# bp = axes[2].boxplot(precision, patch_artist=True, showfliers=False)
# axes[2].set_xticklabels(['PLS-DA', "SVM", "RF", "NN", "CNN"])
# axes[2].set_ylim(0.5, 1.01)
# # outline color 
# bp['boxes'][0].set(color='#b0bf1a') # PLSDA
# bp['boxes'][1].set(color='#b284be') # SVM
# bp['boxes'][2].set(color='#72a0c1') # RF
# bp['boxes'][3].set(color='#d3212d') # NN
# bp['boxes'][4].set(color='#d58044') # CNN
# # face color 
# bp['boxes'][0].set(facecolor='#b0bf1a') # PLSDA
# bp['boxes'][1].set(facecolor='#b284be') # SVM
# bp['boxes'][2].set(facecolor='#72a0c1') # RF
# bp['boxes'][3].set(facecolor='#d3212d') # NN
# bp['boxes'][4].set(facecolor='#d58044') # CNN

# axes[3].get_xaxis().tick_bottom()
# axes[3].get_yaxis().tick_left()
# axes[3].set_ylabel("Recall")
# bp = axes[3].boxplot(recall, patch_artist=True, showfliers=False)
# axes[3].set_xticklabels(['PLS-DA', "SVM", "RF", "NN", "CNN"])
# axes[3].set_ylim(0.5, 1.01)
# # outline color 
# bp['boxes'][0].set(color='#b0bf1a') # PLSDA
# bp['boxes'][1].set(color='#b284be') # SVM
# bp['boxes'][2].set(color='#72a0c1') # RF
# bp['boxes'][3].set(color='#d3212d') # NN
# bp['boxes'][4].set(color='#d58044') # CNN
# # face color 
# bp['boxes'][0].set(facecolor='#b0bf1a') # PLSDA
# bp['boxes'][1].set(facecolor='#b284be') # SVM
# bp['boxes'][2].set(facecolor='#72a0c1') # RF
# bp['boxes'][3].set(facecolor='#d3212d') # NN
# bp['boxes'][4].set(facecolor='#d58044') # CNN

# axes[4].get_xaxis().tick_bottom()
# axes[4].get_yaxis().tick_left()
# axes[4].set_ylabel("F1 Score")
# bp = axes[4].boxplot(f1, patch_artist=True,  showfliers= False)
# axes[4].set_xticklabels(['PLS-DA', "SVM", "RF", "NN", "CNN"])
# axes[4].set_ylim(0.5, 1.01)
# # outline color 
# bp['boxes'][0].set(color='#b0bf1a') # PLSDA
# bp['boxes'][1].set(color='#b284be') # SVM
# bp['boxes'][2].set(color='#72a0c1') # RF
# bp['boxes'][3].set(color='#d3212d') # NN
# bp['boxes'][4].set(color='#d58044') # CNN
# # face color 
# bp['boxes'][0].set(facecolor='#b0bf1a') # PLSDA
# bp['boxes'][1].set(facecolor='#b284be') # SVM
# bp['boxes'][2].set(facecolor='#72a0c1') # RF
# bp['boxes'][3].set(facecolor='#d3212d') # NN
# bp['boxes'][4].set(facecolor='#d58044') # CNN

plt.tight_layout()

plt.savefig('benign_aggressive_results.pdf')

pdb.set_trace()


