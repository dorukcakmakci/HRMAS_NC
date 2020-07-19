import matplotlib.pyplot as plt
import pandas as pd

import pdb
import sys
import os 

seeds = [228, 1849, 2251]

seed_0_prefix = f"./seed_{seeds[0]}"
seed_1_prefix = f"./seed_{seeds[1]}"
seed_2_prefix = f"./seed_{seeds[2]}"

plsda_prefix = "/logs/PLSDA/"
svm_prefix = "/logs/SVM/"
rf_prefix = "/logs/RF/"
nn_prefix = "/logs/NN/"
cnn_prefix = "/logs/CNN/"



auroc = []
aupr = []

temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + plsda_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + plsda_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + plsda_prefix, "auroc_scores.txt")).values[:,0].tolist()
auroc.append(temp)
temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + plsda_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + plsda_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + plsda_prefix, "aupr_scores.txt")).values[:,0].tolist()
aupr.append(temp)

temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + svm_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + svm_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + svm_prefix, "auroc_scores.txt")).values[:,0].tolist()
auroc.append(temp)
temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + svm_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + svm_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + svm_prefix, "aupr_scores.txt")).values[:,0].tolist()
aupr.append(temp)

temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + rf_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + rf_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + rf_prefix, "auroc_scores.txt")).values[:,0].tolist()
auroc.append(temp)
temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + rf_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + rf_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + rf_prefix, "aupr_scores.txt")).values[:,0].tolist()
aupr.append(temp)

temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + nn_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + nn_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + nn_prefix, "auroc_scores.txt")).values[:,0].tolist()
auroc.append(temp)
temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + nn_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + nn_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + nn_prefix, "aupr_scores.txt")).values[:,0].tolist()
aupr.append(temp)

temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + cnn_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + cnn_prefix, "auroc_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + cnn_prefix, "auroc_scores.txt")).values[:,0].tolist()
auroc.append(temp)
temp = []
temp += pd.read_csv(os.path.join(seed_0_prefix + cnn_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_1_prefix + cnn_prefix, "aupr_scores.txt")).values[:,0].tolist()
temp += pd.read_csv(os.path.join(seed_2_prefix + cnn_prefix, "aupr_scores.txt")).values[:,0].tolist()
aupr.append(temp)



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
axes[0].set_ylim(0.6, 1.01)
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
axes[1].set_ylim(0.6, 1.01)
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

plt.tight_layout()

plt.savefig('fig_2_panel_b.pdf')