import os 
import pdb
import numpy as np 
import pickle

# configuration variables
task_name = "control_tumor"
dataset_seed = 1648
fold_ids = list(range(1,9))

base_data_path = f"../../../fold_data/{task_name}/seed_{dataset_seed}/"

# fold 1
with open(os.path.join(base_data_path, "fold_1_hydrogens"), "rb") as f:
    fold_1_hydrogens = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_1_labels"), "rb") as f:
    fold_1_labels = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_1_nrmn"), "rb") as f:
    fold_1_nrmn = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_1_pathology_classes"), "rb") as f:
    fold_1_pathology_classes = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_1_patients"), "rb") as f:
    fold_1_patients = pickle.load(f).squeeze(1)

# fold 2
with open(os.path.join(base_data_path, "fold_2_hydrogens"), "rb") as f:
    fold_2_hydrogens = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_2_labels"), "rb") as f:
    fold_2_labels = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_2_nrmn"), "rb") as f:
    fold_2_nrmn = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_2_pathology_classes"), "rb") as f:
    fold_2_pathology_classes = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_2_patients"), "rb") as f:
    fold_2_patients = pickle.load(f).squeeze(1)

# fold 3
with open(os.path.join(base_data_path, "fold_3_hydrogens"), "rb") as f:
    fold_3_hydrogens = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_3_labels"), "rb") as f:
    fold_3_labels = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_3_nrmn"), "rb") as f:
    fold_3_nrmn = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_3_pathology_classes"), "rb") as f:
    fold_3_pathology_classes = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_3_patients"), "rb") as f:
    fold_3_patients = pickle.load(f).squeeze(1)

# fold 4
with open(os.path.join(base_data_path, "fold_4_hydrogens"), "rb") as f:
    fold_4_hydrogens = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_4_labels"), "rb") as f:
    fold_4_labels = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_4_nrmn"), "rb") as f:
    fold_4_nrmn = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_4_pathology_classes"), "rb") as f:
    fold_4_pathology_classes = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_4_patients"), "rb") as f:
    fold_4_patients = pickle.load(f).squeeze(1)

# fold 5
with open(os.path.join(base_data_path, "fold_5_hydrogens"), "rb") as f:
    fold_5_hydrogens = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_5_labels"), "rb") as f:
    fold_5_labels = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_5_nrmn"), "rb") as f:
    fold_5_nrmn = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_5_pathology_classes"), "rb") as f:
    fold_5_pathology_classes = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_5_patients"), "rb") as f:
    fold_5_patients = pickle.load(f).squeeze(1)

# fold 6
with open(os.path.join(base_data_path, "fold_6_hydrogens"), "rb") as f:
    fold_6_hydrogens = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_6_labels"), "rb") as f:
    fold_6_labels = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_6_nrmn"), "rb") as f:
    fold_6_nrmn = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_6_pathology_classes"), "rb") as f:
    fold_6_pathology_classes = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_6_patients"), "rb") as f:
    fold_6_patients = pickle.load(f).squeeze(1)

# fold 7
with open(os.path.join(base_data_path, "fold_7_hydrogens"), "rb") as f:
    fold_7_hydrogens = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_7_labels"), "rb") as f:
    fold_7_labels = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_7_nrmn"), "rb") as f:
    fold_7_nrmn = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_7_pathology_classes"), "rb") as f:
    fold_7_pathology_classes = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_7_patients"), "rb") as f:
    fold_7_patients = pickle.load(f).squeeze(1)

# fold 8
with open(os.path.join(base_data_path, "fold_8_hydrogens"), "rb") as f:
    fold_8_hydrogens = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_8_labels"), "rb") as f:
    fold_8_labels = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_8_nrmn"), "rb") as f:
    fold_8_nrmn = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_8_pathology_classes"), "rb") as f:
    fold_8_pathology_classes = pickle.load(f).squeeze(1)
with open(os.path.join(base_data_path, "fold_8_patients"), "rb") as f:
    fold_8_patients = pickle.load(f).squeeze(1)

if __name__ == "__main__":
    pdb.set_trace()