import pdb

import numpy as np
import matplotlib.pyplot as plt

RAW_SPECTRUM_LENGTH = 16314
PROCESSED_SPECTRUM_LENGTH = 8172
MIN_PPM = -2
MAX_PPM = 12


def find_ppm_value(idx):
    return round((14 * (idx) / RAW_SPECTRUM_LENGTH -2), 2)

def plot_all_shap_values(shap_values, spectrum, save_name):

    # take absolute value
    # shap_values = np.absolute(shap_values)

    # normalize each ppm 
    spectrum = spectrum / np.amax(spectrum,axis=0,keepdims=True)

    # # prepare scatter plot entries
    xs = []
    ys = []
    vals = []
    sizes = []
    for i in range(shap_values.shape[1]):
        for j in range(shap_values.shape[0]):
            xs.append((i+1))
            ys.append(shap_values[j,i])
            vals.append(spectrum[j,i])
            sizes.append(1)
    
    plt.figure()
    res = plt.scatter(xs,ys,c=vals,s=sizes,marker='o',cmap="cool",alpha=0.3)
    
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Signal Amplitude', rotation=90)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.ax.text(0.5, -0.01, 'Low', transform=cbar.ax.transAxes, 
        va='top', ha='center')
    cbar.ax.text(0.5, 1.01, 'High', transform=cbar.ax.transAxes, 
        va='bottom', ha='center')

    # change x axis scale 
    locs, labels = plt.xticks()
    locs = np.arange(-1000, 9000, 500)
    labels = [find_ppm_value(float(item)) for item in locs]
    plt.xticks(locs, labels,rotation = (45), fontsize = 10, va='top', ha='center')

    # set axis labels
    plt.xlabel("ppm")
    plt.ylabel("SHAP Value")


    plt.tight_layout()
    plt.savefig(save_name+".pdf")
    
    plt.close()

def plot_top_k_shap_values(shap_values, spectrum, k, save_name):
    # take absolute value
    abs_shap_values = np.absolute(shap_values)
    max_abs_shap_values = np.amax(abs_shap_values, axis=0)
    top_k_ind  = max_abs_shap_values.argsort()[(-1)*k:][::-1]
    print("Indices with top " + str(k) +" maximum shap value:")
    temp = [find_ppm_value(x) for x in top_k_ind]
    temp.sort()
    print(temp)

    # normalize each ppm 
    spectrum = spectrum / np.amax(spectrum,axis=0,keepdims=True)

    # # prepare scatter plot entries
    xs = []
    ys = []
    vals = []
    sizes = []
    xs_ = []
    ys_ = []
    sizes_ = []
    for i in range(shap_values.shape[1]):
        if i in top_k_ind:
            for j in range(shap_values.shape[0]):
                xs.append((i+1))
                ys.append(shap_values[j,i])
                vals.append(spectrum[j,i])
                sizes.append(1)
        else:
            xs_.append((i+1))
            ys_.append(0)
            # vals.append(0)
            sizes_.append(1)
    
    plt.figure()
    res = plt.scatter(xs,ys,c=vals,s=sizes,marker='o',cmap="cool",alpha=0.3)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Signal Amplitude', rotation=90)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.ax.text(0.5, -0.01, 'Low', transform=cbar.ax.transAxes, 
        va='top', ha='center')
    cbar.ax.text(0.5, 1.01, 'High', transform=cbar.ax.transAxes, 
        va='bottom', ha='center')

    res = plt.scatter(xs_,ys_,c='k',s=sizes_,marker='o',alpha=0.3)
    
    

    # change x axis scale 
    locs, labels = plt.xticks()
    locs = np.arange(-1000, 9000, 500)
    labels = [find_ppm_value(float(item)) for item in locs]
    plt.xticks(locs, labels,rotation = (45), fontsize = 10, va='top', ha='center')

    # set axis labels
    plt.xlabel("ppm")
    plt.ylabel("SHAP Value")


    plt.tight_layout()
    plt.savefig(save_name+".pdf")
    
    plt.close()
    
