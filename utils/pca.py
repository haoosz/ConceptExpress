import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def make_color(label):
    colors = ['darkorange','black','royalblue','forestgreen','crimson']
    
    label_list = list(label.detach().cpu().numpy())
    
    color_list = []
    for i in label_list:
        color_list.append(colors[i])
    
    return color_list

def pca_visual(X, label, info, out_dir):
    X = X.detach().cpu().numpy()
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.fit_transform(X)
    
    color = make_color(label)
    fig = plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=50, alpha=0.5, color=color)
    plt.savefig(os.path.join(out_dir, 'pca_vis/embed-{}.png'.format(info)))
    plt.clf()