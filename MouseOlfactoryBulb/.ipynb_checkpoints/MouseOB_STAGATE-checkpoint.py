#!/usr/bin/env python
# coding: utf-8

# # Preparation
import sys
print(sys.path)


import tensorflow as tf

import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import warnings
warnings.filterwarnings("ignore")

import STAGATE

# the location of R (used for the mclust clustering)
# os.environ['R_HOME'] = 'D:\\anaconda\envs\STAligner\Lib\R'
# os.environ['R_USER'] = 'D:\\anaconda\envs\STAligner\Lib\site-packages\rpy2'

sample_names = ["10X","slide","stereo"]
input_dir = '/home/lixiangyu/wjk/data/mouseOB/25um/'
output_dir = '/home/lixiangyu/wjk/data/mouseOB/25um/STAGATE/'
experiment_name = 'MouseOB25um'


# # load data

import anndata as ad
adata_list = {}

for dataset in sample_names:
    # adata = sc.read_visium(input_dir + dataset)
    # adata.var_names_make_unique()
    # adata.obs_names_make_unique()

    adata = sc.read_h5ad(input_dir + dataset + '.h5ad')
    df = adata.obs[['x','y']].astype('float32')
    adata.obsm['spatial'] = df.values
    
    # adata.obs['Ground Truth'] = adata.obs.loc[adata.obs_names, 'Classification'].astype('category')
    
    # adata = adata[~pd.isna(adata.obs['x'])]
    
    # make spot name unique
    adata.obs_names = [x + '_' + dataset for x in adata.obs_names]
    
    adata_list[dataset] = adata.copy()

print(adata_list)

# fig, axs = plt.subplots(1, len(sample_names), figsize=(12, 3))
# it=0
# for section_id in sample_names:
#     if it == len(sample_names)-1:
#         sc.pl.spatial(adata_list[section_id], img_key="hires", ax=axs[it],
#                       color=["x"], title=section_id, show=False, spot_size = 1)
#     else:
#         sc.pl.spatial(adata_list[section_id], img_key="hires", ax=axs[it], legend_loc=None,
#                       color=["x"], title=section_id, show=False, spot_size = 100)
#     it+=1


for section_id in sample_names:
    if(section_id == '10X'):
        STAGATE.Cal_Spatial_Net(adata, rad_cutoff=200)
    elif(section_id == 'slide'):
        STAGATE.Cal_Spatial_Net(adata, rad_cutoff=50)
    else:
        STAGATE.Cal_Spatial_Net(adata, rad_cutoff=3)
    STAGATE.Stats_Spatial_Net(adata_list[section_id])


# # Conbat the scanpy objects and spatial networks
del adata

adata = sc.concat([adata_list[x] for x in datasets], keys=None)
print(adata)
adata.uns['Spatial_Net'] = pd.concat([adata_list[x].uns['Spatial_Net'] for x in datasets])
STAGATE.Stats_Spatial_Net(adata)

print(adata.obsm['spatial'])


#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


# # Running STAGATE

adata_Vars =  adata[:, adata.var['highly_variable']]
X = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
print(X)
cells = np.array(X.index)
print(cells)

adata = STAGATE.train_STAGATE(adata, alpha=0)
print(adata)
print(type(adata.obsm['STAGATE']))

# save embedding
np.savetxt(output_dir + experiment_name + '_STAGATE.csv', adata.obsm['STAGATE'], delimiter=",")

# # save

print(type(adata))
print(adata.obsm)
print(adata)

adata.obs = adata.obs.astype('str')

print(adata.isbacked)
adata.filename = output_dir + experiment_name + '.h5ad'
print(adata.isbacked)

