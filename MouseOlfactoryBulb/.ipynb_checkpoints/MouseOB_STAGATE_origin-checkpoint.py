#!/usr/bin/env python
# coding: utf-8

# # Preparation

import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings("ignore")



import STAGATE

output_dir = './data/'
experiment_name = 'mouseOB_origin'

# # Load Data
adata_list = {}


# ### Slide-seqV2
input_dir = 'G:/dataset/06-Mouse olfactory bulb/input/slide-seq/'
counts_file = os.path.join(input_dir, 'Puck_200127_15.digital_expression.txt')
coor_file = os.path.join(input_dir, 'Puck_200127_15_bead_locations.csv')


counts = pd.read_csv(counts_file, sep='\t', index_col=0)
# coor_df = pd.read_csv(coor_file, index_col=0)
coor_df = pd.read_csv(coor_file, index_col=3)
print(counts.shape, coor_df.shape)


adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
adata.obsm["spatial"] = coor_df.to_numpy()

sc.pp.calculate_qc_metrics(adata, inplace=True)
print(adata)

plt.rcParams["figure.figsize"] = (6,5)
sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts",s=6, show=False)
plt.title('')
plt.axis('off')

# can be downloaded from https://drive.google.com/drive/folders/10lhz5VY7YfvHrtV40MwaqLmWz56U9eBP?usp=sharing
used_barcode = pd.read_csv(os.path.join(input_dir, 'used_barcodes.txt'), sep='\t', header=None)
used_barcode = used_barcode[0]
print(len(used_barcode))


adata = adata[used_barcode,]
adata

plt.rcParams["figure.figsize"] = (5,5)
sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts",s=10, show=False, title='Removing spots outside the main tissue area')

plt.axis('off')


sc.pp.filter_genes(adata, min_cells=50)
print('After flitering: ', adata.shape)


# make spot name unique
adata.obs_names = [x+'_SlideSeqV2' for x in adata.obs_names]

adata_list['SlideSeqV2'] = adata.copy()


# ### Stereo-seq


input_dir = 'G:/dataset/06-Mouse olfactory bulb/input/stereo-seq/'
counts_file = os.path.join(input_dir, 'RNA_counts.tsv')
coor_file = os.path.join(input_dir, 'position.tsv')


counts = pd.read_csv(counts_file, sep='\t', index_col=0)
coor_df = pd.read_csv(coor_file, sep='\t')
print(counts.shape, coor_df.shape)

counts.columns = ['Spot_'+str(x) for x in counts.columns]
coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
coor_df = coor_df.loc[:, ['x','y']]

coor_df.head()

adata = sc.AnnData(counts.T)
adata.var_names_make_unique()


print(adata)


coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
adata.obsm["spatial"] = coor_df.to_numpy()
sc.pp.calculate_qc_metrics(adata, inplace=True)


plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
plt.title("")
plt.axis('off')


used_barcode = pd.read_csv(os.path.join(input_dir, 'used_barcodes.txt'), sep='\t', header=None)
used_barcode = used_barcode[0]
adata = adata[used_barcode,]


# print(adata)


plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
plt.title("")
plt.axis('off')


sc.pp.filter_genes(adata, min_cells=50)
print('After flitering: ', adata.shape)


# make spot name unique
adata.obs_names = [x+'_StereoSeq' for x in adata.obs_names]


adata_list['StereoSeq'] = adata.copy()


# # Constructing the spatial network for each secion

# ### Slide-seqV2

STAGATE.Cal_Spatial_Net(adata_list['SlideSeqV2'], rad_cutoff=50)
STAGATE.Stats_Spatial_Net(adata_list['SlideSeqV2'])

STAGATE.Cal_Spatial_Net(adata_list['StereoSeq'], rad_cutoff=50)
STAGATE.Stats_Spatial_Net(adata_list['StereoSeq'])

adata_list['SlideSeqV2'].uns['Spatial_Net']


# # Conbat the scanpy objects and spatial networks

adata = sc.concat([adata_list['SlideSeqV2'], adata_list['StereoSeq']], keys=None)

adata.uns['Spatial_Net'] = pd.concat([adata_list['SlideSeqV2'].uns['Spatial_Net'], adata_list['StereoSeq'].uns['Spatial_Net']])


STAGATE.Stats_Spatial_Net(adata)


# # Normalization


#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


# # Running STAGATE

adata = STAGATE.train_STAGATE(adata, alpha=0)

sc.pp.neighbors(adata, use_rep='STAGATE')
sc.tl.umap(adata)


adata.obs['Tech'] = [x.split('_')[-1] for x in adata.obs_names]

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata, color='Tech', title='Unintegrated')


# # save
adata.obs = adata.obs.astype('str')

print(adata.isbacked)
adata.filename = output_dir + experiment_name + '.h5ad'
print(adata.isbacked)

# save embedding
np.savetxt(output_dir + experiment_name + '_STAGATE.csv', adata.obsm['STAGATE'], delimiter=",")



