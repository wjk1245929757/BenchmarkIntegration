#!/usr/bin/env python
# coding: utf-8

import os 

print(os.getcwd())#显示当前路径
os.chdir('D:/bio/DeepST/DeepST-main2/deepst')#更改路径，''里面为更改的路径
print(os.getcwd())#显示当前路径

from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData


data_name_list = ['E9.5_E1S1', 'E10.5_E2S1', 'E11.5_E1S1', 'E12.5_E1S1']
# data_path = '/home/lixiangyu/benchmark/data/MOSTA/'
# save_path = '/home/lixiangyu/benchmark/data/MOSTA/STAligner/'
# n_domains = 10

data_path = 'G:/dataset/08-MOSTA/input/'
save_path = 'G:/dataset/08-MOSTA/output/DeepST/'


deepen = run(save_path = save_path, 
	task = "Integration",
	pre_epochs = 800, 
	epochs = 1000, 
	use_gpu = True,
	)

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
# import stlearn
from _compat import Literal
import scanpy
import scipy


def read_stereoSeq_h5ad(path,
                bin_size=100,
                is_sparse=True,
                library_id=None,
                scale=None,
                quality="hires",
                spot_diameter_fullres=1,
                background_color="white",
                ):
    
    adata = sc.read_h5ad(path)

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 20 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "StereoSeq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata


###### Generate an augmented list of multiple datasets
augement_data_list = []
graph_list = []
for i in range(len(data_name_list)):
	print(data_name_list[i])
	adata = read_stereoSeq_h5ad(path = os.path.join(data_path, data_name_list[i] + '.MOSTA.h5ad'))
	# adata = deepen._get_adata(platform="stereoSeq", data_path=data_path, data_name=data_name_list[i])
	adata = deepen._get_image_crop(adata, data_name=data_name_list[i])
	adata = deepen._get_augment(adata, spatial_type="BallTree", use_morphological=False)
	graph_dict = deepen._get_graph(adata.obsm["spatial"], distType = "BallTree")
        
# 	save_data_path = Path(os.path.join(save_path, "Data", data_name_list[i]))
# 	save_data_path.mkdir(parents=True, exist_ok=True)
# 	adata.write(os.path.join(save_data_path, f'{data_name_list[i]}_raw.h5ad'), compression="gzip")
    
	augement_data_list.append(adata)
	graph_list.append(graph_dict)



######## Synthetic Datasets and Graphs
multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list = augement_data_list, data_name_list = data_name_list, graph_list = graph_list)



###### Enhanced data preprocessing
data = deepen._data_process(multiple_adata, pca_n_comps = 200)


deepst_embed = deepen._fit(
		data = data,
		graph_dict = multiple_graph,
		domains = multiple_adata.obs["batch"].values,  ##### Input to Domain Adversarial Model
		n_domains = len(data_name_list))
multiple_adata.obsm["DeepST_embed"] = deepst_embed
multiple_adata = deepen._get_cluster_data(multiple_adata, n_domains=n_domains, priori = True)


sc.pp.neighbors(multiple_adata, use_rep='DeepST_embed')
sc.tl.umap(multiple_adata)
sc.pl.umap(multiple_adata, color=["DeepST_refine_domain","batch_name"])
plt.savefig(os.path.join(save_path, f'{"_".join(data_name_list)}_umap.pdf'), bbox_inches='tight', dpi=300)


for data_name in data_name_list:
	adata = multiple_adata[multiple_adata.obs["batch_name"]==data_name]
	sc.pl.spatial(adata, color='DeepST_refine_domain', frameon = False, spot_size=150)
	plt.savefig(os.path.join(save_path, f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=300)


print(multiple_adata.isbacked)
multiple_adata.filename = save_path + '/MOSTA.h5ad'
print(multiple_adata.isbacked)

# try:   
#     get_ipython().system('jupyter nbconvert --to python file_name.ipynb')
#     # python即转化为.py，script即转化为.html
#     # file_name.ipynb即当前module的文件名
# except:
#     pass

