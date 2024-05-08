import scanpy as sc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import SPACEL
from SPACEL import Scube
    
sample_names = ["embryo1-2","embryo1-5","embryo2-2","embryo2-5","embryo3-2","embryo3-5"]
input_dir = '/home/lixiangyu/wjk/data/MouseAtlas/'
output_dir = '/home/lixiangyu/wjk/data/MouseAtlas/STAligner/'
experiment_name = 'all'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

adata_list = []
# meta = list()
for dataset in sample_names:
    print(dataset)
    adata = sc.read_h5ad(input_dir + dataset + ".h5ad")
    adata_list.append(adata)

Scube.align(adata_list,
      cluster_key='spatial_domain',
      n_neighbors = 15,
      n_threads=10,
      p=2,
      write_loc_path=output_dir+ experiment_name+'_aligned_coordinates.csv'
     )
sc.concat(adata_list).write(output_dir+experiment_name+'.h5ad')

coo = pd.DataFrame()
for i in range(len(adata_list)):
    loc = adata_list[i].obsm['spatial_aligned'].copy()
    loc['Z'] = i
    coo = pd.concat([coo,loc],axis=0)
coo.to_csv(output_dir+ experiment_name +'.csv')