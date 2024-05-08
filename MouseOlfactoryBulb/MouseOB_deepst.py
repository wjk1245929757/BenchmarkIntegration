import os

print(os.getcwd())  # 显示当前路径
os.chdir('/home/lixiangyu/wjk/method/DeepST-main/deepst')  # 更改路径，''里面为更改的路径
print(os.getcwd())  # 显示当前路径

from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData

data_path = "/home/lixiangyu/wjk/data/mouseOB/25um/"
data_name_list = ["10X", "slide", "stereo"]
save_path = "/home/lixiangyu/wjk/data/mouseOB/25um/DeepST"
n_domains = 10

deepen = run(save_path=save_path,
             task="Integration",
             pre_epochs=800,
             epochs=1000,
             use_gpu=True,
             )

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from scipy.sparse import issparse, csr_matrix
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


def read_10X_Visium(path,
                    genome=None,
                    count_file='filtered_feature_bc_matrix.h5',
                    library_id=None,
                    load_images=True,
                    quality='hires',
                    image_path=None):
    adata = sc.read_visium(path,
                           genome=genome,
                           count_file=count_file,
                           library_id=library_id,
                           load_images=load_images,
                           )
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata


def read_SlideSeq(path,
                  library_id=None,
                  scale=None,
                  quality="hires",
                  spot_diameter_fullres=50,
                  background_color="white", ):
    count = pd.read_csv(os.path.join(path, "count_matrix.count"), sep='\t', index_col=0)
    meta = pd.read_csv(os.path.join(path, "spatial.idx"), index_col=0)

    adata = AnnData(count.T)

    # adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["barcode"].values

    if scale == None:
        max_coor = np.max(meta[["xcoord", "ycoord"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["xcoord"].values * scale
    adata.obs["imagerow"] = meta["ycoord"].values * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"] = scale

    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    adata.obsm["spatial"] = meta[["xcoord", "ycoord"]].values

    return adata


def read_stereoSeq(path,
                   bin_size=100,
                   is_sparse=True,
                   library_id=None,
                   scale=None,
                   quality="hires",
                   spot_diameter_fullres=1,
                   background_color="white",
                   ):
    from scipy import sparse
    count = pd.read_csv(os.path.join(path, "count.txt"), sep='\t', comment='#', header=0)
    count.dropna(inplace=True)
    if "MIDCounts" in count.columns:
        count.rename(columns={"MIDCounts": "UMICount"}, inplace=True)
    count['x1'] = (count['x'] / bin_size).astype(np.int32)
    count['y1'] = (count['y'] / bin_size).astype(np.int32)
    count['pos'] = count['x1'].astype(str) + "-" + count['y1'].astype(str)
    bin_data = count.groupby(['pos', 'geneID'])['UMICount'].sum()
    cells = set(x[0] for x in bin_data.index)
    genes = set(x[1] for x in bin_data.index)
    cellsdic = dict(zip(cells, range(0, len(cells))))
    genesdic = dict(zip(genes, range(0, len(genes))))
    rows = [cellsdic[x[0]] for x in bin_data.index]
    cols = [genesdic[x[1]] for x in bin_data.index]
    exp_matrix = sparse.csr_matrix((bin_data.values, (rows, cols))) if is_sparse else \
        sparse.csr_matrix((bin_data.values, (rows, cols))).toarray()
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    adata = AnnData(X=exp_matrix, obs=obs, var=var)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
    adata.obsm['spatial'] = pos

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
    if data_name_list[i] == "10X":
        adata = read_10X_Visium(path=os.path.join(data_path, data_name_list[i]))
        # 		adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name_list[i])
        adata = deepen._get_image_crop(adata, data_name=data_name_list[i])
        adata = deepen._get_augment(adata, spatial_type="LinearRegress")
        graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
    if data_name_list[i] == "stereo":
        adata = read_stereoSeq(path=os.path.join(data_path, data_name_list[i]))
        adata = deepen._get_adata(platform="stereoSeq", data_path=data_path, data_name=data_name_list[i])
        adata = deepen._get_augment(adata, spatial_type="BallTree", use_morphological=False)
        graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="BallTree")
    if data_name_list[i] == "slide":
        adata = read_SlideSeq(path=os.path.join(data_path, data_name_list[i]))
        # 		adata = deepen._get_adata(platform="slideSeq", data_path=data_path, data_name=data_name_list[i])
        adata = deepen._get_augment(adata, spatial_type="BallTree", use_morphological=False)
        graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="BallTree")

    save_data_path = Path(os.path.join(save_path, "Data", data_name_list[i]))
    save_data_path.mkdir(parents=True, exist_ok=True)
    adata.write(os.path.join(save_data_path, f'{data_name_list[i]}_raw.h5ad'), compression="gzip")

    augement_data_list.append(adata)
    graph_list.append(graph_dict)

######## Synthetic Datasets and Graphs
multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list,
                                                            data_name_list=data_name_list, graph_list=graph_list)

###### Enhanced data preprocessing
data = deepen._data_process(multiple_adata, pca_n_comps=200)

deepst_embed = deepen._fit(
    data=data,
    graph_dict=multiple_graph,
    domains=multiple_adata.obs["batch"].values,  ##### Input to Domain Adversarial Model
    n_domains=len(data_name_list))
multiple_adata.obsm["DeepST_embed"] = deepst_embed
multiple_adata = deepen._get_cluster_data(multiple_adata, n_domains=n_domains, priori=True)

sc.pp.neighbors(multiple_adata, use_rep='DeepST_embed')
sc.tl.umap(multiple_adata)
sc.pl.umap(multiple_adata, color=["DeepST_refine_domain", "batch_name"])
plt.savefig(os.path.join(save_path, f'{"_".join(data_name_list)}_umap.pdf'), bbox_inches='tight', dpi=300)

for data_name in data_name_list:
    adata = multiple_adata[multiple_adata.obs["batch_name"] == data_name]
    sc.pl.spatial(adata, color='DeepST_refine_domain', frameon=False, spot_size=150)
    plt.savefig(os.path.join(save_path, f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=300)

print(multiple_adata.isbacked)
multiple_adata.filename = save_path + '/mouseOB.h5ad'
print(multiple_adata.isbacked)
