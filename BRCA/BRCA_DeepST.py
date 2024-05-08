import os 

print(os.getcwd())#显示当前路径
os.chdir('/home/lixiangyu/benchmark/DeepST/DeepST-main/deepst')#更改路径，''里面为更改的路径
print(os.getcwd())#显示当前路径

from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData


from matplotlib.image import imread
import json

def add_visium_image(
    adata,
    path: str,
    library_id = None,
    load_images = True,
    source_image_path = None,
    quality ='hires',
) -> AnnData:
    
    path = Path(path)

    adata.uns["spatial"] = dict()

    from h5py import File

    if library_id is None:
        library_id = 'deepst'

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        tissue_positions_file = (
            path / "spatial/tissue_positions.csv"
            if (path / "spatial/tissue_positions.csv").exists()
            else path / "spatial/tissue_positions_list.csv"
        )
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            hires_image=path / "spatial/tissue_hires_image.png",
            lowres_image=path / "spatial/tissue_lowres_image.png",
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logg.warning(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            try:
                adata.uns["spatial"][library_id]["images"][res] = imread(
                    str(files[f"{res}_image"])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

        # read coordinates
        positions = pd.read_csv(
            files["tissue_positions_file"],
            header=0 if tissue_positions_file.name == "tissue_positions.csv" else None,
            index_col=0,
        )
        positions.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm["spatial"] = adata.obs[
            ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        adata.obs.drop(
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )
            
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


data_name_list = ['1142243F','1160920F']
data_path = '/home/lixiangyu/benchmark/data/BRCA/'
save_path = '/home/lixiangyu/benchmark/data/BRCA/DeepST/'


deepen = run(save_path = save_path, 
	task = "Integration",
	pre_epochs = 800, 
	epochs = 1000, 
	use_gpu = True,
	)

###### Generate an augmented list of multiple datasets
augement_data_list = []
graph_list = []
for i in range(len(data_name_list)):
# 	adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name_list[i])
	adata = sc.read_h5ad(data_path+ data_name_list[i] +'.h5ad')
	adata = add_visium_image(adata, data_path + data_name_list[i])
	adata = deepen._get_image_crop(adata, data_name=data_name_list[i])
	adata = deepen._get_augment(adata, spatial_type="LinearRegress")
	graph_dict = deepen._get_graph(adata.obsm["spatial"], distType = "KDTree")
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
np.savetxt(os.path.join(save_path, "deepst_embedding.csv"), deepst_embed, delimiter=",")
multiple_adata.obsm["DeepST_embed"] = deepst_embed


print(multiple_adata.isbacked)
multiple_adata.filename = save_path + '/BRCA.h5ad'
print(multiple_adata.isbacked)