# Portal
[![DOI](https://zenodo.org/badge/423325112.svg)](https://zenodo.org/badge/latestdoi/423325112)

*Adversarial domain translation networks for integrating large-scale atlas-level single-cell datasets*

An efficient, accurate and flexible method for single-cell data integration.

Check out our manuscript in Nature Computational Science:
+ [Nature Computational Science website](https://www.nature.com/articles/s43588-022-00251-y)
+ [Read fulltext link](https://rdcu.be/cOCbU)
+ [Preprint in bioRxiv](https://www.biorxiv.org/content/10.1101/2021.11.16.468892v2)

## Reproducibility
We provide [source codes](https://github.com/jiazhao97/Portal-reproducibility) for reproducing the experiments of the paper "Adversarial domain translation networks for fast and accurate integration of large-scale atlas-level single-cell datasets".

+ [Integration of mouse spleen datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-Spleen.html) (we reproduce the result of performance metrics in this notebook as an example). [Benchmarking](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Benchmarking-Spleen.html).  
+ [Integration of mouse marrow datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-Marrow.html).
+ [Integration of mouse bladder datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-Bladder.html).
+ [Integration of mouse brain cerebellum datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-MouseBrain-CB.html).
+ [Integration of mouse brain hippocampus datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-MouseBrain-HC.html).
+ [Integration of mouse brain thalamus datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-MouseBrain-TH.html).
+ [Integration of human PBMC datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-sensitivity.html) (sensitivity analysis).
+ [Integration of entire mouse cell atlases from the Tablula Muris project](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-TabulaMuris-full.html).
+ [Integration of mouse brain scRNA-seq and snRNA-seq datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-MouseBrain-CellNuclei.html).
+ [Integration of human PBMC scRNA-seq and human brain snRNA-seq datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-BloodCell-BrainNuclei.html).
+ [Integration of scRNA-seq and scATAC-seq datasets](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-PBMC-ATACseq.html).
+ [Integration of developmental trajectories](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-trajectory.html).
+ [Integration of spermatogenesis differentiation process across multiple species](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Reproduce-Spermatogenesis.html). Gene lists from Ensembl Biomart (we only use genes that are assigned with the type "ortholog_one2one" in the lists): [orthologues (human vs mouse)](https://github.com/jiazhao97/Portal-reproducibility/raw/main/orthologues_human_mouse.txt), [orthologues (human vs macaque)](https://github.com/jiazhao97/Portal-reproducibility/raw/main/orthologues_human_macaque.txt).

## Installation
To run Portal, please follow the installation instruction:
```bash
git clone https://github.com/YangLabHKUST/Portal.git
cd Portal
conda env update --f environment.yml
conda activate portal
```
Normally the installation time is less than 5 minutes.

## Quick Start
### Basic Usage
Starting with raw count matrices formatted as AnnData objects, Portal uses a standard pipline adopted by Seurat and Scanpy to preprocess data, followed by PCA for dimensionality reduction. After preprocessing, Portal can be trained via `model.train()`.
```python
import portal
import scanpy as sc

# read AnnData
adata_1 = sc.read_h5ad("adata_1.h5ad")
adata_2 = sc.read_h5ad("adata_2.h5ad")

model = portal.model.Model()
model.preprocess(adata_1, adata_2) # perform preprocess and PCA
model.train() # train the model
model.eval() # get integrated latent representation of cells
```
The evaluating procedure `model.eval()` saves the integrated latent representation of cells in `model.latent`, which can be used for downstream integrative analysis.

#### Parameters in `portal.model.Model()`:
* `lambdacos`: Coefficient of the regularizer for preserving cosine similarity across domains. *Default*: `20.0`.
* `training_steps`: Number of steps for training. *Default*: `2000`. Use `training_steps=1000` for datasets with sample size < 20,000.
* `npcs`: Dimensionality of the embeddings in each domain (number of PCs). *Default*: `30`.
* `n_latent`: Dimensionality of the shared latent space. *Default*: `20`.
* `batch_size`: Batch size for training. *Default*: `500`.
* `seed`: Random seed. *Default*: `1234`.

The default setting of the parameter `lambdacos` works in general. We also enable tuning of this parameter to achieve a better performance, see [**Tuning `lambdacos` (optional)**](#tuning-lambdacos-optional). For the integration task where the cosine similarity is not a reliable cross-domain correspondance (such as cross-species integration), we recommend to use a lower value such as `lambdacos=10.0`.

### Memory-efficient Version
To deal with large single-cell datasets, we also developed a memory-efficient version by reading mini-batches from the disk:
```python
model = portal.model.Model()
model.preprocess_memory_efficient(adata_A_path="adata_1.h5ad", adata_B_path="adata_2.h5ad")
model.train_memory_efficient()
model.eval_memory_efficient()
```

### Integrating Multiple Datasets
Portal integrates multiple datasets incrementally. Given `adata_list = [adata_1, ..., adata_n]` is a list of AnnData objects, they can be integrated by running the following commands: 
```python
lowdim_list = portal.utils.preprocess_datasets(adata_list)
integrated_data = portal.utils.integrate_datasets(lowdim_list)
```

### Tuning `lambdacos` (optional)
An optional choice is to tune the parameter `lambdacos` in the range [15.0, 50.0]. Users can run the following command to search for an optimal parameter that yields the best integration result in terms of the mixing metric:
```python
lowdim_list = portal.utils.preprocess_datasets(adata_list)
integrated_data = portal.utils.integrate_datasets(lowdim_list, search_cos=True)
```

### Recovering expression matrices
Portal can provide harmonized expression matrices (in scaled level or log-normalized level):
```python
lowdim_list, hvg, mean, std, pca = portal.utils.preprocess_recover_expression(adata_list)
expression_scaled, expression_log_normalized = portal.utils.integrate_recover_expression(lowdim_list, mean, std, pca)
```

### Demos
We provide demos for users to get a quick start: [Demo 1](https://jiazhao97.github.io/Portal_demo1/), [Demo 2](https://htmlpreview.github.io/?https://github.com/jiazhao97/Portal-reproducibility/blob/main/Portal_recover_expression.html).

## Development
This package is developed by Jia Zhao (jzhaoaz@connect.ust.hk) and Gefei Wang (gwangas@connect.ust.hk). 

## Citation
Jia Zhao, Gefei Wang, Jingsi Ming, Zhixiang Lin, Yang Wang, The Tabula Microcebus Consortium, Angela Ruohao Wu, Can Yang. Adversarial domain translation networks for integrating large-scale atlas-level single-cell datasets. Nature Computational Science 2, 317–330 (2022).
