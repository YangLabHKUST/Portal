# Portal
A fast, accurate and scalable method for single-cell data integration.

## Installation
To run Portal, please follow the installation instruction:
```bash
git clone https://github.com/YangLabHKUST/Portal.git
cd Portal
conda env update --f environment.yml
conda activate portal
```

## Quick Start
### Basic Usage
Starting with raw count matrices formatted as AnnData objects, Portal uses a standard pipline adopted by Seurat and Scanpy to preprocess data, followed by PCA for dimensionality reduction. After preprocessing, Portal can be trained via ```model.train()```.
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
The evaluating procedure ```model.eval()``` saves the integrated latent representation of cells in ```model.latent```, which can be used for downstream integrative analysis.

### Memory-efficient Version
To deal with large single-cell datasets, we developed a memory-efficient version of Portal. In this version, by reading mini-batches from the disk.
```python
model = portal.model.Model()
model.preprocess_memory_efficient(adata_A_path="adata_1.h5ad", adata_B_path="adata_2.h5ad")
model.train_memory_efficient()
model.eval_memory_efficient()
```

### Demos
We provide demos for users to get a quick start: [Demo 1](https://jiazhao97.github.io/Portal_demo1/).

## Development
This package is developed by Jia Zhao (jzhaoaz@connect.ust.hk) and Gefei Wang (gwangas@connect.ust.hk). 

## Citation
