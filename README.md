# Portal
Adversarial domain translation networks enable fast and accurate large-scale atlas-level single-cell data integration.

preprint: [https://www.biorxiv.org/content/10.1101/2021.11.16.468892v1](https://www.biorxiv.org/content/10.1101/2021.11.16.468892v1).

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

### Demos
We provide demos for users to get a quick start: [Demo 1](https://jiazhao97.github.io/Portal_demo1/).

## Development
This package is developed by Jia Zhao (jzhaoaz@connect.ust.hk) and Gefei Wang (gwangas@connect.ust.hk). 

## Citation
Jia Zhao, Gefei Wang, Jingsi Ming, Zhixiang Lin, Yang Wang, Tabula Microcebus Consortium, Angela Ruohao Wu, Can Yang. Adversarial domain translation networks enable fast and accurate large-scale atlas-level single-cell data integration. bioRxiv 2021.11.16.468892; doi: [https://doi.org/10.1101/2021.11.16.468892](https://doi.org/10.1101/2021.11.16.468892).
