import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import umap
from portal.model import *
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import anndata2ri
from sklearn.neighbors import NearestNeighbors

def preprocess_datasets(adata_list, # list of anndata to be integrated
                        hvg_num=4000, # number of highly variable genes for each anndata
                        save_embedding=False, # save low-dimensional embeddings or not
                        data_path="data"
                        ):

    if len(adata_list) < 2:
        raise ValueError("There should be at least two datasets for integration!")

    sample_size_list = []

    print("Finding highly variable genes...")
    for i, adata in enumerate(adata_list):
        sample_size_list.append(adata.shape[0])
        # adata = adata_input.copy()
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=hvg_num)
        hvg = adata.var[adata.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        if i == 0:
            hvg_total = hvg
        else:
            hvg_total = hvg_total & hvg
        if len(hvg_total) < 100:
            raise ValueError("The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(hvg_total))

    print("Normalizing and scaling...")
    for i, adata in enumerate(adata_list):
        # adata = adata_input.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, hvg_total]
        sc.pp.scale(adata, max_value=10)
        if i == 0:
            adata_total = adata
        else:
            adata_total = adata_total.concatenate(adata, index_unique=None)

    print("Dimensionality reduction via PCA...")
    npcs = 30
    pca = PCA(n_components=npcs, svd_solver="arpack", random_state=0)
    adata_total.obsm["X_pca"] = pca.fit_transform(adata_total.X)

    indices = np.cumsum(sample_size_list)

    data_path = os.path.join(data_path, "preprocess")

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if save_embedding:
        for i in range(len(indices)):
            if i == 0:
                np.save(os.path.join(data_path, "lowdim_1.npy"), 
                        adata_total.obsm["X_pca"][:indices[0], :npcs])
            else:
                np.save(os.path.join(data_path, "lowdim_%d.npy" % (i + 1)), 
                        adata_total.obsm["X_pca"][indices[i-1]:indices[i], :npcs])

    lowdim = adata_total.obsm["X_pca"].copy()
    lowdim_list = [lowdim[:indices[0], :npcs] if i == 0 else lowdim[indices[i - 1]:indices[i], :npcs] for i in range(len(indices))]

    return lowdim_list


def integrate_datasets(lowdim_list, # list of low-dimensional representations
                       search_cos=False, # searching for an optimal lambdacos
                       lambda_cos=10.0,
                       training_steps=None,
                       space=None, # None or "reference" or "latent"
                       data_path="data"
                       ):

    if training_steps == None:
        training_steps = 2000

    if space == None:
        if len(lowdim_list) == 2:
            space = "latent"
        else:
            space = "reference"

    print("Incrementally integrating %d datasets..." % len(lowdim_list))

    if not search_cos:
        # if not search hyperparameter lambdacos
        for i in range(len(lowdim_list) - 1):
            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            model = Model(lambdacos=lambda_cos,
                          training_steps=training_steps, 
                          data_path=os.path.join(data_path, "preprocess"), 
                          model_path="models/%d_datasets" % (i + 2), 
                          result_path="results/%d_datasets" % (i + 2))
            if i == 0:
                model.emb_A = lowdim_list[0]
            else:
                model.emb_A = emb_total
            model.emb_B = lowdim_list[i + 1]
            model.train()
            model.eval()
            emb_total = model.data_Aspace
        if space == "reference":
            return emb_total
        elif space == "latent":
            return model.latent
        else:
            raise ValueError("Space should be either 'reference' or 'latent'.")
    else:
        for i in range(len(indices) - 1):
            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            for lambda_cos in [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
                model = Model(lambdacos=lambda_cos,
                              training_steps=training_steps, 
                              data_path=os.path.join(data_path, "preprocess"), 
                              model_path="models/%d_datasets" % (i + 2), 
                              result_path="results/%d_datasets" % (i + 2))
                if i == 0:
                    model.emb_A = lowdim_list[0]
                else:
                    model.emb_A = emb_total
                model.emb_B = lowdim_list[i + 1]
                model.train()
                model.eval()
                meta = pd.DataFrame(index=np.arange(model.emb_A.shape[0] + model.emb_B.shape[0]))
                meta["method"] = ["A"] * model.emb_A.shape[0] + ["B"] * model.emb_B.shape[0]
                mixing = calculate_mixing_metric(model.latent, meta, k=5, max_k=300, methods=list(set(meta.method)))
                print("lambda_cos: %f, mixing metric: %f \n" % (lambda_cos, mixing))
                if lambda_cos == 10.0:
                    model_opt = model
                    mixing_metric_opt = mixing
                elif mixing < mixing_metric_opt:
                    model_opt = model
                    mixing_metric_opt = mixing
            emb_total = model_opt.data_Aspace
        if space == "reference":
            return emb_total
        elif space == "latent":
            return model_opt.latent
        else:
            raise ValueError("Space should be either 'reference' or 'latent'.")


def calculate_mixing_metric(data, meta, methods, k=5, max_k=300, subsample=True):
    if subsample:
        if data.shape[0] >= 1e5:
            np.random.seed(1234)
            subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
            data = data[subsample_idx]
            meta = meta.iloc[subsample_idx]
    lowdim = data

    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='kd_tree').fit(lowdim)
    _, indices = nbrs.kneighbors(lowdim)
    indices = indices[:, 1:]
    mixing = np.zeros((data.shape[0], 2))
    for i in range(data.shape[0]):
        if len(np.where(meta.method[indices[i, :]] == methods[0])[0]) > k-1:
            mixing[i, 0] = np.where(meta.method[indices[i, :]] == methods[0])[0][k-1]
        else: mixing[i, 0] = max_k - 1
        if len(np.where(meta.method[indices[i, :]] == methods[1])[0]) > k-1:
            mixing[i, 1] = np.where(meta.method[indices[i, :]] == methods[1])[0][k-1]
        else: mixing[i, 1] = max_k - 1
    return np.mean(np.median(mixing, axis=1) + 1)

def calculate_ARI(data, meta, anno_A="drop_subcluster", anno_B="subcluster"):
    # np.random.seed(1234)
    if data.shape[0] > 1e5:
        np.random.seed(1234)
        subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
        data = data[subsample_idx]
        meta = meta.iloc[subsample_idx]
    lowdim = data

    cellid = meta.index.astype(str)
    method = meta["method"].astype(str)
    cluster_A = meta[anno_A].astype(str)
    if (anno_B != anno_A):
        cluster_B = meta[anno_B].astype(str)

    rpy2.robjects.numpy2ri.activate()
    nr, nc = lowdim.shape
    lowdim = ro.r.matrix(lowdim, nrow=nr, ncol=nc)
    ro.r.assign("data", lowdim)
    rpy2.robjects.numpy2ri.deactivate()

    cellid = ro.StrVector(cellid)
    ro.r.assign("cellid", cellid)
    method = ro.StrVector(method)
    ro.r.assign("method", method)
    cluster_A = ro.StrVector(cluster_A)
    ro.r.assign("cluster_A", cluster_A)
    if (anno_B != anno_A):
        cluster_B = ro.StrVector(cluster_B)
        ro.r.assign("cluster_B", cluster_B)

    ro.r("set.seed(1234)")
    ro.r['library']("Seurat")
    ro.r['library']("mclust")

    ro.r("comb_normalized <- t(data)")
    ro.r('''rownames(comb_normalized) <- paste("gene", 1:nrow(comb_normalized), sep = "")''')
    ro.r("colnames(comb_normalized) <- as.vector(cellid)")

    ro.r("comb_raw <- matrix(0, nrow = nrow(comb_normalized), ncol = ncol(comb_normalized))")
    ro.r("rownames(comb_raw) <- rownames(comb_normalized)")
    ro.r("colnames(comb_raw) <- colnames(comb_normalized)")

    ro.r("comb <- CreateSeuratObject(comb_raw)")
    ro.r('''scunitdata <- Seurat::CreateDimReducObject(
                embeddings = t(comb_normalized),
                stdev = as.numeric(apply(comb_normalized, 2, stats::sd)),
                assay = "RNA",
                key = "scunit")''')
    ro.r('''comb[["scunit"]] <- scunitdata''')

    ro.r("comb@meta.data$method <- method")

    ro.r("comb@meta.data$cluster_A <- cluster_A")
    if (anno_B != anno_A):
        ro.r("comb@meta.data$cluster_B <- cluster_B")

    ro.r('''comb <- FindNeighbors(comb, reduction = "scunit", dims = 1:ncol(data), force.recalc = TRUE, verbose = FALSE)''')
    ro.r('''comb <- FindClusters(comb, verbose = FALSE)''')

    if (anno_B != anno_A):
        method_set = pd.unique(meta["method"])
        method_A = method_set[0]
        ro.r.assign("method_A", method_A)
        method_B = method_set[1]
        ro.r.assign("method_B", method_B)
        ro.r('''indx_A <- which(comb$method == method_A)''')
        ro.r('''indx_B <- which(comb$method == method_B)''')

        ro.r("ARI_A <- adjustedRandIndex(comb$cluster_A[indx_A], comb$seurat_clusters[indx_A])")
        ro.r("ARI_B <- adjustedRandIndex(comb$cluster_B[indx_B], comb$seurat_clusters[indx_B])")
        ARI_A = np.array(ro.r("ARI_A"))[0]
        ARI_B = np.array(ro.r("ARI_B"))[0]

        return ARI_A, ARI_B
    else:
        ro.r("ARI_A <- adjustedRandIndex(comb$cluster_A, comb$seurat_clusters)")
        ARI_A = np.array(ro.r("ARI_A"))[0]

        return ARI_A

def calculate_kBET(data, meta):
    cellid = meta.index.astype(str)
    method = meta["method"].astype(str)

    rpy2.robjects.numpy2ri.activate()
    nr, nc = data.shape
    data = ro.r.matrix(data, nrow=nr, ncol=nc)
    ro.r.assign("data", data)
    rpy2.robjects.numpy2ri.deactivate()

    cellid = ro.StrVector(cellid)
    ro.r.assign("cellid", cellid)
    method = ro.StrVector(method)
    ro.r.assign("method", method)

    ro.r("set.seed(1234)")
    ro.r['library']("kBET")

    accept_rate = []
    for _ in range(100):
        ro.r("subset_id <- sample.int(n = length(method), size = 1000, replace=FALSE)")

        ro.r("batch.estimate <- kBET(data[subset_id,], method[subset_id], do.pca = FALSE, plot=FALSE)")
        accept_rate.append(np.array(ro.r("mean(batch.estimate$results$kBET.pvalue.test > 0.05)")))

    return np.median(accept_rate)

def plot_UMAP(data, meta, space="latent", score=None, colors=["method"], subsample=False,
              save=False, result_path=None, filename_suffix=None):
    if filename_suffix is not None:
        filenames = [os.path.join(result_path, "%s-%s-%s.pdf" % (space, c, filename_suffix)) for c in colors]
    else:
        filenames = [os.path.join(result_path, "%s-%s.pdf" % (space, c)) for c in colors]

    if subsample:
        if data.shape[0] >= 1e5:
            np.random.seed(1234)
            subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
            data = data[subsample_idx]
            meta = meta.iloc[subsample_idx]
            if score is not None:
                score = score[subsample_idx]
    
    adata = anndata.AnnData(X=data)
    adata.obs.index = meta.index
    adata.obs = pd.concat([adata.obs, meta], axis=1)
    adata.var.index = "dim-" + adata.var.index
    adata.obsm["latent"] = data
    
    # run UMAP
    reducer = umap.UMAP(n_neighbors=30,
                        n_components=2,
                        metric="correlation",
                        n_epochs=None,
                        learning_rate=1.0,
                        min_dist=0.3,
                        spread=1.0,
                        set_op_mix_ratio=1.0,
                        local_connectivity=1,
                        repulsion_strength=1,
                        negative_sample_rate=5,
                        a=None,
                        b=None,
                        random_state=1234,
                        metric_kwds=None,
                        angular_rp_forest=False,
                        verbose=True)
    embedding = reducer.fit_transform(adata.obsm["latent"])
    adata.obsm["X_umap"] = embedding

    n_cells = embedding.shape[0]
    if n_cells >= 10000:
        size = 120000 / n_cells
    else:
        size = 12

    for i, c in enumerate(colors):
        groups = sorted(set(adata.obs[c].astype(str)))
        if "nan" in groups:
            groups.remove("nan")
        palette = "rainbow"
        if save:
            fig = sc.pl.umap(adata, color=c, palette=palette, groups=groups, return_fig=True, size=size)
            fig.savefig(filenames[i], bbox_inches='tight', dpi=300)
        else:
            sc.pl.umap(adata, color=c, palette=palette, groups=groups, size=size)

    if space == "Aspace":
        method_set = pd.unique(meta["method"])
        adata.obs["score"] = score
        adata.obs["margin"] = (score < -5.0) * 1
        fig = sc.pl.umap(adata[meta["method"]==method_set[1]], color="score", palette=palette, groups=groups, return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-score.pdf" % space), bbox_inches='tight', dpi=300)
        fig = sc.pl.umap(adata[meta["method"]==method_set[1]], color="margin", palette=palette, groups=groups, return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-margin.pdf" % space), bbox_inches='tight', dpi=300)
    if space == "Bspace":
        method_set = pd.unique(meta["method"])
        adata.obs["score"] = score
        adata.obs["margin"] = (score < -5.0) * 1
        fig = sc.pl.umap(adata[meta["method"]==method_set[0]], color="score", palette=palette, groups=groups, return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-score.pdf" % space), bbox_inches='tight', dpi=300)
        fig = sc.pl.umap(adata[meta["method"]==method_set[0]], color="margin", palette=palette, groups=groups, return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-margin.pdf" % space), bbox_inches='tight', dpi=300)

def read_TM(tissue, path_SS2, path_10X, data_path):
    meta_SS2 = pd.read_csv("/import/home/jzhaoaz/jiazhao/scUNIT-v2/Data/TabulaMuris/rawdata/annotations_facs.csv", keep_default_na=False)
    meta_10X = pd.read_csv("/import/home/jzhaoaz/jiazhao/scUNIT-v2/Data/TabulaMuris/rawdata/annotations_droplet.csv", keep_default_na=False)

    if tissue == "Heart_and_Aorta":
        # Read SS2 cell-by-gene counts
        adata_SS2_Heart = sc.read_csv(os.path.join(path_SS2, "Heart-counts.csv")).transpose()
        ERCC_idx = pd.Series(adata_SS2_Heart.var.index).str.startswith('ERCC')
        cell_idx = adata_SS2_Heart.obs.index.isin(meta_SS2[(meta_SS2.cell_ontology_class != 0) & 
                                                         (meta_SS2.cell_ontology_class != "")].cell)
        adata_SS2_Heart = adata_SS2_Heart[cell_idx, -ERCC_idx]

        adata_SS2_Aorta = sc.read_csv(os.path.join(path_SS2, "Aorta-counts.csv")).transpose()
        ERCC_idx = pd.Series(adata_SS2_Aorta.var.index).str.startswith('ERCC')
        cell_idx = adata_SS2_Aorta.obs.index.isin(meta_SS2[(meta_SS2.cell_ontology_class != 0) & 
                                                         (meta_SS2.cell_ontology_class != "")].cell)
        adata_SS2_Aorta = adata_SS2_Aorta[cell_idx, -ERCC_idx]

        gene = adata_SS2_Heart.var.index & adata_SS2_Aorta.var.index
        adata_SS2_Heart = adata_SS2_Heart[:, gene]
        adata_SS2_Aorta = adata_SS2_Aorta[:, gene]
        adata_SS2 = adata_SS2_Heart.concatenate(adata_SS2_Aorta, index_unique=None)
    else:
        # Read SS2 cell-by-gene counts
        adata_SS2 = sc.read_csv(os.path.join(path_SS2, "%s-counts.csv" % tissue)).transpose()
        ERCC_idx = pd.Series(adata_SS2.var.index).str.startswith('ERCC')
        cell_idx = adata_SS2.obs.index.isin(meta_SS2[(meta_SS2.cell_ontology_class != 0) & 
                                                         (meta_SS2.cell_ontology_class != "")].cell)
        adata_SS2 = adata_SS2[cell_idx, -ERCC_idx]

    # Read 10X cell-by-gene counts
    channels = sorted(set(meta_10X[meta_10X.tissue == tissue].channel))
    for i, channel in enumerate(channels):
        if i == 0:
            adata_10X = sc.read_10x_mtx(path_10X + '/%s-%s/' % (tissue, channel), 
                                        var_names='gene_symbols',cache=False)
            adata_10X.obs.index = channel + "_" + adata_10X.obs.index
            adata_10X.obs.index = adata_10X.obs.index.map(lambda x: x[:-2])
            cell_idx = adata_10X.obs.index.isin(meta_10X[(meta_10X.cell_ontology_class != 0) &
                                                         (meta_10X.cell_ontology_class != "")].cell)
            adata_10X = adata_10X[cell_idx, :]
        else:
            tmp = sc.read_10x_mtx(path_10X + '/%s-%s/' % (tissue, channel), 
                                  var_names='gene_symbols',cache=False)
            tmp.obs.index = channel + "_" + tmp.obs.index
            tmp.obs.index = tmp.obs.index.map(lambda x: x[:-2])
            cell_idx = tmp.obs.index.isin(meta_10X[(meta_10X.cell_ontology_class != 0) &
                                                   (meta_10X.cell_ontology_class != "")].cell)
            adata_10X = adata_10X.concatenate(tmp[cell_idx, :], index_unique=None)

    celltype_SS2 = meta_SS2[meta_SS2.cell.isin(adata_SS2.obs.index)][["cell", "cell_ontology_class"]].set_index("cell")
    celltype_SS2["method"] = "SS2"
    celltype_10X = meta_10X[meta_10X.cell.isin(adata_10X.obs.index)][["cell", "cell_ontology_class"]].set_index("cell")
    celltype_10X["method"] = "10X"
    meta = pd.concat([celltype_SS2, celltype_10X]).rename(columns={"cell_ontology_class": "celltype"})
    meta.to_pickle(os.path.join(data_path, "meta_raw.pkl"))

    return adata_SS2, adata_10X

def read_TM_full(path_SS2, path_10X, data_path):
    meta_SS2 = pd.read_csv("/import/home/jzhaoaz/jiazhao/scUNIT-v2/Data/TabulaMuris/rawdata/annotations_facs.csv", keep_default_na=False)
    meta_10X = pd.read_csv("/import/home/jzhaoaz/jiazhao/scUNIT-v2/Data/TabulaMuris/rawdata/annotations_droplet.csv", keep_default_na=False)

    # SS2
    for tissue in ["Aorta", "Bladder", "Brain_Myeloid", "Brain_Non-Myeloid", "Diaphragm", "Fat", "Heart",  
                   "Kidney", "Large_Intestine", "Limb_Muscle", "Liver", "Lung", "Mammary_Gland", "Marrow", 
                   "Pancreas", "Skin", "Spleen", "Thymus", "Tongue", "Trachea"]:

        # Read SS2 cell-by-gene counts
        adata_SS2 = sc.read_csv(os.path.join(path_SS2, "%s-counts.csv" % tissue)).transpose()
        ERCC_idx = pd.Series(adata_SS2.var.index).str.startswith('ERCC')
        cell_idx = adata_SS2.obs.index.isin(meta_SS2[(meta_SS2.cell_ontology_class != 0) & 
                                                         (meta_SS2.cell_ontology_class != "")].cell)
        adata_SS2 = adata_SS2[cell_idx, -ERCC_idx]
        if tissue == "Aorta":
            adata_SS2_all = adata_SS2.copy()
        else:
            genes = adata_SS2_all.var.index & adata_SS2.var.index
            adata_SS2_all = adata_SS2_all[:, genes].concatenate(adata_SS2[:, genes], index_unique=None)

    # 10X
    for tissue in ["Bladder", "Heart_and_Aorta", "Kidney", "Limb_Muscle", "Liver", "Lung", "Mammary_Gland", 
                   "Marrow", "Spleen", "Thymus", "Tongue", "Trachea"]:

        # Read 10X cell-by-gene counts
        channels = sorted(set(meta_10X[meta_10X.tissue == tissue].channel))
        for i, channel in enumerate(channels):
            if i == 0:
                adata_10X = sc.read_10x_mtx(path_10X + '/%s-%s/' % (tissue, channel), 
                                            var_names='gene_symbols',cache=False)
                adata_10X.obs.index = channel + "_" + adata_10X.obs.index
                adata_10X.obs.index = adata_10X.obs.index.map(lambda x: x[:-2])
                cell_idx = adata_10X.obs.index.isin(meta_10X[(meta_10X.cell_ontology_class != 0) &
                                                             (meta_10X.cell_ontology_class != "")].cell)
                adata_10X = adata_10X[cell_idx, :]
            else:
                tmp = sc.read_10x_mtx(path_10X + '/%s-%s/' % (tissue, channel), 
                                      var_names='gene_symbols',cache=False)
                tmp.obs.index = channel + "_" + tmp.obs.index
                tmp.obs.index = tmp.obs.index.map(lambda x: x[:-2])
                cell_idx = tmp.obs.index.isin(meta_10X[(meta_10X.cell_ontology_class != 0) &
                                                       (meta_10X.cell_ontology_class != "")].cell)
                adata_10X = adata_10X.concatenate(tmp[cell_idx, :], index_unique=None)
        if tissue == "Bladder":
            adata_10X_all = adata_10X.copy()
        else:
            genes = adata_10X_all.var.index & adata_10X.var.index
            adata_10X_all = adata_10X_all[:, genes].concatenate(adata_10X[:, genes], index_unique=None)

    celltype_SS2 = meta_SS2[meta_SS2.cell.isin(adata_SS2_all.obs.index)][["cell", "tissue", "cell_ontology_class"]].set_index("cell")
    celltype_SS2["method"] = "SS2"
    celltype_10X = meta_10X[meta_10X.cell.isin(adata_10X_all.obs.index)][["cell", "tissue", "cell_ontology_class"]].set_index("cell")
    celltype_10X["method"] = "10X"
    meta = pd.concat([celltype_SS2, celltype_10X]).rename(columns={"cell_ontology_class": "celltype"})
    meta.to_pickle(os.path.join(data_path, "meta_raw.pkl"))

    return adata_SS2_all, adata_10X_all

def read_MouseBrainAtlas(tissue, datapath_Drop, datapath_10X, data_path):
    # Drop-seq
    dropviz_RDSname_dict = {"TH": "F_GRCm38.81.P60Thalamus.subcluster.assign.RDS",
                            "HC": "F_GRCm38.81.P60Hippocampus.subcluster.assign.RDS",
                            "CB": "F_GRCm38.81.P60Cerebellum_ALT.subcluster.assign.RDS"}
    dropviz_dgename_dict = {"TH": "F_GRCm38.81.P60Thalamus.raw.dge.txt.gz",
                            "HC": "F_GRCm38.81.P60Hippocampus.raw.dge.txt.gz",
                            "CB": "F_GRCm38.81.P60Cerebellum_ALT.raw.dge.txt.gz"}

    dropviz_anno = ro.r['readRDS'](datapath_Drop + "/annotation.BrainCellAtlas_Saunders_version_2018.04.01.RDS")
    ro.r.assign("dropviz_anno", dropviz_anno)
    ro.r.assign("tissue", tissue)
    ro.r('''dropviz_anno_class <- dropviz_anno[which(dropviz_anno$tissue == tissue), c("class")]''')
    dropviz_anno_class = list(ro.r("dropviz_anno_class"))
    ro.r('''dropviz_anno_class_marker <- dropviz_anno[which(dropviz_anno$tissue == tissue), c("class_marker")]''')
    dropviz_anno_class_marker = list(ro.r("dropviz_anno_class_marker"))
    ro.r('''dropviz_anno_subcluster <- dropviz_anno[which(dropviz_anno$tissue == tissue), c("subcluster")]''')
    dropviz_anno_subcluster = list(ro.r("dropviz_anno_subcluster"))

    subcluster_assign = ro.r['readRDS'](datapath_Drop + "/" + dropviz_RDSname_dict[tissue])
    ro.r.assign("subcluster_assign", subcluster_assign)
    ro.r("subcluster_assign_name <- names(subcluster_assign)")
    ro.r("subcluster_assign <- as.vector(subcluster_assign)")
    subcluster_assign_name = list(ro.r("subcluster_assign_name"))
    subcluster_assign = list(ro.r("subcluster_assign"))

    meta_dropseq_dict = pd.DataFrame({"class": dropviz_anno_class, "marker": dropviz_anno_class_marker}, index=dropviz_anno_subcluster)
    meta_dropseq_dict["subclass"] = meta_dropseq_dict["class"] + " (" + meta_dropseq_dict["marker"] + ")"

    meta_dropseq = pd.DataFrame({"subcluster": subcluster_assign}, index=subcluster_assign_name)
    meta_dropseq["drop_class"] = meta_dropseq_dict.loc[list(meta_dropseq["subcluster"].values)]["class"].values
    meta_dropseq["drop_subclass"] = meta_dropseq_dict.loc[list(meta_dropseq["subcluster"].values)]["subclass"].values
    meta_dropseq["drop_subcluster"] = (meta_dropseq["drop_class"] + "-" + meta_dropseq["subcluster"]).astype(str)

    ro.r['library']("DropSeq.util")
    ro.r['library']("Seurat")
    ro.r['library']("SingleCellExperiment")
    ro.r.assign("dge.path", datapath_Drop + "/" + dropviz_dgename_dict[tissue])
    ro.r('''dge <- loadSparseDge(dge.path) ''')
    ro.r("sce <- CreateSeuratObject(counts = dge)")
    ro.r("sce <- as.SingleCellExperiment(sce)")

    anndata2ri.activate()
    adata_dropseq = ro.r('as(sce, "SingleCellExperiment")')
    adata_dropseq = adata_dropseq[meta_dropseq.index]


    # 10X
    tissuename_10X_dict = {"TH": "Thal", "HC": "HC", "CB": "CB"}

    adata_10X = sc.read_loom(datapath_10X + "/l5_all.loom")

    meta_10X = adata_10X.obs[["Tissue", "Class", "Clusters"]]
    meta_10X["subclass"] = meta_10X["Class"] + "_" + meta_10X["Clusters"].astype(str)

    sub_meta_10X = meta_10X[meta_10X.Tissue == tissuename_10X_dict[tissue]]
    sub_adata_10X = adata_10X[meta_10X.Tissue == tissuename_10X_dict[tissue]]

    sub_adata_10X = sub_adata_10X[~sub_adata_10X.obs.index.duplicated(), ~sub_adata_10X.var.index.duplicated()]
    sub_meta_10X = sub_meta_10X[~sub_meta_10X.index.duplicated()]

    print(np.sum(sub_adata_10X.obs.index.duplicated()), np.sum(sub_adata_10X.var.index.duplicated()))
    print(np.sum(adata_dropseq.obs.index.duplicated()), np.sum(adata_dropseq.var.index.duplicated()))

    meta_dropseq["method"] = "Drop"
    sub_meta_10X["method"] = "10X"
    meta = pd.concat([meta_dropseq, sub_meta_10X])
    meta.to_pickle(os.path.join(data_path, "meta_raw.pkl"))

    return adata_dropseq, sub_adata_10X

def read_MouseBrainAtlas_full(datapath_Drop, datapath_10X, data_path):
    # Drop-seq
    ro.r['library']("DropSeq.util")
    ro.r['library']("Seurat")
    ro.r['library']("SingleCellExperiment")

    ro.r('''load("%s")''' % (datapath_Drop + "/Drop-seq_all.RData"))
    ro.r("sce <- as.SingleCellExperiment(sce)")
    anndata2ri.activate()
    adata_dropseq = ro.r('as(sce, "SingleCellExperiment")')

    ro.r('''load("%s")''' % (datapath_Drop + "/Drop-seq_meta_all.RData"))
    ro.r("cellid <- as.vector(meta_all$cellid)")
    ro.r("subcluster <- as.vector(meta_all$subcluster)")
    ro.r("celltype <- as.vector(meta_all$class)")

    cellid = list(ro.r("cellid"))
    subcluster = list(ro.r("subcluster"))
    celltype = list(ro.r("celltype"))
    meta_dropseq = pd.DataFrame({"celltype": celltype, "subcluster": subcluster}, index=cellid)

    # 10X
    adata_10X = sc.read_loom(datapath_10X + "/l5_all.loom")

    meta_10X = adata_10X.obs[["Tissue", "Class", "Clusters"]]
    meta_10X["subclass"] = meta_10X["Class"] + "_" + meta_10X["Clusters"].astype(str)

    adata_10X = adata_10X[~adata_10X.obs.index.duplicated(), ~adata_10X.var.index.duplicated()]
    meta_10X = meta_10X[~meta_10X.index.duplicated()]

    print(np.sum(adata_10X.obs.index.duplicated()), np.sum(adata_10X.var.index.duplicated()))
    print(np.sum(adata_dropseq.obs.index.duplicated()), np.sum(adata_dropseq.var.index.duplicated()))

    print(adata_10X.shape, len(meta_10X))
    print(adata_dropseq.shape, len(meta_dropseq))

    meta_dropseq["method"] = "Drop"
    meta_10X["method"] = "10X"
    meta = pd.concat([meta_dropseq, meta_10X])

    meta.to_pickle(os.path.join(data_path, "meta_raw.pkl"))

    return adata_dropseq, adata_10X

def read_MouseBrainAtlas_1M_full(datapath_Drop, datapath_10X, data_path, subsample_num):
    # Drop-seq
    ro.r['library']("DropSeq.util")
    ro.r['library']("Seurat")
    ro.r['library']("SingleCellExperiment")

    ro.r('''load("%s")''' % (datapath_Drop + "/Drop-seq_all_withNA.RData"))
    ro.r("sce <- as.SingleCellExperiment(sce)")
    anndata2ri.activate()
    adata_dropseq = ro.r('as(sce, "SingleCellExperiment")')

    ro.r('''load("%s")''' % (datapath_Drop + "/Drop-seq_meta_all_withNA.RData"))
    ro.r("cellid <- as.vector(meta_all$cellid)")
    ro.r("subcluster <- as.vector(meta_all$subcluster)")
    ro.r("celltype <- as.vector(meta_all$class)")

    cellid = list(ro.r("cellid"))
    subcluster = list(ro.r("subcluster"))
    celltype = list(ro.r("celltype"))
    meta_dropseq = pd.DataFrame({"celltype": celltype, "subcluster": subcluster}, index=cellid)

    # 10X
    adata_10X = sc.read_loom(datapath_10X + "/l5_all.loom")

    meta_10X = adata_10X.obs[["Tissue", "Class", "Clusters"]]
    meta_10X["subclass"] = meta_10X["Class"] + "_" + meta_10X["Clusters"].astype(str)

    adata_10X = adata_10X[~adata_10X.obs.index.duplicated(), ~adata_10X.var.index.duplicated()]
    meta_10X = meta_10X[~meta_10X.index.duplicated()]

    print(np.sum(adata_dropseq.obs.index.duplicated()), np.sum(adata_dropseq.var.index.duplicated()))
    print(np.sum(adata_10X.obs.index.duplicated()), np.sum(adata_10X.var.index.duplicated()))
    
    print(adata_dropseq.shape, len(meta_dropseq))
    print(adata_10X.shape, len(meta_10X))

    total_num = adata_dropseq.shape[0] + adata_10X.shape[0]

    np.random.seed(1234)
    if subsample_num == None:
        data_path = os.path.join(data_path, "full")
        sample_idx = np.random.choice(total_num, size=total_num, replace=False)
    else:
        data_path = os.path.join(data_path, "sub-%d" % subsample_num)
        sample_idx = np.random.choice(total_num, size=subsample_num, replace=False)

    sample_idx_dropseq = sample_idx[np.where(sample_idx < adata_dropseq.shape[0])]
    sample_idx_10X = sample_idx[np.where(sample_idx >= adata_dropseq.shape[0])] - adata_dropseq.shape[0]

    adata_dropseq = adata_dropseq[sample_idx_dropseq, :].copy()
    adata_10X = adata_10X[sample_idx_10X, :].copy()

    meta_dropseq = meta_dropseq.loc[list(adata_dropseq.obs.index), ]
    meta_10X = meta_10X.loc[list(adata_10X.obs.index), ]

    print(adata_dropseq.shape, len(meta_dropseq))
    print(adata_10X.shape, len(meta_10X))

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    meta_dropseq["method"] = "Drop"
    meta_10X["method"] = "10X"
    meta = pd.concat([meta_dropseq, meta_10X])

    meta.to_pickle(os.path.join(data_path, "meta_raw.pkl"))

    return adata_dropseq, adata_10X

def read_HumanHeartAtlas(raw_path,
                         data_path, 
                         subsample_num=None,
                         dataset1="Sanger-Nuclei", dataset2="Sanger-Cells"):

    adata = sc.read_h5ad(raw_path)
    adata = adata[adata.obs["cell_source"].isin([dataset1, dataset2]), :]

    np.random.seed(1234)
    if subsample_num == None:
        data_path = os.path.join(data_path, "full")
        sample_idx = np.random.choice(adata.X.shape[0], size=adata.X.shape[0], replace=False)
    else:
        data_path = os.path.join(data_path, "sub-%d" % subsample_num)
        sample_idx = np.random.choice(adata.X.shape[0], size=subsample_num, replace=False)

    adata = adata[sample_idx, :]

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    adata1 = adata[adata.obs["cell_source"] == dataset1, :]
    adata2 = adata[adata.obs["cell_source"] == dataset2, :]
    
    meta = adata.obs[["cell_source", "cell_type"]]

    meta = meta.rename(columns={"cell_source": "method", "cell_type": "celltype"})
    meta.to_pickle(os.path.join(data_path, "meta_raw.pkl"))

    return adata1, adata2

def read_HumanHeartAtlas_full(raw_path, data_path):

    adata = sc.read_h5ad(raw_path)
    adata = adata[(adata.obs["cell_type"] != "doublets") & (adata.obs["cell_type"] != "NotAssigned"), :]

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    adata1 = adata[adata.obs["cell_source"] == "Harvard-Nuclei", :]
    adata2 = adata[adata.obs["cell_source"] == "Sanger-Nuclei", :]
    adata3 = adata[adata.obs["cell_source"] == "Sanger-Cells", :]
    adata4 = adata[adata.obs["cell_source"] == "Sanger-CD45", :]
    
    meta = adata.obs[["cell_source", "cell_type"]]

    meta = meta.rename(columns={"cell_source": "method", "cell_type": "celltype"})
    meta.to_pickle(os.path.join(data_path, "meta_raw.pkl"))

    return adata1, adata2, adata3, adata4

def read_seurat_obj(data_path,
                    datapath_1="/home/jingsi/ref-success-fail/ref.csv", 
                    datapath_2="/home/jingsi/ref-success-fail/success.csv",
                    method_1 = "reference",
                    method_2 = "success"):

    # ro.r['library']("Seurat")

    # data1
    # ro.r.assign("datapath_1", datapath_1)
    # obj_1 = ro.r['readRDS'](datapath_1)

    # ro.r("cnts_1 <- obj_1@assays$RNA@counts")
    # ro.r("cnts_1 <- as.matrix(cnts_1)")
    # counts_1 = np.array(ro.r('''as.matrix(cnts_1)'''))
    # genes_1 = list(ro.r("as.vector(rownames(cnts_1))"))
    # cells_1 = list(ro.r("as.vector(colnames(cnts_1))"))

    df_1 = pd.read_csv(datapath_1, index_col=0, encoding = 'utf8')

    adata_1 = anndata.AnnData(df_1.values.T)
    adata_1.obs.index = df_1.columns
    adata_1.var.index = df_1.index

    # data2
    # ro.r.assign("datapath_2", datapath_2)
    # obj_2 = ro.r['readRDS'](datapath_2)

    # ro.r("cnts_2 <- obj_2@assays$RNA@counts")
    # ro.r("cnts_2 <- as.matrix(cnts_2)")
    # counts_2 = np.array(ro.r('''as.matrix(cnts_2)'''))
    # genes_2 = list(ro.r("as.vector(rownames(cnts_2))"))
    # cells_2 = list(ro.r("as.vector(colnames(cnts_2))"))

    df_2 = pd.read_csv(datapath_2, index_col=0, encoding = 'utf8')

    adata_2 = anndata.AnnData(df_2.values.T)
    adata_2.obs.index = df_2.columns
    adata_2.var.index = df_2.index

    meta_1 = adata_1.obs.copy()
    meta_2 = adata_2.obs.copy()

    meta_1["method"] = method_1
    meta_2["method"] = method_2

    meta = pd.concat([meta_1, meta_2])
    meta.to_pickle(os.path.join(data_path, "meta_raw.pkl"))

    return adata_1, adata_2

def calculate_ASW(data, meta, anno_A="drop_subcluster", anno_B="subcluster"):
    if data.shape[0] >= 1e5:
        np.random.seed(1234)
        subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
        data = data[subsample_idx]
        meta = meta.iloc[subsample_idx]
    lowdim = data

    cellid = meta.index.astype(str)
    method = meta["method"].astype(str)
    cluster_A = meta[anno_A].astype(str)
    if (anno_B != anno_A):
        cluster_B = meta[anno_B].astype(str)

    rpy2.robjects.numpy2ri.activate()
    nr, nc = lowdim.shape
    lowdim = ro.r.matrix(lowdim, nrow=nr, ncol=nc)
    ro.r.assign("data", lowdim)
    rpy2.robjects.numpy2ri.deactivate()

    cellid = ro.StrVector(cellid)
    ro.r.assign("cellid", cellid)
    method = ro.StrVector(method)
    ro.r.assign("method", method)
    cluster_A = ro.StrVector(cluster_A)
    ro.r.assign("cluster_A", cluster_A)
    if (anno_B != anno_A):
        cluster_B = ro.StrVector(cluster_B)
        ro.r.assign("cluster_B", cluster_B)

    ro.r("set.seed(1234)")
    ro.r['library']("cluster")

    if (anno_B != anno_A):
        method_set = pd.unique(meta["method"])
        method_A = method_set[0]
        ro.r.assign("method_A", method_A)
        method_B = method_set[1]
        ro.r.assign("method_B", method_B)
        ro.r('''indx_A <- which(method == method_A)''')
        ro.r('''indx_B <- which(method == method_B)''')
        ro.r('''ASW_A <- summary(silhouette(as.numeric(as.factor(cluster_A[indx_A])), dist(data[indx_A, 1:20])))[["avg.width"]]''')
        ro.r('''ASW_B <- summary(silhouette(as.numeric(as.factor(cluster_B[indx_B])), dist(data[indx_B, 1:20])))[["avg.width"]]''')
        ASW_A = np.array(ro.r("ASW_A"))[0]
        ASW_B = np.array(ro.r("ASW_B"))[0]

        return ASW_A, ASW_B
    else:
        ro.r('''ASW_A <- summary(silhouette(as.numeric(as.factor(cluster_A)), dist(data[, 1:20])))[["avg.width"]]''')
        ASW_A = np.array(ro.r("ASW_A"))[0]

        return ASW_A