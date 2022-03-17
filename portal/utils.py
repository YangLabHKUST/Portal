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
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.linear_model import LinearRegression
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

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
                       lambda_cos=20.0,
                       training_steps=2000,
                       space=None, # None or "reference" or "latent"
                       data_path="data",
                       mixingmetric_subsample=True
                       ):

    if space == None:
        if len(lowdim_list) == 2:
            space = "latent"
        else:
            space = "reference"

    print("Incrementally integrating %d datasets..." % len(lowdim_list))

    if not search_cos:
        # if not search hyperparameter lambdacos
        if isinstance(lambda_cos, float) or isinstance(lambda_cos, int):
            lambda_cos_tmp = lambda_cos

        for i in range(len(lowdim_list) - 1):

            if isinstance(lambda_cos, list):
                lambda_cos_tmp = lambda_cos[i]

            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            model = Model(lambdacos=lambda_cos_tmp,
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
        for i in range(len(lowdim_list) - 1):
            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            for lambda_cos in [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
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
                mixing = calculate_mixing_metric(model.latent, meta, k=5, max_k=300, methods=list(set(meta.method)), subsample=mixingmetric_subsample)
                print("lambda_cos: %f, mixing metric: %f \n" % (lambda_cos, mixing))
                if lambda_cos == 15.0:
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
        if data.shape[0] >= 1e4:
            np.random.seed(1234)
            subsample_idx = np.random.choice(data.shape[0], 10000, replace=False)
            data = data[subsample_idx]
            meta = meta.iloc[subsample_idx]
            meta.index = np.arange(len(subsample_idx))
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
      
      
def calculate_cellcycleconservation(data, meta, adata_raw, organism="mouse", resources_path="./cell_cycle_resources"):
    #adata
    cellid = list(meta.index.astype(str))
    geneid = ["gene_"+str(i) for i in range(data.shape[1])]
    adata = anndata.AnnData(X=data, obs=cellid, var=geneid)

    #score cell cycle
    cc_files = {'mouse': [os.path.join(resources_path, 's_genes_tirosh.txt'),
                          os.path.join(resources_path, 'g2m_genes_tirosh.txt')]}
    with open(cc_files[organism][0], "r") as f:
        s_genes = [x.strip() for x in f.readlines() if x.strip() in adata_raw.var.index]
    with open(cc_files[organism][1], "r") as f:
        g2m_genes = [x.strip() for x in f.readlines() if x.strip() in adata_raw.var.index]
    sc.tl.score_genes_cell_cycle(adata_raw, s_genes, g2m_genes)

    adata_raw.obs["method"] = meta["method"].values.astype(str)
    adata.obs["method"] = meta["method"].values.astype(str)
    batches = adata_raw.obs["method"].unique()
    
    scores_final = []
    scores_before = []
    scores_after = []
    for batch in batches:
        raw_sub = adata_raw[adata_raw.obs["method"] == batch]
        int_sub = adata[adata.obs["method"] == batch].copy()
        int_sub = int_sub.X

        #regression variable
        covariate_values = raw_sub.obs[['S_score', 'G2M_score']]
        if pd.api.types.is_numeric_dtype(covariate_values):
            covariate_values = np.array(covariate_values).reshape(-1, 1)
        else:
            covariate_values = pd.get_dummies(covariate_values)

        #PCR on data before integration
        n_comps = 50
        svd_solver = 'arpack'
        pca = sc.tl.pca(raw_sub.X, n_comps=n_comps, use_highly_variable=False, return_info=True, svd_solver=svd_solver, copy=True)
        X_pca = pca[0].copy()
        pca_var = pca[3].copy()
        del pca

        r2 = []
        for i in range(n_comps):
            pc = X_pca[:, [i]]
            lm = LinearRegression()
            lm.fit(covariate_values, pc)
            r2_score = np.maximum(0, lm.score(covariate_values, pc))
            r2.append(r2_score)

        Var = pca_var / sum(pca_var) * 100
        before = sum(r2 * Var) / 100

        #PCR on data after integration
        n_comps = min(data.shape)
        svd_solver = 'full'
        pca = sc.tl.pca(int_sub, n_comps=n_comps, use_highly_variable=False, return_info=True, svd_solver=svd_solver, copy=True)
        X_pca = pca[0].copy()
        pca_var = pca[3].copy()
        del pca

        r2 = []
        for i in range(n_comps):
            pc = X_pca[:, [i]]
            lm = LinearRegression()
            lm.fit(covariate_values, pc)
            r2_score = np.maximum(0, lm.score(covariate_values, pc))
            r2.append(r2_score)

        Var = pca_var / sum(pca_var) * 100
        after = sum(r2 * Var) / 100

        #scale result
        score = 1 - abs(before - after) / before
        if score < 0:
            score = 0
        scores_before.append(before)
        scores_after.append(after)
        scores_final.append(score)

    score_out = np.mean(scores_final) 
    return score_out


def calculate_isolatedASW(data, meta, anno):
    tmp = meta[[anno, "method"]].drop_duplicates()
    batch_per_lab = tmp.groupby(anno).agg({"method": "count"})
    iso_threshold = batch_per_lab.min().tolist()[0]
    labels = batch_per_lab[batch_per_lab["method"] <= iso_threshold].index.tolist()

    scores = {}
    for label_tar in labels:
        iso_label = np.array(meta[anno] == label_tar).astype(int)
        asw = silhouette_score(
            X=data,
            labels=iso_label,
            metric='euclidean'
        )
        asw = (asw + 1) / 2
        scores[label_tar] = asw

    scores = pd.Series(scores)
    score = scores.mean()
    return score


def calculate_isolatedF1(data, meta, anno):
    if data.shape[0] > 1e5:
        np.random.seed(1234)
        subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
        data = data[subsample_idx]
        meta = meta.iloc[subsample_idx]
    lowdim = data

    tmp = meta[[anno, "method"]].drop_duplicates()
    batch_per_lab = tmp.groupby(anno).agg({"method": "count"})
    iso_threshold = batch_per_lab.min().tolist()[0]
    labels = batch_per_lab[batch_per_lab["method"] <= iso_threshold].index.tolist()

    cellid = meta.index.astype(str)
    method = meta["method"].astype(str)
    cluster_A = meta[anno].astype(str)

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

    ro.r("set.seed(1234)")
    ro.r['library']("Seurat")

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

    ro.r('''comb <- FindNeighbors(comb, reduction = "scunit", dims = 1:ncol(data), force.recalc = TRUE, verbose = FALSE)''')
    ro.r('''comb <- FindClusters(comb, verbose = FALSE)''')

    louvain_clusters = np.array(ro.r("comb$seurat_clusters")).astype("str")
    louvain_list = list(set(louvain_clusters))

    scores = {}
    for label_tar in labels:
        max_f1 = 0
        for cluster in louvain_list:
            y_pred = louvain_clusters == cluster
            y_true = meta[anno].values.astype(str) == label_tar
            f1 = f1_score(y_pred, y_true)
            if f1 > max_f1:
                max_f1 = f1
        scores[label_tar] = max_f1

    scores = pd.Series(scores)
    score = scores.mean()
    return score


def calculate_graphconnectivity(data, meta, anno):
    cellid = list(meta.index.astype(str))
    geneid = ["gene_"+str(i) for i in range(data.shape[1])]
    adata = anndata.AnnData(X=data, obs=cellid, var=geneid)

    adata.obsm["X_emb"] = data
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_emb")

    adata.obs["anno"] = meta[anno].values.astype(str)
    anno_list = list(set(adata.obs["anno"]))

    clust_res = []

    for label in anno_list:
        adata_sub = adata[adata.obs["anno"].isin([label])]
        _, labels = connected_components(
            adata_sub.obsp['connectivities'],
            connection='strong'
        )
        tab = pd.value_counts(labels)
        clust_res.append(tab.max() / sum(tab))

    score = np.mean(clust_res)
    return score


def calculate_PCRbatch(data, meta, data_before=None):
    covariate_values = meta["method"]

    n_comps = min(data.shape)
    svd_solver = 'full'
    pca = sc.tl.pca(data, n_comps=n_comps, use_highly_variable=False, return_info=True, svd_solver=svd_solver, copy=True)
    X_pca = pca[0].copy()
    pca_var = pca[3].copy()
    del pca

    if pd.api.types.is_numeric_dtype(covariate_values):
        covariate_values = np.array(covariate_values).reshape(-1, 1)
    else:
        covariate_values = pd.get_dummies(covariate_values)

    r2 = []
    for i in range(n_comps):
        pc = X_pca[:, [i]]
        lm = LinearRegression()
        lm.fit(covariate_values, pc)
        r2_score = np.maximum(0, lm.score(covariate_values, pc))
        r2.append(r2_score)

    Var = pca_var / sum(pca_var) * 100
    R2Var = sum(r2 * Var) / 100

    if data_before is not None:
        n_comps = 50
        svd_solver = 'arpack'
        pca = sc.tl.pca(data_before, n_comps=n_comps, use_highly_variable=False, return_info=True, svd_solver=svd_solver, copy=True)
        X_pca = pca[0].copy()
        pca_var = pca[3].copy()
        del pca

        r2 = []
        for i in range(n_comps):
            pc = X_pca[:, [i]]
            lm = LinearRegression()
            lm.fit(covariate_values, pc)
            r2_score = np.maximum(0, lm.score(covariate_values, pc))
            r2.append(r2_score)

        Var = pca_var / sum(pca_var) * 100
        R2Var_before = sum(r2 * Var) / 100

        score = (R2Var_before - R2Var) / R2Var_before
        return score, R2Var, R2Var_before
    else:
        return R2Var
      

def calculate_NMI(data, meta, anno_A="drop_subcluster", anno_B="subcluster"):
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

    np.random.seed(1234)
    if (anno_B != anno_A):
        method_set = pd.unique(meta["method"])
        method_A = method_set[0]
        ro.r.assign("method_A", method_A)
        method_B = method_set[1]
        ro.r.assign("method_B", method_B)
        ro.r('''indx_A <- which(comb$method == method_A)''')
        ro.r('''indx_B <- which(comb$method == method_B)''')

        #A
        louvain_A = np.array(ro.r("comb$seurat_clusters[indx_A]")).astype("str")
        cluster_A = np.array(ro.r("comb$cluster_A[indx_A]")).astype("str")
        df_A = pd.DataFrame({'louvain_A': louvain_A, 'cluster_A': cluster_A})
        df_A.louvain_A = pd.Categorical(df_A.louvain_A)
        df_A.cluster_A = pd.Categorical(df_A.cluster_A)
        df_A['louvain_code'] = df_A.louvain_A.cat.codes
        df_A['A_code'] = df_A.cluster_A.cat.codes
        NMI_A = NMI(df_A['A_code'].values, df_A['louvain_code'].values)

        #B
        louvain_B = np.array(ro.r("comb$seurat_clusters[indx_B]")).astype("str")
        cluster_B = np.array(ro.r("comb$cluster_B[indx_B]")).astype("str")
        df_B = pd.DataFrame({'louvain_B': louvain_B, 'cluster_B': cluster_B})
        df_B.louvain_B = pd.Categorical(df_B.louvain_B)
        df_B.cluster_B = pd.Categorical(df_B.cluster_B)
        df_B['louvain_code'] = df_B.louvain_B.cat.codes
        df_B['B_code'] = df_B.cluster_B.cat.codes
        NMI_B = NMI(df_B['B_code'].values, df_B['louvain_code'].values)

        return NMI_A, NMI_B
    else:
        louvain_clusters = np.array(ro.r("comb$seurat_clusters")).astype("str")
        cluster_A = np.array(ro.r("comb$cluster_A")).astype("str")

        df_fornmi = pd.DataFrame({'louvain_clusters': louvain_clusters, 
                                  'cluster_A': cluster_A})
        df_fornmi.louvain_clusters = pd.Categorical(df_fornmi.louvain_clusters)
        df_fornmi.cluster_A = pd.Categorical(df_fornmi.cluster_A)
        df_fornmi['louvain_code'] = df_fornmi.louvain_clusters.cat.codes
        df_fornmi['A_code'] = df_fornmi.cluster_A.cat.codes

        NMI_A = NMI(df_fornmi['A_code'].values, df_fornmi['louvain_code'].values)
        return NMI_A

     
def annotate_by_nn(vec_tar, vec_ref, label_ref, k=20, metric='cosine'):
    dist_mtx = cdist(vec_tar, vec_ref, metric=metric)
    idx = dist_mtx.argsort()[:, :k]
    labels = [max(list(label_ref[i]), key=list(label_ref[i]).count) for i in idx]
    return labels

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

