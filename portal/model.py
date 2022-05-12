import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import tables
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA, IncrementalPCA

from portal.networks import *

class Model(object):
    def __init__(self, batch_size=500, training_steps=2000, seed=1234, npcs=30, n_latent=20, lambdacos=20.0,
                 model_path="models", data_path="data", result_path="results"):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.npcs = npcs
        self.n_latent = n_latent
        self.lambdacos = lambdacos
        self.lambdaAE = 10.0
        self.lambdaLA = 10.0
        self.lambdaGAN = 1.0
        self.margin = 5.0
        self.model_path = model_path
        self.data_path = data_path
        self.result_path = result_path


    def preprocess(self, 
                   adata_A_input, 
                   adata_B_input, 
                   hvg_num=4000, # number of highly variable genes for each anndata
                   save_embedding=False # save low-dimensional embeddings or not
                   ):
        '''
        Performing preprocess for a pair of datasets.
        To integrate multiple datasets, use function preprocess_multiple_anndata in utils.py
        '''
        adata_A = adata_A_input.copy()
        adata_B = adata_B_input.copy()

        print("Finding highly variable genes...")
        sc.pp.highly_variable_genes(adata_A, flavor='seurat_v3', n_top_genes=hvg_num)
        sc.pp.highly_variable_genes(adata_B, flavor='seurat_v3', n_top_genes=hvg_num)
        hvg_A = adata_A.var[adata_A.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_B = adata_B.var[adata_B.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_total = hvg_A & hvg_B
        if len(hvg_total) < 100:
            raise ValueError("The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(hvg_total))

        print("Normalizing and scaling...")
        sc.pp.normalize_total(adata_A, target_sum=1e4)
        sc.pp.log1p(adata_A)
        adata_A = adata_A[:, hvg_total]
        sc.pp.scale(adata_A, max_value=10)

        sc.pp.normalize_total(adata_B, target_sum=1e4)
        sc.pp.log1p(adata_B)
        adata_B = adata_B[:, hvg_total]
        sc.pp.scale(adata_B, max_value=10)

        adata_total = adata_A.concatenate(adata_B, index_unique=None)

        print("Dimensionality reduction via PCA...")
        pca = PCA(n_components=self.npcs, svd_solver="arpack", random_state=0)
        adata_total.obsm["X_pca"] = pca.fit_transform(adata_total.X)

        self.emb_A = adata_total.obsm["X_pca"][:adata_A.shape[0], :self.npcs].copy()
        self.emb_B = adata_total.obsm["X_pca"][adata_A.shape[0]:, :self.npcs].copy()

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if save_embedding:
            np.save(os.path.join(self.data_path, "lowdim_A.npy"), self.emb_A)
            np.save(os.path.join(self.data_path, "lowdim_B.npy"), self.emb_B)


    def preprocess_memory_efficient(self, 
                                    adata_A_path, 
                                    adata_B_path, 
                                    hvg_num=4000, 
                                    chunk_size=20000,
                                    save_embedding=True
                                    ):
        '''
        Performing preprocess for a pair of datasets with efficient memory usage.
        To improve time efficiency, use a larger chunk_size.
        '''
        adata_A_input = sc.read_h5ad(adata_A_path, backed="r+", chunk_size=chunk_size)
        adata_B_input = sc.read_h5ad(adata_B_path, backed="r+", chunk_size=chunk_size)

        print("Finding highly variable genes...")
        subsample_idx_A = np.random.choice(adata_A_input.shape[0], size=np.minimum(adata_A_input.shape[0], chunk_size), replace=False)
        subsample_idx_B = np.random.choice(adata_B_input.shape[0], size=np.minimum(adata_B_input.shape[0], chunk_size), replace=False)

        adata_A_subsample = adata_A_input[subsample_idx_A].to_memory().copy()
        adata_B_subsample = adata_B_input[subsample_idx_B].to_memory().copy()

        sc.pp.highly_variable_genes(adata_A_subsample, flavor='seurat_v3', n_top_genes=hvg_num)
        sc.pp.highly_variable_genes(adata_B_subsample, flavor='seurat_v3', n_top_genes=hvg_num)

        hvg_A = adata_A_subsample.var[adata_A_subsample.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_B = adata_B_subsample.var[adata_B_subsample.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg = hvg_A & hvg_B

        del adata_A_subsample, adata_B_subsample, subsample_idx_A, subsample_idx_B

        print("Normalizing and scaling...")
        adata_A = adata_A_input.copy(adata_A_path)
        adata_B = adata_B_input.copy(adata_B_path)

        adata_A_hvg_idx = adata_A.var.index.get_indexer(hvg)
        adata_B_hvg_idx = adata_B.var.index.get_indexer(hvg)

        mean_A = np.zeros((1, len(hvg)))
        sq_A = np.zeros((1, len(hvg)))
        mean_B = np.zeros((1, len(hvg)))
        sq_B = np.zeros((1, len(hvg)))

        for i in range(adata_A.shape[0] // chunk_size):
            X_norm = sc.pp.normalize_total(adata_A[i * chunk_size: (i + 1) * chunk_size].to_memory(), target_sum=1e4, inplace=False)["X"]
            X_norm = X_norm[:, adata_A_hvg_idx]
            X_norm = sc.pp.log1p(X_norm)
            mean_A = mean_A + X_norm.sum(axis=0) / adata_A.shape[0]
            sq_A = sq_A + X_norm.power(2).sum(axis=0) / adata_A.shape[0]

        if (adata_A.shape[0] % chunk_size) > 0:
            X_norm = sc.pp.normalize_total(adata_A[(adata_A.shape[0] // chunk_size) * chunk_size: adata_A.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
            X_norm = X_norm[:, adata_A_hvg_idx]
            X_norm = sc.pp.log1p(X_norm)
            mean_A = mean_A + X_norm.sum(axis=0) / adata_A.shape[0]
            sq_A = sq_A + X_norm.power(2).sum(axis=0) / adata_A.shape[0]

        std_A = np.sqrt(sq_A - np.square(mean_A))

        for i in range(adata_B.shape[0] // chunk_size):
            X_norm = sc.pp.normalize_total(adata_B[i * chunk_size: (i + 1) * chunk_size].to_memory(), target_sum=1e4, inplace=False)["X"]
            X_norm = X_norm[:, adata_B_hvg_idx]
            X_norm = sc.pp.log1p(X_norm)
            mean_B = mean_B + X_norm.sum(axis=0) / adata_B.shape[0]
            sq_B = sq_B + X_norm.power(2).sum(axis=0) / adata_B.shape[0]

        if (adata_B.shape[0] % chunk_size) > 0:
            X_norm = sc.pp.normalize_total(adata_B[(adata_B.shape[0] // chunk_size) * chunk_size: adata_B.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
            X_norm = X_norm[:, adata_B_hvg_idx]
            X_norm = sc.pp.log1p(X_norm)
            mean_B = mean_B + X_norm.sum(axis=0) / adata_B.shape[0]
            sq_B = sq_B + X_norm.power(2).sum(axis=0) / adata_B.shape[0]

        std_B = np.sqrt(sq_B - np.square(mean_B))

        del X_norm, sq_A, sq_B 

        print("Dimensionality reduction via Incremental PCA...")
        ipca = IncrementalPCA(n_components=self.npcs, batch_size=chunk_size)
        total_ncells = adata_A.shape[0] + adata_B.shape[0]
        order = np.arange(total_ncells)
        np.random.RandomState(1234).shuffle(order)

        for i in range(total_ncells // chunk_size):
            idx = order[i * chunk_size : (i + 1) * chunk_size]
            idx_is_A = (idx < adata_A.shape[0])
            data_A = sc.pp.normalize_total(adata_A[idx[idx_is_A]].to_memory(), target_sum=1e4, inplace=False)["X"]
            data_A = data_A[:, adata_A_hvg_idx]
            data_A = sc.pp.log1p(data_A)
            data_A = np.clip((data_A - mean_A) / std_A, -10, 10)
            idx_is_B = (idx >= adata_A.shape[0])
            data_B = sc.pp.normalize_total(adata_B[idx[idx_is_B] - adata_A.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
            data_B = data_B[:, adata_B_hvg_idx]
            data_B = sc.pp.log1p(data_B)
            data_B = np.clip((data_B - mean_B) / std_B, -10, 10)
            data = np.concatenate((data_A, data_B), axis=0)
            ipca.partial_fit(data)

        if (total_ncells % chunk_size) > 0:
            idx = order[(total_ncells // chunk_size) * chunk_size: total_ncells]
            idx_is_A = (idx < adata_A.shape[0])
            data_A = sc.pp.normalize_total(adata_A[idx[idx_is_A]].to_memory(), target_sum=1e4, inplace=False)["X"]
            data_A = data_A[:, adata_A_hvg_idx]
            data_A = sc.pp.log1p(data_A)
            data_A = np.clip((data_A - mean_A) / std_A, -10, 10)
            idx_is_B = (idx >= adata_A.shape[0])
            data_B = sc.pp.normalize_total(adata_B[idx[idx_is_B] - adata_A.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
            data_B = data_B[:, adata_B_hvg_idx]
            data_B = sc.pp.log1p(data_B)
            data_B = np.clip((data_B - mean_B) / std_B, -10, 10)
            data = np.concatenate((data_A, data_B), axis=0)
            ipca.partial_fit(data)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if save_embedding:
            h5filename_A = os.path.join(self.data_path, "lowdim_A.h5")
            f = tables.open_file(h5filename_A, mode='w')
            atom = tables.Float64Atom()
            f.create_earray(f.root, 'data', atom, (0, self.npcs))
            f.close()
            # transform
            f = tables.open_file(h5filename_A, mode='a')
            for i in range(adata_A.shape[0] // chunk_size):
                data_A = sc.pp.normalize_total(adata_A[i * chunk_size: (i + 1) * chunk_size].to_memory(), target_sum=1e4, inplace=False)["X"]
                data_A = data_A[:, adata_A_hvg_idx]
                data_A = sc.pp.log1p(data_A)
                data_A = np.clip((data_A - mean_A) / std_A, -10, 10)
                data_A = ipca.transform(data_A)
                f.root.data.append(data_A)
            if (adata_A.shape[0] % chunk_size) > 0:
                data_A = sc.pp.normalize_total(adata_A[(adata_A.shape[0] // chunk_size) * chunk_size: adata_A.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
                data_A = data_A[:, adata_A_hvg_idx]
                data_A = sc.pp.log1p(data_A)
                data_A = np.clip((data_A - mean_A) / std_A, -10, 10)
                data_A = ipca.transform(data_A)
                f.root.data.append(data_A)
                f.close()
            del data_A

            h5filename_B = os.path.join(self.data_path, "lowdim_B.h5")
            f = tables.open_file(h5filename_B, mode='w')
            atom = tables.Float64Atom()
            f.create_earray(f.root, 'data', atom, (0, self.npcs))
            f.close()
            # transform
            f = tables.open_file(h5filename_B, mode='a')
            for i in range(adata_B.shape[0] // chunk_size):
                data_B = sc.pp.normalize_total(adata_B[i * chunk_size: (i + 1) * chunk_size].to_memory(), target_sum=1e4, inplace=False)["X"]
                data_B = data_B[:, adata_B_hvg_idx]
                data_B = sc.pp.log1p(data_B)
                data_B = np.clip((data_B - mean_B) / std_B, -10, 10)
                data_B = ipca.transform(data_B)
                f.root.data.append(data_B)
            if (adata_B.shape[0] % chunk_size) > 0:
                data_B = sc.pp.normalize_total(adata_B[(adata_B.shape[0] // chunk_size) * chunk_size: adata_B.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
                data_B = data_B[:, adata_B_hvg_idx]
                data_B = sc.pp.log1p(data_B)
                data_B = np.clip((data_B - mean_B) / std_B, -10, 10)
                data_B = ipca.transform(data_B)
                f.root.data.append(data_B)
                f.close()
            del data_B


    def train(self):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.D_A = discriminator(self.npcs).to(self.device)
        self.D_B = discriminator(self.npcs).to(self.device)
        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.)
        params_D = list(self.D_A.parameters()) + list(self.D_B.parameters())
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.)
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]

        for step in range(self.training_steps):
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)

            # discriminator loss:
            optimizer_D.zero_grad()
            if step <= 5:
                # Warm-up
                loss_D_A = (torch.log(1 + torch.exp(-self.D_A(x_A))) + torch.log(1 + torch.exp(self.D_A(x_BtoA)))).mean()
                loss_D_B = (torch.log(1 + torch.exp(-self.D_B(x_B))) + torch.log(1 + torch.exp(self.D_B(x_AtoB)))).mean()
            else:
                loss_D_A = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_A), -self.margin, self.margin))) + torch.log(1 + torch.exp(torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean()
                loss_D_B = (torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_B), -self.margin, self.margin))) + torch.log(1 + torch.exp(torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
            loss_D = loss_D_A + loss_D_B
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # autoencoder loss:
            loss_AE_A = torch.mean((x_Arecon - x_A)**2)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2)
            loss_AE = loss_AE_A + loss_AE_B

            # cosine correspondence:
            loss_cos_A = (1 - torch.sum(F.normalize(x_AtoB, p=2) * F.normalize(x_A, p=2), 1)).mean()
            loss_cos_B = (1 - torch.sum(F.normalize(x_BtoA, p=2) * F.normalize(x_B, p=2), 1)).mean()
            loss_cos = loss_cos_A + loss_cos_B

            # latent align loss:
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_LA = loss_LA_AtoB + loss_LA_BtoA

            # generator loss
            optimizer_G.zero_grad()
            if step <= 5:
                # Warm-up
                loss_G_GAN = (torch.log(1 + torch.exp(-self.D_A(x_BtoA))) + torch.log(1 + torch.exp(-self.D_B(x_AtoB)))).mean()
            else:
                loss_G_GAN = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin))) + torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdacos * loss_cos + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA
            loss_G.backward()
            optimizer_G.step()

            if not step % 200:
                print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_cos=%f, loss_LA=%f"
                 % (step, loss_D, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdacos*loss_cos, self.lambdaLA*loss_LA))

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'D_A': self.D_A.state_dict(), 'D_B': self.D_B.state_dict(), 
                 'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict()}

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))

    def train_memory_efficient(self):
        import tables
        f_A = tables.open_file(os.path.join(self.data_path, "lowdim_A.h5"))
        f_B = tables.open_file(os.path.join(self.data_path, "lowdim_B.h5"))

        self.emb_A = np.array(f_A.root.data)
        self.emb_B = np.array(f_B.root.data)

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]
        
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.D_A = discriminator(self.npcs).to(self.device)
        self.D_B = discriminator(self.npcs).to(self.device)
        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.)
        params_D = list(self.D_A.parameters()) + list(self.D_B.parameters())
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.)
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()

        for step in range(self.training_steps):
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size, replace=False)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size, replace=False)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)

            # discriminator loss:
            optimizer_D.zero_grad()
            if step <= 5:
                # Warm-up
                loss_D_A = (torch.log(1 + torch.exp(-self.D_A(x_A))) + torch.log(1 + torch.exp(self.D_A(x_BtoA)))).mean()
                loss_D_B = (torch.log(1 + torch.exp(-self.D_B(x_B))) + torch.log(1 + torch.exp(self.D_B(x_AtoB)))).mean()
            else:
                loss_D_A = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_A), -self.margin, self.margin))) + torch.log(1 + torch.exp(torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean()
                loss_D_B = (torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_B), -self.margin, self.margin))) + torch.log(1 + torch.exp(torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
            loss_D = loss_D_A + loss_D_B
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # autoencoder loss:
            loss_AE_A = torch.mean((x_Arecon - x_A)**2)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2)
            loss_AE = loss_AE_A + loss_AE_B

            # cosine correspondence:
            loss_cos_A = (1 - torch.sum(F.normalize(x_AtoB, p=2) * F.normalize(x_A, p=2), 1)).mean()
            loss_cos_B = (1 - torch.sum(F.normalize(x_BtoA, p=2) * F.normalize(x_B, p=2), 1)).mean()
            loss_cos = loss_cos_A + loss_cos_B

            # latent align loss:
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_LA = loss_LA_AtoB + loss_LA_BtoA

            # generator loss
            optimizer_G.zero_grad()
            if step <= 5:
                # Warm-up
                loss_G_GAN = (torch.log(1 + torch.exp(-self.D_A(x_BtoA))) + torch.log(1 + torch.exp(-self.D_B(x_AtoB)))).mean()
            else:
                loss_G_GAN = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin))) + torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdacos * loss_cos + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA
            loss_G.backward()
            optimizer_G.step()

            if not step % 200:
                print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_cos=%f, loss_LA=%f"
                 % (step, loss_D, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdacos*loss_cos, self.lambdaLA*loss_LA))

        f_A.close()
        f_B.close()

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'D_A': self.D_A.state_dict(), 'D_B': self.D_B.state_dict(), 
                 'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict()}

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))


    def eval(self, D_score=False, save_results=False):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.E_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_A'])
        self.E_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_B'])
        self.G_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_A'])
        self.G_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_B'])

        x_A = torch.from_numpy(self.emb_A).float().to(self.device)
        x_B = torch.from_numpy(self.emb_B).float().to(self.device)

        z_A = self.E_A(x_A)
        z_B = self.E_B(x_B)

        x_AtoB = self.G_B(z_A)
        x_BtoA = self.G_A(z_B)

        end_time = time.time()
        
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)
        self.data_Aspace = np.concatenate((self.emb_A, x_BtoA.detach().cpu().numpy()), axis=0)
        self.data_Bspace = np.concatenate((x_AtoB.detach().cpu().numpy(), self.emb_B), axis=0)

        if D_score:
            self.D_A = discriminator(self.npcs).to(self.device)
            self.D_B = discriminator(self.npcs).to(self.device)
            self.D_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['D_A'])
            self.D_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['D_B'])

            score_D_A_A = self.D_A(x_A)
            score_D_B_A = self.D_B(x_AtoB)
            score_D_B_B = self.D_B(x_B)
            score_D_A_B = self.D_A(x_BtoA)

            self.score_Aspace = np.concatenate((score_D_A_A.detach().cpu().numpy(), score_D_A_B.detach().cpu().numpy()), axis=0)
            self.score_Bspace = np.concatenate((score_D_B_A.detach().cpu().numpy(), score_D_B_B.detach().cpu().numpy()), axis=0)

        if save_results:
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)

            np.save(os.path.join(self.result_path, "latent_A.npy"), z_A.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "latent_B.npy"), z_B.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "x_AtoB.npy"), x_AtoB.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "x_BtoA.npy"), x_BtoA.detach().cpu().numpy())
            if D_score:
                np.save(os.path.join(self.result_path, "score_Aspace_A.npy"), score_D_A_A.detach().cpu().numpy())
                np.save(os.path.join(self.result_path, "score_Bspace_A.npy"), score_D_B_A.detach().cpu().numpy())
                np.save(os.path.join(self.result_path, "score_Bspace_B.npy"), score_D_B_B.detach().cpu().numpy())
                np.save(os.path.join(self.result_path, "score_Aspace_B.npy"), score_D_A_B.detach().cpu().numpy())


    def eval_memory_efficient(self):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_A'])
        self.E_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_B'])

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        f_A = tables.open_file(os.path.join(self.data_path, "lowdim_A.h5"))
        f_B = tables.open_file(os.path.join(self.data_path, "lowdim_B.h5"))

        N_A = f_A.root.data.shape[0]
        N_B = f_B.root.data.shape[0]

        h5_latent_A = os.path.join(self.result_path, "latent_A.h5")
        f_latent_A = tables.open_file(h5_latent_A, mode='w')
        atom = tables.Float64Atom()
        f_latent_A.create_earray(f_latent_A.root, 'data', atom, (0, self.n_latent))
        f_latent_A.close()

        f_latent_A = tables.open_file(h5_latent_A, mode='a')
        # f_x_AtoB = tables.open_file(h5_x_AtoB, mode='a')
        for i in range(N_A // self.batch_size):
            x_A = torch.from_numpy(f_A.root.data[i * self.batch_size: (i + 1) * self.batch_size]).float().to(self.device)
            z_A = self.E_A(x_A)
            f_latent_A.root.data.append(z_A.detach().cpu().numpy())
        if (N_A % self.batch_size) > 0:
            x_A = torch.from_numpy(f_A.root.data[(N_A // self.batch_size) * self.batch_size: N_A]).float().to(self.device)
            z_A = self.E_A(x_A)
            f_latent_A.root.data.append(z_A.detach().cpu().numpy())
            f_latent_A.close()

        h5_latent_B = os.path.join(self.result_path, "latent_B.h5")
        f_latent_B = tables.open_file(h5_latent_B, mode='w')
        atom = tables.Float64Atom()
        f_latent_B.create_earray(f_latent_B.root, 'data', atom, (0, self.n_latent))
        f_latent_B.close()

        f_latent_B = tables.open_file(h5_latent_B, mode='a')
        for i in range(N_B // self.batch_size):
            x_B = torch.from_numpy(f_B.root.data[i * self.batch_size: (i + 1) * self.batch_size]).float().to(self.device)
            z_B = self.E_B(x_B)
            f_latent_B.root.data.append(z_B.detach().cpu().numpy())
        if (N_B % self.batch_size) > 0:
            x_B = torch.from_numpy(f_B.root.data[(N_B // self.batch_size) * self.batch_size: N_B]).float().to(self.device)
            z_B = self.E_B(x_B)
            f_latent_B.root.data.append(z_B.detach().cpu().numpy())
            f_latent_B.close()

        end_time = time.time()

        f_A.close()
        f_B.close()
        
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)
