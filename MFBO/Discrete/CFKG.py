import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from FidelityFusion_Models import *
from FidelityFusion_Models.GP_DMF import *

class discrete_fidelity_knowledgement_gradient(torch.nn.Module):
    def __init__(self, fidelity_num, GP_model_list, cost, data_model, data_manager, model_name, xdim, search_range, seed):
        super(discrete_fidelity_knowledgement_gradient, self).__init__()

        self.GP_model_pre = GP_model_list[0]
        self.data_model = data_model
        self.GP_model_new = GP_model_list[1]
        self.cost = cost
        self.seed = seed
        self.search_range = search_range
        self.data_manager = data_manager
        self.model_name = model_name
        self.x_dim = xdim
        self.fidelity_num = fidelity_num

    def negative_cfkg(self, x, s):

        xall = torch.rand(100, self.x_dim,dtype=torch.float64) * (self.search_range[1] - self.search_range[0]) + self.search_range[0]
        # mean_y, sigma_y = self.GP_model_pre(xall, self.total_fid_num)  # 预测最高精度
        # xall = self.data_manager.normalizelayer[self.GP_model_pre.fidelity_num-1].normalize_x(xall)
        with torch.no_grad():
            if self.model_name in ['CMF_CAR','GP','CMF_CAR_dkl']:
                mu_pre, var_pre = self.GP_model_pre(self.data_manager, xall, torch.ones(100).reshape(-1,1), normal = False)
            else:
                mu_pre, var_pre = self.GP_model_pre(self.data_manager, xall, normal = False)
        # mu_pre, var_pre = self.data_manager.normalizelayer[self.GP_model_pre.fidelity_num-1].denormalize(mu_pre, var_pre)
        min_pre = torch.min(mu_pre)
        
        x = x.reshape(1,-1)
        with torch.no_grad():
            x = x.double()
            ypred, _ = self.GP_model_pre(self.data_manager, x, s, normal = False)
        
        if self.model_name in ['CMF_CAR','GP','CMF_CAR_dkl']:
            x = torch.cat((x, torch.tensor([[s]], dtype=torch.float64)),dim = 1)
            self.data_manager.add_data(raw_fidelity_name = '0',fidelity_index= 0 , x=x, y=ypred)
        else:
            self.data_manager.add_data(raw_fidelity_name=str(s), fidelity_index=s, x = x, y=ypred)
        ## can it be imporved?
        print('train new GP model')
        if self.model_name == 'ResGP':
            train_ResGP(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        elif self.model_name == 'AR':
            train_AR(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        # elif self.model_name == 'CAR':
        #     train_CAR(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        elif self.model_name == 'DMF_CAR':
            train_DMFCAR(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        elif self.model_name == 'DMF_CAR_dkl':
            train_DMFCAR_dkl(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        
        elif self.model_name == 'CMF_CAR':
            train_CAR_large(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        elif self.model_name == 'CMF_CAR_dkl':
            train_CMFCAR_dkl(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        elif self.model_name == 'GP':
            train_GP(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal=False)
        
        with torch.no_grad():
            if self.model_name in ['CMF_CAR','GP','CMF_CAR_dkl']:
                mu, var = self.GP_model_new(self.data_manager, xall, torch.ones(100).reshape(-1,1), normal=False)
            else:
                mu, var = self.GP_model_new(self.data_manager, xall, normal=False)
        # mu, _ = self.data_manager.normalizelayer[self.GP_model_new.fidelity_num-1].denormalize(mu, var)
        
        if self.model_name in ['CMF_CAR','GP','CMF_CAR_dkl']:
            self.data_manager.data_dict['0']['X'] = self.data_manager.data_dict['0']['X'][:-1]
            self.data_manager.data_dict['0']['Y'] = self.data_manager.data_dict['0']['Y'][:-1]
        else:
            self.data_manager.data_dict[str(s)]['X'] = self.data_manager.data_dict[str(s)]['X'][:-1]
            self.data_manager.data_dict[str(s)]['Y'] = self.data_manager.data_dict[str(s)]['Y'][:-1]
        min_mu = torch.min(mu)
        c = self.cost.compute_cost(s / self.fidelity_num)
        cfkg = (min_pre - min_mu) / c

        return cfkg

    def compute_next(self):
        # torch.manual_seed(self.seed)
        N = 3
        sample_x = []
        for i in range(self.x_dim):
            torch.manual_seed(self.seed + 86 + i)
            sample_x.append(torch.rand(N, 1) * (self.search_range[1] - self.search_range[0]) + self.search_range[0])
        sample_x = torch.cat(sample_x, axis=1)

        # s = torch.ones(N) + 1
        torch.manual_seed(self.seed + 86 + 37)
        s = torch.randint(0, self.fidelity_num, (N,), dtype=torch.float)
        
        for i in range(N):
            cfkg = self.negative_cfkg(sample_x[i], int(s[i]))
            if i == 0:
                max_cfkg = cfkg
                new_x = sample_x[i].reshape(-1,1)
                new_s = int(s[i])
            else:
                if cfkg > max_cfkg:
                    max_cfkg = cfkg
                    new_x = sample_x[i].reshape(-1,1)
                    new_s = int(s[i])

        return new_x.reshape(1,-1), new_s

