import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from FidelityFusion_Models.GP_DMF import *
from FidelityFusion_Models.CMF_CAR import *
from FidelityFusion_Models.CMF_CAR_dkl import *
from GaussianProcess.cigp_v10 import *

class discrete_fidelity_knowledgement_gradient(torch.nn.Module):
    def __init__(self, fidelity_num, GP_model_list, cost, data_model, data_manager, model_name, xdim, seed):
        super(discrete_fidelity_knowledgement_gradient, self).__init__()

        self.GP_model_pre = GP_model_list[0]
        self.data_model = data_model
        self.GP_model_new = GP_model_list[1]
        self.cost = cost
        self.seed = seed
        self.search_range = [[-1,1], [-1,1]]
        self.data_manager = data_manager
        self.model_name = model_name
        self.x_dim = xdim
        self.fidelity_num = fidelity_num

    def negative_cfkg(self, x, s):

        xall = torch.rand(100, self.x_dim,dtype=torch.float64) * (self.search_range[0][1] - self.search_range[0][0]) + self.search_range[0][0]
        # mean_y, sigma_y = self.GP_model_pre(xall, self.total_fid_num)  # 预测最高精度
        with torch.no_grad():
            mu_pre, _ = self.GP_model_pre(self.data_manager, xall, [s]*xall.shape[0], normal = False)
        min_pre = torch.min(mu_pre)
        
        x = x.reshape(1,-1)
        with torch.no_grad():
            x = x.double()
            y, _ = self.GP_model_pre(self.data_manager, x, s, normal = False)
        x = torch.cat((x, torch.tensor(s).reshape(1,1)), dim=1)
        
        self.data_manager.add_data(raw_fidelity_name='0', fidelity_index=0, x = x, y = y)
        ## can it be imporved?
        print('train new GP model')
        
        if self.model_name == 'CMF_CAR':
            train_CAR_large(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2)
        elif self.model_name == 'CMF_CAR_dkl':
            train_CMFCAR_dkl(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2)
        else:
            train_GP(self.GP_model_new, self.data_manager, max_iter=200, lr_init=1e-2, normal = False)
        
        with torch.no_grad():
            mu, _ = self.GP_model_new(self.data_manager, xall,[s]*xall.shape[0], normal = False)
        self.data_manager.data_dict['0']['X'] = self.data_manager.data_dict['0']['X'][:-1]
        self.data_manager.data_dict['0']['Y'] = self.data_manager.data_dict['0']['Y'][:-1]
        min_mu = torch.min(mu)
        c = self.cost.compute_cost(s)
        cfkg = (min_pre - min_mu) / c

        return cfkg

    def compute_next(self):
        # torch.manual_seed(self.seed)
        N = 3
        sample_x = []
        for i in range(self.x_dim):
            sample_x.append(torch.rand(N, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0])
        sample_x = torch.cat(sample_x, axis=1)

        # s = torch.ones(N) + 1
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

