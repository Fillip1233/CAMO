import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import GaussianProcess.kernel as kernel
from FidelityFusion_Models import *
import matplotlib.pyplot as plt
from Data_simulation.Synthetic_MF_Function.Forrester import forrester
from MFBO.Discrete.MF_EI import expected_improvement as EI
from MFBO.Discrete.CFKG import discrete_fidelity_knowledgement_gradient as DMF_KG

if __name__ == '__main__':
    torch.manual_seed(0)
    seed = 0
    total_fid_num = 2
    Data_model = forrester(total_fidelity_num=total_fid_num)
    Data_cost = Data_model.cost
    xtr, ytr = Data_model.Initiate_data(index={0: 10, 1: 4}, seed = seed)

    initial_data = [
                        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': xtr[0], 'Y': ytr[0]},
                        {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': xtr[1], 'Y': ytr[1]},
                    ]
    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = [kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.) for _ in range(total_fid_num)]
    kernel_init = [kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.) for _ in range(total_fid_num)]
    # model = ResGP(fidelity_num = total_fid_num, kernel_list = kernel1, if_nonsubset = True)
    # model_init = ResGP(fidelity_num = total_fid_num, kernel_list = kernel_init, if_nonsubset = True)
    model = AR(fidelity_num = total_fid_num, kernel_list = kernel1, if_nonsubset = True)
    model_init = AR(fidelity_num = total_fid_num, kernel_list = kernel_init, if_nonsubset = True)

    key_iterations = [2,4,5,6,8,10]
    predictions = []
    observed_l = []
    observed_h = []

    # xall = torch.rand(100, 1)
    test_points = torch.linspace(0, 1.5, 100).reshape(-1, 1)
    lowf_y = Data_model.get_data(test_points, 0)
    true_y = Data_model.get_data(test_points, 1)

    for iter in range(10):
        # train_ResGP(model, fidelity_manager, max_iter=200, lr_init=1e-2, normal= True)
        train_AR(model, fidelity_manager, max_iter=200, lr_init=1e-2, normal= True)

        # Acq_fun = EI(x_dimension=xtr[0].shape[1],fidelity_num=len(initial_data),GP_model=model,cost=Data_cost,data_manager=fidelity_manager,threshold=1e-6,seed=seed)

        Acq_fun = DMF_KG(fidelity_num=len(initial_data), GP_model_list=[model, model_init], cost=Data_cost, data_model=Data_model, data_manager=fidelity_manager,model_name='AR', seed=seed)

        new_x, new_s = Acq_fun.compute_next()
        print('new_x:', new_x, 'new_s:', new_s)
        new_y = Data_model.get_data(new_x, new_s)
        fidelity_manager.add_data(raw_fidelity_name=str(new_s), fidelity_index=new_s, x=new_x, y=new_y)
        
        if (iter + 1) in key_iterations:
            model.eval()
            with torch.no_grad():
                test_x = fidelity_manager.normalizelayer[model.fidelity_num-1].normalize_x(test_points)
                ypred, ypred_var = model(fidelity_manager, test_x)
                ypred, ypred_var = fidelity_manager.normalizelayer[model.fidelity_num-1].denormalize(ypred, ypred_var)
                predictions.append((ypred, ypred_var))
                observed_l.append((fidelity_manager.get_data(0)))
                observed_h.append((fidelity_manager.get_data(1)))

    plt.figure(figsize=(15, 12))
    for i, (ypred, ypred_var) in enumerate(predictions):
        plt.subplot(3, 2, i+1)
        # plt.ylim(-2, 2)
        plt.plot(test_points, true_y, 'k-', label='True function')
        plt.plot(test_points, lowf_y, 'r-', label='Low fidelity function')
        plt.plot(test_points, ypred, 'b--', label='GP mean')
        
        plt.fill_between(test_points.reshape(-1),
                        ypred.reshape(-1) - ypred_var.diag().sqrt(),
                        ypred.reshape(-1) + ypred_var.diag().sqrt(),
                        color='blue', alpha=0.2, label='GP uncertainty')

        plt.scatter(observed_h[i][0], observed_h[i][1], c='g', zorder=2, label='Observed points (High)')
        plt.scatter(observed_l[i][0], observed_l[i][1], c='r', zorder=2, label='Observed points (Low)')
        plt.title(f'Samples: {key_iterations[i]}')
        # plt.legend()
        plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()