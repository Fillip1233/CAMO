import torch
import matplotlib.pyplot as plt
import numpy as np
from MF_model.CMF_CAR import *
from MF_model.GP_SE2 import *
from MF_model.AR_autoRegression import *
from MF_model.MF_data import *
from utils.calculate_metrix import calculate_metrix
import pandas as pd
if __name__ == "__main__":
    torch.manual_seed(1)

    # generate the data
    x_all = torch.rand(500, 1) * 3
    xlow_indices = torch.randperm(500)[:300]
    xlow_indices = torch.sort(xlow_indices).values
    x_low = x_all[xlow_indices]
    xhigh1_indices = torch.randperm(500)[:300]
    xhigh1_indices = torch.sort(xhigh1_indices).values
    x_high1 = x_all[xhigh1_indices]
    xhigh2_indices = torch.randperm(500)[:250]
    xhigh2_indices = torch.sort(xhigh2_indices).values
    x_high2 = x_all[xhigh2_indices]
    x_test = torch.linspace(0, 3, 100).reshape(-1, 1)

    y_low = torch.sin(2*torch.pi*x_low) * (1-torch.exp(torch.tensor(-3 * 0.2))) + torch.rand(300, 1) * 0.01 - 0.005
    y_high1 = torch.sin(2*torch.pi*x_high1) * (1-torch.exp(torch.tensor(-3 * 0.5))) + torch.rand(300, 1) * 0.01 - 0.005
    y_high2 = torch.sin(2*torch.pi*x_high2) + torch.rand(250, 1) * 0.01 - 0.005
    y_test = torch.sin(2*torch.pi*x_test)

    initial_data2 = [
        {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x_low.double(), 'Y': y_low.double()},
        {'raw_fidelity_name': '1','fidelity_indicator': 1, 'X': x_high1.double(), 'Y': y_high1.double()},
        {'raw_fidelity_name': '2','fidelity_indicator': 2, 'X': x_high2.double(), 'Y': y_high2.double()},
    ]
    
    # plt.figure(figsize=(12, 6))
    # x_low_np = x_low.numpy().flatten()
    # y_low_np = y_low.numpy().flatten()
    # x_high1_np = x_high1.numpy().flatten()
    # y_high1_np = y_high1.numpy().flatten()
    # x_high2_np = x_high2.numpy().flatten()
    # y_high2_np = y_high2.numpy().flatten()
    
    # plt.scatter(x_low_np, y_low_np, color='blue', s=10, alpha=0.6, label='Low Fidelity (Points)')
    # plt.scatter(x_high1_np, y_high1_np, color='green', s=10, alpha=0.6, label='High Fidelity 1 (Points)')
    # plt.scatter(x_high2_np, y_high2_np, color='red', s=10, alpha=0.6, label='High Fidelity 2 (Points)')

    # x_smooth = np.linspace(0, 3, 300)
    # plt.plot(x_smooth, np.sin(2*np.pi*x_smooth) * (1-np.exp(-3 * 0.2)), 'b-', lw=2, label='Low Fidelity (Trend)')
    # plt.plot(x_smooth, np.sin(2*np.pi*x_smooth) * (1-np.exp(-3 * 0.5)), 'g--', lw=2, label='High Fidelity 1 (Trend)')
    # plt.plot(x_smooth, np.sin(2*np.pi*x_smooth), 'r-.', lw=2, label='High Fidelity 2 (Trend)')

    # plt.title("Multi-Fidelity Data Visualization", fontsize=14)
    # plt.xlabel("X", fontsize=12)
    # plt.ylabel("Y", fontsize=12)
    # plt.legend(fontsize=10, loc='upper right')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.xlim(0, 3)

    # plt.tight_layout()
    # plt.savefig('multi_fidelity_data.png', dpi=300)

    y_train = torch.cat((y_low, y_high1, y_high2), 0)
    y_test = torch.sin(2*torch.pi*x_test)

    x_low = torch.cat((x_low, torch.ones(x_low.shape[0]).reshape(-1,1)), 1)
    x_high1 = torch.cat((x_high1, 2*torch.ones(x_high1.shape[0]).reshape(-1,1)), 1)
    x_high2 = torch.cat((x_high2, 3*torch.ones(x_high2.shape[0]).reshape(-1,1)), 1)
    x_test2 = torch.cat((x_test, 3*torch.ones(x_test.shape[0]).reshape(-1,1)), 1)
    x = torch.cat((x_low, x_high1, x_high2), 0)
    y = torch.cat((y_low, y_high1, y_high2), 0)
    initial_data = [
        {'raw_fidelity_name': '0', 'fidelity_indicator': 0, 'X': x.double(), 'Y': y.double()}
    ]
    fidelity_manager = MultiFidelityDataManager(initial_data)
    
    fidelity_manager2 = MultiFidelityDataManager(initial_data2)

    kernel_x = kernel.SquaredExponentialKernel()
    CAR = ContinuousAutoRegression_large(kernel_x=kernel_x, b_init=1.0)

    kernel_1 = kernel.SquaredExponentialKernel()
    kernel_2 = kernel.SquaredExponentialKernel()
    GPSE2 = GP_SE2(kernel1=kernel_1, kernel2=kernel_2)

    fidelity_num = 3
    kernel_list = [kernel.SquaredExponentialKernel() for _ in range(fidelity_num)]
    myAR = AR(fidelity_num = fidelity_num, kernel_list = kernel_list, rho_init=1.0, if_nonsubset=False)

    train_CAR_large(CAR, fidelity_manager, max_iter=200, lr_init=2e-2)
    train_GPSE2(GPSE2, fidelity_manager, max_iter=200, lr_init=2e-2)
    train_AR(myAR, fidelity_manager2, max_iter=200, lr_init=3e-2)
    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[]}
    with torch.no_grad():
        ypred, ypred_var = CAR(fidelity_manager,x_test2.double())
        metrics = calculate_metrix(y_test = y_test[:,0], y_mean_pre = ypred)
        recording['rmse'].append(metrics['rmse'])
        recording['nrmse'].append(metrics['nrmse'])
        recording['r2'].append(metrics['r2'])
        recording['mae'].append(metrics['mae'])
        ypred_GPSE2, ypred_var_GPSE2 = GPSE2(fidelity_manager,x_test2.double())
        metrics = calculate_metrix(y_test = y_test[:,0], y_mean_pre = ypred_GPSE2)
        recording['rmse'].append(metrics['rmse'])
        recording['nrmse'].append(metrics['nrmse'])
        recording['r2'].append(metrics['r2'])
        recording['mae'].append(metrics['mae'])
        x_test = fidelity_manager2.normalizelayer[myAR.fidelity_num-1].normalize_x(x_test.double())
        ypredAR, ypred_varAR = myAR(fidelity_manager2,x_test.double())
        ypredAR, ypred_varAR = fidelity_manager2.normalizelayer[myAR.fidelity_num-1].denormalize(ypredAR, ypred_varAR)
        metrics = calculate_metrix(y_test = y_test[:,0], y_mean_pre = ypredAR)
        recording['rmse'].append(metrics['rmse'])
        recording['nrmse'].append(metrics['nrmse'])
        recording['r2'].append(metrics['r2'])
        recording['mae'].append(metrics['mae'])
        record = pd.DataFrame(recording)
        record.to_csv('res.csv', index=False)
 
    plt.plot(x_test2[:, 0].flatten(), y_test[:, 0], 'k*', markersize=3, label='Ground Truth')
    plt.plot(x_test2[:, 0].flatten(), ypred[:, 0].reshape(-1).detach(), 
            'r-', linewidth=2, label='GP-LiFiDE(CAMO)')
    plt.plot(x_test2[:, 0].flatten(), ypred_GPSE2[:, 0].reshape(-1).detach(),
            'g--', linewidth=2, label='GP-SExSE')
    plt.plot(x_test2[:, 0].flatten(), ypredAR[:, 0].reshape(-1).detach(),
            'b-.', linewidth=2, label='AR')
    std_dev = ypred_var.diag().sqrt().squeeze().detach()
    std_dev_GPSE2 = ypred_var_GPSE2.diag().sqrt().squeeze().detach()
    std_devAR = ypred_varAR.diag().sqrt().squeeze().detach()
    plt.fill_between(x_test2[:, 0].flatten(), 
                    ypred[:, 0].detach() - std_dev,
                    ypred[:, 0].detach() + std_dev,
                    color='salmon', alpha=0.3)
    plt.fill_between(x_test2[:, 0].flatten(),
                    ypred_GPSE2[:, 0].detach() - std_dev_GPSE2,
                    ypred_GPSE2[:, 0].detach() + std_dev_GPSE2,
                    color='lightgreen', alpha=0.3)
    plt.fill_between(x_test2[:, 0].flatten(),
                    ypredAR[:, 0].detach() - std_devAR,
                    ypredAR[:, 0].detach() + std_devAR,
                    color='lightblue', alpha=0.3)

    plt.title("Prediction and Variance of GP-LiFiDE, GP SExSE, and AR Methods", fontsize=14, pad=20)
    plt.xlabel("Input Feature (x)", fontsize=12)
    plt.ylabel("Target Value (y)", fontsize=12)
    plt.legend(fontsize=10, framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.2)

    x_min, x_max = x_test2.min(), x_test2.max()
    plt.xlim(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min))

    plt.tight_layout()
    plt.savefig('CAR_kernel.png', bbox_inches='tight', transparent=False)
    plt.close()
    pass