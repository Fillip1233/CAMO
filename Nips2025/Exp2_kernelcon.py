import torch
import matplotlib.pyplot as plt
import numpy as np
from MF_model.CMF_CAR import *
from MF_model.GP_SE2 import *
from MF_model.AR_autoRegression import *
from MF_model.MF_data import *
from utils.calculate_metrix import calculate_metrix
import pandas as pd
import math

def f(x, t):
    """目标函数:f(x,t) = sin(2πx) x (1 - exp(-3t))"""
    return torch.sin(2 * math.pi * x) * (1 - torch.exp(-3 * t))

def bohachevsky(x, t):
    """Bohachevsky function: f(x1, x2) = x1^2 + 2*x2^2 - 0.3*cos(3πx1) - 0.4*cos(4πx2) + 0.7"""
    x1, x2 = x[:, 0].reshape(-1,1), x[:, 1].reshape(-1,1)
    y_high = x1**2 + 2 * x2**2 - 0.3 * torch.cos(3 * math.pi * x1) - 0.4 * torch.cos(4 * math.pi * x2) + 0.7
    y_low = (x1*0.7)**2 + 2 * x2**2 - 0.3 * torch.cos(3 * math.pi * (x1*0.7)) - 0.4 * torch.cos(4 * math.pi * x2) + 0.7 + x1 * x2 - 12
    y = y_high * torch.log10(9 * t + 1) + y_low * (1 - torch.log10(9 * t + 1))
    return y

if __name__ == "__main__":
    torch.manual_seed(0)
    test_fidelity = 3.0  # 测试点的 t 值
    n_samples = 200

    x = torch.rand(n_samples) * 2
    x = x.reshape(-1, 1)  # 确保 x 是二维的
    t = torch.rand(n_samples) * 3
    t = t.reshape(-1, 1)  # 确保 t 是二维的
    x_test = torch.linspace(0, 2, 100).reshape(-1, 1)  # 测试点
    y = f(x, t)
    y_test = f(x_test, torch.tensor([test_fidelity]))  # 使用 t=1.0 进行测试
    x = torch.cat((x,t), dim=1)
    t_test = torch.tensor([test_fidelity]).repeat(x_test.shape[0], 1)  # 测试点的 t 值
    x_test = torch.cat((x_test, t_test), dim=1)

    # x = torch.rand(n_samples, 2) * 10 - 5  # [-5, 5] 区间
    # t = torch.rand(n_samples, 1)  # [0, 3] 区间
    # x_test = torch.linspace(-5, 5, 100).reshape(-1, 2)  # 测试点
    # y = bohachevsky(x, t)
    # y_test = bohachevsky(x_test, torch.tensor([test_fidelity]).repeat(x_test.shape[0], 1))  # 使用 t=1.0 进行测试
    # x = torch.cat((x, t), dim=1)
    # t_test = torch.tensor([test_fidelity]).repeat(x_test.shape[0], 1)  # 测试点的 t 值
    # x_test = torch.cat((x_test, t_test), dim=1)


    initial_data = [
        {'raw_fidelity_name': '0', 'fidelity_indicator': 0, 'X': x.double(), 'Y': y.double()}
    ]
    fidelity_manager = MultiFidelityDataManager(initial_data) 
    kernel_x = kernel.SquaredExponentialKernel()
    CAR = ContinuousAutoRegression_large(kernel_x=kernel_x, b_init=1.0)

    kernel_1 = kernel.SquaredExponentialKernel()
    kernel_2 = kernel.SquaredExponentialKernel()
    GPSE2 = GP_SE2(kernel1=kernel_1, kernel2=kernel_2)

    #exp1
    #iter = 300 lr=2e-2 
    #exp2
    #iter = 400 lr=1e-1
    train_CAR_large(CAR, fidelity_manager, max_iter=300, lr_init=2e-2)
    train_GPSE2(GPSE2, fidelity_manager, max_iter=300, lr_init=2e-2)
    
    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[]}

    with torch.no_grad():
        ypred, ypred_var = CAR(fidelity_manager,x_test.double())
        metrics = calculate_metrix(y_test = y_test[:,0], y_mean_pre = ypred)
        recording['rmse'].append(metrics['rmse'])
        recording['nrmse'].append(metrics['nrmse'])
        recording['r2'].append(metrics['r2'])
        recording['mae'].append(metrics['mae'])
        ypred_GPSE2, ypred_var_GPSE2 = GPSE2(fidelity_manager,x_test.double())
        metrics = calculate_metrix(y_test = y_test[:,0], y_mean_pre = ypred_GPSE2)
        recording['rmse'].append(metrics['rmse'])
        recording['nrmse'].append(metrics['nrmse'])
        recording['r2'].append(metrics['r2'])
        recording['mae'].append(metrics['mae'])
        record = pd.DataFrame(recording)
        record.to_csv(f'res_level{test_fidelity}.csv', index=False)
    

    
    # x1_grid, x2_grid = np.meshgrid(x_test[:, 0].reshape(-1, 1), x_test[:, 1].reshape(-1, 1))
    # # 绘图
    # plt.figure(figsize=(10, 6))
    # ax = plt.subplot(111, projection='3d')
    # ax.plot_surface(x1_grid, x2_grid, y_test,
    #                color='lightgray', alpha=0.5, label='Ground Truth')
    # ax.plot_surface(x1_grid, x2_grid, ypred,
    #                  color='salmon', alpha=0.7, label='GP-LiFiDE(CAMO)')
    # ax.plot_surface(x1_grid, x2_grid, ypred_GPSE2,
    #                  color='lightgreen', alpha=0.7, label='GP SExSE')
    # ax.set_title(f"Prediction of GP-LiFiDE, GP SExSE Methods, test_level = {test_fidelity}", fontsize=14, pad=20)
    # ax.set_xlabel("x1", fontsize=12)
    # ax.set_ylabel("x2", fontsize=12)
    # ax.set_zlabel("y", fontsize=12)
    # ax.legend(fontsize=10, framealpha=1)
    # plt.tight_layout()
    # plt.savefig(f'Bohachevsky_kernel_con_level{test_fidelity}.png', bbox_inches='tight', transparent=False)
    # plt.close()
    
    # plt.plot(x_test[:, 0].flatten(), y_test[:, 0], 'k*', markersize=3, label='Ground Truth')
    # plt.plot(x_test[:, 0].flatten(), ypred[:, 0].reshape(-1).detach(), 
    #         'r-', linewidth=2, label='GP-LiFiDE(CAMO)')
    # plt.plot(x_test[:, 0].flatten(), ypred_GPSE2[:, 0].reshape(-1).detach(),
    #         'g--', linewidth=2, label='GP-SExSE')
    # std_dev = ypred_var.diag().sqrt().squeeze().detach()
    # std_dev_GPSE2 = ypred_var_GPSE2.diag().sqrt().squeeze().detach()
    # plt.fill_between(x_test[:, 0].flatten(), 
    #                 ypred[:, 0].detach() - std_dev,
    #                 ypred[:, 0].detach() + std_dev,
    #                 color='salmon', alpha=0.3)
    # plt.fill_between(x_test[:, 0].flatten(),
    #                 ypred_GPSE2[:, 0].detach() - std_dev_GPSE2,
    #                 ypred_GPSE2[:, 0].detach() + std_dev_GPSE2,
    #                 color='lightgreen', alpha=0.3)
    # plt.title(f"Prediction and Variance of GP-LiFiDE, GP SExSE Methods, test_level = {test_fidelity}", fontsize=14, pad=20)
    # plt.xlabel("Input Feature (x)", fontsize=12)
    # plt.ylabel("Target Value (y)", fontsize=12)
    # plt.legend(fontsize=10, framealpha=1)
    # plt.grid(True, linestyle='--', alpha=0.2)

    # x_min, x_max = x_test[:,0].min(), x_test[:,0].max()
    # plt.xlim(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min))

    # plt.tight_layout()
    # plt.savefig(f'CAR_kernel_con_level{test_fidelity}.png', bbox_inches='tight', transparent=False)
    # plt.close()
    pass