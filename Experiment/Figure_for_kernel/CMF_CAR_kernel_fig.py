import torch
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import GaussianProcess.kernel as kernel

def ard_forward(x1, x2):
        log_length_scales = torch.zeros(1)

        X1 = x1[:, 0].reshape(-1, 1)
        X2 = x2[:, 0].reshape(-1, 1)
        fidelity_indicator_1 = x1[:, 1].reshape(-1, 1) # t'
        fidelity_indicator_2 = x2[:, 1].reshape(-1, 1) # t

        scaled_x1 = fidelity_indicator_1 / log_length_scales.exp()
        scaled_x2 = fidelity_indicator_2 / log_length_scales.exp()
        sqdist = torch.cdist(scaled_x1, scaled_x2, p=2)**2
        # sqdist = torch.cdist(scaled_x1, scaled_x2, p=2)**0.5

        kernel_x = kernel.ARDKernel(x1.shape[1])

        # return  torch.exp(-0.5 * sqdist) * kernel_x(X1, X2)
        return torch.exp(-0.5 * sqdist)

def gamma_forward(x1, x2):
        log_length_scales = torch.zeros(1)

        X1 = x1[:, 0].reshape(-1, 1)
        X2 = x2[:, 0].reshape(-1, 1)
        fidelity_indicator_1 = x1[:, 1].reshape(-1, 1) # t'
        fidelity_indicator_2 = x2[:, 1].reshape(-1, 1) # t

        scaled_x1 = fidelity_indicator_1 / log_length_scales.exp()
        scaled_x2 = fidelity_indicator_2 / log_length_scales.exp()
        sqdist = torch.cdist(fidelity_indicator_1, fidelity_indicator_2, p=2)**2
        from scipy.stats import gamma
        import numpy as np

        final_part = -torch.from_numpy(gamma.cdf(sqdist.numpy(), 1, scale = 1)).exp() + 2

        # return  torch.exp(-0.5 * sqdist) * kernel_x(X1, X2)
        return final_part

def Integral_ard_forward(x1, x2):

        fidelity_indicator_1 = x1[:, 1].reshape(-1, 1) # t'
        fidelity_indicator_2 = x2[:, 1].reshape(-1, 1) # t

        import numpy as np
        from scipy.integrate import dblquad

        kernel = []
        for i in range(20):
            t = 4
            t_1 = fidelity_indicator_2[i]

            def integrand(y, x):
                tem = np.exp(- 1 * (t-x) - (t_1-y) -  np.exp(2* 1e-9) * (x-y)**2)
                return tem
            
            result, error = dblquad(integrand, 0, t, 0, t_1)
            kernel.append(result)


        # return  torch.exp(-0.5 * sqdist) * kernel_x(X1, X2)
        return np.asarray(kernel)


def derive_forward(x1, x2):

    log_length_scales = torch.zeros(1)
    b = torch.ones(1)
    v = log_length_scales.exp() * b * 0.5

    def h(t, t_1):
        tem_1 = (v**2).exp() / (2*b)
        tem_2 = (-b * t).exp() 
        tem_3 = (b * t_1).exp() * (torch.erf((t-t_1)/log_length_scales.exp() - v) + torch.erf((t_1)/log_length_scales.exp() + v))
        tem_4 = (-b * t_1).exp() * (torch.erf((t/log_length_scales.exp()) - v) + torch.erf(v))

        return tem_1 * tem_2 * (tem_3 - tem_4)
        
    fidelity_indicator_1 = x1[:, 1].reshape(-1, 1) # t'
    fidelity_indicator_2 = x2[:, 1].reshape(-1, 1) # t

    tem = [fidelity_indicator_1 for i in range(fidelity_indicator_2.size(0))]
    T1 = torch.cat(tem, dim=1)
    tem = [fidelity_indicator_2 for i in range(fidelity_indicator_1.size(0))]
    T2 = torch.cat(tem, dim=1).T

    h_part_1 = h(T1, T2)
    h_part_2 = h(T2, T1)

    final_part = 0.5 * torch.sqrt(torch.tensor(torch.pi)) * log_length_scales.exp() * (h_part_1 + h_part_2)


    return final_part

def derive_forward_2(x1, x2):

    log_length_scales = torch.zeros(1)
    b = torch.ones(1)
    v = log_length_scales.exp() * b * 0.5

    def h(t, t_1):
        tem_1 = (v**2).exp() / (2*b)
        tem_2 = (-b * t).exp() 
        tem_3 = (b * t_1).exp() * (torch.erf((t-t_1)/log_length_scales.exp() - v) + torch.erf((t_1)/log_length_scales.exp() + v))
        tem_4 = (-b * t_1).exp() * (torch.erf((t/log_length_scales.exp()) - v) + torch.erf(v))

        return tem_1 * tem_2 * (tem_3 - tem_4)
        
    fidelity_indicator_1 = x1[:, 1].reshape(-1, 1) # t'
    fidelity_indicator_2 = x2[:, 1].reshape(-1, 1) # t

    tem = [fidelity_indicator_1 for i in range(fidelity_indicator_2.size(0))]
    T1 = torch.cat(tem, dim=1)
    tem = [fidelity_indicator_2 for i in range(fidelity_indicator_1.size(0))]
    T2 = torch.cat(tem, dim=1).T

    h_part_1 = h(T1, T2)
    h_part_2 = h(T2, T1)

    final_part = 0.5 * torch.sqrt(torch.tensor(torch.pi)) * log_length_scales.exp() * (h_part_1 + h_part_2)


    return final_part + (-b * (T1+T2)).exp()


t_all = torch.linspace(0, 1, 20).reshape(-1, 1) *3+ 1
x_all = torch.rand(20, 1) * 20
x_test = torch.cat((x_all, t_all), 1)

t_all_2 = torch.linspace(0, 1, 10).reshape(-1, 1) * 3 + 1
x_all_2 = torch.rand(10, 1) * 20
x_test_2 = torch.cat((x_all_2, t_all_2), 1)

Sigma_ard = ard_forward(x_test, x_test)
Sigma_gamma = gamma_forward(x_test, x_test)
Sigma_integral = Integral_ard_forward(x_test, x_test)

Sigma_derive = derive_forward(x_test, x_test_2)
Sigma_derive_2 = derive_forward_2(x_test, x_test_2)
 
plt.figure()
plt.plot(t_all.numpy().flatten(), Sigma_ard[:, -1].detach().numpy().flatten(), 'r-.', label = 'ARD_Kernel')
# plt.plot(t_all.numpy().flatten(), Sigma_gamma[:, -1].detach().numpy().flatten(), 'b-.', label = 'Gamma_Kernel')
plt.plot(t_all.numpy().flatten(), Sigma_integral, 'g-.', label = 'Integral_ARD_Kernel')
plt.plot(t_all.numpy().flatten(), Sigma_derive[:, -1].detach().numpy().flatten(), 'b-.', label = 'Derivation_Kernel')

plt.legend()
# plt.show()
plt.savefig('Kernel_compare.png') 
