import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


def get_data(type, data_name, method_name, file_name):
    path = os.path.join(sys.path[-1], 'exp', type, data_name, method_name, file_name + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    return data

# Dic = {'ResGP_UCB': ['#ff7f0e', "*", "ResGP_UCB"],
#        'ResGP_EI': ['#708090', "*", "ResGP_EI"],
#        'ResGP_cfKG': ['#17becf', "*", "ResGP_cfKG"],
#        'AR_UCB': ['#8c564b', "s", "AR_UCB"],
#        'AR_EI': ['#2ca02c', "s", "AR_EI"],
#        'AR_cfKG': ['#DC143C', "s", "AR_cfKG"],
#         'DMF_CAR_UCB': ['#FFD700', "+", "CAR_UCB"],
#         'DMF_CAR_EI': ['#FF4500', "+", "CAR_EI"],
#         'DMF_CAR_cfKG': ['black', "+", "CAR_cfKG"],
#         'DNN': ['deeppink', "X", "DNN"],
#         'GP_UCB': ['#FF6347', "X", "GP_UCB"],
#         'GP_EI': ['#B51B75', "X", "GP_EI"],
#         'GP_cfKG': ['#E65C19', "X", "GP_cfKG"],
#         'CMF_CAR_UCB': ['green', "o", "CMF_CAR_UCB"],
#         'CMF_CAR_EI': ['red', "o", "CMF_CAR_EI"],
#         'CMF_CAR_cfKG': ['orange', "o", "CMF_CAR_cfKG"],
#         'CMF_CAR_dkl_UCB': ['blue', "v", "CMF_CAR_dkl_UCB"],
#         'CMF_CAR_dkl_EI': ['purple', "v", "CMF_CAR_dkl_EI"],
#         'CMF_CAR_dkl_cfKG': ['grey', "v", "CMF_CAR_dkl_cfKG"],
#         'GP_UCB_con': ['#7469B6', "s", "Con_GP_UCB"],
#         'GP_cfKG_con': ['#E1AFD1', "s", "Con_GP_cfKG"],
#         'CMF_CAR_UCB_con': ['#FF5F00', "*", "Con_CMF_CAR_UCB"],
#         'CMF_CAR_cfKG_con': ['#002379', "*", "Con_CMF_CAR_cfKG"],
#         'CMF_CAR_dkl_UCB_con': ['#948979', "p", "Con_CMF_CAR_dkl_UCB"],
#         'CMF_CAR_dkl_cfKG_con': ['#153448', "p", "Con_CMF_CAR_dkl_cfKG"],
#         }

# UCB * EI s cfkg o
Dic = {'ResGP_UCB': ['#006769', "*", "ResGP-UCB"],
       'ResGP_EI': ['#006769', "s", "ResGP-EI"],
       'ResGP_cfKG': ['#006769', "o", "ResGP-cfKG"],
       'AR_UCB': ['purple', "*", "AR-UCB"],
       'AR_EI': ['purple', "s", "AR-EI"],
       'AR_cfKG': ['purple', "o", "AR-cfKG"],
        'DMF_CAR_UCB': ['#FF8A08', "*", "CAR-UCB"],
        'DMF_CAR_EI': ['#FFC100', "s", "CAR-EI"],
        'DMF_CAR_cfKG': ['#FF6500', "o", "CAR-cfKG"],
        
        'DNN_MFBO': ['orange', "X", "DNN"],
        
        'GP_UCB': ['black', "*", "GP-UCB"],
        'GP_EI': ['black', "s", "GP-EI"],
        'GP_cfKG': ['black', "o", "GP-cfKG"],
        'CMF_CAR_UCB': ['#FF8A08', "*", "CMF_CAR-UCB"],
        'CMF_CAR_EI': ['#FFC100', "s", "CMF-CAR_EI"],
        'CMF_CAR_cfKG': ['#FF6500', "o", "CMF-CAR_cfKG"],
        
        'CMF_CAR_dkl_UCB': ['#1E0342', "*", "CMF_CAR_dkl-UCB"],
        'CMF_CAR_dkl_EI': ['#0E46A3', "s", "CMF-CAR_dkl-EI"],
        'CMF_CAR_dkl_cfKG': ['#68D2E8', "o", "CMF-CAR_dkl_cfKG"],
        'GP_UCB_con': ['#75A47F', "*", "Con_GP-UCB"],
        'GP_cfKG_con': ['#90D26D', "o", "Con_GP-cfKG"],
        
        'CMF_CAR_UCB_con': ['red', "*", "CAR-UCB"],
        'CMF_CAR_cfKG_con': ['red', "o", "CAR-cfKG"],
        'CMF_CAR_dkl_UCB_con': ['blue', "*", "CAR_dkl-UCB"],
        'CMF_CAR_dkl_cfKG_con': ['blue', "o", "CAR_dkl-cfKG"],
        }

# data_name = 'Park'
data_name = 'Forrester'
# data_name = 'Branin'
# data_name = 'non_linear_sin'
cost_name = 'pow_10'

max_dic = {'non_linear_sin':0, 'forrester': 50}
add_dict = {'Forrester': 7 ,'non_linear_sin': 0.15}
lim_x = {'Forrester': [48, 135], 'non_linear_sin': [15, 135]}
lim_y = {'Forrester': [0, 55], 'non_linear_sin': [0, 0.30]}
# cost_lim_y = {'Forrester': [0, 0], 'non_linear_sin': [0, -0.25]}


methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG', 'ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG',
                     'GP_UCB', 'GP_EI', 'GP_cfKG']
cmf_methods_name_list = ['CMF_CAR_UCB','CMF_CAR_cfKG',
                        #  'GP_UCB', 'GP_cfKG',
                         'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']

line = []
plt.figure(figsize=(10, 6))
for methods_name in methods_name_list:
    cost_collection = []
    # SR_collection = []
    inter_collection = []
    for seed in [0, 1, 2,3,4]:
        path = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        # cost = data['cost'].to_numpy().reshape(-1, 1)
        # SR = data['SR'].to_numpy().reshape(-1, 1)
        cost = data['cost'].to_numpy()
        
        SR = data['SR'].to_numpy()
        inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
        # SR_collection.append(SR)
        cost_collection.append(cost)
        inter_collection.append(inter)

    cost_x = np.unique(np.concatenate(cost_collection))
    SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
    SR_new = np.array(SR_new)
    mean = np.mean(SR_new, axis=0)
    var = np.std(SR_new, axis=0)
    ll = plt.plot(cost_x, mean + add_dict[data_name], ls='dashed', color=Dic[methods_name][0],
                   label=Dic[methods_name][2],
                   marker=Dic[methods_name][1], markersize=7,markevery=7)
    plt.fill_between(cost_x,
                     mean + add_dict[data_name] - 0.96 * var,
                     mean + add_dict[data_name] + 0.96 * var,
                     alpha=0.1, color=Dic[methods_name][0])
    # line.append(ll)
    plt.xlabel("Cost", fontsize=20)
    plt.ylabel("Simple regret", fontsize=20)
    # plt.xticks(labelsize=20)
    # plt.yticks(labelsize=20)
for methods_name in cmf_methods_name_list:
    cost_collection = []
    # SR_collection = []
    inter_collection = []
    for seed in [0, 1, 2,3,4]:
        path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        # cost = data['cost'].to_numpy().reshape(-1, 1)
        # SR = data['SR'].to_numpy().reshape(-1, 1)
        cost = data['cost'].to_numpy()
        if methods_name == 'fabolas':
            SR = max_dic['data_name'] - data['SR'].to_numpy()
        SR = data['SR'].to_numpy()
        inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
        # SR_collection.append(SR)
        cost_collection.append(cost)
        inter_collection.append(inter)

    cost_x = np.unique(np.concatenate(cost_collection))
    SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
    SR_new = np.array(SR_new)
    mean = np.mean(SR_new, axis=0)
    var = np.std(SR_new, axis=0)
    ll = plt.plot(cost_x, mean + add_dict[data_name], ls='dashed', color=Dic[methods_name+'_con'][0],
                   label=Dic[methods_name+'_con'][2],
                   marker=Dic[methods_name+'_con'][1], markersize=7,markevery=7)
    plt.fill_between(cost_x,
                     mean + add_dict[data_name] - 0.96 * var,
                     mean + add_dict[data_name] + 0.96 * var,
                     alpha=0.1, color=Dic[methods_name+'_con'][0])
    # line.append(ll)
    plt.xlabel("Cost", fontsize=20)
    plt.ylabel("Simple regret", fontsize=20)

# plt.yscale('log')
plt.xlim(lim_x[data_name][0], lim_x[data_name][1])
plt.ylim(lim_y[data_name][0], lim_y[data_name][1])
label = [Dic[i][-1] for i in methods_name_list]
label = label + [Dic[i+'_con'][-1] for i in cmf_methods_name_list]
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
plt.grid()
# seed = '1'
plt.tight_layout()
# plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper') + '/' + data_name +'_'+ cost_name +'_SR_Interpolation.png', bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper') + '/' +'DMF_'+ data_name +'_'+ cost_name +'_SR.pdf', bbox_inches='tight')
