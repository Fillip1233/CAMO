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


# UCB * EI s cfkg o
Dic = {'AR_UCB': ['#000080', "^", "AR-UCB"],
       'AR_EI': ['#000080', "s", "AR-EI"],
       'AR_cfKG': ['#000080', "o", "AR-cfKG"],
       'ResGP_UCB': ['#00CCFF', "^", "ResGP-UCB"],
       'ResGP_EI': ['#00CCFF', "s", "ResGP-EI"],
       'ResGP_cfKG': ['#00CCFF', "o", "ResGP-cfKG"],
        
        'DNN_MFBO': ['#228B22', "X", "DNN"],
        
        'GP_UCB': ['#4169E1', "^", "GP-UCB"],
        'GP_EI': ['#4169E1', "s", "GP-EI"],
        'GP_cfKG': ['#4169E1', "o", "GP-cfKG"],
        'GP_UCB_con': ['#4169E1', "^", "Con_GP-UCB"],
        'GP_cfKG_con': ['#4169E1', "o", "Con_GP-cfKG"],
        
        'CMF_CAR_UCB_con': ['#FF0000', "^", "CAR-UCB"], # red
        'CMF_CAR_cfKG_con': ['#FF0000', "o", "CAR-cfKG"],
        'CMF_CAR_dkl_UCB_con': ['#FF5E00', "^", "CAR_dkl-UCB"], # orange
        'CMF_CAR_dkl_cfKG_con': ['#FF5E00', "o", "CAR_dkl-cfKG"],
        }


# data_name = 'Park'
# data_name = 'Forrester'
# data_name = 'Branin'
data_name = 'non_linear_sin'
cost_name = 'pow_10'

max_dic = {'non_linear_sin':0, 'forrester': 50}
add_dict = {'Forrester': 7 ,'non_linear_sin': 0.15}
lim_x = {'Forrester': [48, 135], 'non_linear_sin': [50, 135]}
lim_y = {'Forrester': [0, 55], 'non_linear_sin': [0, 0.32]}
# cost_lim_y = {'Forrester': [0, 0], 'non_linear_sin': [0, -0.25]}


methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG', 'ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG', 'GP_UCB', 'GP_EI', 'GP_cfKG', 'DNN_MFBO']
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
    ll = plt.plot(cost_x, mean + add_dict[data_name], ls='solid', color=Dic[methods_name][0],
                   label=Dic[methods_name][2],
                   marker=Dic[methods_name][1], markersize=7,markevery=7)
    plt.fill_between(cost_x,
                     mean + add_dict[data_name] - 0.96 * var,
                     mean + add_dict[data_name] + 0.96 * var,
                     alpha=0.05, color=Dic[methods_name][0])
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
    ll = plt.plot(cost_x, mean + add_dict[data_name], ls='solid', color=Dic[methods_name+'_con'][0],
                   label=Dic[methods_name+'_con'][2],
                   marker=Dic[methods_name+'_con'][1], markersize=7,markevery=7)
    plt.fill_between(cost_x,
                     mean + add_dict[data_name] - 0.96 * var,
                     mean + add_dict[data_name] + 0.96 * var,
                     alpha=0.07, color=Dic[methods_name+'_con'][0])
    # line.append(ll)
    plt.xlabel("Cost", fontsize=20)
    plt.ylabel("Simple regret", fontsize=20)

# plt.yscale('log')
plt.xlim(lim_x[data_name][0], lim_x[data_name][1])
plt.ylim(lim_y[data_name][0], lim_y[data_name][1])
label = [Dic[i][-1] for i in methods_name_list]
label = label + [Dic[i+'_con'][-1] for i in cmf_methods_name_list]
plt.tick_params(axis='both', labelsize=15)
# if data_name == 'non_linear_sin':
#     plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=15)
# plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=15)
plt.grid()
# seed = '1'
plt.tight_layout()
# plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper') + '/' + data_name +'_'+ cost_name +'_SR_Interpolation.png', bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper_yx') + '/' +'DMF_'+ data_name +'_'+ cost_name +'_SR.pdf', bbox_inches='tight')
