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

Dic = {'ResGP_UCB': ['#006769', "*", "ResGP_UCB"],
       'ResGP_EI': ['#40A578', "*", "ResGP_EI"],
       'ResGP_cfKG': ['#9DDE8B', "*", "ResGP_cfKG"],
       'AR_UCB': ['#561F55', "s", "AR_UCB"],
       'AR_EI': ['#8B2F97', "s", "AR_EI"],
       'AR_cfKG': ['#CF56A1', "s", "AR_cfKG"],
        'DMF_CAR_UCB': ['#FF8A08', "+", "CAR_UCB"],
        'DMF_CAR_EI': ['#FFC100', "+", "CAR_EI"],
        'DMF_CAR_cfKG': ['#FF6500', "+", "CAR_cfKG"],
        'DNN_MFBO': ['#FF204E', "X", "DNN"],
        'GP_UCB': ['#006769', "X", "GP_UCB"],
        'GP_EI': ['#40A578', "X", "GP_EI"],
        'GP_cfKG': ['#9DDE8B', "X", "GP_cfKG"],
        'CMF_CAR_UCB': ['#FF8A08', "o", "CMF_CAR_UCB"],
        'CMF_CAR_EI': ['#FFC100', "o", "CMF_CAR_EI"],
        'CMF_CAR_cfKG': ['#FF6500', "o", "CMF_CAR_cfKG"],
        'CMF_CAR_dkl_UCB': ['#1E0342', "v", "CMF_CAR_dkl_UCB"],
        'CMF_CAR_dkl_EI': ['#0E46A3', "v", "CMF_CAR_dkl_EI"],
        'CMF_CAR_dkl_cfKG': ['#9AC8CD', "v", "CMF_CAR_dkl_cfKG"],
        'GP_UCB_con': ['#0A6847', "s", "Con_GP_UCB"],
        'GP_cfKG_con': ['#90D26D', "s", "Con_GP_cfKG"],
        'CMF_CAR_UCB_con': ['#C40C0C', "*", "Con_CMF_CAR_UCB"],
        'CMF_CAR_cfKG_con': ['#FC4100', "*", "Con_CMF_CAR_cfKG"],
        'CMF_CAR_dkl_UCB_con': ['#5356FF', "p", "Con_CMF_CAR_dkl_UCB"],
        'CMF_CAR_dkl_cfKG_con': ['#1640D6', "p", "Con_CMF_CAR_dkl_cfKG"],
        }

# data_name = 'forrester'
# data_name = 'Branin'
data_name = 'non_linear_sin'

methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG','ResGP_UCB', 'ResGP_EI', 
                     'ResGP_cfKG','DMF_CAR_UCB', 'DMF_CAR_EI', 
                     'GP_UCB', 'GP_EI', 'GP_cfKG',
                     'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_EI', 'CMF_CAR_dkl_cfKG','DNN_MFBO']

cost_name = 'pow_10'
line = []
plt.figure(figsize=(10, 6))
for methods_name in methods_name_list:
    cost_collection = []
    # SR_collection = []
    inter_collection = []
    for seed in [0, 1, 2, 3, 4]:
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
    ll = plt.plot(cost_x, mean, ls='dashed', color=Dic[methods_name][0],
                   label=Dic[methods_name][2],
                   marker=Dic[methods_name][1], markersize=6)
    plt.fill_between(cost_x,
                     mean - 0.96 * var,
                     mean + 0.96 * var,
                     alpha=0.1, color=Dic[methods_name][0])
    # line.append(ll)
    plt.xlabel("Cost", fontsize=20)
    plt.ylabel("Simple regret", fontsize=20)
    # plt.xticks(labelsize=20)
    # plt.yticks(labelsize=20)
label = [Dic[i][-1] for i in methods_name_list]
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
plt.grid()
# seed = '1'
plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Graphs', 'Interpolation') + '/' + data_name +'_'+ cost_name +'_SR_Interpolation.png', bbox_inches='tight')

