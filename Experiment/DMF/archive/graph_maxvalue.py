import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


def get_data(type, data_name, method_name, file_name):
    path = os.path.join(sys.path[-1], 'exp', type, data_name, method_name, file_name + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    return data


# Dic = {'ResGP_UCB': ['navy', "o", "ResGP_UCB"],
#        'ResGP_EI': ['saddlebrown', "v", "ResGP_UCB"],
#        'ResGP_cfKG': ['green', "s", "ResGP_cfKG"],
#     #    'FiDEs_cfKG': ['orange', "*", "FiDEs_CFKG"],
#        'AR_UCB': ['darkgreen', "+", "AR_UCB"],
#        'AR_EI': ['green', "+", "AR_EI"],
#        'AR_cfKG': ['lime', "+", "AR_cfKG"],
#     #    'ResGP_UCB': ['saddlebrown', "+", "ResGP_UCB"],
#     #    'ResGP_EI': ['chocolate', "+", "ResGP_EI"],
#     #    'ResGP_cfKG': ['sandybrown', "+", "ResGP_cfKG"],
#        'smac': ['deeppink', "X", "SMAC"],
#        'fabolas': ['fuchsia', "+", "FABOLAS"], }

Dic = {'ResGP_UCB': ['#ff7f0e', "*", "ResGP_UCB"],
       'ResGP_EI': ['#708090', "*", "ResGP_EI"],
       'ResGP_cfKG': ['#17becf', "*", "ResGP_cfKG"],
       'AR_UCB': ['#8c564b', "s", "AR_UCB"],
       'AR_EI': ['#2ca02c', "s", "AR_EI"],
       'AR_cfKG': ['#DC143C', "s", "AR_cfKG"],
        'CAR_UCB': ['#FFD700', "+", "CAR_UCB"],
        'CAR_EI': ['#FF4500', "+", "CAR_EI"],
        'CAR_cfKG': ['black', "+", "CAR_cfKG"],
        'DNN': ['deeppink', "X", "DNN"],
        }

# data_name = 'forrester'
# data_name = 'Branin'
data_name = 'non_linear_sin'
# methods_name_list = ['FiDEs_UCB', 'fides_EI', 'fides_ES', 'fides_CFKG', 'smac']
# methods_name_list = ['FiDEs_UCB', 'FiDEs_EI', 'FiDEs_ES', 'FiDEs_cfKG']
methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG','ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG','CAR_UCB', 'CAR_EI', 'CAR_cfKG']
cost_name = 'pow_10'
line = []
for methods_name in methods_name_list:
    ct = []
    tem = []
    for seed in [0,1,2,3,4]:
        path = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy().reshape(-1, 1)
        incumbents = data['max_value'].to_numpy().reshape(-1, 1)
        tem.append(incumbents)
        ct.append(cost)
    tem = np.array(tem)
    mean = np.mean(tem, axis=0)
    var = np.std(tem, axis=0)
    ll = plt.plot(ct[0].flatten(), mean.flatten(), ls='dashed', color=Dic[methods_name][0],
                   label=Dic[methods_name][2],
                   marker=Dic[methods_name][1], markersize=8)
    # plt.fill_between(ct[0].flatten(),
    #                  mean.flatten() - 0.96 * var.flatten(),
    #                  mean.flatten() + 0.96 * var.flatten(),
    #                  alpha=0.1, color=Dic[methods_name][0])
    # line.append(ll)
    plt.xlabel("Cost", fontsize=20)
    plt.ylabel("Max_value", fontsize=20)
    # plt.xticks(labelsize=20)
    # plt.yticks(labelsize=20)
label = [Dic[i][-1] for i in methods_name_list]
plt.legend(loc="upper left", fontsize=10)
# plt.legend(loc="lower right", fontsize=10)
plt.grid()
seed = 'all'
plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Graphs') + '/' + data_name+'_'+ cost_name +'_Max-value' +'_seed_' + str(seed) + '.png',
            bbox_inches='tight')
