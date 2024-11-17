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
        'DMF_CAR_UCB': ['#FFD700', "+", "CAR_UCB"],
        'DMF_CAR_EI': ['#FF4500', "+", "CAR_EI"],
        'DMF_CAR_cfKG': ['black', "+", "CAR_cfKG"],
        'DNN': ['deeppink', "X", "DNN"],
        'GP_UCB': ['#FF6347', "X", "GP_UCB"],
        'GP_EI': ['#B51B75', "X", "GP_EI"],
        'GP_cfKG': ['#E65C19', "X", "GP_cfKG"],
        'CMF_CAR_UCB': ['green', "o", "CMF_CAR_UCB"],
        'CMF_CAR_EI': ['red', "o", "CMF_CAR_EI"],
        'CMF_CAR_cfKG': ['orange', "o", "CMF_CAR_cfKG"],
        'CMF_CAR_dkl_UCB': ['blue', "v", "CMF_CAR_dkl_UCB"],
        'CMF_CAR_dkl_EI': ['purple', "v", "CMF_CAR_dkl_EI"],
        'CMF_CAR_dkl_cfKG': ['grey', "v", "CMF_CAR_dkl_cfKG"],
        }

# data_name = 'forrester'
# data_name = 'Branin'
data_name = 'non_linear_sin'
# data_name = 'tl2'
# data_name = 'maolin1'
# methods_name_list = ['FiDEs_UCB', 'fides_EI', 'fides_ES', 'fides_CFKG', 'smac']
# methods_name_list = ['FiDEs_UCB', 'FiDEs_EI', 'FiDEs_ES', 'FiDEs_cfKG']
# acq_name = '_ALL'
# methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG','ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG','CAR_UCB', 'CAR_EI', 'CAR_cfKG','GP_UCB', 'GP_EI', 'GP_cfKG']
methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG','DMF_CAR_UCB', 'DMF_CAR_EI', 
                     'GP_UCB', 'GP_EI', 'GP_cfKG',
                     'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_EI', 'CMF_CAR_dkl_cfKG']
# methods_name_list_withoutEI = ['AR_UCB', 'AR_cfKG','ResGP_UCB', 
#                      'ResGP_cfKG','DMF_CAR_UCB', 'DMF_CAR_cfKG',
#                      'GP_UCB', 'GP_cfKG','CMF_CAR_UCB',
#                      'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']
# methods_name_list = ['AR_UCB','ResGP_UCB','DMF_CAR_UCB','GP_UCB','CMF_CAR_UCB']
# methods_name_list = ['AR_EI','ResGP_EI','DMF_CAR_EI','GP_EI','CMF_CAR_EI']
# methods_name_list = ['AR_cfKG','ResGP_cfKG','DMF_CAR_cfKG','GP_cfKG','CMF_CAR_cfKG']
cost_name = 'pow_10'
line = []
plt.figure(figsize=(10, 6))
seed = 3
for methods_name in methods_name_list:
    ct = []
    tem = []
    
    path = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results',
                        data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    cost = data['cost'].to_numpy().reshape(-1, 1)
    incumbents = data['SR'].to_numpy().reshape(-1, 1)
    tem.append(incumbents)
    ct.append(cost)
    tem = np.array(tem)
    mean = np.mean(tem, axis=0)
    var = np.std(tem, axis=0)
    ll = plt.plot(ct[0].flatten(), mean.flatten(), ls='dashed', color=Dic[methods_name][0],
                   label=Dic[methods_name][2],
                   marker=Dic[methods_name][1], markersize=6)
    # plt.fill_between(ct[0].flatten(),
    #                  mean.flatten() - 0.96 * var.flatten(),
    #                  mean.flatten() + 0.96 * var.flatten(),
    #                  alpha=0.1, color=Dic[methods_name][0])
    # line.append(ll)
    plt.xlabel("Cost", fontsize=20)
    plt.ylabel("Simple regret", fontsize=20)
    # plt.xticks(labelsize=20)
    # plt.yticks(labelsize=20)
label = [Dic[i][-1] for i in methods_name_list]
# plt.legend(loc="upper right", fontsize=10)
# plt.legend(loc="lower left", fontsize=10)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
plt.grid()
# seed = '1'
plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Graphs') + '/' + data_name +'_'+ cost_name +'_SR_' +'_seed_' + str(seed) + '.png',
            bbox_inches='tight')
# plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Graphs','GP') + '/' + data_name +'_'+ cost_name +'_SR_' +'_seed_' + str(seed)+acq_name + '.png',
#             bbox_inches='tight')
