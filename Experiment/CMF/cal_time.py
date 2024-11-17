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

# Dic = {'ResGP_UCB': ['#ff7f0e', "*", "ResGP_UCB"],
#        'ResGP_EI': ['#708090', "*", "ResGP_EI"],
#        'ResGP_cfKG': ['#17becf', "*", "ResGP_cfKG"],
#        'AR_UCB': ['#8c564b', "s", "AR_UCB"],
#        'AR_EI': ['#2ca02c', "s", "AR_EI"],
#        'AR_cfKG': ['#DC143C', "s", "AR_cfKG"],
#         'CAR_UCB': ['#FFD700', "+", "CAR_UCB"],
#         'CAR_EI': ['#FF4500', "+", "CAR_EI"],
#         'CAR_cfKG': ['black', "+", "CAR_cfKG"],
#         'DNN': ['deeppink', "X", "DNN"],
#         'GP_UCB': ['#FF6347', "X", "GP_UCB"],
#         'GP_EI': ['#B51B75', "X", "GP_EI"],
#         'GP_cfKG': ['#E65C19', "X", "GP_cfKG"],
#         }
# colors = ['#FFD700','#FDE49E','#E1AFD1','black','#B51B75', '#E65C19', 'blue', 'green']
colors = ['#FFD700','#FDE49E','#B51B75', '#E65C19', 'blue', 'green']

data_name = 'forrester'

# data_name = 'VibratePlate2'
# data_name = 'HeatedBlock'


methods_name_list = ['GP_UCB','GP_cfKG','CMF_CAR_UCB', 'CMF_CAR_cfKG',
                     'CMF_CAR_dkl_UCB','CMF_CAR_dkl_cfKG']

cost_name = 'pow_10'
line = []
values = []
for methods_name in methods_name_list:
    ct = []
    tem = []
    for seed in [1,2,3,6]:
        path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy().reshape(-1, 1)
        if methods_name == 'DNN_MFBO':
            time = data['operation_time'].to_numpy().reshape(-1, 1)[-1]
            tem.append((time/len(cost)).item())
        else:
            time = data['operation_time'].to_numpy().reshape(-1, 1)
            tem.append(np.mean(time))
    tem = np.array(tem)
    values.append(np.mean(tem, axis=0))
plt.bar(methods_name_list, values, color=colors,width = 0.6)
for i in range(len(methods_name_list)):
    plt.text(i, values[i], f"{values[i]:.2f}", ha='center', va='bottom')
plt.xticks(rotation=45, ha='right')
# plt.title('Bar Chart Example')
# plt.xlabel('Categories')
plt.yscale('log')
plt.ylabel('Time (seconds)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper') + '/' + data_name +'_query_time' + '.eps',
            bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper') + '/' + data_name +'_query_time' + '.png',
            bbox_inches='tight')
