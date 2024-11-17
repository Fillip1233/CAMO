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
colors = ['#ff7f0e', '#708090', '#17becf', '#8c564b', '#2ca02c', '#DC143C', '#FFD700', '#FF4500', 'black', 'deeppink', '#FF6347', '#B51B75', '#E65C19']

# data_name = 'forrester'
# data_name = 'Branin'
data_name = 'non_linear_sin'
# methods_name_list = ['FiDEs_UCB', 'fides_EI', 'fides_ES', 'fides_CFKG', 'smac']
# methods_name_list = ['FiDEs_UCB', 'FiDEs_EI', 'FiDEs_ES', 'FiDEs_cfKG']
# acq_name = '_ALL'
# methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG','ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG','CAR_UCB', 'CAR_EI', 'CAR_cfKG','GP_UCB', 'GP_EI', 'GP_cfKG']
methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG','ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG','DMF_CAR_UCB', 'DMF_CAR_EI', 'DMF_CAR_cfKG']
# methods_name_list = ['AR_UCB','ResGP_UCB','CAR_UCB','GP_UCB']
# methods_name_list = ['AR_EI','ResGP_EI','CAR_EI','GP_EI']
# methods_name_list = ['AR_cfKG','ResGP_cfKG','CAR_cfKG','GP_cfKG']
cost_name = 'log'
line = []
values = []
for methods_name in methods_name_list:
    ct = []
    tem = []
    for seed in [0,1,2,3,4]:
        path = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy().reshape(-1, 1)
        tem.append(cost.shape[0])
    tem = np.array(tem)
    values.append(int(np.mean(tem, axis=0)))

plt.bar(methods_name_list, values, color=colors,width = 0.6)
for i in range(len(methods_name_list)):
    plt.text(i, values[i], str(int(values[i])), ha='center', va='bottom')
plt.xticks(rotation=45, ha='right')
# plt.title('Bar Chart Example')
# plt.xlabel('Categories')
plt.ylabel('Query number', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Graphs') + '/' + data_name +'_'+ cost_name +'_query_' + '.png',
            bbox_inches='tight')

