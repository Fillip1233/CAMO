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
Dic = { 'fabolas':['#808000', "*", "Fabolas"],
        'smac':['#006400', "*", "SMAC3"],
        
        'GP_UCB_con': ['#4169E1', "^", "Con_GP-UCB"],
        'GP_cfKG_con': ['#4169E1', "o", "Con_GP-cfKG"],
        
        'CMF_CAR_UCB_con': ['#FF0000', "^", "CAR-UCB"], # red
        'CMF_CAR_cfKG_con': ['#FF0000', "o", "CAR-cfKG"],
        'CMF_CAR_dkl_UCB_con': ['#FF5E00', "^", "CAR_dkl-UCB"], # orange
        'CMF_CAR_dkl_cfKG_con': ['#FF5E00', "o", "CAR_dkl-cfKG"],
        }

data_name = 'Park'
# data_name = 'Currin'
# data_name = 'Forrester'
# data_name = 'Branin'
# data_name = 'non_linear_sin'
cost_name = 'pow_10'

max_dic = {'non_linear_sin':0, 'Forrester': 50,'Branin2': 55,'Currin': 14,'Park': 2.2}
add_dict = {'Forrester': 7 ,'non_linear_sin': 0.15,'Branin2': 0,'Currin': 0,'Park': 0}
lim_x = {'Forrester': [48, 135], 'non_linear_sin': [15, 135],
         'Branin2':[48,150],'Currin':[48,150],'Park':[48,150]}
lim_y = {'Forrester': [0, 55], 'non_linear_sin': [0, 0.30],'Currin':[0,2],'Park':[0,1.4]}

cmf_methods_name_list = ['fabolas', 'smac',
                         'GP_UCB', 'GP_cfKG', 
                         'CMF_CAR_UCB','CMF_CAR_cfKG',
                         'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']

line = []
plt.figure(figsize=(10, 6))
lable_name = []
for methods_name in cmf_methods_name_list:
    cost_collection = []
    # SR_collection = []
    inter_collection = []
    if data_name == 'Currin':
        seed_list = [3, 5]
    elif data_name == 'Park':
        seed_list = [0, 1, 4]
    else:
        seed_list = [0, 1, 2, 3, 4]
    for seed in seed_list:
        path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy()
        if methods_name in ['fabolas']:
            SR = max_dic[data_name] - data['incumbents'].to_numpy()
        elif methods_name in ['smac']:
            SR = (max_dic[data_name] - data['incumbents'].to_numpy())
        else:
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
    if methods_name in ['fabolas', 'smac']:
        new_method_name = methods_name
        lable_name.append(new_method_name)
    else:
        new_method_name = methods_name + '_con'
        lable_name.append(new_method_name)
    
    ll = plt.plot(cost_x, mean + add_dict[data_name], ls='solid', color=Dic[new_method_name][0],
                   label=Dic[new_method_name][2],
                   marker=Dic[new_method_name][1], markersize=7,markevery=7)
    plt.fill_between(cost_x,
                     mean + add_dict[data_name] - 0.96 * var,
                     mean + add_dict[data_name] + 0.96 * var,
                     alpha=0.05, color=Dic[new_method_name][0])
    # line.append(ll)
    plt.xlabel("Cost", fontsize=20)
    plt.ylabel("Simple regret", fontsize=20)

# plt.yscale('log')
plt.xlim(lim_x[data_name][0], lim_x[data_name][1])
plt.ylim(lim_y[data_name][0], lim_y[data_name][1])

label = [Dic[i][-1] for i in lable_name]
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper_yx') + '/' +'CMF_'+ data_name +'_'+ cost_name +'_SR.pdf', bbox_inches='tight')
