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
Dic = { 'fabolas':['#808000', "*", "Fabolas", 'solid'],
        'smac':['#006400', "*", "SMAC3", 'solid'],
        
        'GP_UCB_con': ['#4169E1', "^", "BOCA", 'solid'],
        'GP_cfKG_con': ['#4169E1', "X", "cfKG", 'solid'],
        
        'CMF_CAR_UCB_con': ['#FF0000', "^", "CAMO-BOCA", 'dashed'], # red
        'CMF_CAR_cfKG_con': ['#FF0000', "X", "CAMO-cfKG", 'dashed'],
        'CMF_CAR_dkl_UCB_con': ['#FF5E00', "^", "CAMO-DKL-BOCA", 'dashed'], # orange
        'CMF_CAR_dkl_cfKG_con': ['#FF5E00', "X", "CAMO-DKL-cfKG", 'dashed'],
        }


add_dic = {'VibratePlate2': 0, 'HeatedBlock': 1}
lim_x = {'VibratePlate2': [48, 150], 'HeatedBlock': [48, 150]}
lim_y = {'VibratePlate2': [28, 41.8], 'HeatedBlock': [0,1.44]}
seed_dic = {'VibratePlate2': [0,1,2,4,5], 'HeatedBlock': [1,2,3,6]}

cmf_methods_name_list = ['GP_UCB', 'GP_cfKG', 
                         'CMF_CAR_UCB','CMF_CAR_cfKG',
                         'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']

data_list = ['VibratePlate2', 'HeatedBlock']
cost_name = 'pow_10'
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

for kk in range(2):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['cost'].to_numpy()
            SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        
        methods_name = methods_name + '_con'

        ll = axs[0, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=17)
        axs[0, kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        
    axs[0, kk].set_xlabel("Cost", fontsize=25)
    axs[0, kk].set_ylabel("Simple regret", fontsize=25)
    axs[0, kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[0, kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[0, kk].tick_params(axis='both', labelsize=20)
    axs[0, kk].grid()

lim_x = {'VibratePlate2': [80, 350], 'HeatedBlock': [22, 200]}
for kk in range(2):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost_tem = data['operation_time'].to_numpy()
            cost = np.cumsum(cost_tem)
            SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        
        methods_name = methods_name + '_con'

        ll = axs[1, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=17)
        axs[1, kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        
    axs[1, kk].set_xlabel("Wall clock time (s)", fontsize=25)
    axs[1, kk].set_ylabel("Simple regret", fontsize=25)
    axs[1, kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[1, kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[1, kk].tick_params(axis='both', labelsize=20)
    axs[1, kk].grid()


lines, labels = axs[1, 1].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.11), fancybox=True, mode='normal', ncol=3, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper_yx') + '/' + 'CMF_real_' + cost_name +'_SR_together_4.pdf', bbox_inches='tight')