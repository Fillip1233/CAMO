import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
       'GP_UCB': ['#4169E1', "^", "MF-GP-UCB"],
       'GP_EI': ['#4169E1', "s", "GP-EI"],
       'GP_cfKG': ['#4169E1', "o", "cfKG"],
       'GP_UCB_con': ['#4169E1', "^", "Con_GP-UCB"],
       'GP_cfKG_con': ['#4169E1', "o", "Con_GP-cfKG"],
       'CMF_CAR_UCB_con': ['#FF0000', "^", "CAMO-UCB"], # red
       'CMF_CAR_cfKG_con': ['#FF0000', "o", "CAMO-cfKG"],
       'CMF_CAR_dkl_UCB_con': ['#FF5E00', "^", "CAMO_dkl-UCB"], # orange
       'CMF_CAR_dkl_cfKG_con': ['#FF5E00', "o", "CAMO_dkl-cfKG"],
      }

data_name = 'Forrester'
cost_name = 'pow_10'

max_dic = {'non_linear_sin': 0, 'forrester': 50}
add_dict = {'Forrester': 7, 'non_linear_sin': 0.15}
lim_x = {'Forrester': [48, 135], 'non_linear_sin': [50, 135]}
lim_y = {'Forrester': [0, 55], 'non_linear_sin': [0, 0.32]}

methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG', 'ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG', 'GP_UCB', 'GP_EI', 'GP_cfKG']
cmf_methods_name_list = ['CMF_CAR_UCB', 'CMF_CAR_cfKG', 'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

seeds = [0, 1, 2, 3]

for i, seed in enumerate(seeds):
    ax = axes[i]
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
        ll = ax.plot(ct[0].flatten(), mean.flatten() + add_dict[data_name], ls='dashed', color=Dic[methods_name][0],
                     label=Dic[methods_name][2],
                     marker=Dic[methods_name][1], markersize=6)
        ax.fill_between(ct[0].flatten(),
                        mean.flatten() + add_dict[data_name] - 0.96 * var.flatten(),
                        mean.flatten() + add_dict[data_name] + 0.96 * var.flatten(),
                        alpha=0.1, color=Dic[methods_name][0])

    for methods_name in cmf_methods_name_list:
        ct = []
        tem = []
        path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy().reshape(-1, 1)
        if methods_name == 'fabolas':
            incumbents = max_dic['data_name'] - data['SR'].to_numpy()
        else:    
            incumbents = data['SR'].to_numpy().reshape(-1, 1)
        tem.append(incumbents)
        ct.append(cost)
        tem = np.array(tem)
        mean = np.mean(tem, axis=0)
        var = np.std(tem, axis=0)
        ll = ax.plot(ct[0].flatten(), mean.flatten() + add_dict[data_name], ls='dashed', color=Dic[methods_name + '_con'][0],
                     label=Dic[methods_name + '_con'][2],
                     marker=Dic[methods_name + '_con'][1], markersize=6)
        ax.fill_between(ct[0].flatten(),
                        mean.flatten() + add_dict[data_name] - 0.96 * var.flatten(),
                        mean.flatten() + add_dict[data_name] + 0.96 * var.flatten(),
                        alpha=0.1, color=Dic[methods_name + '_con'][0])

    ax.set_xlabel("Cost", fontsize=15)
    ax.set_ylabel("Simple regret", fontsize=15)
    ax.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    ax.tick_params(axis='both', labelsize=12)
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid()

lines, labels = ax.get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.51, 1.25), fancybox=True, mode='normal', ncol=4, markerscale = 1.3, fontsize=25)
for line in leg.get_lines():
    line.set_linewidth(2.5)
plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper_seed') + '/' + 'DMF_' + data_name + '_' + cost_name + '_seedsall.pdf', bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper_seed') + '/' + 'DMF_' + data_name + '_' + cost_name + '_seedsall.eps', bbox_inches='tight')
