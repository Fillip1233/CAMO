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
Dic = { 'fabolas':['#808000', "*", "Fabolas", 'solid'],
        'smac':['#006400', "*", "SMAC3", 'solid'],
        
        'GP_UCB': ['#4169E1', "^", "BOCA", 'solid'],
        'GP_cfKG': ['#4169E1', "X", "cfKG", 'solid'],
        
        'CMF_CAR_UCB': ['#FF0000', "^", "CAMO-UCB", 'dashed'], # red
        'CMF_CAR_cfKG': ['#FF0000', "X", "CAMO-cfKG", 'dashed'],
        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAMO-DKL-UCB", 'dashed'], # orange
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "X", "CAMO-DKL-cfKG", 'dashed'],
        }

data_name = 'Forrester'
cost_name = 'pow_10'

max_dic = {'Forrester': 50, 'non_linear_sin':0,'Branin': 55,'Currin': 14,'Park': 2.2}
opt_dic = {'Forrester': 48.4998, 'non_linear_sin':0.133398,'Branin': 54.7544,'Currin': 13.7978,'Park': 2.1736}
add_dic = {'Forrester': 7 , 'non_linear_sin': 0.15,'Branin': 0,'Currin': 0,'Park': 0}
lim_x = {'Forrester': [48, 135], 'non_linear_sin': [48, 135], 'Branin':[48,140],'Currin':[48,140],'Park':[48,140]}
lim_y = {'Forrester': [0, 55], 'non_linear_sin': [0,0.3], 'Branin':[2,12], 'Currin':[0,1.75],'Park':[0.2,1.4]}
seed_dic = {'Forrester': [0,1,2,3], 'non_linear_sin': [0,1,2,3,4], 'Branin':[1,2], 'Currin':[3,5],'Park':[0,1,4]}

# methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG', 'ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG', 'GP_UCB', 'GP_EI', 'GP_cfKG']
cmf_methods_name_list = ['fabolas','smac','GP_UCB','GP_cfKG','CMF_CAR_UCB', 'CMF_CAR_cfKG','CMF_CAR_dkl_UCB','CMF_CAR_dkl_cfKG']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes = axes.flatten()

seeds = [1,2]

for i, seed in enumerate(seeds):
    ax = axes[i]
    ax.set_yscale('log')
    for methods_name in cmf_methods_name_list:
        ct = []
        tem = []
        path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy().reshape(-1, 1)
        if methods_name in ['fabolas',"smac"]:
            incumbents = max_dic[data_name] - data['incumbents'].to_numpy()
        else:    
            incumbents = data['SR'].to_numpy().reshape(-1, 1)
        tem.append(incumbents)
        ct.append(cost)
        tem = np.array(tem)
        mean = np.mean(tem, axis=0)
        var = np.std(tem, axis=0)
        ll = ax.plot(ct[0].flatten(), mean.flatten() + add_dic[data_name], ls='dashed', color=Dic[methods_name][0],
                     label=Dic[methods_name ][2],
                     marker=Dic[methods_name ][1], markersize=6)
        ax.fill_between(ct[0].flatten(),
                        mean.flatten() + add_dic[data_name] - 0.96 * var.flatten(),
                        mean.flatten() + add_dic[data_name] + 0.96 * var.flatten(),
                        alpha=0.1, color=Dic[methods_name ][0])

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
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper_seed') + '/' + 'CMF_' + data_name + '_' + cost_name + '_seedsall.pdf', bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper_seed') + '/' + 'CMF_' + data_name + '_' + cost_name + '_seedsall.eps', bbox_inches='tight')
