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

data_list = ['non_linear_sin', 'Forrester', 'Branin', 'Currin', 'Park']
cost_name = 'pow_10'

max_dic = {'Forrester': 50, 'non_linear_sin':0,'Branin': 55,'Currin': 14,'Park': 2.2}
opt_dic = {'Forrester': 48.4998, 'non_linear_sin':0.133398,'Branin': 54.7544,'Currin': 13.7978,'Park': 2.1736}
add_dic = {'Forrester': 7 , 'non_linear_sin': 0.15,'Branin': 0,'Currin': 0,'Park': 0}
lim_x = {'Forrester': [48, 135], 'non_linear_sin': [48, 135], 'Branin':[48,140],'Currin':[48,140],'Park':[48,140]}
lim_y = {'Forrester': [0, 55], 'non_linear_sin': [0,0.3], 'Branin':[2,12], 'Currin':[0,1.75],'Park':[0.2,1.4]}
seed_dic = {'Forrester': [0,1,2,3], 'non_linear_sin': [0,1,2,3,4], 'Branin':[1,2], 'Currin':[3,5],'Park':[0,1,4]}

cmf_methods_name_list = ['fabolas', 'smac',
                         'GP_UCB', 'GP_cfKG', 
                         'CMF_CAR_UCB',
                         'CMF_CAR_dkl_UCB']

cost_name = 'pow_10'
fig, axs = plt.subplots(2, 2, figsize=(20, 12))
for kk in range(2):
    data_name = data_list[kk]

def draw_seed(axs, seed, data_name):
    label_name = []
    for methods_name in cmf_methods_name_list:
        path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy()
        if methods_name in ['fabolas', 'smac']:
            SR = max_dic[data_name] - data['incumbents'].to_numpy()
        # elif methods_name in ['smac']:
        #     SR = (max_dic[data_name] - data['incumbents'].to_numpy())
        else:
            SR = data['SR'].to_numpy()

        if methods_name in ['fabolas', 'smac']:
            continue
        else:
            SR = np.insert(SR, 0, opt_dic[data_name])
            cost_x = np.insert(cost, 0, 50)

        if methods_name in ['fabolas', 'smac']:
            new_method_name = methods_name
            label_name.append(new_method_name)
        else:
            new_method_name = methods_name + '_con'
            label_name.append(new_method_name)

        axs.plot(cost_x, np.log(SR + add_dic[data_name]), ls=Dic[new_method_name][-1], color=Dic[new_method_name][0],
            label=Dic[new_method_name][2],
            marker=Dic[new_method_name][1], markersize=12)

 
    # plt.yscale('log')
    axs.set_xlabel("Cost", fontsize=25)
    axs.set_ylabel("Simple regret", fontsize=25)
    axs.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    # axs.set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs.tick_params(axis='both', labelsize=20)
    axs.grid()

draw_seed(axs[0, 0], 1, 'Branin')
draw_seed(axs[0, 1], 2, 'Branin')
draw_seed(axs[1, 0], 3, 'Currin')
draw_seed(axs[1, 1], 5, 'Currin')

# 共享图例
lines, labels = axs[0,0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.52, 1.17), fancybox=True, mode='normal', ncol=5, markerscale = 1.3, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper_appendix') + '/Branin_Currin_seed.pdf', bbox_inches='tight')