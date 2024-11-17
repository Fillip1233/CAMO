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
Dic = {'AR_UCB': ['#000080', "^", "AR-UCB", "solid"],
       'AR_EI': ['#000080', "s", "AR-EI", "solid"],
       'AR_cfKG': ['#000080', "X", "AR-cfKG", "solid"],
       'ResGP_UCB': ['#00CCFF', "^", "ResGP-UCB", "solid"],
       'ResGP_EI': ['#00CCFF', "s", "ResGP-EI", "solid"],
       'ResGP_cfKG': ['#00CCFF', "X", "ResGP-cfKG", "solid"],
        
        'DNN_MFBO': ['#228B22', "*", "DNN", "solid"],
        
        'GP_UCB': ['#4169E1', "^", "GP-UCB", "solid"],
        'GP_EI': ['#4169E1', "s", "GP-EI", "solid"],
        'GP_cfKG': ['#4169E1', "X", "GP-cfKG", "solid"],
        'GP_UCB_con': ['#4169E1', "^", "Con_GP-UCB", "solid"],
        'GP_cfKG_con': ['#4169E1', "X", "Con_GP-cfKG", "solid"],
        
        'CMF_CAR_UCB_con': ['#FF0000', "^", "CAMO-UCB", "dashed"], # red
        'CMF_CAR_cfKG_con': ['#FF0000', "X", "CAMO-cfKG", "dashed"],
        'CMF_CAR_dkl_UCB_con': ['#FF5E00', "^", "CAMO-DKL-UCB", "dashed"], # orange
        'CMF_CAR_dkl_cfKG_con': ['#FF5E00', "X", "CAMO-DKL-cfKG", "dashed"],
        }



max_dic = {'non_linear_sin':0, 'Forrester': 50}
add_dict = {'Forrester': 7 ,'non_linear_sin': 0.15}
lim_x = {'Forrester': [48, 135], 'non_linear_sin': [50, 135]}
lim_y = {'Forrester': [0, 55], 'non_linear_sin': [0, 0.32]}



methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG', 'ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG', 'GP_UCB', 'GP_EI', 'GP_cfKG']
cmf_methods_name_list = ['CMF_CAR_UCB','CMF_CAR_cfKG',
                        #  'GP_UCB', 'GP_cfKG',
                         'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']

data_list = ['non_linear_sin', 'Forrester']
cost_name = 'pow_10'
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
for kk in range(2):
    data_name = data_list[kk]
    for methods_name in methods_name_list:
        cost_collection = []
        # SR_collection = []
        inter_collection = []
        for seed in [0, 1, 2,3,4]:
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
        ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12, markevery=10)
        axs[kk].fill_between(cost_x,
                        mean + add_dict[data_name] - 0.96 * var,
                        mean + add_dict[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        
    for methods_name in ['DNN_MFBO']:
        cost_collection = []
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
        ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=10)
        axs[kk].fill_between(cost_x,
                        mean + add_dict[data_name] - 0.96 * var,
                        mean + add_dict[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])


    for methods_name in cmf_methods_name_list:
        cost_collection = []
        # SR_collection = []
        inter_collection = []
        for seed in [0, 1, 2, 3, 4]:
            path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            # cost = data['cost'].to_numpy().reshape(-1, 1)
            # SR = data['SR'].to_numpy().reshape(-1, 1)
            cost = data['cost'].to_numpy()
            if methods_name == 'fabolas':
                SR = max_dic['data_name'] - data['SR'].to_numpy()
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
        ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name+'_con'][-1], color=Dic[methods_name+'_con'][0],
                    label=Dic[methods_name+'_con'][2],
                    marker=Dic[methods_name+'_con'][1], markersize=12,markevery=10)
        axs[kk].fill_between(cost_x,
                        mean + add_dict[data_name] - 0.96 * var,
                        mean + add_dict[data_name] + 0.96 * var,
                        alpha=0.07, color=Dic[methods_name+'_con'][0])
        
    # plt.yscale('log')
    axs[kk].set_xlabel("Cost", fontsize=25)
    axs[kk].set_ylabel("Simple regret", fontsize=25)
    axs[kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[kk].tick_params(axis='both', labelsize=20)
    axs[kk].grid()

# label = [Dic[i][-1] for i in methods_name_list]
# label = label + [Dic[i+'_con'][-1] for i in cmf_methods_name_list]

# 共享图例
lines, labels = axs[0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.52, 1.35), fancybox=True, mode='normal', ncol=5, markerscale = 1.3, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper_yx') + '/' +'DMF_together.pdf', bbox_inches='tight')