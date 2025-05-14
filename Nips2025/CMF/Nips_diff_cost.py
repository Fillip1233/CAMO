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
Dic = {
    'fabolas': ['#808000', "*", "Fabolas", 'solid'],  # 深红 + 五边形
    'smac': ['#2E8B57', "H", "SMAC3", 'dashdot'],     # 深绿色 + 六边形
    
    'GP_UCB': ['#1E90FF', "s", "BOCA", 'solid'],       # 深蓝 + 方形
    'GP_cfKG': ['#1E90FF', "o", "cfKG", 'solid'],      # 深蓝 + 圆形
    
    'CMF_CAR_UCB': ['#FF4500', "D", "CAMO-BOCA", 'dashed'],   # 橙红 + 菱形
    'CMF_CAR_cfKG': ['#FF4500', "v", "CAMO-cfKG", 'dashed'],  # 橙红 + 下三角形
    'CMF_CAR_dkl_UCB': ['#BA55D3', "p", "CAMO-DKL-BOCA", 'solid'],  # 淡紫色 + 五边形
    'CMF_CAR_dkl_cfKG': ['#BA55D3', "^", "CAMO-DKL-cfKG", 'dashed'],  # 淡紫色 + 上三角形
}



max_dic = {'Branin': 55,'Currin': 13.798,'Park': 2.2, 'VibratePlate': 250, 'HeatedBlock': 2,'bohachevsky': 72.15,'borehole':0}
add_dict = {'Branin': 3,'Currin': 0.02,'Park': 0.3, 'VibratePlate': 0, 'HeatedBlock': 0,'bohachevsky': 4,'borehole':0}
cost_lim_y = {'Branin': [2,12], 'Currin': [0, 3], 'Park': [0.2, 1.4],'bohachevsky': [0, 32], 'borehole': [0, 0.5]}
cost_lim_x = {'Branin': [13, 300], 'Currin': [13, 150], 'Park': [13, 300],'bohachevsky': [13, 150], 'borehole': [13, 300]}


##Branin
# data_name = 'Branin'
# seed_dic ={'pow_10':[2,4,5,7,8],'linear':[4,5,7,8],'log':[2,3,5,7,8]}
##Currin
data_name = 'Currin'
# seed_dic ={'pow_10':[2,3,4,6,8,9,11,12,13,16,17,18,19,20,21,22,23,24,25,26,28],'linear':[2,3,4,6,8,9,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28],
#            'log':[2,3,4,6,8,9,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28]}
seed_dic ={'pow_10':[3,4,6,8,9,11,12,13,16,17,18,19,20,21,22,23,24,25,26,28],'linear':[3,4,6,8,9,11,12,13,16,17,18,19,20,21,22,23,24,25,26,28],
           'log':[3,4,6,8,9,11,12,13,16,17,18,19,20,21,22,23,24,25,26,28]}
##Park
# data_name = 'Park'
# seed_dic ={'pow_10':[0,2,4,5,7],'linear':[0,4,7,9],'log':[0,1,2,3,4,7]}

#bohachevsky
# data_name = 'bohachevsky'
# seed_dic ={'pow_10':[0,1,2,3,4,7,9,10,11,12,15,17,19,21,22,24,26,27],'linear':[0,1,2,3,4,7,8,10,11,12,15,17,19,21,22,24],'log':[0,1,2,3,4,7,9,10,11,12,15,17,19,21,22,24]}

##
# data_name = 'borehole'
# seed_dic ={'pow_10':[0,1,2,6],'linear':[0,1,2,6],'log':[0,1,2,6]}

methods_name_list = [ 
                     'GP_UCB', 
                     'GP_cfKG',
                     'fabolas',
                    'smac',
                     'CMF_CAR_UCB',
                    #  'CMF_CAR_cfKG',
                        
                    #  'CMF_CAR_dkl_UCB',
                    #  'CMF_CAR_dkl_cfKG'
                    ]
# baseline_list = ['fabolas','smac']

cost_list = ['log', 'linear', 'pow_10']
cost_label_dic = {'log': 'Log', 'linear': 'Linear', 'pow_10': 'Power 10'}
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

Exp_marker = "Norm_res"

for kk in range(3):
    cost_name = cost_list[kk]
    for methods_name in methods_name_list:
        print(methods_name)
        cost_collection = []
        # SR_collection = []
        inter_collection = []  
        for seed in seed_dic[cost_name]:
            path = os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Exp_results',Exp_marker,
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            # cost = data['cost'].to_numpy().reshape(-1, 1)
            # SR = data['SR'].to_numpy().reshape(-1, 1)
            cost = data['cost'].to_numpy()
            
            if methods_name == 'fabolas':
                SR = max_dic[data_name] - data['incumbents'].to_numpy()
            else:
                SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            # inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            # SR_collection.append(SR)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        if  cost_name == 'log':
            if methods_name in ['fabolas']:
                makervery_index = 120
            else:
                makervery_index = 360
        elif  cost_name == 'linear':
            makervery_index = 30
        else:
            makervery_index = 30
        ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12, markevery=makervery_index,linewidth=2.2)
        # axs[kk].fill_between(cost_x,
        #                 mean + add_dict[data_name] - 0.96 * var,
        #                 mean + add_dict[data_name] + 0.96 * var,
        #                 alpha=0.05, color=Dic[methods_name][0])
        
    # for methods_name in baseline_list:
    #     print(methods_name)
    #     cost_collection = []
    #     # SR_collection = []
    #     inter_collection = []
    #     for seed in [0,1,2,3,4]:
    #         path = os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Exp_results',Exp_marker,
    #                             data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
    #         data = pd.DataFrame(pd.read_csv(path))
    #         # cost = data['cost'].to_numpy().reshape(-1, 1)
    #         # SR = data['SR'].to_numpy().reshape(-1, 1)
    #         cost = data['cost'].to_numpy()
            
    #         if methods_name == 'fabolas':
    #             SR = max_dic[data_name] - data['incumbents'].to_numpy()
    #         else:
    #             SR = data['SR'].to_numpy()
    #         inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
    #         # inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
    #         # SR_collection.append(SR)
    #         cost_collection.append(cost)
    #         inter_collection.append(inter)

    #     cost_x = np.unique(np.concatenate(cost_collection))
    #     SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
    #     SR_new = np.array(SR_new)
    #     mean = np.mean(SR_new, axis=0)
    #     var = np.std(SR_new, axis=0)
    #     if  cost_name == 'log':
    #         if methods_name in ['CMF_CAR_UCB','CMF_CAR_cfKG','CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']:
    #             makervery_index = 90
    #         else:
    #             makervery_index = 90
    #     elif  cost_name == 'linear':
    #         makervery_index = 20
    #     else:
    #         makervery_index = 14
    #     ll = axs[kk].plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
    #                 label=Dic[methods_name][2],
    #                 marker=Dic[methods_name][1], markersize=12, markevery=makervery_index)
    #     axs[kk].fill_between(cost_x,
    #                     mean + add_dict[data_name] - 0.96 * var,
    #                     mean + add_dict[data_name] + 0.96 * var,
    #                     alpha=0.05, color=Dic[methods_name][0])
        
    # plt.yscale('log')
    axs[kk].set_xlabel("Cost: " + cost_label_dic[cost_name], fontsize=25)
    axs[kk].set_ylabel("Simple regret", fontsize=25)
    # axs[kk].set_yscale('log')
    axs[kk].set_xlim(cost_lim_x[data_name][0], cost_lim_x[data_name][1])
    axs[kk].set_ylim(cost_lim_y[data_name][0], cost_lim_y[data_name][1])
    axs[kk].tick_params(axis='both', labelsize=20)
    axs[kk].grid()

# label = [Dic[i][-1] for i in methods_name_list]
# label = label + [Dic[i+'_con'][-1] for i in cmf_methods_name_list]

# 共享图例
lines, labels = axs[0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, mode='normal', ncol=5, markerscale = 1.3, fontsize=25)

# change the line width for the legend
# for line in leg.get_lines():
#     line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Nips2025', 'CMF', 'Graphs') + '/' +'CMF_' + data_name + '_cost.png', bbox_inches='tight')
