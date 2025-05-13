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
    'GP_dkl_UCB': ['black', "^", "GP_dkl_UCB","dashed"],
    'GP_dkl_cfKG': ['black', "o", "GP_dkl_cfKG","dashed"],
    
    'CMF_CAR_UCB': ['#FF4500', "D", "CAMO-BOCA", 'dashed'],   # 橙红 + 菱形
    'CMF_CAR_cfKG': ['#FF4500', "v", "CAMO-cfKG", 'dashed'],  # 橙红 + 下三角形
    'CMF_CAR_dkl_UCB': ['#BA55D3', "p", "CAMO-DKL-BOCA", 'solid'],  # 淡紫色 + 五边形
    'CMF_CAR_dkl_cfKG': ['#BA55D3', "^", "CAMO-DKL-cfKG", 'dashed'],  # 淡紫色 + 上三角形
}

max_dic = {'forrester': 48.4495, 'non_linear_sin':0.03338,'Branin': 54.75,'Currin': 13.798,'Park': 2.174,'himmelblau':303.5,'bohachevsky': 72.15,'colvile':609.26}
add_dict = {'forrester': 0.8 , 'non_linear_sin': 0,'Branin': 0.86,'Currin': 0.01,'Park': 0.1, 'himmelblau': 1,'bohachevsky': 4,'VibratePlate': 0, 'HeatedBlock': 1.2,'borehole':0,'colvile':125}
lim_x = {'forrester': [48, 300], 'non_linear_sin': [48, 300],
         'Branin':[48,300],'Currin':[48,300],'Park':[0,300],'VibratePlate':[48,150],'HeatedBlock':[48,150],
         'borehole':[48,300],'booth':[48,150],'hartmann':[48,150],"bohachevsky":[48,300],'himmelblau':[48,300],'colvile':[48,300]}
lim_y = {'forrester': [0, 40], 'non_linear_sin': [0,0.04], 'Branin':[0,10], 'Currin':[0,3],'Park':[0,1.2],'himmelblau':[0, 150],'bohachevsky':[0,32],'colvile': [0, 425]}
seed_dic ={'Currin':[6,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,28],'Branin':[4,5,7,8,10,11,13,14,15,16,17,18,19,20,21,22,24,25,27,29],'Park':[0,1,2,3,4,5,6,7,8,9,11,14,16,20,21,22,23,24,25,28,29],
           'non_linear_sin':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],'forrester':[2,4,5,9,11,13,14,15,16,17,18,19,21,22,23,24,25,27,28,29],
           'bohachevsky':[0,1,2,3,4,7,9,10,11,12,15,17,19,21,22,24,26,27],'himmelblau':[0,1,2,3,4,6,7,8,9,11,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28],
           'borehole':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],'colvile':[0,3,4,5,8,9,12,13,14,15,19,20,21,22,24,25,29]}


cmf_methods_name_list = [
                    'GP_UCB', 
                    # 'GP_cfKG',
                    'CMF_CAR_UCB',
                    # 'CMF_CAR_cfKG',
                    # 'fabolas',
                    # 'smac'
                         ]

data_list = ['non_linear_sin', 'bohachevsky', 'Branin']
cost_name = 'pow_10'
fig, axs = plt.subplots(1, 1, figsize=(20, 6))
Exp_marker = 'Norm_res'

for kk in range(1):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            if data_name =='colvile':
                path = os.path.join(sys.path[-1], 'Nips2025', 'CMF', 'Exp_results','Norm_res_60',
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            else:
                path = os.path.join(sys.path[-1], 'Nips2025', 'CMF', 'Exp_results',Exp_marker,
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['cost'].to_numpy()
            if methods_name == 'fabolas':
                SR = max_dic[data_name]-data['incumbents'].to_numpy()
            else:
                SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
    
        ll = axs.plot(cost_x, mean + add_dict[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=25)
        axs.fill_between(cost_x,
                        mean + add_dict[data_name] - 0.96 * var,
                        mean + add_dict[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        
    axs.set_xlabel("Cost", fontsize=25)
    axs.set_ylabel("Simple regret", fontsize=25)
    axs.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs.set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs.tick_params(axis='both', labelsize=20)
    axs.text(0.5, 1.02, data_name, transform=axs.transAxes, ha='center', fontsize=25)
    axs.grid()


lines, labels = axs.get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.23), fancybox=True, mode='normal', ncol=5, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Nips2025', 'CMF','Graphs') + '/' + data_list[0]+ '_' + cost_name +'_SR.png', bbox_inches='tight')