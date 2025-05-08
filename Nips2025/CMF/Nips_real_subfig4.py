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


add_dic = {'VibratePlate': 0, 'HeatedBlock': 0}
max_dic = {'VibratePlate': 250, 'HeatedBlock': 2}
lim_x = {'VibratePlate': [48, 300], 'HeatedBlock': [48, 300]}
lim_y = {'VibratePlate': [28, 41.8], 'HeatedBlock': [0.3,1]}

seed_dic = {'VibratePlate': [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], 
            'HeatedBlock': [0,1,2,3,4,5,7,8,9,10,13,14,15,16,18,19,20,22,24,25,26,28,29]}

cmf_methods_name_list = [
                        'GP_UCB', 'GP_cfKG', 
                         'CMF_CAR_UCB','CMF_CAR_cfKG',
                        #  'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG'
                         ]
baseline_list = [
    'fabolas',
    'smac']

data_list = ['VibratePlate', 'HeatedBlock']
cost_name = 'pow_10'
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

for kk in range(2):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'ICLR_exp2', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['cost'].to_numpy()
            SR = data['SR'].to_numpy()
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            # inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        

        ll = axs[0, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=60)
        axs[0, kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        axs[0,kk].text(0.5, 1.02, data_name, transform=axs[0,kk].transAxes, ha='center', fontsize=25)
        
    for methods_name in baseline_list:
        cost_collection = []
        inter_collection = []
    
        for seed in seed_dic[data_name]:
            
            path = os.path.join(sys.path[-1], 'ICLR_exp2', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost = data['cost'].to_numpy()
            if methods_name == 'fabolas':
                SR = max_dic[data_name] - data['incumbents'].to_numpy()
            else:
                SR = data['SR'].to_numpy() + 1.27
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            # inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        
        methods_name = methods_name

        ll = axs[0, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=17)
        axs[0, kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        axs[0,kk].text(0.5, 1.02, data_name, transform=axs[0,kk].transAxes, ha='center', fontsize=25)
        
    axs[0, kk].set_xlabel("Cost", fontsize=25)
    axs[0, kk].set_ylabel("Simple regret", fontsize=25)
    # axs[0, kk].set_yscale('log')
    axs[0, kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[0, kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[0, kk].tick_params(axis='both', labelsize=20)
    axs[0, kk].grid()
    
    

# lim_x = {'VibratePlate': [80, 350], 'HeatedBlock': [22, 200]}
for kk in range(2):
    data_name = data_list[kk]
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'ICLR_exp2', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            cost_tem = data['operation_time'].to_numpy()
            cost = np.cumsum(cost_tem)
            SR = data['SR'].to_numpy()
            # inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)
        

        ll = axs[1, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=60)
        axs[1, kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
    
    for methods_name in baseline_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
            path = os.path.join(sys.path[-1], 'ICLR_exp2', 'CMF', 'Exp_results','Norm_res',
                                data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
            data = pd.DataFrame(pd.read_csv(path))
            if methods_name == 'fabolas':
                cost_tem = data['operation_time'].to_numpy()
            elif methods_name == 'smac' and data_name == 'VibratePlate':
                cost_tem = data['time'].to_numpy()
            else:
                cost_tem = data['time'].to_numpy()
            
            if methods_name == 'fabolas':
                cost = cost_tem
            else:
                cost = np.cumsum(cost_tem)
            if methods_name == 'smac' and data_name == 'VibratePlate':
                cost = cost+106
            elif methods_name == 'smac' and data_name == 'HeatedBlock':
                cost = cost+50
            
            if methods_name == 'fabolas':
                SR = max_dic[data_name] - data['incumbents'].to_numpy()
            elif methods_name == 'smac' and data_name == 'HeatedBlock':
                SR = data['SR'].to_numpy()+ 1.27
            else:
                SR = data['SR'].to_numpy()
            # inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            inter = interp1d(cost, SR, kind='linear', fill_value=np.nan, bounds_error=False)
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        var = np.std(SR_new, axis=0)

        ll = axs[1, kk].plot(cost_x, mean + add_dic[data_name], ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=12,markevery=7)
        axs[1, kk].fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[methods_name][0])
        
    axs[1, kk].set_xlabel("Wall clock time (s)", fontsize=25)
    axs[1, kk].set_ylabel("Simple regret", fontsize=25)
    axs[1, kk].set_xscale('log')
    # axs[0, kk].set_yscale('log')
    # axs[1, kk].set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs[1, kk].set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs[1, kk].tick_params(axis='both', labelsize=20)
    axs[1, kk].grid()


lines, labels = axs[1, 1].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.11), fancybox=True, mode='normal', ncol=6, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
# plt.savefig(os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Graphs') + '/' + 'CMF_real_' + cost_name +'_SR_together_4.pdf', bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Nips2025', 'CMF', 'Graphs') + '/' + 'CMF_real_' + cost_name +'_SR_together_4.png', bbox_inches='tight')