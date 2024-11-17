import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerLine2D
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


def get_data(type, data_name, method_name, file_name):
    path = os.path.join(sys.path[-1], 'exp', type, data_name, method_name, file_name + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    return data

def draw_plots(axs, data_name, cmf_methods_name_list):
    label_name = []
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []

        for seed in seed_dic[data_name]:
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
            inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
            cost_collection.append(cost)
            inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        if methods_name in ['fabolas', 'smac']:
            var = np.std(SR_new, axis=0)
        else:
            mean = np.insert(mean, 0, opt_dic[data_name])
            cost_x = np.insert(cost_x, 0, 50)
            var = np.std(SR_new, axis=0)
            var = np.insert(var, 0, 0.5)
        if methods_name in ['fabolas', 'smac']:
            new_method_name = methods_name
            label_name.append(new_method_name)
        else:
            new_method_name = methods_name + '_con'
            label_name.append(new_method_name)
        
        axs.plot(cost_x, mean + add_dic[data_name], ls=Dic[new_method_name][-1], color=Dic[new_method_name][0],
                    label=Dic[new_method_name][2],
                    marker=Dic[new_method_name][1], markersize=12, markevery=17)
        axs.fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.05, color=Dic[new_method_name][0])


    axs.set_xlabel("Cost", fontsize=25)
    axs.set_ylabel("Simple regret", fontsize=25)
    axs.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs.set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs.tick_params(axis='both', labelsize=20)
    axs.grid()

    return label_name

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

fig = plt.figure(figsize=(20, 13))

# 创建图形
gs = gridspec.GridSpec(2, 6) # 创立2 * 6 网格
gs.update(wspace=0.8)

# 对第一行进行绘制
ax1 = plt.subplot(gs[0,  :2]) # gs(哪一行，绘制网格列的范围)
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[0, 4:6])

# 对第二行进行绘制
ax4 = plt.subplot(gs[1, 1:3])
ax5 = plt.subplot(gs[1, 3:5])

# 开始画图
draw_plots(ax1, 'Branin', cmf_methods_name_list)
draw_plots(ax2, 'Currin', cmf_methods_name_list)
draw_plots(ax3, 'Park', cmf_methods_name_list)
draw_plots(ax4, 'non_linear_sin', cmf_methods_name_list)
draw_plots(ax5, 'Forrester', cmf_methods_name_list)

lines, labels = ax5.get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), fancybox=True, mode='normal', ncol=3, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper_yx') + '/' + 'CMF_' + cost_name +'_SR_together.pdf', bbox_inches='tight')
# plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'exp_521') + '/' + 'CMF_' + cost_name +'_SR_together.pdf', bbox_inches='tight')