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

def draw_plots(axs, data_name, cmf_methods_name_list, exp_marker):
    label_name = []
    for methods_name in cmf_methods_name_list:
        cost_collection = []
        inter_collection = []
        
        # if data_name in ['bohachevsky']:
        #     smac_dict =[1,17,19,21,22]
        # else:
        #     smac_dict = [0,1,2,3,4]
        if methods_name in ['smac']:
            for seed in seed_dic[data_name]:
                path = os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Exp_results',exp_marker,
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
                data = pd.DataFrame(pd.read_csv(path))
                cost = data['cost'].to_numpy()
                SR = data['SR'].to_numpy()
                inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
                cost_collection.append(cost)
                inter_collection.append(inter)
        else:
            for seed in seed_dic[data_name]:
                path = os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Exp_results',exp_marker,
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
                data = pd.DataFrame(pd.read_csv(path))
                cost = data['cost'].to_numpy()
                if methods_name in ['fabolas', 'smac']:
                    SR = max_dic[data_name] - data['incumbents'].to_numpy()
                else:
                    SR = data['SR'].to_numpy()
                inter = interp1d(cost, SR, kind='linear', fill_value="extrapolate")
                cost_collection.append(cost)
                inter_collection.append(inter)

        cost_x = np.unique(np.concatenate(cost_collection))
        SR_new = [inter_collection[i](cost_x) for i in range(len(inter_collection))]
        SR_new = np.array(SR_new)
        mean = np.mean(SR_new, axis=0)
        # if methods_name in ['fabolas', 'smac']:
        var = np.std(SR_new, axis=0)
        # else:
        #     mean = np.insert(mean, 0, opt_dic[data_name])
        #     cost_x = np.insert(cost_x, 0, 50)
        #     var = np.std(SR_new, axis=0)
        #     var = np.insert(var, 0, 0.5)

        new_method_name = methods_name
        label_name.append(new_method_name)
        
        axs.plot(cost_x, mean + add_dic[data_name], ls=Dic[new_method_name][-1], color=Dic[new_method_name][0],
                    label=Dic[new_method_name][2],
                    marker=Dic[new_method_name][1], markersize=12, markevery=30)
        axs.fill_between(cost_x,
                        mean + add_dic[data_name] - 0.96 * var,
                        mean + add_dic[data_name] + 0.96 * var,
                        alpha=0.07, color=Dic[new_method_name][0])
        # markevery_indices = range(0, len(cost_x), 180)
        # errorbar_x = [cost_x[i] for i in markevery_indices]
        # errorbar_y = [mean[i] + add_dic[data_name] for i in markevery_indices]
        # errorbar_yerr = [0.96 * var[i] for i in markevery_indices]

        # axs.errorbar(
        #     errorbar_x, 
        #     errorbar_y, 
        #     yerr=errorbar_yerr, 
        #     fmt=Dic[new_method_name][1],  # 鏍囪鏍峰紡
        #     markersize=12, 
        #     color=Dic[new_method_name][0], 
        #     capsize=5,  # 璇樊鏉＄鐐归暱锟�??
        #     alpha=0.8  # 绾挎潯閫忔槑锟�??
        #     )


    axs.set_xlabel("Cost", fontsize=25)
    axs.set_ylabel("Simple regret", fontsize=25)
    axs.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs.set_ylim(lim_y[data_name][0], lim_y[data_name][1])
    axs.tick_params(axis='both', labelsize=20)
    axs.grid()

    return label_name

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


data_list = ['Branin', 'Currin', 'Park', 'non_linear_sin', 'Forrester','bohachevsky']
cost_name = 'pow_10'

max_dic = {'forrester': 48.4495, 'non_linear_sin':0.03338,'Branin': 54.75,'Currin': 13.798,'Park': 2.174,'himmelblau':303.5,'bohachevsky': 72.15}
add_dic = {'forrester': 0.8 , 'non_linear_sin': 0,'Branin': 0.9,'Currin': 0.01,'Park': 0.1, 'himmelblau': 1,'bohachevsky': 4}
lim_x = {'forrester': [48, 300], 'non_linear_sin': [48, 300], 'Branin':[48,300],'Currin':[48,300],'Park':[48,300], 'himmelblau':[48, 300],'bohachevsky':[48,300]}
lim_y = {'forrester': [0, 40], 'non_linear_sin': [0,0.04], 'Branin':[0,10], 'Currin':[0,3],'Park':[0,1.2],'himmelblau':[0, 150],'bohachevsky':[0,32]}
# seed_dic = {'forrester': [0,1,2,3,5,9], 'non_linear_sin': [1,4,5,6,9], 'Branin':[2,4,5,7,8], 'Currin':[2,3,4,5,6,7,8],'Park':[0,2,4,5,7],'himmelblau':[0,1,2,3,8],
#             'bohachevsky':[1,17,19,21,22]}
seed_dic ={
        # 'Currin':[6,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,28],##24
           'Currin':[3,4,6,8,9,11,12,13,16,17,18,19,20,21,22,23,24,25,26,28],
           'Branin':[4,5,7,8,10,11,13,14,15,16,17,18,19,20,21,22,24,25,27,29],##20
           'Park':[0,1,2,3,4,5,6,7,8,9,11,14,16,20,21,22,23,24,25,28,29],##21
           'non_linear_sin':[1,3,5,6,7,11,14,16,17,18,19,20,21,22,24,25,26,27,29],#19
           'forrester':[2,4,5,9,11,13,14,15,16,17,18,19,21,22,23,24,25,27,28,29],##21
           'bohachevsky':[0,1,2,3,4,7,9,10,11,12,15,17,19,21,22,24,26,27]}##18

cmf_methods_name_list = [  
                        'GP_UCB','GP_cfKG',
                        'CMF_CAR_UCB','CMF_CAR_cfKG',
                        'CMF_CAR_dkl_UCB','CMF_CAR_dkl_cfKG',
                        # 'fabolas',
                        'smac'
                         ]
Exp_marker = 'Norm_res'


fig = plt.figure(figsize=(30, 16))

# 创建图形
gs = gridspec.GridSpec(2, 6) # 创立2 * 6 网格
gs.update(wspace=0.9)

# 对第一行进行绘�?
ax1 = plt.subplot(gs[0,  :2]) # gs(哪一行，绘制网格列的范围)
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[0, 4:6])

# 对第二行进行绘制
ax4 = plt.subplot(gs[1, :2])
ax5 = plt.subplot(gs[1, 2:4])
ax6 = plt.subplot(gs[1, 4:6])

# 开始画�?
draw_plots(ax1, 'Branin', cmf_methods_name_list, Exp_marker)
ax1.text(0.5, 1.02, 'Branin', transform=ax1.transAxes, ha='center', fontsize=25)
draw_plots(ax2, 'Currin', cmf_methods_name_list, Exp_marker)
ax2.text(0.5, 1.02, 'Currin', transform=ax2.transAxes, ha='center', fontsize=25)
draw_plots(ax3, 'Park', cmf_methods_name_list, Exp_marker)
ax3.text(0.5, 1.02, 'Park', transform=ax3.transAxes, ha='center', fontsize=25)
draw_plots(ax4, 'non_linear_sin', cmf_methods_name_list, Exp_marker)
ax4.text(0.5, 1.02, 'Non-linear Sin', transform=ax4.transAxes, ha='center', fontsize=25)
draw_plots(ax5, 'forrester', cmf_methods_name_list, Exp_marker)
ax5.text(0.5, 1.02, 'Forrester', transform=ax5.transAxes, ha='center', fontsize=25)
draw_plots(ax6, 'bohachevsky', cmf_methods_name_list, Exp_marker)
ax6.text(0.5, 1.02, 'Bohachevsky', transform=ax6.transAxes, ha='center', fontsize=25)

lines, labels = ax1.get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), fancybox=True, mode='normal', ncol=4, markerscale = 1.5, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
# plt.savefig(os.path.join(sys.path[-1], 'Rebuttal_Experiment', 'CMF', 'Graph_show') + '/' + 'CMF_' + cost_name +'_SR_together.pdf', bbox_inches='tight')
# plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'exp_521') + '/' + 'CMF_' + cost_name +'_SR_together.pdf', bbox_inches='tight')
# plt.savefig(os.path.join(sys.path[-1], 'ICLR_exp3', 'CMF', 'Graphs') + '/' + 'CMF_' + cost_name +'_SR_6.pdf', bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'ICLR_exp3', 'CMF', 'Graphs') + '/' + 'CMF_' + cost_name +'_SR_6.png', bbox_inches='tight')