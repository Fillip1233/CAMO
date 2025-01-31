import pandas as pd
# import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

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


max_dic = {'forrester': 48.4495, 'non_linear_sin':0.03338,'Branin': 54.75,'Currin': 13.798,'Park': 2.174,
           'himmelblau':303.5,'bohachevsky': 72.15,'borehole':244,'colvile':609.26}
add_dic = {'forrester': 0.8 , 'non_linear_sin': 0,'Branin': 0.9,'Currin': 0.01,'Park': 0.1, 
           'himmelblau': 1,'bohachevsky': 4,'colvile': 125,'borehole':4}
lim_x = {'forrester': [48, 300], 'non_linear_sin': [48, 300], 'Branin':[48,300],'Currin':[48,300],
         'Park':[48,300], 'himmelblau':[48, 300],'bohachevsky':[48,300],'borehole':[48,300],'colvile': [48, 300]}
# lim_y = {'forrester': [0, 40], 'non_linear_sin': [0,0.04], 'Branin':[0,10], 'Currin':[0,3],'Park':[0,1.2],'himmelblau':[0, 150],'bohachevsky':[0,32]}

seed_dic ={
           'Currin':[3,4,6,8],
           'Branin':[4,5,7,8],
           'Park': [0,1,2,3],
           'non_linear_sin':[1,3,5,6],
           'forrester':[2,4,5,9],
           'bohachevsky':[0,1,2,3],
           'himmelblau':[0,3,4,6,7],
           'borehole':[11,13,14,15,17],
           'colvile':[0,3,4,5,8]
           }

cmf_methods_name_list = [
                         'GP_UCB', 'GP_cfKG', 
                         'CMF_CAR_UCB',
                         'CMF_CAR_cfKG',
                         'CMF_CAR_dkl_UCB',
                         'CMF_CAR_dkl_cfKG',
                         'fabolas', 
                         'smac',
                         ]

cost_name = 'pow_10'
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    
data_name = 'colvile'

def draw_seed(axs, seed, data_name):
    label_name = []
    ymax = 0
    for methods_name in cmf_methods_name_list:
        
        path = os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Exp_results','Norm_res_60',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy()
        if methods_name in ['fabolas']:
            SR = max_dic[data_name] - data['incumbents'].to_numpy()
        else:
            SR = data['SR'].to_numpy()

        label_name.append(methods_name)

        axs.plot(cost, (SR + add_dic[data_name]), ls=Dic[methods_name][-1], color=Dic[methods_name][0],
            label=Dic[methods_name][2],
            marker=Dic[methods_name][1], markersize=12)
        if methods_name not in ['smac']:
            ymax = max(ymax, max(SR + add_dic[data_name]))
            
    # plt.yscale('log')
    axs.set_xlabel("Cost", fontsize=25)
    axs.set_ylabel("Simple regret", fontsize=25)
    axs.set_xlim(lim_x[data_name][0], lim_x[data_name][1])
    axs.set_ylim(0, ymax+7)
    axs.tick_params(axis='both', labelsize=20)
    axs.grid()

draw_seed(axs[0, 0], seed_dic[data_name][0], data_name)
draw_seed(axs[0, 1], seed_dic[data_name][1], data_name)
draw_seed(axs[1, 0], seed_dic[data_name][2], data_name)
draw_seed(axs[1, 1], seed_dic[data_name][3], data_name)

# 共享图例
lines, labels = axs[0,0].get_legend_handles_labels()
leg = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.52, 1.12), fancybox=True, mode='normal', ncol=4, markerscale = 1.3, fontsize=25)

# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'ICLR_exp', 'CMF', 'Graphs',data_name) + '_4seed.pdf', bbox_inches='tight')