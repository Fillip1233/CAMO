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

Dic = { 'fabolas':['#808000', "*", "Fabolas", 'solid'],
        'smac_':['#006400', "*", "SMAC3", 'solid'],
        
        'GP_UCB': ['#4169E1', "^", "MF-GP-UCB", 'solid'],
        'GP_cfKG': ['#4169E1', "X", "MF-GP-cfKG", 'solid'],
        
        'CAMO-': ['#FF0000', "^", "CAMO-UCB", 'dashed'], # red
        'CMF_CAR_cfKG': ['#FF0000', "X", "CAMO-cfKG", 'dashed'],
        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAMO-DKL-UCB", 'dashed'], # orange
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "X", "CAMO-DKL-cfKG", 'dashed'],
        'RS-': ['#0000FF', "o", "Random Search", 'dotted'],
        'Bayes-': ["#806D00", "s", "BO-GP", 'dotted'],
        'GP-': ['#800080', "D", "BOCA", 'dotted'],
        'CAMO2-': ['#FF00FF', "D", "CAMO2", 'dotted'],
        }
seed_num = 5
seed_dic = {
     'CAMO-': [i for i in range(seed_num)],
     'RS-': [j for j in range(seed_num)],
     'Bayes-': [k for k in range(seed_num)],
     'GP-': [i for i in range(seed_num)],
     'smac_' : [i for i in range(seed_num)],
     'CAMO2-': [i for i in range(seed_num)],
}

method_list = [
                
               'RS-','Bayes-','smac_','GP-','CAMO-',
]
plt.figure(figsize=(8, 6))
Exp_marker = 'ANN-HPO'
for methods_name in method_list:
    iter_collection = []
    mse_collection = []

    for seed in seed_dic[methods_name]:
        if methods_name == 'smac_':
            path = os.path.join('/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res/', Exp_marker,
                            methods_name + 'seed_' + str(seed) + '.csv')
        else:
            path = os.path.join('/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res/', Exp_marker,
                            methods_name + 'seed_' + str(seed) + '_res.csv')
        
        data = pd.DataFrame(pd.read_csv(path))
        
        mse = data['err'].to_numpy()
        mse_collection.append(mse)

    # 获取所有iteration点的并集
    if methods_name in ['CAMO-', 'GP-','Bayes-']:
        iter_x = np.arange(10, len(data)+10)
    else:
        iter_x = np.arange(1, len(data)+1)
    mse_new = np.array(mse_collection)
    mean = np.mean(mse_new, axis=0)
    var = np.std(mse_new, axis=0)

    ll = plt.plot(iter_x, mean, ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                label=Dic[methods_name][2],
                marker=Dic[methods_name][1], markersize=10, markevery=10)
plt.xlabel('Iteration', fontsize=16)  # 修改x轴标签
plt.ylabel('Error', fontsize=16)
# plt.title('Regression HPO on Boston dataset using Random Forest', fontsize=16)
plt.title('ANN Classifier HPO on MNIST dataset', fontsize=16)
# plt.ylim(0.058, 0.072)
plt.xlim(0, len(iter_x))  # 调整x轴范围
plt.tick_params(labelsize=16)
plt.legend(fontsize=16, loc='upper right')
plt.grid()
plt.tight_layout()
save_path = os.path.join('/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res', Exp_marker, 'HPO_Comparison_iter.png')
plt.savefig(save_path, dpi=300)
plt.close()