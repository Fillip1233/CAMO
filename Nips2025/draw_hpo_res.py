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
                'CAMO-',
               'Bayes-','RS-','GP-','smac_',
]
plt.figure(figsize=(8, 6))
Exp_marker = 'ANN-HPO'
for methods_name in method_list:
    clock_collection = []
    inter_collection = []

    for seed in seed_dic[methods_name]:
        if methods_name == 'smac_':
            path = os.path.join('/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res/', Exp_marker,
                            methods_name + 'seed_' + str(seed) + '.csv')
        else:
            path = os.path.join('/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res/', Exp_marker,
                            methods_name + 'seed_' + str(seed) + '_res.csv')
        
        data = pd.DataFrame(pd.read_csv(path))
        clock = data['wallclocktime'].to_numpy()
        
        err = data['err'].to_numpy()
        inter = interp1d(clock, err, kind='linear', fill_value="extrapolate")
        clock_collection.append(clock)
        inter_collection.append(inter)

    clock_x = np.unique(np.concatenate(clock_collection))
    clock_new = [inter_collection[i](clock_x) for i in range(len(inter_collection))]
    clock_new = np.array(clock_new)
    mean = np.mean(clock_new, axis=0)
    var = np.std(clock_new, axis=0)

    ll = plt.plot(clock_x, mean, ls=Dic[methods_name][-1], color=Dic[methods_name][0],
                label=Dic[methods_name][2],
                marker=Dic[methods_name][1], markersize=10,markevery=40)
plt.xlabel('Wallclock Time (s)', fontsize=16)
plt.ylabel('Error', fontsize=16)
# plt.title('Regression HPO on Boston dataset using Random Forest', fontsize=16)
plt.title('ANN Classifier HPO on MNIST', fontsize=16)
plt.ylim(0.050, 0.10)
# plt.xlim(3.8, 250)
plt.tick_params(labelsize=16)
plt.legend(fontsize=16, loc='upper right')
plt.grid()
plt.tight_layout()
save_path = os.path.join('/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res', Exp_marker, 'HPO_Comparison.png')
plt.savefig(save_path, dpi=300)
plt.close()