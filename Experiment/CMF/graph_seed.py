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
Dic = {'AR_UCB': ['#000080', "^", "AR-UCB"],
       'AR_EI': ['#000080', "s", "AR-EI"],
       'AR_cfKG': ['#000080', "o", "AR-cfKG"],
       'ResGP_UCB': ['#00CCFF', "^", "ResGP-UCB"],
       'ResGP_EI': ['#00CCFF', "s", "ResGP-EI"],
       'ResGP_cfKG': ['#00CCFF', "o", "ResGP-cfKG"],
        
        'DNN_MFBO': ['#228B22', "X", "DNN"],
        'fabolas':['#808000', "*", "Fabolas"],
        'smac':['#006400', "*", "SMAC3"],
        
        'GP_UCB': ['#4169E1', "^", "GP-UCB"],
        'GP_EI': ['#4169E1', "s", "GP-EI"],
        'GP_cfKG': ['#4169E1', "o", "GP-cfKG"],
        'GP_UCB_con': ['#4169E1', "^", "Con_GP-UCB"],
        'GP_cfKG_con': ['#4169E1', "o", "Con_GP-cfKG"],
        
        'CMF_CAR_UCB': ['#FF0000', "^", "CAR-UCB"], # red
        'CMF_CAR_cfKG': ['#FF0000', "o", "CAR-cfKG"],
        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAR_dkl-UCB"], # orange
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "o", "CAR_dkl-cfKG"],
        }


data_name = 'Park'
# data_name = 'Forrester'
# data_name = 'Branin'
# data_name = 'HeatedBlock'
cost_name = 'pow_10'

max_dic = {'non_linear_sin':0, 'forrester': 50,'Branin2': 55,'Currin': 14,'Park': 2.2}
add_dict = {'Forrester': 7 ,'non_linear_sin': 0.15,'Branin2': 0,'Currin': 0,'Park': 0, 'VibratePlate2': 0, 'HeatedBlock': 1.2}
lim_x = {'Forrester': [48, 135], 'non_linear_sin': [15, 135],
         'Branin2':[48,150],'Currin':[48,150],'Park':[48,150]}
lim_y = {'Forrester': [0, 55], 'non_linear_sin': [0, 0.30],'Currin':[0,2],'Park':[0,1.4]}
# cost_lim_y = {'Forrester': [0, 0], 'non_linear_sin': [0, -0.25]}


# methods_name_list = ['AR_UCB', 'AR_EI', 'AR_cfKG', 'ResGP_UCB', 'ResGP_EI', 'ResGP_cfKG', 'GP_UCB', 'GP_EI', 'GP_cfKG']
cmf_methods_name_list = ['GP_UCB', 'GP_cfKG','CMF_CAR_UCB','CMF_CAR_cfKG',
                         'CMF_CAR_dkl_UCB', 'CMF_CAR_dkl_cfKG']

line = []
# seed = 4
for seed in [0,1]:
    plt.figure(figsize=(10, 6))
    for methods_name in cmf_methods_name_list:
        ct = []
        tem = []
        
        path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        # cost = data['cost'].to_numpy().reshape(-1, 1)
        # SR = data['SR'].to_numpy().reshape(-1, 1)
        cost = data['cost'].to_numpy().reshape(-1, 1)
        if methods_name in ['fabolas',"smac"]:
            incumbents = max_dic[data_name] - data['incumbents'].to_numpy()
        else:    
            incumbents = data['SR'].to_numpy().reshape(-1, 1)
        tem.append(incumbents)
        ct.append(cost)
        tem = np.array(tem)
        mean = np.mean(tem, axis=0)
        var = np.std(tem, axis=0)
        ll = plt.plot(ct[0].flatten(), mean.flatten()+ add_dict[data_name], ls='dashed', color=Dic[methods_name][0],
                    label=Dic[methods_name][2],
                    marker=Dic[methods_name][1], markersize=6)
        plt.fill_between(ct[0].flatten(),
                        mean.flatten()+ add_dict[data_name] - 0.96 * var.flatten(),
                        mean.flatten()+ add_dict[data_name] + 0.96 * var.flatten(),
                        alpha=0.1, color=Dic[methods_name][0])
        line.append(ll)
        plt.xlabel("Cost", fontsize=20)
        plt.ylabel("Simple regret", fontsize=20)

    # plt.yscale('log')
    # plt.xlim(lim_x[data_name][0], lim_x[data_name][1])
    # plt.ylim(lim_y[data_name][0], lim_y[data_name][1])
    label = [Dic[i][-1] for i in cmf_methods_name_list]
    plt.tick_params(axis='both', labelsize=15)
    # if data_name == 'non_linear_sin':
    #     plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=15)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=15)
    plt.grid()
    # seed = '1'
    plt.tight_layout()
    # plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper') + '/' + data_name +'_'+ cost_name +'_SR_Interpolation.png', bbox_inches='tight')
    plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper_seed') + '/' +'CMF_'+ data_name +'_'+ cost_name +'_seed'+str(seed)+'_.pdf', bbox_inches='tight')
    plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'CMF', 'paper_seed') + '/' +'CMF_'+ data_name +'_'+ cost_name +'_seed'+str(seed)+'_.eps', bbox_inches='tight')
