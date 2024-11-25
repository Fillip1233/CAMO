import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_data(type, data_name, method_name, file_name):
    path = os.path.join(sys.path[-1], 'exp', type, data_name, method_name, file_name + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    return data

Dic = { 'fabolas':['#808000', "*", "Fabolas", 'solid'],
        'smac':['#006400', "*", "SMAC3", 'solid'],
        'AR_UCB': ['#000080', "^", "AR-UCB", "solid"],
        'AR_EI': ['#000080', "s", "AR-EI", "solid"],
        'AR_cfKG': ['#000080', "X", "AR-cfKG", "solid"],
        'ResGP_UCB': ['#00CCFF', "^", "ResGP-UCB", "solid"],
        'ResGP_EI': ['#00CCFF', "s", "ResGP-EI", "solid"],
        'ResGP_cfKG': ['#00CCFF', "X", "ResGP-cfKG", "solid"],
        
        'DNN_MFBO': ['#228B22', "*", "MutualInfo", "solid"],
        
        'GP_UCB': ['#4169E1', "^", "BOCA", "solid"],
        'GP_EI': ['#4169E1', "s", "GP-EI", "solid"],
        'GP_cfKG': ['#4169E1', "X", "cfKG", "solid"],
        'GP_UCB_con': ['#4169E1', "^", "Con_GP-UCB", "solid"],
        'GP_cfKG_con': ['#4169E1', "X", "Con_GP-cfKG", "solid"],

        'DMF_CAR_UCB': ['#FF0000', "^", "CAMO-UCB", "dashed"], # red
        'DMF_CAR_cfKG': ['#FF0000', "X", "CAMO-cfKG", "dashed"],
        'DMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAMO-DKL-UCB", "dashed"], # orange
        'DMF_CAR_dkl_cfKG': ['#FF5E00', "X", "CAMO-DKL-cfKG", "dashed"],
        
        'CMF_CAR_UCB': ['#FF0000', "^", "CAMO-UCB", "dashed"], # red
        'CMF_CAR_cfKG': ['#FF0000', "X", "CAMO-cfKG", "dashed"],
        'CMF_CAR_dkl_UCB': ['#FF5E00', "^", "CAMO-DKL-UCB", "dashed"], # orange
        'CMF_CAR_dkl_cfKG': ['#FF5E00', "X", "CAMO-DKL-cfKG", "dashed"],
        }

# DMF

def paint_dmf_query(axs):
    colors = ['#ff7f0e', '#708090', '#17becf', '#8c564b']
    UCB_list = ['AR_UCB', 'ResGP_UCB', 'GP_UCB', 'CMF_CAR_UCB', 'CMF_CAR_dkl_UCB']
    cfKG_list = ['AR_cfKG', 'ResGP_cfKG', 'GP_cfKG', 
                 'CMF_CAR_cfKG', 'CMF_CAR_dkl_cfKG'
                 ]
    EI_list = ['AR_EI', 'ResGP_EI', 'GP_EI']
    DNN_list = ['DNN_MFBO']

    dmf_methods_list = [UCB_list, cfKG_list, EI_list, DNN_list]
    cmf_data_name_list = ['forrester', 'non_linear_sin']
    acq_name_list = ['MF-UCB', 'cfKG', 'MF-EI', 'MutualInfo']

    values_mean =[]
    values_std = []

    for acq_list in dmf_methods_list:
        m = []
        for methods_name in acq_list:
            for data_name in cmf_data_name_list:
                for seed in seed_dic[data_name]:
                    if methods_name in ['CMF_CAR_UCB', 'CMF_CAR_dkl_UCB', 'CMF_CAR_cfKG', 'CMF_CAR_dkl_cfKG']:
                        path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
                    else:
                        path = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results',
                                            data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
                    data = pd.DataFrame(pd.read_csv(path))
                    cost = data['cost'].to_numpy().reshape(-1, 1)
                    if methods_name == 'DNN_MFBO':
                        time = data['operation_time'].to_numpy().reshape(-1, 1)[-1]
                        m.append((time/len(cost)).item())
                    else:
                        time = data['operation_time'].to_numpy().reshape(-1, 1)
                        m.append(np.mean(time))
                        # s.append(np.std(time))

        values_mean.append(np.mean(np.array(m)))
        values_std.append(np.std(np.array(m)))

    axs.bar(acq_name_list, values_mean, yerr=values_std, capsize=15, color=colors, width = 0.6)
    for i in range(len(acq_name_list)):
        axs.text(i, values_mean[i],f"{values_mean[i]:.2f}",ha='center', va='bottom', fontsize=25)
    axs.set_xticklabels(acq_name_list, rotation=45, ha='right', fontsize=30)
    axs.set_yscale('log')
    axs.tick_params(axis='both', labelsize=25)
    axs.set_ylabel('Query time (seconds)', fontsize=30)

# CMF
def paint_cmf_query(axs):
    methods_name_list = ['GP_UCB','GP_cfKG',
                        'fabolas', 'smac',
                        'CMF_CAR_UCB', 'CMF_CAR_cfKG',
                        # 'CMF_CAR_dkl_UCB','CMF_CAR_dkl_cfKG'
                        ]
    cmf_data_name_list = ['forrester', 'non_linear_sin', 'Park', 'Currin', 'Branin2']
    cost_name = 'pow_10'
    line = []
    values_mean = []
    values_std = []
    for methods_name in methods_name_list:
        m = []
        for data_name in cmf_data_name_list:
            tem = []
            for seed in seed_dic[data_name]:
                path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                                    data_name, cost_name, methods_name + '_seed_' + str(seed) + '.csv')
                data = pd.DataFrame(pd.read_csv(path))
                cost = data['cost'].to_numpy().reshape(-1, 1)
                if methods_name == 'DNN_MFBO':
                    time = data['operation_time'].to_numpy().reshape(-1, 1)[-1]
                    tem.append((time/len(cost)).item())
                else:
                    time = data['operation_time'].to_numpy().reshape(-1, 1)
                    tem.append(np.mean(time))
            tem = np.array(tem)
            m.append(np.mean(tem, axis=0))
        values_mean.append(np.mean(np.array(m)))
        values_std.append(np.std(np.array(m)))

    colors = [Dic[i][0] for i in methods_name_list]
    axs.bar(methods_name_list, values_mean, yerr=values_std, capsize=15, color=colors,width = 0.6)
    for i in range(len(methods_name_list)):
        if methods_name_list[i] == 'smac':
            axs.text(i, values_mean[i], f"{8e-5}",ha='center', va='bottom', fontsize=25)
        else:
            axs.text(i, values_mean[i],f"{values_mean[i]:.2f}",ha='center', va='bottom', fontsize=25)
    axs.set_xticklabels([Dic[i][2] for i in methods_name_list], rotation=45, ha='right', fontsize=30)
    axs.set_yscale('log')
    # axs[1].set_yticklabels([1.0])
    axs.tick_params(axis='both', labelsize=25)
    axs.set_ylabel('Query time (seconds)', fontsize=30)


def paint_dmf_training(axs):

    dmf_methods_list = ['AR_UCB', 'ResGP_UCB', 'GP_UCB', 'DNN_MFBO', 'DMF_CAR_UCB', 'DMF_CAR_dkl_UCB']
    colors = [Dic[i][0] for i in dmf_methods_list]
    labels = ['AR', 'ResGP', 'GP', 'DNN-MFBO', 'CAMO', 'CAMO-DKL']

    values_mean =[]
    values_std = []
    for methods_name in dmf_methods_list:

        path = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results_time',
                            cost_name, methods_name + '_seed_0.csv')
        
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy().reshape(-1, 1)
        if methods_name == 'DNN_MFBO':
            time = data['operation_time'].to_numpy().reshape(-1, 1)
            values_mean.append(np.mean((time/len(cost)).flatten()))
            values_std.append(np.std((time/len(cost)).flatten()))
        else:
            time = data['operation_time'].to_numpy().reshape(-1, 1)
            values_mean.append(np.mean(time))
            values_std.append(np.std(time))

    axs.bar(labels, values_mean, yerr=values_std, capsize=15, color=colors, width = 0.6)
    for i in range(len(labels)):
        axs.text(i, values_mean[i],f"{values_mean[i]:.2f}",ha='center', va='bottom', fontsize=25)

    axs.set_xticklabels(labels, rotation=45, ha='right', fontsize=30)
    axs.set_yscale('log')
    axs.tick_params(axis='both', labelsize=25)
    axs.set_ylabel('Training time (seconds)', fontsize=30)

def paint_cmf_training(axs):

    dmf_methods_list = ['GP_UCB',  'fabolas', 'smac', 'CMF_CAR_UCB', 'CMF_CAR_dkl_UCB']
    colors = [Dic[i][0] for i in dmf_methods_list]
    labels = ['GP', 'Fabolas', 'SMAC3', 'CAMO', 'CAMO-DKL']

    values_mean =[]
    values_std = []
    for methods_name in dmf_methods_list:
        if methods_name == 'smac':
            values_mean.append(8.485810369506553e-05)
            values_std.append(2.5076552318485867e-05)
        else:
            path = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results_time',
                                cost_name, methods_name + '_seed_0.csv')
            
            data = pd.DataFrame(pd.read_csv(path))
            time = data['operation_time'].to_numpy().reshape(-1, 1)
            values_mean.append(np.mean(time))
            values_std.append(np.std(time))

    axs.bar(labels, values_mean, yerr=values_std, capsize=15, color=colors, width = 0.6)
    for i in range(len(labels)):
        if labels[i] == 'SMAC3':
            axs.text(i, values_mean[i], f"{8e-5}",ha='center', va='bottom', fontsize=25)
        else:
            axs.text(i, values_mean[i],f"{values_mean[i]:.2f}",ha='center', va='bottom', fontsize=25)
    axs.set_xticklabels(labels, rotation=45, ha='right', fontsize=30)
    axs.set_yscale('log')
    axs.tick_params(axis='both', labelsize=25)
    axs.set_ylabel('Training time (seconds)', fontsize=30)


seed_dic = {'forrester': [0,1,2,3], 'non_linear_sin': [0,1,2,3,4], 'Branin2':[1,2], 'Currin':[3,5],'Park':[0,1,4]}
cost_name = 'pow_10'

fig, axs = plt.subplots(1, 4, figsize=(35, 8))
paint_dmf_query(axs[0])
paint_cmf_query(axs[1])

paint_dmf_training(axs[2])
paint_cmf_training(axs[3])

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'paper_yx') + '/query_time_heng' + '.pdf',
            bbox_inches='tight')


