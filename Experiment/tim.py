import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


seed_dic = {'forrester': [0,1,2,3], 'non_linear_sin': [0,1,2,3,4], 'Branin2':[1,2], 'Currin':[3,5],'Park':[0,1,4]}
methods_name_list = ['GP_UCB','GP_cfKG',
                        'fabolas', 'smac',
                        'CMF_CAR_UCB', 'CMF_CAR_cfKG',
                        'CMF_CAR_dkl_UCB','CMF_CAR_dkl_cfKG']
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


