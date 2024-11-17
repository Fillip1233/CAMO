import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def get_data(type, data_name, method_name, file_name):
    path = os.path.join(sys.path[-1], 'exp', type, data_name, method_name, file_name + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    return data

colors = ['#ff7f0e', '#708090', '#17becf', '#8c564b', '#2ca02c', '#DC143C',
          '#FFD700', '#FF4500', 'black', 'deeppink', '#FF6347', '#B51B75', '#E65C19', 'blue', 'green']

methods_name_list = ['AR_UCB', 'ResGP_UCB', 'GP_UCB', 'DMF_CAR_dkl_UCB', 'DMF_CAR_UCB']
cost_name = 'pow_10'
values = []
errors = []

for methods_name in methods_name_list:
    tem = []
    path = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results_time',
                        cost_name, methods_name + '_seed_0.csv')
    data = pd.DataFrame(pd.read_csv(path))
    if methods_name == 'DNN_MFBO':
        time = data['operation_time'].to_numpy().reshape(-1, 1)[-1]
        tem.append((time / len(data['cost'])).item())
    else:
        time = data['operation_time'].to_numpy().reshape(-1, 1)
        tem.append(time)

    tem = np.concatenate(tem)
    values.append(np.mean(tem))
    errors.append(np.std(tem))

values = np.array(values)
errors = np.array(errors)

plt.bar(methods_name_list, values, yerr=errors, color=colors[:len(methods_name_list)], width=0.5, capsize=5)

# Adjust the vertical position of text labels
for i in range(len(methods_name_list)):
    plt.text(i, values[i] + errors[i] + 0.005 * max(values), f"{values[i]:.2f}", ha='center', va='bottom')

new_xticks = ['AR', 'ResGP', 'DMF_GP', 'DMF_CAR_dkl', 'DMF_CAR']
plt.xticks(range(len(new_xticks)), new_xticks, rotation=45, ha='right')
plt.yscale('log')
plt.ylabel('Time (seconds)', fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper') + '/' + 'train_time' + '.eps',
            bbox_inches='tight')
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'DMF', 'paper') + '/' + 'train_time' + '.png',
            bbox_inches='tight')
# plt.show()
