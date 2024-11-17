import torch
import pandas as pd
import sys
import time
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from FidelityFusion_Models import *
from Data_simulation.Synthetic_MF_Function import *
import GaussianProcess.kernel as kernel
from Data_simulation.Synthetic_MF_Function import *
from MFBO.Continue import *
from sklearn.metrics import mean_squared_error, r2_score

MF_model_list = {'CAR': ContinuousAutoRegression_large}
Acq_list = {'UCB': CMF_UCB, 'cfKG': CMF_KG}
Data_list = {'Branin': Branin, 'Hartmann': Hartmann}

def MF_BO_continuous(exp_config):
    seed = exp_config["seed"]

    '''Initiate Setting'''
    data_model = exp_config["data_model"]
    num_fi = exp_config['initial_num']
    BO_iterations = exp_config['BO_iterations']
    MF_iterations = exp_config['MF_iterations']
    MF_learning_rate = exp_config['MF_learning_rate']

    '''prepare initial data'''
    data = data_model()
    search_range = data.search_range
    xtr, ytr, s_index = data.Initiate_data(num_fi, seed)
    model_cost = data.cost
    data_max, data_test, data_gt = data.find_max_value_in_range(seed)

    xtr = torch.cat((xtr, s_index), dim=1)
    # ytr = torch.cat((ytr, s_index), dim=1)
    recording = {"cost": [],
                 "SR": [],
                 "IR": [],
                 "rmse": [],
                 "r2":[],
                 "max_value":[],
                 "operation_time": []}
    
    initial_data = [
                    {'fidelity_indicator': 0,'raw_fidelity_name': 'data_with_fi', 'X': xtr, 'Y': ytr},
                ]
    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)
    kernel_init = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)
    
    flag = True
    i = 0
    while flag:
        print('iteration:', i + 1)
        T1 = time.time()
        
        # Fit the Gaussian process model to the sampled points
        if exp_config["MF_model"] == "CAR":
            model_objective = MF_model_list[exp_config["MF_model"]](input_dim=xtr.shape[1]-1, kernel_x = kernel1)
            train_CAR_large(model_objective, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate)

        if exp_config["Acq_function"] == "ES":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr.shape[1],
                                                                search_range=search_range,
                                                                model_objective=model_objective,
                                                                model_cost=model_cost,
                                                                seed=seed + i + 1234)

        elif exp_config["Acq_function"] == "UCB":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr.shape[1] - 1,
                                                                search_range=search_range,
                                                                posterior_function=model_objective,
                                                                model_cost=model_cost,
                                                                data_manager=fidelity_manager,
                                                                seed=[seed + i + 1234, i])
            
        elif exp_config["Acq_function"] == "cfKG":

            model_objective_new = MF_model_list[exp_config["MF_model"]](input_dim=xtr.shape[1]-1, kernel_x = kernel_init)
            Acq_function = Acq_list[exp_config["Acq_function"]](posterior_function=model_objective,
                                                                model_objective_new=model_objective_new,
                                                                data_model=data,
                                                                model_cost=model_cost,
                                                                search_range=search_range,
                                                                data_manager=fidelity_manager,
                                                                seed=seed + i + 1234)
            
        new_x, new_s = Acq_function.compute_next()
        new_y = data.get_data(new_x, new_s)

        print(f"Optimization finished {i} times. New x: {new_x}, New s: {new_s}, New y: {new_y}")
        new_x = torch.cat((new_x, new_s.reshape(-1,1)), dim=1)
        # new_y = torch.cat((new_y, new_s.reshape(-1,1)), dim=1)

        fidelity_manager.add_data(raw_fidelity_name='data_with_fi', fidelity_index = 0, x = new_x, y = new_y)

        T2 = time.time()

        X = fidelity_manager.get_data(0)[0]
        Y = fidelity_manager.get_data(0)[1]

        with torch.no_grad():
            mu, var = model_objective(fidelity_manager, data_test[:100].double())
        rmse = torch.sqrt(torch.tensor(mean_squared_error(data_gt.reshape(-1,1), mu.detach())))
        y_formax = data.get_data(X[:,:-1], torch.tensor([1]*X.shape[0]).reshape(-1,1))
        recording["cost"].append(model_cost.compute_model_cost(dataset = Y[:,0], s_index = Y[:,1]))
        recording['rmse'].append(rmse.item())
        recording["SR"].append((data_max - max(y_formax[:,0])).item())
        recording["IR"].append((data_max - max(mu)).item())
        recording["max_value"].append(max(y_formax[:,0]).item())
        recording["operation_time"].append(T2 - T1)
        i += 1

    return recording


if __name__ == '__main__':

    # "Branin", "Hartmann", "mln_mnist", "cnn_cifar"
    data_name = 'Branin'
    for mf_model in ["CMF_CAR","CMF_CAR_dkl"]:
        for seed in [0,1,2,3,4]:
            for acq in ["UCB","cfKG"]:
                exp_config = {
                    'seed': seed,
                    'data_model': Data_list[data_name],
                    'MF_model': mf_model,
                    'Acq_function': acq,
                    'initial_num': [10,4,2],
                    'BO_iterations': 10,
                    'MF_iterations': 100,
                    'MF_learning_rate': 0.0001,
                }
                record = MF_BO_continuous(exp_config)

                path_csv = os.path.join(sys.path[-1], 'Experiment', 'CMF', 'Exp_results',
                                        data_name)
                if not os.path.exists(path_csv):
                    os.makedirs(path_csv)

                df = pd.DataFrame(record)
                df.to_csv(path_csv + '/'+ mf_model + exp_config['Acq_function'] + '_seed_' + str(seed) + '.csv', index=False)