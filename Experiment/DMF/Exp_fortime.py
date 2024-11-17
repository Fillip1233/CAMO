import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import time
import torch
from FidelityFusion_Models import *
from FidelityFusion_Models.DMF_CAR import *
from Data_simulation.Synthetic_MF_Function import *
import GaussianProcess.kernel as kernel
from GaussianProcess.cigp_v10 import *
from FidelityFusion_Models.DMF_CAR_dkl import *
from MFBO.Discrete import *
from sklearn.metrics import mean_squared_error, r2_score
import argparse

MF_model_list = {'CAR': ContinuousAutoRegression, 'ResGP': ResGP, 'AR': AR, 'GP': cigp,
                 'DMF_CAR': DMF_CAR,'DMF_CAR_dkl': DMF_CAR_dkl}
Acq_list = {'UCB': DMF_UCB, 'EI': DMF_EI, 'cfKG': DMF_KG}
Data_list = {'non_linear_sin': non_linear_sin, 'forrester': forrester}


def MF_BO_discrete(exp_config):
    seed = exp_config["seed"]

    '''Initiate Setting'''
    data_model = exp_config["data_model"]
    total_fidelity_num = exp_config['total_fidelity_num']
    initial_index = exp_config['initial_index']
    BO_iterations = exp_config['BO_iterations']
    MF_iterations = exp_config['MF_iterations']
    MF_learning_rate = exp_config['MF_learning_rate']
    cost_type = exp_config['cost_type']
    search_range = exp_config['search_range']

    '''prepare initial data'''
    data = data_model(cost_type,total_fidelity_num)
    index = initial_index
    xtr, ytr = data.Initiate_data(index, seed)

    model_cost = data.cost
    data_max, data_test = data.find_max_value_in_range()
    recording = {"cost": [],
                    "SR": [],
                    "IR": [],
                    "rmse": [],
                    "r2":[],
                    "max_value":[],
                    "operation_time": []}
    
    initial_data = [
                    {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': xtr[0], 'Y': ytr[0]},
                    {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': xtr[1], 'Y': ytr[1]},
                    # {'fidelity_indicator': 2, 'raw_fidelity_name': '2','X': xtr[2], 'Y': ytr[2]},
                ]
    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = [kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.) for _ in range(total_fidelity_num)]
    kernel_init = [kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.) for _ in range(total_fidelity_num)]

    flag = True
    i = 0
    while flag:
        print('iteration:', i + 1)
        T1 = time.time()
        if exp_config["MF_model"] == "DMF_CAR_dkl":
            GP_model = MF_model_list[exp_config["MF_model"]](fidelity_num = total_fidelity_num,input_dim = xtr[0].shape[1], kernel_list = kernel1, if_nonsubset = True)
        else:
            GP_model = MF_model_list[exp_config["MF_model"]](fidelity_num = total_fidelity_num, kernel_list = kernel1, if_nonsubset = True)
        # Fit the Gaussian process model to the sampled points
        if exp_config["MF_model"] == "ResGP":
            train_ResGP(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)
        if exp_config["MF_model"] == "AR":
            train_AR(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)
        if exp_config["MF_model"] == "CAR":
            train_CAR(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)
        if exp_config["MF_model"] == "DMF_CAR":
            train_DMFCAR(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)
        if exp_config["MF_model"] == "DMF_CAR_dkl":
            train_DMFCAR_dkl(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate, normal= False)

        if exp_config["Acq_function"] == "ES":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                fidelity_num=total_fidelity_num,
                                                                model_objective=GP_model,
                                                                model_cost=model_cost,
                                                                seed=(seed + 1234 + i, i))
                
        elif exp_config["Acq_function"] == "UCB":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                fidelity_num=total_fidelity_num,
                                                                posterior_function=GP_model,
                                                                data_manager=fidelity_manager,
                                                                search_range = search_range,
                                                                seed=(seed + 1234 + i, i))
            
        elif exp_config["Acq_function"] == "EI":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                fidelity_num=total_fidelity_num,
                                                                GP_model= GP_model,
                                                                cost = model_cost,
                                                                data_manager = fidelity_manager,
                                                                search_range = search_range,
                                                                model_name = exp_config["MF_model"],
                                                                seed= seed + i + 1234)
            
        elif exp_config["Acq_function"] == "cfKG":
            if exp_config["MF_model"] == "DMF_CAR_dkl":
                GP_model_new = MF_model_list[exp_config["MF_model"]](fidelity_num = total_fidelity_num,input_dim = xtr[0].shape[1], kernel_list = kernel_init, if_nonsubset = True)
            else:
                GP_model_new = MF_model_list[exp_config["MF_model"]](fidelity_num = total_fidelity_num, kernel_list = kernel_init, if_nonsubset = True)
            Acq_function = Acq_list[exp_config["Acq_function"]](GP_model_list=[GP_model, GP_model_new],
                                                                data_model=data,
                                                                cost=model_cost,
                                                                fidelity_num=total_fidelity_num,
                                                                data_manager=fidelity_manager,
                                                                xdim=xtr[0].shape[1],
                                                                model_name = exp_config["MF_model"],
                                                                seed=seed + i + 1234)
        
        new_x, new_s = Acq_function.compute_next()
        new_y = data.get_data(new_x, new_s)

        print(f"Optimization finished {i} times. New x: {new_x.item()}, New s: {new_s}, New y: {new_y.item()}")
        fidelity_manager.add_data(raw_fidelity_name=str(new_s), fidelity_index=new_s, x=new_x, y=new_y)

        # Calculate evaluation indicators
        x_train = torch.cat((fidelity_manager.get_data(0)[0], fidelity_manager.get_data(1)[0]), dim=0)
        y_high_for_train = data.get_data(x_train, 1)
        best_y_high_train = max(y_high_for_train)
        
        T2 = time.time()

        cost_iter = model_cost.compute_model_cost([fidelity_manager.get_data(0)[1],fidelity_manager.get_data(1)[1]])
        if cost_iter >= 150:
            flag = False
        recording["cost"].append(cost_iter.item())
        recording["max_value"].append(max(y_high_for_train).item())
        recording["SR"].append((data_max - best_y_high_train).item())

        with torch.no_grad():
            # data_test = fidelity_manager.normalizelayer[GP_model.fidelity_num-1].normalize_x( data_test[:100].double())
            mu, var = GP_model(fidelity_manager, data_test[:100].double(), normal = False)
            # mu, _ = fidelity_manager.normalizelayer[GP_model.fidelity_num-1].denormalize(mu, var)
        y_gt = data.get_data(data_test[:100], total_fidelity_num - 1)
        rmse = torch.sqrt(torch.tensor(mean_squared_error(y_gt.reshape(-1,1), mu.detach())))
        r2 = r2_score(y_gt.reshape(-1,1), mu.detach())
        recording["IR"].append((data_max - max(mu)).item())
        recording['r2'].append(r2.item())
        recording['rmse'].append(rmse.item())
        recording["operation_time"].append(T2 - T1)
        i += 1

    return recording


if __name__ == '__main__':

    # data_name = "non_linear_sin"
    # data_name = "forrester"
    # data_name = "Branin"
    # data_name = "maolin1"
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--data_name", type=str, default="forrester")
    parser.add_argument("--cost_type", type=str, default="pow_10")
    args = parser.parse_args()
    data_name = args.data_name
    for mf_model in ["GP"]:
    # for mf_model in ["DMF_CAR"]:
        # for acq in ["EI"]:
        for acq in ["UCB"]:
            for seed in [0]:
                exp_config = {
                            'seed': seed,
                            'data_model': Data_list[args.data_name],
                            'cost_type':args.cost_type,
                            'MF_model': mf_model,
                            'Acq_function': acq,
                            'total_fidelity_num': 2,
                            'initial_index': {0: 10, 1: 4},
                            'search_range': [0, 1.5],   
                            'BO_iterations': 10,
                            'MF_iterations': 200,
                            'MF_learning_rate': 0.01,
                    }

                record = MF_BO_discrete(exp_config)

                path_csv = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results_time',
                                        data_name,exp_config['cost_type'])
                if not os.path.exists(path_csv):
                    os.makedirs(path_csv)

                df = pd.DataFrame(record)
                df.to_csv(path_csv + '/'+ mf_model+'_' + exp_config['Acq_function'] + '_seed_' + str(seed) + '.csv',
                          index=False)