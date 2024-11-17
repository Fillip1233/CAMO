import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import time
import torch
from FidelityFusion_Models import *
from Data_simulation.Synthetic_MF_Function import *
import GaussianProcess.kernel as kernel
from FidelityFusion_Models.GP_DMF import cigp, train_GP
from MFBO.Discrete import *
from GP_CFKG import discrete_fidelity_knowledgement_gradient as GP_KG
from sklearn.metrics import mean_squared_error
import argparse

MF_model_list = {'CAR': ContinuousAutoRegression, 'ResGP': ResGP, 'AR': AR, 'GP': cigp}
Acq_list = {'UCB': DMF_UCB, 'EI': DMF_EI, 'cfKG': GP_KG}
Data_list = {'non_linear_sin': non_linear_sin, 'forrester': forrester,'Park3':Park,
             'tl2':tl2, 'tl3':tl3, 'test3':test3, 'test4':test4, 'maolin1':maolin1}


def MF_BO_discrete(exp_config):
    seed = exp_config["seed"]

    '''Initiate Setting'''
    data_model = exp_config["data_model"]
    total_fidelity_num = exp_config['total_fidelity_num']
    initial_index = exp_config['initial_index']
    BO_iterations = exp_config['BO_iterations']
    MF_iterations = exp_config['MF_iterations']
    MF_learning_rate = exp_config['MF_learning_rate']
    search_range = exp_config['search_range']
    cost_type = exp_config['cost_type']

    '''prepare initial data'''
    data = data_model(cost_type,total_fidelity_num + 1)
    index = initial_index
    xtr, ytr = data.Initiate_data(index, seed)

    model_cost = data.cost
    data_max, data_test = data.find_max_value_in_range()
    recording = {"cost": [],
                    "SR": [],
                    "IR": [],
                    "rmse": [],
                    "max_value":[],
                    "operation_time": []}
    x = []
    y = []
    for f in range(total_fidelity_num +1):
        x.append(torch.cat((xtr[f], torch.full((xtr[f].shape[0], 1), f+1)), dim=1))
        y.append(ytr[f])
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    
    initial_data = [
                    {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x, 'Y': y},
                ]
    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = [kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.) for _ in range(total_fidelity_num)]
    kernel_init = [kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.) for _ in range(total_fidelity_num)]

    flag = True
    i = 0
    while flag:
        print('iteration:', i + 1)
        T1 = time.time()
        
        GP_model = MF_model_list[exp_config["MF_model"]](kernel = kernel1[0], log_beta = 1.0)
        train_GP(GP_model, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate)
        T2 = time.time()
        
        if exp_config["Acq_function"] == "ES":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                fidelity_num=total_fidelity_num,
                                                                model_objective=GP_model,
                                                                model_cost=model_cost,
                                                                seed=(seed + 1234 + i, i))
                
        elif exp_config["Acq_function"] == "UCB":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                fidelity_num=2,
                                                                posterior_function=GP_model,
                                                                data_manager=fidelity_manager,
                                                                search_range=search_range,
                                                                seed=(seed + 1234 + i, i))
            
        elif exp_config["Acq_function"] == "EI":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                fidelity_num=2,
                                                                GP_model= GP_model,
                                                                cost = model_cost,
                                                                search_range = search_range,
                                                                model_name = exp_config["MF_model"],
                                                                data_manager = fidelity_manager,
                                                                seed= seed + i + 1234)
            
        elif exp_config["Acq_function"] == "cfKG":
            # GP_model_new = MF_model_list[exp_config["MF_model"]](fidelity_num = total_fidelity_num, kernel_list = kernel_init, if_nonsubset = True)
            GP_model_new = MF_model_list[exp_config["MF_model"]](kernel = kernel_init[0], log_beta = 1.0)
            Acq_function = Acq_list[exp_config["Acq_function"]](GP_model_list=[GP_model, GP_model_new],
                                                                data_model=data,
                                                                cost=model_cost,
                                                                fidelity_num=2,
                                                                data_manager=fidelity_manager,
                                                                xdim=xtr[0].shape[1],
                                                                model_name = exp_config["MF_model"],
                                                                seed=seed + i + 1234)
        
        new_x, new_s = Acq_function.compute_next()
        new_y = data.get_data(new_x, new_s)
        new_x = torch.cat((new_x, torch.tensor(new_s+1).reshape(1,1)), dim=1)

        print(f"Optimization finished {i} times. New x: {new_x}, New s: {new_s}, New y: {new_y.item()}")
        fidelity_manager.add_data(raw_fidelity_name=str(0), fidelity_index=0, x=new_x, y=new_y)

        # Calculate evaluation indicators
        x_train = fidelity_manager.get_data(0)[0]
        if args.data_name in ["Branin","Park3","Currin3"]:
            y_high_for_train = data.get_data(x_train[:,:-1], torch.tensor([1]*x_train.shape[0]))
        else:
            y_high_for_train = data.get_data(x_train[:,:-1], 1)
        best_y_high_train = max(y_high_for_train.reshape(-1,1))
        
        # T2 = time.time()

        cost_iter = model_cost.compute_gp_cost([fidelity_manager.get_data(0)[0]])
        if cost_iter >= 150:
            flag = False
        recording["cost"].append(cost_iter.item())
        recording["max_value"].append(max(y_high_for_train.reshape(-1,1)).item())
        recording["SR"].append((data_max - best_y_high_train).item())

        with torch.no_grad():
            mu, var = GP_model(fidelity_manager, data_test[:100].double(),torch.tensor([1]*100).double())
            # mu, _ = fidelity_manager.normalizelayer[0].denormalize(mu, var)
        if args.data_name in ["Branin","Park3","Currin3"]:
            y_gt = data.get_data(data_test[:100], torch.tensor([1]*data_test[:100].shape[0]))
        else:
            y_gt = data.get_data(data_test[:100], total_fidelity_num - 1)

        rmse = torch.sqrt(torch.tensor(mean_squared_error(y_gt.reshape(-1,1), mu.detach())))
        recording["IR"].append((data_max - max(mu)).item())
        recording['rmse'].append(rmse.item())
        recording["operation_time"].append(T2 - T1)
        i += 1

    return recording


if __name__ == '__main__':

    # data_name = "non_linear_sin"
    data_name = "forrester"
    # data_name = "Branin"
    # data_name = "maolin1"
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--data_name", type=str, default="Park3")
    parser.add_argument("--cost_type", type=str, default="pow_10")
    args = parser.parse_args()
    data_name = args.data_name
    for mf_model in ["GP"]:
        for acq in ["UCB","EI","cfKG"]:
            for seed in [0,1,4]:
                exp_config = {
                            'seed': seed,
                            'data_model': Data_list[args.data_name],
                            'cost_type':args.cost_type,
                            'MF_model': mf_model,
                            'Acq_function': acq,
                            'total_fidelity_num': 1,
                            'search_range': [0, 1.5],
                            'initial_index': {0: 10, 1: 4},
                            'BO_iterations': 10,
                            'MF_iterations': 200,
                            'MF_learning_rate': 0.01,
                    }

                record = MF_BO_discrete(exp_config)

                path_csv = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results',
                                        data_name,exp_config['cost_type'])
                if not os.path.exists(path_csv):
                    os.makedirs(path_csv)

                df = pd.DataFrame(record)
                df.to_csv(path_csv + '/'+ mf_model+'_' + exp_config['Acq_function'] + '_seed_' + str(seed) + '.csv',
                          index=False)