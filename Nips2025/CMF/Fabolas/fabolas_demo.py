import json
import logging
import numpy as np
import pandas as pd
import torch
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import george
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))

from Data_simulation.Synthetic_MF_Function import *
# from Data_simulation.Real_Application.HeatedBlock import HeatedBlock
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.information_gain_per_unit_cost import InformationGainPerUnitCost
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.maximizers.random_sampling import RandomSampling
from robo.models.fabolas_gp import FabolasGPMCMC
from robo.priors.env_priors import EnvPrior
import argparse
from robo.util.incumbent_estimation import projected_incumbent_estimation

from Data_simulation.Real_Application.HeatedBlock import HeatedBlock
from Data_simulation.Real_Application.VibratePlate import VibPlate



def fabolas(exp_config):
    data_name = exp_config["data_name"]
    path_csv = sys.path[-1] + '/Nips2025/CMF/Exp_results/'+exp_config['Exp_marker']+'/' + data_name + '/' + exp_config['cost_type'] + '/'
    seed = exp_config["seed"]
    data_model = exp_config["data_model"]
    initial_index = exp_config['initial_num']
    BO_iterations = exp_config['BO_iterations']

    cost_type = exp_config['cost_type']
    data = data_model(cost_type, total_fidelity_num = 2)
    search_range = data.search_range
    index = initial_index
    xtr, ytr = data.Initiate_data(index, seed)
    model_cost_exp = data.cost
    data_max, data_test = data.find_max_value_in_range()


    def transform(s, s_min, s_max):
        s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
        return s_transform

    def retransform(s_transform, s_min, s_max):
        s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
        return int(s)

    '''fabolas 有不同的数据组成结构'''

    # x = np.random.rand(32)[:, None]
    s_min = 1
    s_max = 5
    s_low = transform(2, s_min, s_max)
    s_high = transform(4, s_min, s_max)

    ss_low = retransform(s_low, s_min, s_max)
    ss_high = retransform(s_high, s_min, s_max)

    x = []
    y = []
    for f in range(2):
        x.append(torch.cat((xtr[f], torch.full((xtr[f].shape[0], 1), f+1)), dim=1))
        y.append(ytr[f])
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    # xtr = torch.cat((xtr, s_index), dim=1)

    '''进入fabolas的循'''
    logger = logging.getLogger(__name__)
    n_dims = data.x_dim
    lower = np.array([data.search_range[i][0] for i in range(n_dims)])
    upper = np.array([data.search_range[i][1] for i in range(n_dims)])
    n_hypers = 1
    rng = None
    time_start = time.time()
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    # Bookkeeping
    time_func_eval = []
    time_overhead = []
    incumbents = []
    runtime = []

    X = x.numpy()
    y = y.numpy().flatten()
    c = np.concatenate((np.ones(index[0]).reshape(-1, 1), 5 * np.ones(index[1]).reshape(-1, 1)), axis=0).flatten()
    # c = model_cost_exp.compute_model_cost(dataset = ytr[:,0], s_index = ytr[:,1])
    num_iterations = X.shape[0] + BO_iterations
    
    ## zhenjie change
    if data_name in ['Park', 'Branin', 'Currin']:
        y_high_for_train = data.get_data(torch.from_numpy(X[:,:-1]), torch.tensor([1]*X.shape[0]))
    else:
        y_high_for_train = data.get_data(torch.from_numpy(X[:,:-1]), torch.tensor(1.))
    ##

    recording = {"cost": [model_cost_exp.compute_model_cost_fabolas(X[:, 0], X[:, -1]).item()],
                 "incumbents": [max(y_high_for_train).item()],
                 "operation_time": [0]}

    # Define model for the objective function
    cov_amp = 1  # Covariance amplitude
    kernel = cov_amp

    # ARD Kernel for the configuration space
    for d in range(n_dims):
        kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                ndim=n_dims + 1, axes=d)

    env_kernel = george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                   ndim=n_dims + 1,
                                                   axes=n_dims)

    kernel *= env_kernel

    # Take 3 times more samples than we have hyperparameters
    if n_hypers < 2 * len(kernel):
        n_hypers = 3 * len(kernel)
        if n_hypers % 2 == 1:
            n_hypers += 1

    prior = EnvPrior(len(kernel) + 1,
                         n_ls=n_dims,
                         n_lr=2,
                         rng=rng)

    quadratic_bf = lambda x: (1 - x) ** 2
    linear_bf = lambda x: x

    model_objective = FabolasGPMCMC(kernel,
                                    prior=prior,
                                    burnin_steps=100,
                                    chain_length=100,
                                    n_hypers=n_hypers,
                                    normalize_output=False,
                                    basis_func=quadratic_bf,
                                    lower=lower,
                                    upper=upper,
                                    rng=rng)

    # Define model for the cost function
    cost_cov_amp = 1

    cost_kernel = cost_cov_amp

    # ARD Kernel for the configuration space
    for d in range(n_dims):
        cost_kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                     ndim=n_dims + 1, axes=d)

    cost_env_kernel = george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                    ndim=n_dims + 1,
                                                    axes=n_dims)
    cost_kernel *= cost_env_kernel

    cost_prior = EnvPrior(len(cost_kernel) + 1,
                          n_ls=n_dims,
                          n_lr=2,
                          rng=rng)

    model_cost = FabolasGPMCMC(cost_kernel,
                               prior=cost_prior,
                               burnin_steps=100,
                               chain_length=100,
                               n_hypers=n_hypers,
                               basis_func=linear_bf,
                               normalize_output=False,
                               lower=lower,
                               upper=upper,
                               rng=rng)

    # Extend input space by task variable
    extend_lower = np.append(lower, 0)
    extend_upper = np.append(upper, 1)
    is_env = np.zeros(extend_lower.shape[0])
    is_env[-1] = 1

    # Define acquisition function and maximizer
    ig = InformationGainPerUnitCost(model_objective,
                                    model_cost,
                                    extend_lower,
                                    extend_upper,
                                    sampling_acquisition=EI,
                                    is_env_variable=is_env,
                                    n_representer=50)
    acquisition_func = MarginalizationGPMCMC(ig)
    maximizer = RandomSampling(acquisition_func, extend_lower, extend_upper)
    inc_estimation = "mean"

    output_path = path_csv
    count = 0
    flag = 1
    it = 0
    while flag:
        logger.info("Start iteration %d ... ", it)

        start_time = time.time()

        # Train models
        model_objective.train(X, y, do_optimize=True)
        model_cost.train(X, c, do_optimize=True)

        if inc_estimation == "last_seen":
            # Estimate incumbent as the best observed value so far
            best_idx = np.argmin(y)
            incumbent = X[best_idx][:-1]
            incumbent = np.append(incumbent, 1)
            incumbent_value = y[best_idx]
        else:
            # Estimate incumbent by projecting all observed points to the task of interest and
            # pick the point with the lowest mean prediction
            incumbent, incumbent_value = projected_incumbent_estimation(model_objective, X[:, :-1],
                                                                        proj_value=1)
        incumbents.append(incumbent[:-1])
        logger.info("Current incumbent %s with estimated performance %f",
                    str(incumbent), np.exp(incumbent_value))

        # Maximize acquisition function
        acquisition_func.update(model_objective, model_cost)
        new_x = maximizer.maximize()

        s = retransform(new_x[-1], s_min, s_max)  # Map s from log space to original linear space

        time_overhead.append(time.time() - start_time)
        logger.info("Optimization overhead was %f seconds", time_overhead[-1])

        # Evaluate the chosen configuration
        logger.info("Evaluate candidate %s on subset size %f", str(new_x[:-1]), s)
        start_time = time.time()
        if s > 2.5:
            s = 2
            new_c = 5
        else:
            s = 1
            new_c = 1

        # new_y = gen_data(seed, 'non_linear_sin', new_x[:-1], s, total_fidelity_num)
        # new_y = data.get_data(torch.from_numpy(new_x), s)
        
        
        # new_y = data.get_cmf_data(torch.from_numpy(new_x)[:n_dims], torch.from_numpy(new_x)[-1])
        new_y = data.get_cmf_data(torch.from_numpy(new_x)[:n_dims].reshape(1, -1), torch.from_numpy(new_x)[-1])
        time_func_eval.append(time.time() - start_time)

        logger.info("Configuration achieved a performance of %f with cost %f", new_y, new_c)
        logger.info("Evaluation of this configuration took %f seconds", time_func_eval[-1])

        # Add new observation to the data
        new_x[-1] += 1  #CMF fid start from 1
        
        X = np.concatenate((X, new_x[None, :]), axis=0)
        y = np.concatenate((y, new_y.flatten()), axis=0)  # Model the target function on a logarithmic scale
        c = np.concatenate((c, np.array([new_c])), axis=0)  # Model the cost function on a logarithmic scale

        runtime.append(time.time() - time_start)
        ss_tem = X[:, -1]
        s_tem_aft = 1 + (ss_tem - np.min(ss_tem)) * (2 - 1) / (np.max(ss_tem) - np.min(ss_tem))
        cost = model_cost_exp.compute_model_cost_fabolas(X[:, 0], s_tem_aft).item()
        
        if cost > 300:
            flag = 0
        recording["cost"].append(cost)
        # recording["incumbents"].append(incumbents[count].tolist()[0])
        
        ## zhenjie change
        if data_name in ['Park', 'Branin', 'Currin']:
            y_high_for_train = data.get_data(torch.from_numpy(X[:,:-1]), torch.tensor([1]*X.shape[0]))
        else:
            y_high_for_train = data.get_data(torch.from_numpy(X[:,:-1]), torch.tensor(1.))
        ##
        
        recording["incumbents"].append(max(y_high_for_train).item())
        recording["operation_time"].append(time.time() - time_start)
        print('cost:', cost)
        print('end iteration', it)
        it += 1

        # if output_path is not None:
        #     data = dict()
        #     data["optimization_overhead"] = time_overhead[count]
        #     data["runtime"] = runtime[count]
        #     data["incumbent"] = incumbents[count].tolist()
        #     data["time_func_eval"] = time_func_eval[count]
        #     data["iteration"] = count
        #     count += 1
        #     json.dump(data, open(os.path.join(output_path, "fabolas_iter_%d.json" % it), "w"))

    # Estimate the final incumbent
    model_objective.train(X, y, do_optimize=True)
    incumbent, incumbent_value = projected_incumbent_estimation(model_objective, X[:, :-1],
                                                                proj_value=1)
    logger.info("Final incumbent %s with estimated performance %f",
                str(incumbent), incumbent_value)

    results = dict()
    results["x_opt"] = incumbent[:-1].tolist()
    results["incumbents"] = [inc.tolist() for inc in incumbents]
    results["runtime"] = runtime
    results["overhead"] = time_overhead
    results["time_func_eval"] = time_func_eval
    results["X"] = [x.tolist() for x in X]
    results["y"] = [np.exp(yi).tolist() for yi in y]
    results["c"] = [ci.tolist() for ci in c]

    if not os.path.exists(path_csv):
        os.makedirs(path_csv)

    df = pd.DataFrame(recording)  # 数据初始化成为DataFrame对象
    # df.to_csv(path_csv + '/demo.csv', index=False)  # 将数据写�??
    df.to_csv(path_csv + '/fabolas_seed_' + str(seed) + '.csv',
            index=False)

if __name__ == '__main__':
    Data_list = {'non_linear_sin': non_linear_sin, 'forrester': forrester, 'Park': Park, 'Branin': Branin, 
                 'Currin': Currin, 'borehole':borehole,'booth':booth,"bohachevsky":bohachevsky,
                 'himmelblau':himmelblau,'colvile':colvile,
                 'HeatedBlock': HeatedBlock,'VibratePlate': VibPlate
                 }
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--data_name", type=str, default="HeatedBlock")
    parser.add_argument("--cost_type", type=str, default="pow_10")
    args = parser.parse_args()
    Exp_marker = "Norm_res"
    data_name = args.data_name
    
    seed_dic ={'Currin':[0,2,3,4,6,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,28],'Branin':[0,2,3,4,5,7,8,10,11,13,14,15,16,17,18,19,20,21,22,24,25,27,29],'Park':[0,1,2,3,4,5,6,7,8,9,11,14,16,20,21,22,23,24,25,28,29],
           'non_linear_sin':[1,5,6,7,11,14,16,17,18,19,20,21,22,24,25,26,27,29],'forrester':[1,2,4,5,7,8,9,11,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29],
           'bohachevsky':[0,1,2,3,4,7,9,10,11,12,15,17,19,21,22,24,26,27],'VibratePlate':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
           'HeatedBlock':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]}
    
    # Set up logging
    log_file_path = os.path.join(sys.path[-1], 'Nips2025', 'CMF', 'Exp_results', Exp_marker, data_name, args.cost_type, 'fabolas.log')
    log_dir = os.path.dirname(log_file_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    for seed in seed_dic[data_name]:
        exp_configuration = {'seed': seed, 'data_name': data_name, 'data_model': Data_list[data_name], 'initial_num': [10,4], 'cost_type': args.cost_type,
                             'BO_iterations': 10, 'MF_iterations': 100, 'MF_learning_rate': 0.0001, 'Exp_marker': Exp_marker}

        fabolas(exp_config = exp_configuration)