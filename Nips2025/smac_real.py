import numpy as np
import pandas as pd
import torch
import os
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario
import sys
import os

import time
import argparse
from Data.RF_Classifier import RF_Classifier
from Data.ANN_Classifer import ANN_Classifier

class SyntheticFunction:
    def __init__(self, data_config, seed):
        self.data_model = data_config["data_model"](total_fidelity_num = data_config["total_fidelity_num"])
        self.model_cost = self.data_model.cost
        self.data_name = data_config["data_name"]
        if self.data_name == "HeatedBlock":
            self.x_range = [[0.1, 0.4], [0.1, 0.4], [0, 2*np.pi]]
        elif self.data_name == "RF_Classifier":
            self.x_range = [[5,50],[2,11],[1,11],[0,1],[1,64]]
        elif self.data_name == "ANN_Classifier":
            self.x_range = [[0, 1], [10, 100], [0, 1], [16, 64]]
        
        self.fidelity_range = (10, 100)
        self.seed = seed

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        for i in range(len(self.x_range)):
            x = Float("x" + str(i + 1), self.x_range[i])
            cs.add([x])

        s = Float("s", self.fidelity_range)
        cs.add([s])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = [config["x" + str(i + 1)] for i in range(len(self.x_range))]
        s = config["s"]
        x.append(s)
        result = self.data_model.get_acc(torch.tensor(x))
        return -result
    
    def plot(self, x, s):
        return self.data_model.get_acc(torch.cat([x, torch.tensor([s])],dim=0))
    def find_max(self,):
        return self.data_model.find_max_value_in_range()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--data_name", type=str, default="ANN_Classifier")
    parser.add_argument("--start_seed", type=int, default= 5)
    
    Data_list = {'RF_Classifier': RF_Classifier,'ANN_Classifier': ANN_Classifier}
    args = parser.parse_args()
    data_name = args.data_name
    for seed in range(args.start_seed, 10):
        recording = {"err": [], 'wallclocktime':[], 'operation_time':[]}
        t1 = time.time()

        data_config = {"data_name": data_name, "data_model": Data_list[data_name], "cost_type": "pow_10", "total_fidelity_num": 2}
        model = SyntheticFunction(data_config, seed = seed)

        # Scenario object specifying the optimization "environment"
        scenario = Scenario(model.configspace, deterministic=True, n_trials=100,seed=seed)

        # Now we use SMAC to find the best hyperparameters
        smac = HPOFacade(
            scenario,
            model.train,  # We pass the target function here
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        )

        incumbent = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(model.configspace.get_default_configuration())
        print(f"Default cost: {default_cost}")

        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        print(f"Incumbent cost: {incumbent_cost}")
        t2 = time.time()

        # Store the results
        data = []
        data_y = []
        best_acc = 0
        for k, v in smac.runhistory.items():
            config = smac.runhistory.get_config(k.config_id)
            x = [config["x" + str(i + 1)] for i in range(len(model.x_range))]
            s = [config["s"]]
            data.append(x + s)
            y = v.cost  # type: ignore # 因为y是相反数-y
            acc = model.plot(torch.tensor(x), torch.tensor(model.fidelity_range[-1])).item()
            
            if acc > best_acc:
                best_acc = acc
            t4 = time.time()
            recording["err"].append(1-best_acc)
            recording["operation_time"].append(v.time)
            recording["wallclocktime"].append(time.time()-t1)

        df = pd.DataFrame(recording)
        directory = '/home/fillip/桌面/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/Exp_res/ANN-HPO'
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(directory + '/smac_seed_' + str(seed) + '.csv', index=False)

