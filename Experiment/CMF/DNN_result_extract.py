import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

def read_txt_file(file_path, data_name, max_value, seed):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            recording = {'cost':[], 'SR': [], 'operation_time': []}
            for line in file:
                content = line.strip()
                if content[:19] == '* simple optimum = ':
                    recording['SR'].append(max_value - float(content[20:-1]))
                elif content[:15] == '* cost so far: ':
                    recording['cost'].append(float(content[-3:]))
                elif content[:15] == "* time spent = ":
                    recording['operation_time'].append(float(content[15:]))
            df = pd.DataFrame(recording)
            if data_name == 'NonLinearSin':
                data_to_name = 'non_linear_sin'
            elif data_name == 'Forrester':
                data_to_name = 'forrester'
            else:
                data_to_name = data_name
            df.to_csv(sys.path[-1] + '/Exp_results/' + data_to_name + '/pow_10/DNN_MFBO_seed_' + str(seed-1) + '.csv', index=False)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # 指定要读取的txt文件的路径
    file_path = sys.path[-1]
    data_name = 'NonLinearSin'
    if data_name == 'NonLinearSin':
        max_value = 0.033
    elif data_name == 'Forrester':
        max_value = 48.51
    for seed in range(1, 7):
        read_txt_file(file_path + '/we_used/log-' + data_name + str(seed) + '.txt', data_name, max_value, seed)
