import os
import re

def process_files(folder_path, seed_list, new_seed_list):
    if len(seed_list) != len(new_seed_list):
        raise ValueError("seed_list and new_seed_list must have the same length.")

    # 将 seed_list 和 new_seed_list 转换为字符串形式，方便匹配文件名
    seed_map = {str(old): str(new) for old, new in zip(seed_list, new_seed_list)}

    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 正则匹配文件名格式 "XX_seed_数字.csv"
    pattern = re.compile(r"^.*_seed_(\d+)\.csv$")

    # 遍历文件，删除不在 seed_list 中的文件
    for file in files:
        match = pattern.match(file)
        if match:
            seed = match.group(1)  # 提取 seed 值
            if seed not in seed_map:
                os.remove(os.path.join(folder_path, file))
                print(f"Deleted: {file}")
        else:
            print(f"Skipped: {file} (not matching the pattern)")

    # 获取剩余文件并排序
    remaining_files = [f for f in os.listdir(folder_path) if pattern.match(f)]
    remaining_files.sort(key=lambda x: int(pattern.match(x).group(1)))

    # 按 new_seed_list 重命名
    for file in remaining_files:
        match = pattern.match(file)
        if match:
            old_seed = match.group(1)
            new_seed = seed_map.get(old_seed)
            if new_seed is not None:
                old_path = os.path.join(folder_path, file)
                new_name = re.sub(r"_seed_\d+", f"_seed_{new_seed}", file)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {file} -> {new_name}")

# 示例用法
folder_path = "F:\Github_project\my-CAMO\CAMO\Data_process\pow_10"  # 替换为你的文件夹路径
seed_list =     [1,3,4,5,6,7,11,14,16,17,18,19,20,21,22,24,25,26,27,29]           # 替换为你的 seed 列表
new_seed_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]   # 替换为新的 seed 列表
process_files(folder_path, seed_list, new_seed_list)
