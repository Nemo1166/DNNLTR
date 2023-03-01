import numpy as np
import torch

def preproc(file_path: str):
    # 打开数据文件
    with open(file_path) as f:
        data = []
        # 逐行读取数据
        for line in f:
            # 将'#'分隔符之前的部分提取出来，然后按空格分割成若干个字段
            fields = line.strip().split('#')[0].strip().split(' ')
            # 提取标签值
            label = int(fields[0])
            # 提取特征值
            features = np.zeros(46)
            for field in fields[2:]:
                index, value = field.split(':')
                features[int(index) - 1] = float(value)
            # 将特征值和标签值存储为元组，并添加到列表中
            data.append(np.append(label,features))
        # 将列表转换为numpy数组
        return np.array(data)

def preproc_group(file_path: str):
    # 打开数据文件
    with open(file_path) as f:
        grouped_data = {}
        # 逐行读取数据
        for line in f:
            # 将'#'分隔符之前的部分提取出来，然后按空格分割成若干个字段
            fields = line.strip().split('#')[0].strip().split(' ')
            # qid
            qid = int(fields[1].split(':')[1])
            if qid not in grouped_data:
                grouped_data[qid] = []
            # 提取标签值
            label = int(fields[0])
            # 提取特征值
            features = np.zeros(46)
            for field in fields[2:]:
                index, value = field.split(':')
                features[int(index) - 1] = float(value)
                # 将特征值和标签值存储为元组，并添加到列表中
                grouped_data[qid].append(np.append(label,features))
    for qid in grouped_data.keys():
        grouped_data[qid] = np.array(grouped_data[qid])
        grouped_data[qid] = (
            torch.from_numpy(grouped_data[qid][:, 1:]).unsqueeze(1).float(),
            torch.from_numpy(grouped_data[qid][:, 0]).float()
        )
    return grouped_data