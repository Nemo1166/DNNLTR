import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

def train(model, train_loader, val_loader, epochs):
    """
    训练PyTorch模型。

    参数：
    model: 要训练的PyTorch模型。
    train_loader: 训练数据的dataloader。
    val_loader: 验证数据的dataloader。
    epochs: 训练的轮数。

    返回：
    train_losses: 训练过程中每个epoch的训练损失。
    val_losses: 训练过程中每个epoch的验证损失。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader), 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证模型
        val_loss = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += loss_fn(val_outputs, val_labels).item()

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print('[%d] validation loss: %.3f' % (epoch + 1, val_loss / len(val_loader)))

    print('Finished Training')

    return train_losses, val_losses

def train_groupdata(model, train_loader, val_loader, epochs, evaluator, k):
    """
    训练PyTorch模型。

    参数：
    model: 要训练的PyTorch模型。
    train_loader: 训练数据的dataloader。
    val_loader: 验证数据的dataloader。
    epochs: 训练的轮数。
    evaluator: 排序评估器。
        evaluator(predictions: iterable, labels: iterable) -> float

    返回：
    train_losses: 训练过程中每个epoch的训练损失。
    val_losses: 训练过程中每个epoch的验证损失。
    val_rankscore: 训练过程中每个epoch的验证排序得分。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    train_losses = []
    val_losses = []
    val_rankscore = []

    for epoch in trange(epochs, position=0, desc="Training"):
        running_loss = 0.0
        for _, data in enumerate(tqdm(train_loader, delay=3, position=1, leave=False, desc=f"Epoch {epoch}"), 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证模型
        val_loss = 0.0
        val_ranker = 0.0
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                val_inputs, val_labels = val_data
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += loss_fn(val_outputs, val_labels).item()
                val_ranker += evaluator(val_outputs, val_labels, k)

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_rankscore.append(val_ranker / len(val_loader))

        # tqdm.write('[Epoch %d] validation loss: %.3f; Ranking score: %.4f' % (
        #     epoch + 1, val_loss / len(val_loader), val_ranker / len(val_loader)))

    print('Finished Training')

    return train_losses, val_losses, val_rankscore, np.mean(np.array(val_rankscore))

def test(model, rank_evaluator, test_loader, k):
    """
    使用测试集测试 PyTorch 模型

    参数:
        model: PyTorch 模型
        criterion: 损失函数
        test_loader: 测试集数据加载器
        k: k of NDCG@k

    返回:
        test_loss: 测试集的平均损失
        accuracy: 测试集的准确率
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将模型设置为评估模式
    model.eval()

    # 初始化测试损失和正确预测的数量
    rank_scores = []

    # 禁用梯度计算
    with torch.no_grad():
        # 遍历测试集数据
        for data, target in tqdm(test_loader, desc="Testing"):
            # 将数据移动到指定的设备
            data, target = data.to(device), target.to(device)
            # 进行前向传递
            output = model(data)
            # 计算排序得分
            rank_scores.append(rank_evaluator(output, target, k))

    # 返回测试集的排序得分
    return np.mean(np.array(rank_scores))

def ndcg_evaluator(y_pred: torch.Tensor, y_true: torch.Tensor, k: int=10) -> float:
    topk = min(k,len(y_pred))
    _, t_indices = torch.topk(y_pred, topk)
    t_indices = t_indices.cpu().numpy()
    y_true = y_true.cpu().numpy()
    dcg = np.nansum(1.0 / np.log2(np.arange(2, topk + 2)) * (y_true[t_indices]))
    idcg = np.nansum(1.0 / np.log2(np.arange(2, topk + 2)) * (np.sort(y_true, kind='stable')[::-1][:topk]))
    return 0.0 if idcg < 10e-5 else dcg / idcg
    # return dcg / idcg
