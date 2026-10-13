import torch

w = torch.tensor(2.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

for step in range(10):
    # 每轮根据当前的 w，重新计算损失
    loss = (w - 5.0) ** 2

    # TODO 1：清除旧梯度
    optimizer.zero_grad()

    # TODO 2：计算当前梯度
    loss.backward()

    # TODO 3：更新参数
    optimizer.step()

    # 用更新后的参数计算损失，方便观察结果
    w_after = w.item()
    loss_after = (w_after - 5.0) ** 2

    print(
        f"Step {step + 1}: "
        f"w = {w_after:.4f}, loss = {loss_after:.4f}"
    )