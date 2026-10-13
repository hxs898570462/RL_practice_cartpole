import torch

w = torch.tensor(8.0, requires_grad=True)
loss = (w - 5.0) ** 2

print("w before backward:", w.item())
print("Loss:", loss.item())

# TODO 1：对 loss 调用 backward 方法，计算梯度
loss.backward()

# TODO 2：读取 w 的 grad 属性
print("Gradient:", w.grad.item())

print("w after backward:", w.item())