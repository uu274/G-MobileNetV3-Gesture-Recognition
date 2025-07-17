# #开发时间: 2023/12/27 15:14

import torch
from torchsummary import summary
from G-MobileNetV3 import mobilenet_v3_large  # Replace with the actual module where your model is defined

# Initialize the model
model = mobilenet_v3_large(num_classes=1000)

# Print the model summary
summary(model, (3, 224, 224))  # Assuming input size is 224x224 and 3 channels





# import torch
# from torch import nn
# from model_v3 import mobilenet_v3_small
# model = mobilenet_v3_small()
#
# # 统计模型参数量
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# total_params = count_parameters(model)
# print(f"Total trainable parameters: {total_params}")
#
# # 打印每一层的参数量
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()}")
#
# # 打印每一层的参数量和总参数量
# print(model)


# import numpy as np
# from model_v3 import mobilenet_v3_small
# import torch
# from torch.backends import cudnn
# import tqdm
# cudnn.benchmark = True
#
# device = 'cuda:0'
# model = mobilenet_v3_small().to(device)
# repetition)
# s = 300
#
# dummy_input = torch.rand(1, 3, 256, 256).to(device)
#
# # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
# print(r'warm up ...\n')
# with torch.no_grad():
#     for _ in range(100):
#         _ = model(dummy_input)
#
# # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
# torch.cuda.synchronize()
#
#
# # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# # 初始化一个时间容器
# timings = np.zeros((repetitions, 1))
#
# print(r'testing ...\n')
# with torch.no_grad():
#     for rep in tqdm.tqdm(range(repetitions)):
#         starter.record()
#         _ = model(dummy_input)
#         ender.record()
#         torch.cuda.synchronize() # 等待GPU任务完成
#         curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
#         timings[rep] = curr_time
#
# avg = timings.sum()/repetitions
# print(r'\navg={}\n'.format(avg))
