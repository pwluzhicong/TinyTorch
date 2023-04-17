import numpy as np
from tinytorch import Tensor, MLP, Parameter, MomentumOptimizer, Linear, Sigmoid, square_loss

x = np.random.randn(20, 10)
w = np.random.randn(10, 1)
y = np.sin(x@w) + np.random.randn(20, 1) * 0.01
# y = np.sin(w@x)
y.shape
# for i in range(10):
#     mlp.forward()

Parameter.clear()
mlp = MLP(10, 1, [16, 8], Sigmoid, use_bias=False)
output_layer = Linear(8,1, use_bias=False)

optimizer = MomentumOptimizer(Parameter.param_list, 1e-3, 0.9)

params_raw = optimizer.params[0].values.copy()

input_x = Tensor(x, requires_grad=False)

output_y = Tensor(y, requires_grad=False)

for i in range(10000):
    # out_tensor = output_layer.forward(mlp.forward(Tensor(x, requires_grad=False))) - Tensor(y, requires_grad=False)
    # loss = out_tensor.T @ out_tensor
    y_pred = mlp.forward(input_x)
    loss = square_loss(y_pred,  output_y)
    loss.backward()
    # out_tensor.backward()

    
    optimizer.step()
    
    if(i % 1000 == 0):
        print(loss.values[0][0])
    # params2 = optimizer.params[0].values.copy()