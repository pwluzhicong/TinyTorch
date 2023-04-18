import numpy as np
from tinytorch import Tensor, MLP, Parameter, MomentumOptimizer, Linear, Sigmoid, square_loss, log_binary_loss
from dataset import DataLoader

def train_func(**params):
    num_units = [int(x) for x in params["num_units"].split(',')]
    activation_func = params["activation_func"]
    batch_size = int(params["batch_size"])
    num_epochs = int(params["num_epochs"])
    learning_rate = float(params["learning_rate"])
    momentum = float(params["momentum"])
    l2_norm_coef = float(params["l2_norm"])
    
    test_ratio = float(params["test_ratio"])
    input_data = np.genfromtxt(params["input_data"], delimiter=',')[1:]
    
    train_data = input_data[: int(input_data.shape[0] * (1-test_ratio))]
    test_data = input_data[int(input_data.shape[0] * (1-test_ratio)):]
    
    
    log_step = int(params.get("log_step", 1))
    
    loss_func = None
    
    if params["loss_func"] == "square_loss":
        loss_func = square_loss
    elif params["loss_func"] == "log_binary_loss":
        loss_func = log_binary_loss
    assert train_data.shape[-1] == test_data.shape[-1]
    
    print(train_data.shape, test_data.shape)
    
    Parameter.clear()
    mlp = MLP(train_data.shape[-1]-1, 1, num_units, "Sigmoid", use_bias=False)
    output_layer = Linear(num_units[-1], 1, use_bias=False)
    
    optimizer = MomentumOptimizer(Parameter.param_list, learning_rate, momentum)

    
    for epoch in range(num_epochs):
        
        for batch_idx, (batch_X, batch_y) in enumerate(DataLoader(train_data, shuffle=True, batch_size=16)):
            # print(batch_idx)
            input_x = Tensor(batch_X, requires_grad=False)
            output_y = Tensor(np.expand_dims(batch_y,-1), requires_grad=False)
            # print(input_x.values)
            y_pred = mlp.forward(input_x)
            loss = loss_func(y_pred,  output_y)

            l2_loss = Tensor(np.zeros((1,1)))
            for param in optimizer.params:
                l2_loss = l2_loss + param.l2_norm(l2_norm_coef)

            total_loss = loss + l2_loss
            total_loss.backward()
            # l2_loss.backward()
            optimizer.step()
        # break
            
            
        # if(epoch % 10 == 0):
        if epoch % log_step == 0:
            print("Epoch %d" % epoch)
            for batch_idx, (batch_X, batch_y) in enumerate(DataLoader(train_data, shuffle=True, batch_size=train_data.shape[0])):
                input_x = Tensor(batch_X, requires_grad=False)
                output_y = Tensor(np.expand_dims(batch_y,-1), requires_grad=False)

                y_pred = mlp.forward(input_x)
                loss = loss_func(y_pred,  output_y)

                l2_loss = Tensor(np.zeros((1,1)))
                for param in optimizer.params:
                    l2_loss = l2_loss + param.l2_norm(l2_norm_coef)

                # total_loss = loss + l2_loss
                print("train_loss:", loss.values)
                # print("train_l2_loss:", l2_loss.values)
            
            # print("Epoch %d" % epoch)
            for batch_idx, (batch_X, batch_y) in enumerate(DataLoader(test_data, shuffle=True, batch_size=test_data.shape[0])):
                input_x = Tensor(batch_X, requires_grad=False)
                output_y = Tensor(np.expand_dims(batch_y,-1), requires_grad=False)

                y_pred = mlp.forward(input_x)
                loss = loss_func(y_pred,  output_y)

                l2_loss = Tensor(np.zeros((1,1)))
                for param in optimizer.params:
                    l2_loss = l2_loss + param.l2_norm(l2_norm_coef)

                # loss = loss + l2_loss
                print("test_loss:", loss.values)
                # print("test_l2_loss:", l2_loss.values)
#             if(batch_idx % 10) == 0:
#                 print(epoch, batch_idx, loss.values)
    