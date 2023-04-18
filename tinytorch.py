import numpy as np

class Tensor:

    def __init__(self, values, requires_grad=False, dependency=None):
        self._values = np.array(values)
        self.shape = self.values.shape

        self.grad = None
        if requires_grad:
            self.zero_grad()
        self.requires_grad = requires_grad

        if dependency is None:
            dependency = []
        self.dependency = dependency

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        self._values = np.array(new_values)
        self.grad = None
        
    def shape(self):
        return self.values.shape
    
    def dim(self):
        return len(self.shape)

    def zero_grad(self):
        self.grad = np.zeros(self.shape)

    def backward(self, grad=None):
        assert self.requires_grad, "Call backward() on a non-requires-grad tensor."
        grad = np.ones(self.shape) if grad is None else grad
        grad = np.array(grad)

        # accumulate gradient
        self.grad += grad

        # propagate the gradient to its dependencies
        for dep in self.dependency:
            grad_for_dep = dep["grad_fn"](grad)
            dep["tensor"].backward(grad_for_dep)
            
    def __matmul__(self, x):
        res_values = self.values @ x.values
        requires_grad = self.requires_grad or x.requires_grad
        dependency = []
        if self.requires_grad:
            dependency.append({"tensor":self, "grad_fn":lambda grad:grad@x.values.T})
        if x.requires_grad:
            dependency.append({"tensor":x, "grad_fn":lambda grad:self.values.T @ grad})
            
        return Tensor(res_values, requires_grad, dependency)
    
    def __mul__(self, x):
        
        res_values = self.values * x.values
        requires_grad = self.requires_grad or x.requires_grad
        dependency = []
        if self.requires_grad:
            dependency.append({"tensor":self, "grad_fn":lambda grad:grad * x.values})
        if x.requires_grad:
            dependency.append({"tensor":x, "grad_fn":lambda grad:grad * self.values})
            
        return Tensor(res_values, requires_grad, dependency)
    
    
    def __add__(self, x):
        res_values = self.values + x.values
        requires_grad = self.requires_grad or x.requires_grad
        dependency = []
        if self.requires_grad:
            dependency.append({"tensor":self, "grad_fn":lambda grad:grad})
        if x.requires_grad:
            dependency.append({"tensor":x, "grad_fn":lambda grad:grad})
            
        return Tensor(res_values, requires_grad, dependency)
    
    def l2_norm(self, alpha):
        v = np.expand_dims(self.values.flatten(), -1)
        # print(v.shape)
        res_values = v.T @ v / 2
        return Tensor(res_values, self.requires_grad, [{"tensor":self, "grad_fn":lambda grad: grad*alpha * self.values}])
    
    def log(self):
        return Tensor(np.log(self.values), self.requires_grad,  [{"tensor":self, "grad_fn":lambda grad: grad / self.values}])
    
    
    @property
    def T(self):
        return Tensor(self.values.T, self.requires_grad, [{"tensor": self, "grad_fn":lambda grad:grad.T}])
    
    def scalar_mul(self, alpha):
        return Tensor(self.values*alpha, self.requires_grad, [{"tensor": self, "grad_fn":lambda grad:grad*alpha}])
    
    def scalar_add(self, alpha):
        return Tensor(self.values+alpha, self.requires_grad, [{"tensor": self, "grad_fn":lambda grad:grad}])
    
    def __neg__(self):
        return self.scalar_mul(-1)
    
    def __sub__(self, x):
        return self.__add__(x.scalar_mul(-1))
    
    def zero_(self):
        self._values.fill(0)
        
        return self
        
    def sub_(self, other, alpha):
        self._values -= alpha * other
        
        return self
        
    def add_(self, other, alpha):
        self._values += alpha * other
        return self
        
    def mul_(self, alpha):
        self._values *= alpha
        
        return self
             

def zeros_like(t):
    return Tensor(np.zeros(t.shape))
        
def Sigmoid(x):
    val = 1 / (1 + np.exp(-x.values))
    return Tensor(val, x.requires_grad, [{"tensor": x, "grad_fn": lambda grad: val*(1-val)*grad}])
    
def Tanh(x):
    val = np.tanh(x.values)
    return Tensor(val, x.requires_grad, [{"tensor": x, "grad_fn": lambda grad: (1-val**2)*grad}]) 
    
    
class Parameter(Tensor):
    param_list = []
    
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad)
        
        # self.data = data
        # self.requires_grad = requires_grad
        Parameter.param_list.append(self)
        # Parameter.register(self)
        
#     @classmethod
#     def register(cls, obj):
#         cls.param_list.append(obj)
    
#     @classmethod
#     def __new__(cls, data, requires_grad=True):
#         self = super().__new__(cls)
#         self.data = data
#         self.requires_grad = requires_grad
#         cls.param_list.append(self)
        
#         return self
    
    @classmethod
    def clear(cls):
        cls.param_list = []
    
class Linear:
    def __init__(self, in_features, out_features, use_bias=True):
        self.weight = Parameter(np.random.randn(in_features, out_features))
        self.use_bias = use_bias
        if use_bias:
            self.bias = Parameter(np.random.randn(1,out_features))
#         else:
#             # self.bias = Parameter(Tensor(np.zeros(out_features, 1)), False)
#             self.bias = Tensor(np.zeros(out_features, 1), requires_grad=False)
            
        
    def forward(self, input_tensor):
        if self.use_bias:
            
            return input_tensor @ self.weight + self.bias
        else:
            return input_tensor @ self.weight
        
        
    
        
class MLP:
    def __init__(self, input_size, output_size, num_units_list, activative_func, use_bias=True):
        self.layers = []
        
        size_list = [input_size] + num_units_list + [output_size]
        
        for i in range(len(size_list)-1):
            self.layers.append(Linear(size_list[i], size_list[i+1], use_bias=use_bias))
            
        if type(activative_func) == str:
            self.activative_func = get_activate_function_by_name(activative_func)
        else:
            self.activative_func = activative_func
    
    def forward(self, input_tensor):
        res = input_tensor
        for layer in self.layers[:-1]:
            res = layer.forward(res)
            res = self.activative_func(res)
            
        res = self.layers[-1].forward(res)
            
        return res
    

    
    
class MomentumOptimizer:
    def __init__(self, params, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = [zeros_like(param) for param in params ]
        # for i, param in enumerate(params):
            # self.velocities[i] = zeros_like(param)
            
        self.params = params
        
#     def params_and_grads(self):
#         for param in self.params:
#             yield param.values, param.grad
            
    def step(self):
        for i, param in enumerate(self.params):
            velocity = self.velocities[i]
            grad = param.grad
            velocity.mul_(self.momentum).add_(grad, alpha=1 - self.momentum)
            param.sub_(velocity.values, alpha=self.lr)
            
            param.zero_grad()

    def zero_grad(self):
        for velocity in self.velocities:
            velocity.zero_()
            
            
def square_loss(y_pred, y):
    delta_y = y_pred - y
    return (delta_y.T @ delta_y).scalar_mul(1/y_pred.shape[0])


def log_binary_loss(y_pred, y):
    return (- y.T @ Sigmoid(y_pred).log() + (y.scalar_add(-1)).T @ (-(Sigmoid(y_pred).scalar_add(-1))).log()).scalar_mul(1/y_pred.shape[0])



def get_activate_function_by_name(x):
    if x.lower() == "sigmoid":
        return Sigmoid
    elif x.lower() == "tanh":
        return Tanh
    # elif x.lower() == ""
