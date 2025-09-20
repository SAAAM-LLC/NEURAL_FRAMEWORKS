import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod
import pickle
import json

class Tensor:
    """
    Core tensor class with automatic differentiation support
    """
    def __init__(self, data: np.ndarray, requires_grad: bool = False, name: str = ""):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self.name = name
        self._backward_fn = None
        self._parents = []
        
        if requires_grad:
            self.grad = np.zeros_like(self.data)
    
    def backward(self, grad_output: Optional[np.ndarray] = None):
        """Backward pass for automatic differentiation"""
        if not self.requires_grad:
            return
            
        if grad_output is None:
            grad_output = np.ones_like(self.data)
        
        self.grad += grad_output
        
        if self._backward_fn is not None:
            self._backward_fn(grad_output)
    
    def zero_grad(self):
        """Reset gradients to zero"""
        if self.grad is not None:
            self.grad.fill(0)
    
    # Mathematical operations with gradient tracking
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        
        result = Tensor(self.data + other.data, 
                       requires_grad=self.requires_grad or other.requires_grad)
        
        def backward_fn(grad_output):
            if self.requires_grad:
                self.backward(grad_output)
            if other.requires_grad:
                other.backward(grad_output)
        
        result._backward_fn = backward_fn
        return result
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        
        result = Tensor(self.data * other.data,
                       requires_grad=self.requires_grad or other.requires_grad)
        
        def backward_fn(grad_output):
            if self.requires_grad:
                self.backward(grad_output * other.data)
            if other.requires_grad:
                other.backward(grad_output * self.data)
        
        result._backward_fn = backward_fn
        return result
    
    def __matmul__(self, other):
        """Matrix multiplication with gradient support"""
        result = Tensor(np.matmul(self.data, other.data),
                       requires_grad=self.requires_grad or other.requires_grad)
        
        def backward_fn(grad_output):
            if self.requires_grad:
                self.backward(np.matmul(grad_output, other.data.T))
            if other.requires_grad:
                other.backward(np.matmul(self.data.T, grad_output))
        
        result._backward_fn = backward_fn
        return result
    
    def sum(self, axis=None, keepdims=False):
        """Sum with gradient support"""
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                       requires_grad=self.requires_grad)
        
        def backward_fn(grad_output):
            if self.requires_grad:
                if axis is None:
                    grad = np.full_like(self.data, grad_output)
                else:
                    grad = np.broadcast_to(np.expand_dims(grad_output, axis), self.data.shape)
                self.backward(grad)
        
        result._backward_fn = backward_fn
        return result
    
    @property
    def shape(self):
        return self.data.shape
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, name='{self.name}')"


class Module(ABC):
    """Base class for all neural network modules"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.training = True
        self._parameters = {}
        self._modules = {}
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters"""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def train(self):
        self.training = True
        for module in self._modules.values():
            module.train()
    
    def eval(self):
        self.training = False
        for module in self._modules.values():
            module.eval()
    
    def zero_grad(self):
        """Zero gradients for all parameters"""
        for param in self.parameters():
            param.zero_grad()


class Linear(Module):
    """Linear/Dense layer"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name: str = "Linear"):
        super().__init__(name)
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self._parameters['weight'] = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)),
            requires_grad=True, name=f"{name}.weight"
        )
        
        if bias:
            self._parameters['bias'] = Tensor(
                np.zeros(out_features),
                requires_grad=True, name=f"{name}.bias"
            )
        else:
            self._parameters['bias'] = None
    
    def forward(self, x: Tensor) -> Tensor:
        output = x @ self._parameters['weight']
        if self._parameters['bias'] is not None:
            output = output + self._parameters['bias']
        return output


class Activation(Module):
    """Base activation class"""
    pass


class ReLU(Activation):
    """ReLU activation function"""
    
    def __init__(self, name: str = "ReLU"):
        super().__init__(name)
    
    def forward(self, x: Tensor) -> Tensor:
        result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        
        def backward_fn(grad_output):
            if x.requires_grad:
                grad = grad_output * (x.data > 0).astype(np.float32)
                x.backward(grad)
        
        result._backward_fn = backward_fn
        return result


class Sigmoid(Activation):
    """Sigmoid activation function"""
    
    def __init__(self, name: str = "Sigmoid"):
        super().__init__(name)
    
    def forward(self, x: Tensor) -> Tensor:
        sigmoid_data = 1 / (1 + np.exp(-np.clip(x.data, -500, 500)))
        result = Tensor(sigmoid_data, requires_grad=x.requires_grad)
        
        def backward_fn(grad_output):
            if x.requires_grad:
                grad = grad_output * sigmoid_data * (1 - sigmoid_data)
                x.backward(grad)
        
        result._backward_fn = backward_fn
        return result


class Loss(ABC):
    """Base loss function class"""
    
    @abstractmethod
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        pass


class MeanSquaredError(Loss):
    """Mean Squared Error loss"""
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        diff = predictions.data - targets.data
        loss_data = np.mean(diff ** 2)
        loss = Tensor(loss_data, requires_grad=predictions.requires_grad)
        
        def backward_fn(grad_output):
            if predictions.requires_grad:
                grad = 2 * diff / diff.size * grad_output
                predictions.backward(grad)
        
        loss._backward_fn = backward_fn
        return loss


class CrossEntropyLoss(Loss):
    """Cross-entropy loss with softmax"""
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Softmax with numerical stability
        exp_preds = np.exp(predictions.data - np.max(predictions.data, axis=1, keepdims=True))
        softmax = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        # Cross-entropy loss
        log_softmax = np.log(softmax + 1e-15)
        loss_data = -np.mean(np.sum(targets.data * log_softmax, axis=1))
        loss = Tensor(loss_data, requires_grad=predictions.requires_grad)
        
        def backward_fn(grad_output):
            if predictions.requires_grad:
                grad = (softmax - targets.data) / predictions.shape[0] * grad_output
                predictions.backward(grad)
        
        loss._backward_fn = backward_fn
        return loss


class Optimizer(ABC):
    """Base optimizer class"""
    
    def __init__(self, parameters: List[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr
    
    @abstractmethod
    def step(self):
        pass
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in parameters]
    
    def step(self):
        for param, velocity in zip(self.parameters, self.velocities):
            if param.grad is not None:
                velocity *= self.momentum
                velocity += self.lr * param.grad
                param.data -= velocity


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        
        self.m = [np.zeros_like(param.data) for param in parameters]
        self.v = [np.zeros_like(param.data) for param in parameters]
    
    def step(self):
        self.step_count += 1
        
        for param, m, v in zip(self.parameters, self.m, self.v):
            if param.grad is not None:
                # Update biased first moment estimate
                m *= self.beta1
                m += (1 - self.beta1) * param.grad
                
                # Update biased second raw moment estimate
                v *= self.beta2
                v += (1 - self.beta2) * (param.grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - self.beta1 ** self.step_count)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = v / (1 - self.beta2 ** self.step_count)
                
                # Update parameters
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Sequential(Module):
    """Sequential container for modules"""
    
    def __init__(self, *modules, name: str = "Sequential"):
        super().__init__(name)
        for i, module in enumerate(modules):
            self._modules[f"layer_{i}"] = module
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x


# Training utilities
class DataLoader:
    """Simple data loader for batching"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]
    
    def __len__(self):
        return self.n_batches


class Trainer:
    """Training orchestrator"""
    
    def __init__(self, model: Module, loss_fn: Loss, optimizer: Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # Convert to tensors
            X = Tensor(batch_X, requires_grad=False)
            y = Tensor(batch_y, requires_grad=False)
            
            # Forward pass
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.data
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        for batch_X, batch_y in val_loader:
            X = Tensor(batch_X, requires_grad=False)
            y = Tensor(batch_y, requires_grad=False)
            
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)
            total_loss += loss.data
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, epochs: int, val_loader: Optional[DataLoader] = None):
        """Full training loop"""
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
    
    def save_model(self, filepath: str):
        """Save model state"""
        state = {}
        for name, param in self.model._parameters.items():
            state[name] = param.data
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_model(self, filepath: str):
        """Load model state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        for name, param in self.model._parameters.items():
            if name in state:
                param.data = state[name]


# Example usage and demonstration
if __name__ == "__main__":
    print("🚀 Neural Framework Initialized!")
    print("Example: Creating a simple neural network...")
    
    # Create a simple 2-layer network
    model = Sequential(
        Linear(4, 10, name="hidden"),
        ReLU(),
        Linear(10, 3, name="output"),
        Sigmoid()
    )
    
    # Create sample data
    X_train = np.random.randn(100, 4)
    y_train = np.random.randint(0, 3, (100, 3))  # One-hot encoded
    
    # Setup training
    train_loader = DataLoader(X_train, y_train, batch_size=16)
    loss_fn = MeanSquaredError()
    optimizer = Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, loss_fn, optimizer)
    
    print(f"Model has {len(model.parameters())} trainable parameters")
    print("Framework is ready for training! 🔥")
