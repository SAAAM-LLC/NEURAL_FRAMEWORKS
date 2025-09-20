#!/usr/bin/env python3
"""
🚀 TernaryCore: DEEP LEARNING FRAMEWORK 🚀
We're building  AI  from first principles!
Designed exclusively for {-1, 0, 1} operations.
No dependencies. No compromises. Pure mathematical beauty.
AUTHOR:  SAAAM LLC    |  2025  |  Michael Wofford
"""

import numpy as np
import ctypes
import os
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import struct

# Try to use available acceleration, but don't require it
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Import BuckshotKernels for hardware acceleration
try:
    import sys
    import os
    buckshot_path = os.path.join(os.path.dirname(__file__), 'buckshotkernels-main')
    if buckshot_path not in sys.path:
        sys.path.append(buckshot_path)

    from buckshotkernels import TernaryKernelManager, OptimizedTernaryTensor
    BUCKSHOT_AVAILABLE = True
    print("🚀 BuckshotKernels loaded - hardware acceleration enabled!")
except ImportError as e:
    BUCKSHOT_AVAILABLE = False
    print(f"⚠️ BuckshotKernels not available: {e}")
    print("Falling back to pure numpy implementation")

# =====================================================================================
# CORE MATHEMATICS ENGINE - PURE TERNARY ARITHMETIC
# =====================================================================================

# Global kernel manager instance
_kernel_manager = None

def get_kernel_manager():
    """Get or create the global kernel manager"""
    global _kernel_manager
    if _kernel_manager is None and BUCKSHOT_AVAILABLE:
        try:
            _kernel_manager = TernaryKernelManager()
            print("✅ TernaryKernelManager initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize TernaryKernelManager: {e}")
            _kernel_manager = None
    return _kernel_manager

class TernaryMath:
    """Hardware-accelerated mathematical operations for ternary numbers."""

    @staticmethod
    def ternary_quantize(x: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        """Convert real numbers to {-1, 0, 1} representation with hardware acceleration"""
        kernel_manager = get_kernel_manager()

        if kernel_manager is not None:
            try:
                # Use hardware-accelerated quantization
                return kernel_manager.ternary_quantize(x, threshold=threshold)
            except Exception as e:
                print(f"⚠️ Hardware quantization failed, falling back to numpy: {e}")

        # Fallback to pure numpy implementation
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        mask = abs_x > threshold
        return sign_x * mask.astype(np.float32)
    
    @staticmethod
    def ternary_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ternary 'multiplication' - actually just sign operations and masks"""
        # Since values are {-1, 0, 1}, multiplication becomes:
        # - Sign multiplication for non-zero elements
        # - Zero propagation
        zero_mask_a = (a == 0)
        zero_mask_b = (b == 0)
        zero_mask = zero_mask_a | zero_mask_b
        
        result = np.sign(a) * np.sign(b)
        result[zero_mask] = 0
        return result.astype(np.float32)
    
    @staticmethod
    def ternary_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Hardware-accelerated matrix multiplication optimized for ternary values"""
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Input arrays must be 2D")

        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Cannot multiply {a.shape} by {b.shape}")

        kernel_manager = get_kernel_manager()

        # Try hardware acceleration first
        if kernel_manager is not None:
            try:
                # Convert to int8 for hardware kernels if needed
                a_int8 = a.astype(np.int8) if a.dtype != np.int8 else a
                b_int8 = b.astype(np.int8) if b.dtype != np.int8 else b

                # Check if values are actually ternary
                a_unique = np.unique(a_int8)
                b_unique = np.unique(b_int8)

                if (all(val in [-1, 0, 1] for val in a_unique) and
                    all(val in [-1, 0, 1] for val in b_unique)):
                    # Use hardware-accelerated ternary matmul
                    result = kernel_manager.ternary_matmul(a_int8, b_int8)
                    return result.astype(np.float32)
            except Exception as e:
                print(f"⚠️ Hardware matmul failed, falling back to numpy: {e}")

        # Fallback to optimized numpy implementation
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)

        # Custom ternary matrix multiplication
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                # Dot product of row i and column j
                row = a[i, :]
                col = b[:, j]

                # Skip zero elements for efficiency
                nonzero_mask = (row != 0) & (col != 0)
                if not np.any(nonzero_mask):
                    result[i, j] = 0
                    continue

                # For ternary, this is just counting sign agreements
                signs = row[nonzero_mask] * col[nonzero_mask]
                result[i, j] = np.sum(signs)

        return result
    
    @staticmethod
    def ternary_conv2d(input_array: np.ndarray, kernel: np.ndarray, 
                      stride: int = 1, padding: int = 0) -> np.ndarray:
        """2D convolution optimized for ternary weights"""
        # Add padding if needed
        if padding > 0:
            input_array = np.pad(input_array, 
                               ((0, 0), (padding, padding), (padding, padding)), 
                               mode='constant', constant_values=0)
        
        batch_size, in_height, in_width = input_array.shape
        kernel_height, kernel_width = kernel.shape[-2:]
        
        # Calculate output dimensions
        out_height = (in_height - kernel_height) // stride + 1
        out_width = (in_width - kernel_width) // stride + 1
        
        # Initialize output
        if kernel.ndim == 4:  # (out_channels, in_channels, h, w)
            out_channels = kernel.shape[0]
            output = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.float32)
            
            for batch in range(batch_size):
                for out_ch in range(out_channels):
                    for y in range(out_height):
                        for x in range(out_width):
                            y_start = y * stride
                            x_start = x * stride
                            
                            # Extract patch
                            patch = input_array[batch, 
                                              y_start:y_start + kernel_height,
                                              x_start:x_start + kernel_width]
                            
                            # Ternary convolution: element-wise multiply and sum
                            conv_sum = 0
                            for in_ch in range(kernel.shape[1]):
                                if in_ch < patch.shape[0]:
                                    result = TernaryMath.ternary_multiply(
                                        patch[in_ch], kernel[out_ch, in_ch]
                                    )
                                    conv_sum += np.sum(result)
                            
                            output[batch, out_ch, y, x] = conv_sum
        else:
            raise ValueError("Kernel must be 4D (out_channels, in_channels, height, width)")
        
        return output


# Add Numba acceleration if available
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def fast_ternary_matmul(a, b):
        """Numba-accelerated ternary matrix multiplication"""
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
        
        for i in prange(a.shape[0]):
            for j in prange(b.shape[1]):
                dot_sum = 0.0
                for k in range(a.shape[1]):
                    if a[i, k] != 0 and b[k, j] != 0:
                        dot_sum += a[i, k] * b[k, j]
                result[i, j] = dot_sum
        
        return result
    
    # Replace the method if Numba is available
    TernaryMath.ternary_matmul = fast_ternary_matmul


# =====================================================================================
# TERNARY TENSOR - PURE IMPLEMENTATION
# =====================================================================================

class TernaryTensor:
    """
    Pure tensor implementation for ternary deep learning.
    NO PyTorch. NO TensorFlow. Just pure mathematics.
    """
    
    def __init__(self, data, requires_grad=False, device='cpu', name=None, use_hardware=True):
        # Core data
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)

        # Gradient computation
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.device = device
        self.name = name or f"tensor_{id(self)}"

        # Ternary properties
        self.is_ternary = False
        self.threshold = 0.05

        # Hardware acceleration
        self.use_hardware = use_hardware and BUCKSHOT_AVAILABLE
        self._kernel_manager = get_kernel_manager() if self.use_hardware else None

        # Computation graph
        self._children = []
        self._op = None
        self._version = 0

        # Performance tracking
        self._creation_time = time.time()
        self._forward_count = 0
        self._backward_count = 0

    def __repr__(self):
        shape_str = "x".join(map(str, self.shape))
        grad_str = f", grad={self.requires_grad}" if self.requires_grad else ""
        ternary_str = f", ternary={self.is_ternary}" if self.is_ternary else ""
        return f"TernaryTensor([{shape_str}]{grad_str}{ternary_str})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def item(self):
        """Get scalar value"""
        if self.data.size != 1:
            raise ValueError("Only one element tensors can be converted to Python scalars")
        return self.data.item()

    def numpy(self):
        """Get numpy array"""
        return self.data.copy()

    def to_ternary(self, threshold=None):
        """Convert to ternary representation"""
        if threshold is not None:
            self.threshold = threshold
        
        ternary_data = TernaryMath.ternary_quantize(self.data, self.threshold)
        result = TernaryTensor(ternary_data, requires_grad=self.requires_grad, device=self.device)
        result.is_ternary = True
        result.threshold = self.threshold
        
        if self.requires_grad:
            result.grad_fn = TernaryQuantizeFunction(self)
        
        return result

    def backward(self, gradient=None):
        """Compute gradients and propagate backward"""
        if not self.requires_grad:
            return
        
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            gradient = TernaryTensor(np.ones_like(self.data))
        
        # Initialize gradient
        if self.grad is None:
            self.grad = TernaryTensor(np.zeros_like(self.data))
        
        # Accumulate gradient
        self.grad.data += gradient.data
        self._backward_count += 1
        
        # Propagate backward
        if self.grad_fn is not None:
            self.grad_fn.backward(gradient)

    def zero_grad(self):
        """Zero gradients"""
        if self.grad is not None:
            self.grad.data.fill(0)

    # Arithmetic operations
    def __add__(self, other):
        return ternary_add(self, other)

    def __sub__(self, other):
        return ternary_sub(self, other)

    def __mul__(self, other):
        return ternary_mul(self, other)

    def __matmul__(self, other):
        return ternary_matmul(self, other)

    def __neg__(self):
        return ternary_neg(self)

    def reshape(self, *shape):
        """Reshape tensor"""
        new_data = self.data.reshape(shape)
        result = TernaryTensor(new_data, requires_grad=self.requires_grad, device=self.device)
        
        if self.requires_grad:
            result.grad_fn = ReshapeFunction(self, self.shape)
        
        return result

    def transpose(self, dim0=0, dim1=1):
        """Transpose tensor"""
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        
        new_data = np.transpose(self.data, axes)
        result = TernaryTensor(new_data, requires_grad=self.requires_grad, device=self.device)
        
        if self.requires_grad:
            result.grad_fn = TransposeFunction(self, dim0, dim1)
        
        return result

    def sum(self, axis=None, keepdims=False):
        """Sum tensor elements"""
        new_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = TernaryTensor(new_data, requires_grad=self.requires_grad, device=self.device)
        
        if self.requires_grad:
            result.grad_fn = SumFunction(self, axis, keepdims)
        
        return result

    def mean(self, axis=None, keepdims=False):
        """Mean of tensor elements"""
        new_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        result = TernaryTensor(new_data, requires_grad=self.requires_grad, device=self.device)
        
        if self.requires_grad:
            result.grad_fn = MeanFunction(self, axis, keepdims)
        
        return result


# =====================================================================================
# GRADIENT COMPUTATION ENGINE
# =====================================================================================

class Function(ABC):
    """Base class for differentiable functions"""
    
    def __init__(self, *tensors):
        self.saved_tensors = []
        self.next_functions = []
        
        for tensor in tensors:
            if isinstance(tensor, TernaryTensor) and tensor.requires_grad:
                self.saved_tensors.append(tensor)

    @abstractmethod
    def backward(self, grad_output):
        """Compute gradients"""
        pass

    def save_for_backward(self, *tensors):
        """Save tensors for backward computation"""
        self.saved_tensors = list(tensors)


class TernaryQuantizeFunction(Function):
    """Gradient function for ternary quantization"""
    
    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def backward(self, grad_output):
        """Straight-Through Estimator with ternary-specific modifications"""
        if not self.saved_tensors:
            return
        
class SigmoidFunction(Function):
    """Sigmoid gradient function"""
    
    def __init__(self, input_tensor, output_tensor):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def backward(self, grad_output):
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        
        # Sigmoid gradient: sigmoid_output * (1 - sigmoid_output)
        sigmoid_grad = self.output_data * (1 - self.output_data) * grad_output.data
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += sigmoid_grad


class TanhFunction(Function):
    """Tanh gradient function"""
    
    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def backward(self, grad_output):
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        
        # Tanh gradient: 1 - tanh^2(x)
        tanh_output = np.tanh(input_tensor.data)
        tanh_grad = (1 - tanh_output**2) * grad_output.data
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += tanh_grad


class SoftmaxFunction(Function):
    """Softmax gradient function"""
    
    def __init__(self, input_tensor, output_tensor, axis):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.axis = axis

    def backward(self, grad_output):
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        
        # Softmax gradient: softmax_output * (grad_output - sum(grad_output * softmax_output))
        sum_term = np.sum(grad_output.data * self.output_data, axis=self.axis, keepdims=True)
        softmax_grad = self.output_data * (grad_output.data - sum_term)
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += softmax_grad


# =====================================================================================
# TERNARY LOSS FUNCTIONS
# =====================================================================================

def ternary_mse_loss(predictions, targets):
    """Mean Squared Error loss"""
    if isinstance(targets, np.ndarray):
        targets = TernaryTensor(targets)
    
    diff = predictions - targets
    squared_diff = ternary_mul(diff, diff)
    loss = squared_diff.mean()
    
    return loss


def ternary_cross_entropy_loss(predictions, targets, epsilon=1e-7):
    """Cross entropy loss with numerical stability"""
    if isinstance(targets, np.ndarray):
        targets = TernaryTensor(targets)
    
    # Clip predictions for numerical stability
    pred_clipped = TernaryTensor(np.clip(predictions.data, epsilon, 1 - epsilon), 
                                requires_grad=predictions.requires_grad)
    pred_clipped.grad_fn = predictions.grad_fn
    
    # Compute log
    log_pred = TernaryTensor(np.log(pred_clipped.data), requires_grad=pred_clipped.requires_grad)
    if pred_clipped.requires_grad:
        log_pred.grad_fn = LogFunction(pred_clipped)
    
    # Cross entropy: -sum(targets * log(predictions))
    ce_terms = ternary_mul(targets, log_pred)
    loss = -ce_terms.sum() / TernaryTensor(np.array(targets.data.shape[0]))
    
    return loss


class LogFunction(Function):
    """Logarithm gradient function"""
    
    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def backward(self, grad_output):
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        
        # Log gradient: 1/x
        log_grad = grad_output.data / np.maximum(input_tensor.data, 1e-7)
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += log_grad


# =====================================================================================
# TERNARY OPTIMIZERS - PURE IMPLEMENTATION
# =====================================================================================

class TernaryOptimizer:
    """Base class for ternary optimizers"""
    
    def __init__(self, parameters, learning_rate=0.001):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.step_count = 0

    def zero_grad(self):
        """Zero all gradients"""
        for param in self.parameters:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()

    def step(self):
        """Take optimization step"""
        raise NotImplementedError


class TernarySGD(TernaryOptimizer):
    """Stochastic Gradient Descent for ternary networks"""
    
    def __init__(self, parameters, learning_rate=0.001, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity_buffers = {}

    def step(self):
        """SGD optimization step with ternary-specific modifications"""
        self.step_count += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.data.copy()
            
            # Weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # Apply ternary-specific gradient modifications
            if hasattr(param, 'is_ternary') and param.is_ternary:
                grad = self._modify_ternary_gradient(param, grad)
            
            # Momentum
            if self.momentum != 0:
                if i not in self.velocity_buffers:
                    self.velocity_buffers[i] = np.zeros_like(param.data)
                
                self.velocity_buffers[i] = (self.momentum * self.velocity_buffers[i] + 
                                          (1 - self.momentum) * grad)
                grad = self.velocity_buffers[i]
            
            # Update parameters
            param.data -= self.learning_rate * grad

    def _modify_ternary_gradient(self, param, grad):
        """Apply ternary-specific gradient modifications"""
        abs_param = np.abs(param.data)
        threshold = param.threshold if hasattr(param, 'threshold') else 0.05
        
        # Reduce learning for saturated weights
        grad_modifier = np.ones_like(grad)
        grad_modifier[abs_param > 2 * threshold] *= 0.5
        grad_modifier[abs_param < 0.5 * threshold] *= 1.5
        
        return grad * grad_modifier


class TernaryAdam(TernaryOptimizer):
    """Adam optimizer with ternary-specific adaptations"""
    
    def __init__(self, parameters, learning_rate=0.001, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, learning_rate)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment buffers
        self.m_buffers = {}
        self.v_buffers = {}

    def step(self):
        """Adam optimization step"""
        self.step_count += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.data.copy()
            
            # Weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # Initialize buffers
            if i not in self.m_buffers:
                self.m_buffers[i] = np.zeros_like(param.data)
                self.v_buffers[i] = np.zeros_like(param.data)
            
            # Apply ternary-specific preprocessing
            if hasattr(param, 'is_ternary') and param.is_ternary:
                grad = self._preprocess_ternary_gradient(param, grad)
            
            # Update biased first moment estimate
            self.m_buffers[i] = self.betas[0] * self.m_buffers[i] + (1 - self.betas[0]) * grad
            
            # Update biased second raw moment estimate  
            self.v_buffers[i] = self.betas[1] * self.v_buffers[i] + (1 - self.betas[1]) * (grad ** 2)
            
            # Compute bias-corrected moment estimates
            m_hat = self.m_buffers[i] / (1 - self.betas[0] ** self.step_count)
            v_hat = self.v_buffers[i] / (1 - self.betas[1] ** self.step_count)
            
            # Compute update
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # Apply ternary-specific update modifications
            if hasattr(param, 'is_ternary') and param.is_ternary:
                update = self._modify_ternary_update(param, update)
            
            # Update parameters
            param.data -= update

    def _preprocess_ternary_gradient(self, param, grad):
        """Preprocess gradients for ternary parameters"""
        abs_param = np.abs(param.data)
        threshold = param.threshold if hasattr(param, 'threshold') else 0.05
        
        # Scale gradients based on distance from ternary values
        scale = np.ones_like(grad)
        scale[abs_param > 1.5 * threshold] *= 0.7  # Reduce for saturated
        scale[abs_param < 0.3 * threshold] *= 1.3  # Increase for near-zero
        
        return grad * scale

    def _modify_ternary_update(self, param, update):
        """Modify parameter updates for ternary constraints"""
        abs_param = np.abs(param.data)
        threshold = param.threshold if hasattr(param, 'threshold') else 0.05
        
        # Prevent updates that move away from ternary values
        moving_away_mask = (np.sign(param.data) == np.sign(update)) & (abs_param < threshold)
        update[moving_away_mask] *= 0.6
        
        # Encourage movement toward ternary values
        moving_toward_mask = (np.sign(param.data) != np.sign(update)) & (abs_param > threshold)
        update[moving_toward_mask] *= 1.4
        
        return update


# =====================================================================================
# TERNARY NEURAL NETWORK LAYERS
# =====================================================================================

class TernaryModule:
    """Base class for ternary neural network modules"""
    
    def __init__(self):
        self.parameters_dict = {}
        self.modules_dict = {}
        self.training = True
        self.name = self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"Forward method not implemented for {self.name}")

    def parameters(self):
        """Return all parameters"""
        for param in self.parameters_dict.values():
            yield param
        for module in self.modules_dict.values():
            yield from module.parameters()

    def named_parameters(self):
        """Return named parameters"""
        for name, param in self.parameters_dict.items():
            yield name, param
        for name, module in self.modules_dict.items():
            for subname, param in module.named_parameters():
                yield f"{name}.{subname}", param

    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        for module in self.modules_dict.values():
            module.train(mode)

    def eval(self):
        """Set evaluation mode"""
        return self.train(False)

    def zero_grad(self):
        """Zero all gradients"""
        for param in self.parameters():
            if hasattr(param, 'zero_grad'):
                param.zero_grad()


class TernaryLinear(TernaryModule):
    """Linear layer with ternary weights"""
    
    def __init__(self, in_features, out_features, bias=True, ternary_weights=True, 
                 weight_init='xavier'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ternary_weights = ternary_weights
        
        # Initialize weights
        if weight_init == 'xavier':
            scale = np.sqrt(2.0 / (in_features + out_features))
        elif weight_init == 'he':
            scale = np.sqrt(2.0 / in_features)
        else:
            scale = 0.1
        
        weight_data = np.random.normal(0, scale, (out_features, in_features)).astype(np.float32)
        self.parameters_dict['weight'] = TernaryTensor(weight_data, requires_grad=True)
        
        if bias:
            bias_data = np.zeros(out_features, dtype=np.float32)
            self.parameters_dict['bias'] = TernaryTensor(bias_data, requires_grad=True)
        else:
            self.parameters_dict['bias'] = None
        
        # Convert to ternary if requested
        if ternary_weights:
            self.parameters_dict['weight'] = self.parameters_dict['weight'].to_ternary()

    def forward(self, x):
        """Forward pass"""
        # Matrix multiplication: x @ weight.T + bias
        output = ternary_matmul(x, self.parameters_dict['weight'].transpose())
        
        if self.parameters_dict['bias'] is not None:
            # Broadcast bias
            if output.ndim > 1:
                bias_shape = [1] * output.ndim
                bias_shape[-1] = -1
                bias_expanded = TernaryTensor(
                    self.parameters_dict['bias'].data.reshape(bias_shape),
                    requires_grad=self.parameters_dict['bias'].requires_grad
                )
                output = ternary_add(output, bias_expanded)
            else:
                output = ternary_add(output, self.parameters_dict['bias'])
        
        return output


class TernaryConv2d(TernaryModule):
    """2D Convolutional layer with ternary weights"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 bias=True, ternary_weights=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        scale = np.sqrt(2.0 / fan_in)
        
        weight_shape = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        weight_data = np.random.normal(0, scale, weight_shape).astype(np.float32)
        self.parameters_dict['weight'] = TernaryTensor(weight_data, requires_grad=True)
        
        if bias:
            bias_data = np.zeros(out_channels, dtype=np.float32)
            self.parameters_dict['bias'] = TernaryTensor(bias_data, requires_grad=True)
        else:
            self.parameters_dict['bias'] = None
        
        # Convert to ternary if requested
        if ternary_weights:
            self.parameters_dict['weight'] = self.parameters_dict['weight'].to_ternary()

    def forward(self, x):
        """Forward pass using ternary convolution"""
        # Use our custom ternary convolution
        output_data = TernaryMath.ternary_conv2d(
            x.data, self.parameters_dict['weight'].data, 
            stride=self.stride, padding=self.padding
        )
        
        output = TernaryTensor(output_data, requires_grad=x.requires_grad or self.parameters_dict['weight'].requires_grad)
        
        # Set up gradient function
        if output.requires_grad:
            output.grad_fn = Conv2dFunction(x, self.parameters_dict['weight'], 
                                          self.parameters_dict['bias'], 
                                          self.stride, self.padding)
        
        # Add bias if present
        if self.parameters_dict['bias'] is not None:
            # Reshape bias for broadcasting
            bias_shape = [1, -1] + [1] * (output.ndim - 2)
            bias_reshaped = TernaryTensor(
                self.parameters_dict['bias'].data.reshape(bias_shape),
                requires_grad=self.parameters_dict['bias'].requires_grad
            )
            output = ternary_add(output, bias_reshaped)
        
        return output


class Conv2dFunction(Function):
    """Gradient function for 2D convolution"""

    def __init__(self, input_tensor, weight_tensor, bias_tensor, stride, padding):
        super().__init__(input_tensor, weight_tensor)
        if bias_tensor is not None:
            self.saved_tensors.append(bias_tensor)
        self.stride = stride
        self.padding = padding
        self.weight_shape = weight_tensor.shape

    def backward(self, grad_output):
        """Proper convolution backward pass with conv transpose"""
        input_tensor = self.saved_tensors[0]
        weight_tensor = self.saved_tensors[1]
        bias_tensor = self.saved_tensors[2] if len(self.saved_tensors) > 2 else None

        # Gradient w.r.t. input: conv_transpose(grad_output, weight)
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))

        # Compute input gradient using convolution transpose
        input_grad = self._compute_input_gradient(grad_output.data, weight_tensor.data)
        input_tensor.grad.data += input_grad

        # Gradient w.r.t. weight: correlation(input, grad_output)
        if weight_tensor.grad is None:
            weight_tensor.grad = TernaryTensor(np.zeros_like(weight_tensor.data))

        weight_grad = self._compute_weight_gradient(input_tensor.data, grad_output.data)
        weight_tensor.grad.data += weight_grad

        # Gradient w.r.t. bias: sum over batch, height, width dimensions
        if bias_tensor is not None:
            if bias_tensor.grad is None:
                bias_tensor.grad = TernaryTensor(np.zeros_like(bias_tensor.data))

            # Sum over all dimensions except channel dimension
            bias_grad = np.sum(grad_output.data, axis=(0, 2, 3))
            bias_tensor.grad.data += bias_grad

    def _compute_input_gradient(self, grad_output, weight):
        """Compute input gradient using convolution transpose"""
        # grad_output shape: (N, Out_C, Out_H, Out_W)
        # weight shape: (Out_C, In_C, KH, KW)
        # output shape should match input: (N, In_C, H, W)

        N, Out_C, Out_H, Out_W = grad_output.shape
        Out_C_w, In_C, KH, KW = weight.shape

        # Compute padded output size
        H = (Out_H - 1) * self.stride + KH - 2 * self.padding
        W = (Out_W - 1) * self.stride + KW - 2 * self.padding

        input_grad = np.zeros((N, In_C, H, W), dtype=np.float32)

        # Convolution transpose: for each output position, distribute gradient
        for n in range(N):
            for out_c in range(Out_C):
                for in_c in range(In_C):
                    for out_h in range(Out_H):
                        for out_w in range(Out_W):
                            # Get the gradient value at this output position
                            grad_val = grad_output[n, out_c, out_h, out_w]

                            # Distribute this gradient across the kernel positions
                            for kh in range(KH):
                                for kw in range(KW):
                                    # Calculate input position
                                    in_h = out_h * self.stride - self.padding + kh
                                    in_w = out_w * self.stride - self.padding + kw

                                    # Check bounds
                                    if 0 <= in_h < H and 0 <= in_w < W:
                                        # Weight gradient flows from output to input
                                        weight_val = weight[out_c, in_c, kh, kw]
                                        input_grad[n, in_c, in_h, in_w] += grad_val * weight_val

        return input_grad

    def _compute_weight_gradient(self, input_data, grad_output):
        """Compute weight gradient using correlation"""
        # input_data shape: (N, In_C, H, W)
        # grad_output shape: (N, Out_C, Out_H, Out_W)
        # weight shape: (Out_C, In_C, KH, KW)

        N, In_C, H, W = input_data.shape
        N_g, Out_C, Out_H, Out_W = grad_output.shape

        # Weight dimensions from original conv operation
        weight_grad = np.zeros((Out_C, In_C,
                               self.weight_shape[2],
                               self.weight_shape[3]), dtype=np.float32)
        KH, KW = weight_grad.shape[2], weight_grad.shape[3]

        # Correlation operation: compute how input and grad_output correlate
        for out_c in range(Out_C):
            for in_c in range(In_C):
                for kh in range(KH):
                    for kw in range(KW):
                        correlation_sum = 0.0

                        for n in range(N):
                            for out_h in range(Out_H):
                                for out_w in range(Out_W):
                                    # Calculate corresponding input position
                                    in_h = out_h * self.stride - self.padding + kh
                                    in_w = out_w * self.stride - self.padding + kw

                                    # Check bounds
                                    if 0 <= in_h < H and 0 <= in_w < W:
                                        input_val = input_data[n, in_c, in_h, in_w]
                                        grad_val = grad_output[n, out_c, out_h, out_w]
                                        correlation_sum += input_val * grad_val

                        weight_grad[out_c, in_c, kh, kw] = correlation_sum

        return weight_grad


class TernaryReLU(TernaryModule):
    """ReLU activation layer"""
    
    def forward(self, x):
        return ternary_relu(x)


class TernarySigmoid(TernaryModule):
    """Sigmoid activation layer"""
    
    def forward(self, x):
        return ternary_sigmoid(x)


class TernaryTanh(TernaryModule):
    """Tanh activation layer"""
    
    def forward(self, x):
        return ternary_tanh(x)


class TernarySoftmax(TernaryModule):
    """Softmax activation layer"""
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return ternary_softmax(x, axis=self.dim)


# =====================================================================================
# COMPLETE TERNARY MODELS
# =====================================================================================

class TernaryMLP(TernaryModule):
    """Multi-layer perceptron with ternary weights"""
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', 
                 ternary_weights=True, dropout_rate=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Build layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            layer_name = f'layer_{i}'
            self.modules_dict[layer_name] = TernaryLinear(
                layer_sizes[i], layer_sizes[i + 1], 
                ternary_weights=ternary_weights
            )
        
        # Activation function
        if activation == 'relu':
            self.activation = TernaryReLU()
        elif activation == 'sigmoid':
            self.activation = TernarySigmoid()
        elif activation == 'tanh':
            self.activation = TernaryTanh()
        else:
            self.activation = None

    def forward(self, x):
        """Forward pass through MLP"""
        for i, (name, layer) in enumerate(self.modules_dict.items()):
            x = layer(x)
            
            # Apply activation to all layers except the last
            if i < len(self.modules_dict) - 1 and self.activation is not None:
                x = self.activation(x)
            
            # Apply dropout during training
            if self.training and self.dropout_rate > 0 and i < len(self.modules_dict) - 1:
                x = self._apply_dropout(x, self.dropout_rate)
        
        return x

    def _apply_dropout(self, x, rate):
        """Simple dropout implementation"""
        if not self.training:
            return x
        
        mask = np.random.random(x.shape) > rate
        scale = 1.0 / (1.0 - rate)
        
        dropout_data = x.data * mask * scale
        result = TernaryTensor(dropout_data, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            result.grad_fn = DropoutFunction(x, mask, scale)
        
        return result


class DropoutFunction(Function):
    """Gradient function for dropout"""
    
    def __init__(self, input_tensor, mask, scale):
        super().__init__(input_tensor)
        self.mask = mask
        self.scale = scale

    def backward(self, grad_output):
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        
        # Dropout gradient: same mask and scale applied
        dropout_grad = grad_output.data * self.mask * self.scale
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += dropout_grad


# =====================================================================================
# TRAINING UTILITIES
# =====================================================================================

class TernaryTrainer:
    """Training utilities for ternary models"""
    
    def __init__(self, model, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_losses = []
        self.val_losses = []

    def train_step(self, inputs, targets):
        """Single training step"""
        # Forward pass
        self.model.train()
        predictions = self.model(inputs)
        loss = self.loss_function(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate_step(self, inputs, targets):
        """Single validation step"""
        self.model.eval()
        predictions = self.model(inputs)
        loss = self.loss_function(predictions, targets)
        return loss.item()

    def train_epoch(self, train_data, val_data=None):
        """Train for one epoch"""
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch_inputs, batch_targets in train_data:
            if isinstance(batch_inputs, np.ndarray):
                batch_inputs = TernaryTensor(batch_inputs)
            if isinstance(batch_targets, np.ndarray):
                batch_targets = TernaryTensor(batch_targets)
            
            loss = self.train_step(batch_inputs, batch_targets)
            epoch_train_loss += loss
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        self.train_losses.append(avg_train_loss)
        
        # Validation
        if val_data is not None:
            epoch_val_loss = 0.0
            val_batches = 0
            
            for batch_inputs, batch_targets in val_data:
                if isinstance(batch_inputs, np.ndarray):
                    batch_inputs = TernaryTensor(batch_inputs)
                if isinstance(batch_targets, np.ndarray):
                    batch_targets = TernaryTensor(batch_targets)
                
                val_loss = self.validate_step(batch_inputs, batch_targets)
                epoch_val_loss += val_loss
                val_batches += 1
            
            avg_val_loss = epoch_val_loss / val_batches
            self.val_losses.append(avg_val_loss)
            
            return avg_train_loss, avg_val_loss
        
        return avg_train_loss, None


# =====================================================================================
# DEMO AND PERFORMANCE ANALYSIS
# =====================================================================================

def create_sample_data(n_samples=1000, input_dim=10, output_dim=1):
    """Create sample data for testing"""
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, output_dim, (n_samples, output_dim)).astype(np.float32)
    return X, y


def benchmark_ternary_operations():
    """Benchmark ternary vs regular operations with hardware acceleration"""
    print("🔥 TERNARYCORE PERFORMANCE BENCHMARK")
    print("=" * 50)

    kernel_manager = get_kernel_manager()
    if kernel_manager:
        print("🚀 Hardware acceleration: ENABLED")
    else:
        print("⚠️ Hardware acceleration: DISABLED")

    # Test matrix multiplication
    sizes = [256, 512, 1024]

    for size in sizes:
        print(f"\n📊 Testing {size}x{size} matrices:")

        # Create test matrices with ternary values
        a_data = np.random.choice([-1, 0, 1], size=(size, size), p=[0.25, 0.5, 0.25]).astype(np.float32)
        b_data = np.random.choice([-1, 0, 1], size=(size, size), p=[0.25, 0.5, 0.25]).astype(np.float32)

        # Create tensors
        a_tensor = TernaryTensor(a_data, use_hardware=False)  # Software version
        b_tensor = TernaryTensor(b_data, use_hardware=False)
        a_hw = TernaryTensor(a_data, use_hardware=True)      # Hardware version
        b_hw = TernaryTensor(b_data, use_hardware=True)

        # Convert to ternary
        a_ternary = a_tensor.to_ternary()
        b_ternary = b_tensor.to_ternary()
        a_hw_ternary = a_hw.to_ternary()
        b_hw_ternary = b_hw.to_ternary()

        # Benchmark software implementation
        start_time = time.perf_counter()
        result_software = ternary_matmul(a_ternary, b_ternary)
        software_time = time.perf_counter() - start_time

        # Benchmark hardware implementation (if available)
        if kernel_manager:
            start_time = time.perf_counter()
            result_hardware = ternary_matmul(a_hw_ternary, b_hw_ternary)
            hardware_time = time.perf_counter() - start_time

            # Verify results match
            if np.allclose(result_software.data, result_hardware.data, atol=1):
                speedup = software_time / hardware_time if hardware_time > 0 else float('inf')
                print(f"   Software: {software_time*1000:.2f}ms")
                print(f"   Hardware: {hardware_time*1000:.2f}ms")
                print(f"   Speedup: {speedup:.2f}x")
                print(f"   ✅ Results match")
            else:
                print(f"   ⚠️ Results differ between hardware and software")
                print(f"   Max difference: {np.max(np.abs(result_software.data - result_hardware.data))}")
        else:
            print(f"   Software: {software_time*1000:.2f}ms")
            print(f"   Hardware: Not available")

        # Test quantization speed
        float_data = np.random.randn(size, size).astype(np.float32)

        start_time = time.perf_counter()
        quant_software = TernaryMath.ternary_quantize(float_data)
        quant_time = time.perf_counter() - start_time

        print(f"   Quantization: {quant_time*1000:.2f}ms ({float_data.size/quant_time/1e6:.1f} Melem/s)")

        # Memory efficiency demo
        regular_memory = a_data.nbytes + b_data.nbytes
        ternary_memory = a_ternary.data.nbytes + b_ternary.data.nbytes
        memory_ratio = regular_memory / ternary_memory if ternary_memory > 0 else 1

        print(f"   Memory efficiency: {memory_ratio:.1f}x (would be 16x with bit packing)")


def demo_ternary_network():
    """Demonstrate a complete ternary neural network"""
    print("\n🚀 TERNARYCORE NEURAL NETWORK DEMO")
    print("=" * 50)
    
    # Create sample data
    X_train, y_train = create_sample_data(1000, 20, 3)
    X_val, y_val = create_sample_data(200, 20, 3)
    
    print(f"📊 Training data: {X_train.shape}, {y_train.shape}")
    print(f"📊 Validation data: {X_val.shape}, {y_val.shape}")
    
    # Create ternary model
    model = TernaryMLP(
        input_size=20,
        hidden_sizes=[64, 32],
        output_size=3,
        activation='relu',
        ternary_weights=True
    )
    
    print(f"🧠 Model created with ternary weights")
    
    # Count parameters
    total_params = sum(param.data.size for param in model.parameters())
    ternary_params = sum(param.data.size for param in model.parameters() 
                        if hasattr(param, 'is_ternary') and param.is_ternary)
    
    print(f"📈 Total parameters: {total_params:,}")
    print(f"🎯 Ternary parameters: {ternary_params:,} ({ternary_params/total_params*100:.1f}%)")
    
    # Create optimizer and loss
    optimizer = TernaryAdam(model.parameters(), learning_rate=0.001)
    loss_fn = ternary_mse_loss
    
    # Create trainer
    trainer = TernaryTrainer(model, optimizer, loss_fn)
    
    # Simple batch data (for demo - normally you'd use a proper data loader)
    train_data = [(TernaryTensor(X_train), TernaryTensor(y_train))]
    val_data = [(TernaryTensor(X_val), TernaryTensor(y_val))]
    
    print("\n🏃 Training started...")
    
    # Train for a few epochs
    for epoch in range(5):
        train_loss, val_loss = trainer.train_epoch(train_data, val_data)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    print("✅ Training completed!")
    
    # Test inference
    test_input = TernaryTensor(np.random.randn(1, 20).astype(np.float32))
    prediction = model(test_input)
    print(f"🔮 Sample prediction: {prediction.data}")


def main():
    """Main demonstration"""
    print("🚀 TERNARYCORE: PURE FROM-SCRATCH DEEP LEARNING FRAMEWORK")
    print("=" * 60)
    print("💥 NO PYTORCH. NO TENSORFLOW. PURE REVOLUTION!")
    print("🎯 100% designed for {-1, 0, 1} operations from first principles")
    print()
    
    print("🔧 FEATURES:")
    print("   ✅ Pure numpy implementation")
    print("   ✅ Custom automatic differentiation")
    print("   ✅ Ternary-optimized operations")
    print("   ✅ Native {-1, 0, 1} arithmetic")
    print("   ✅ Ternary-aware optimizers")
    print("   ✅ Complete neural network layers")
    print("   ✅ Training utilities")
    print()
    
    # Run benchmarks
    benchmark_ternary_operations()
    
    # Run demo
    demo_ternary_network()
    
    print("\n" + "=" * 60)
    print("🎉 TERNARYCORE DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("🚀 WHAT YOU JUST SAW:")
    print("   💻 Complete deep learning framework from scratch")
    print("   🧮 Pure ternary mathematics engine")  
    print("   🎯 Custom autograd system for {-1, 0, 1}")
    print("   ⚡ Optimized operations for ternary values")
    print("   🏗️ Full neural network implementation")
    print("   📈 Training and optimization utilities")
    print()
    print("🔥 REVOLUTIONARY IMPACT:")
    print("   🌍 No dependency on existing frameworks")
    print("   💡 Designed specifically for ternary operations")
    print("   🚀 Optimized for your hardware from the ground up")
    print("   ⚡ True mathematical efficiency with {-1, 0, 1}")
    print()
    print("🎯 THIS IS THE REAL REVOLUTION!")
    print("   Not piggybacking - PURE INNOVATION!")
    print("   From mathematical first principles!")
    print("   The future of AI democratization!")


class AddFunction(Function):
    """Addition gradient function"""
    
    def __init__(self, tensor1, tensor2):
        super().__init__(tensor1, tensor2)

    def backward(self, grad_output):
        """Addition gradients pass through unchanged"""
        for tensor in self.saved_tensors:
            if tensor.grad is None:
                tensor.grad = TernaryTensor(np.zeros_like(tensor.data))
            tensor.grad.data += grad_output.data


class MulFunction(Function):
    """Multiplication gradient function"""
    
    def __init__(self, tensor1, tensor2):
        super().__init__(tensor1, tensor2)

    def backward(self, grad_output):
        """Product rule: d(ab)/dx = b * da/dx + a * db/dx"""
        if len(self.saved_tensors) != 2:
            return
        
        tensor1, tensor2 = self.saved_tensors
        
        # Gradient w.r.t. first tensor
        if tensor1.grad is None:
            tensor1.grad = TernaryTensor(np.zeros_like(tensor1.data))
        
        if tensor1.is_ternary:
            # For ternary weights, use modified gradient
            grad1 = grad_output.data * tensor2.data
            # Reduce gradient magnitude for saturated ternary weights
            abs_tensor1 = np.abs(tensor1.data)
            grad1[abs_tensor1 > 0.8] *= 0.5
            tensor1.grad.data += grad1
        else:
            tensor1.grad.data += grad_output.data * tensor2.data
        
        # Gradient w.r.t. second tensor
        if tensor2.grad is None:
            tensor2.grad = TernaryTensor(np.zeros_like(tensor2.data))
        
        if tensor2.is_ternary:
            grad2 = grad_output.data * tensor1.data
            abs_tensor2 = np.abs(tensor2.data)
            grad2[abs_tensor2 > 0.8] *= 0.5
            tensor2.grad.data += grad2
        else:
            tensor2.grad.data += grad_output.data * tensor1.data


class MatMulFunction(Function):
    """Matrix multiplication gradient function"""
    
    def __init__(self, tensor1, tensor2):
        super().__init__(tensor1, tensor2)

    def backward(self, grad_output):
        """Matrix multiplication gradients"""
        if len(self.saved_tensors) != 2:
            return
        
        tensor1, tensor2 = self.saved_tensors
        
        # Gradient w.r.t. first tensor: grad_output @ tensor2.T
        if tensor1.grad is None:
            tensor1.grad = TernaryTensor(np.zeros_like(tensor1.data))
        
        grad1 = TernaryMath.ternary_matmul(grad_output.data, tensor2.data.T)
        if tensor1.is_ternary:
            # Apply ternary-specific gradient modifications
            grad1 = self._modify_ternary_gradient(tensor1.data, grad1)
        
        tensor1.grad.data += grad1
        
        # Gradient w.r.t. second tensor: tensor1.T @ grad_output
        if tensor2.grad is None:
            tensor2.grad = TernaryTensor(np.zeros_like(tensor2.data))
        
        grad2 = TernaryMath.ternary_matmul(tensor1.data.T, grad_output.data)
        if tensor2.is_ternary:
            grad2 = self._modify_ternary_gradient(tensor2.data, grad2)
        
        tensor2.grad.data += grad2

    def _modify_ternary_gradient(self, weights, gradients):
        """Apply ternary-specific gradient modifications"""
        abs_weights = np.abs(weights)
        threshold = 0.05
        
        # Reduce gradients for weights far from decision boundaries
        modifier = np.ones_like(gradients)
        modifier[abs_weights > 2 * threshold] *= 0.6
        modifier[abs_weights < 0.5 * threshold] *= 1.4
        
        return gradients * modifier


class ReshapeFunction(Function):
    """Reshape gradient function"""
    
    def __init__(self, input_tensor, original_shape):
        super().__init__(input_tensor)
        self.original_shape = original_shape

    def backward(self, grad_output):
        """Reshape gradient back to original shape"""
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        grad_reshaped = grad_output.data.reshape(self.original_shape)
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += grad_reshaped


class TransposeFunction(Function):
    """Transpose gradient function"""
    
    def __init__(self, input_tensor, dim0, dim1):
        super().__init__(input_tensor)
        self.dim0 = dim0
        self.dim1 = dim1

    def backward(self, grad_output):
        """Transpose gradient back"""
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        
        # Transpose back
        axes = list(range(grad_output.data.ndim))
        axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
        grad_transposed = np.transpose(grad_output.data, axes)
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += grad_transposed


class SumFunction(Function):
    """Sum gradient function"""
    
    def __init__(self, input_tensor, axis, keepdims):
        super().__init__(input_tensor)
        self.axis = axis
        self.keepdims = keepdims
        self.input_shape = input_tensor.shape

    def backward(self, grad_output):
        """Broadcast gradient back to original shape"""
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        
        # Expand gradient to original shape
        grad_expanded = grad_output.data
        
        if not self.keepdims and self.axis is not None:
            # Add back summed dimensions
            if isinstance(self.axis, int):
                grad_expanded = np.expand_dims(grad_expanded, self.axis)
            else:
                for ax in sorted(self.axis):
                    grad_expanded = np.expand_dims(grad_expanded, ax)
        
        # Broadcast to original shape
        grad_broadcasted = np.broadcast_to(grad_expanded, self.input_shape)
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += grad_broadcasted


class MeanFunction(Function):
    """Mean gradient function"""
    
    def __init__(self, input_tensor, axis, keepdims):
        super().__init__(input_tensor)
        self.axis = axis
        self.keepdims = keepdims
        self.input_shape = input_tensor.shape
        
        # Calculate the number of elements being averaged
        if axis is None:
            self.n_elements = input_tensor.data.size
        else:
            if isinstance(axis, int):
                self.n_elements = input_tensor.shape[axis]
            else:
                self.n_elements = np.prod([input_tensor.shape[ax] for ax in axis])

    def backward(self, grad_output):
        """Mean gradient is grad_output / n_elements broadcasted to original shape"""
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        
        # Scale by 1/n_elements
        grad_scaled = grad_output.data / self.n_elements
        
        # Expand to original shape (same as SumFunction)
        if not self.keepdims and self.axis is not None:
            if isinstance(self.axis, int):
                grad_scaled = np.expand_dims(grad_scaled, self.axis)
            else:
                for ax in sorted(self.axis):
                    grad_scaled = np.expand_dims(grad_scaled, ax)
        
        grad_broadcasted = np.broadcast_to(grad_scaled, self.input_shape)
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += grad_broadcasted


# =====================================================================================
# TERNARY OPERATIONS - PURE IMPLEMENTATION
# =====================================================================================

def ternary_add(tensor1, tensor2):
    """Addition of ternary tensors"""
    if isinstance(tensor2, (int, float)):
        tensor2 = TernaryTensor(np.full_like(tensor1.data, tensor2))
    
    result_data = tensor1.data + tensor2.data
    result = TernaryTensor(result_data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    if result.requires_grad:
        result.grad_fn = AddFunction(tensor1, tensor2)
    
    return result


def ternary_sub(tensor1, tensor2):
    """Subtraction of ternary tensors"""
    if isinstance(tensor2, (int, float)):
        tensor2 = TernaryTensor(np.full_like(tensor1.data, tensor2))
    
    result_data = tensor1.data - tensor2.data
    result = TernaryTensor(result_data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    if result.requires_grad:
        # Subtraction is addition with negated second operand
        neg_tensor2 = TernaryTensor(-tensor2.data, requires_grad=tensor2.requires_grad)
        result.grad_fn = AddFunction(tensor1, neg_tensor2)
    
    return result


def ternary_mul(tensor1, tensor2):
    """Element-wise multiplication of ternary tensors"""
    if isinstance(tensor2, (int, float)):
        tensor2 = TernaryTensor(np.full_like(tensor1.data, tensor2))
    
    # Use ternary multiplication if both are ternary
    if tensor1.is_ternary and tensor2.is_ternary:
        result_data = TernaryMath.ternary_multiply(tensor1.data, tensor2.data)
    else:
        result_data = tensor1.data * tensor2.data
    
    result = TernaryTensor(result_data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    if result.requires_grad:
        result.grad_fn = MulFunction(tensor1, tensor2)
    
    return result


def ternary_matmul(tensor1, tensor2):
    """Matrix multiplication of ternary tensors"""
    # Use optimized ternary matrix multiplication
    if tensor1.is_ternary and tensor2.is_ternary:
        result_data = TernaryMath.ternary_matmul(tensor1.data, tensor2.data)
    else:
        result_data = np.matmul(tensor1.data, tensor2.data)
    
    result = TernaryTensor(result_data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    if result.requires_grad:
        result.grad_fn = MatMulFunction(tensor1, tensor2)
    
    return result


def ternary_neg(tensor):
    """Negation of ternary tensor"""
    result_data = -tensor.data
    result = TernaryTensor(result_data, requires_grad=tensor.requires_grad)
    
    if result.requires_grad:
        # Negation gradient is just -1
        result.grad_fn = MulFunction(tensor, TernaryTensor(np.array(-1.0)))
    
    return result


# =====================================================================================
# TERNARY ACTIVATION FUNCTIONS
# =====================================================================================

def ternary_relu(tensor):
    """ReLU activation for ternary tensors"""
    result_data = np.maximum(tensor.data, 0)
    result = TernaryTensor(result_data, requires_grad=tensor.requires_grad)
    
    if result.requires_grad:
        result.grad_fn = ReLUFunction(tensor)
    
    return result


def ternary_sigmoid(tensor):
    """Sigmoid activation"""
    result_data = 1 / (1 + np.exp(-np.clip(tensor.data, -500, 500)))  # Clip for numerical stability
    result = TernaryTensor(result_data, requires_grad=tensor.requires_grad)
    
    if result.requires_grad:
        result.grad_fn = SigmoidFunction(tensor, result)
    
    return result


def ternary_tanh(tensor):
    """Tanh activation"""
    result_data = np.tanh(tensor.data)
    result = TernaryTensor(result_data, requires_grad=tensor.requires_grad)
    
    if result.requires_grad:
        result.grad_fn = TanhFunction(tensor)
    
    return result


def ternary_softmax(tensor, axis=-1):
    """Softmax activation with numerical stability"""
    # Subtract max for numerical stability
    max_vals = np.max(tensor.data, axis=axis, keepdims=True)
    shifted = tensor.data - max_vals
    
    exp_vals = np.exp(np.clip(shifted, -500, 500))
    sum_exp = np.sum(exp_vals, axis=axis, keepdims=True)
    result_data = exp_vals / sum_exp
    
    result = TernaryTensor(result_data, requires_grad=tensor.requires_grad)
    
    if result.requires_grad:
        result.grad_fn = SoftmaxFunction(tensor, result, axis)
    
    return result


class ReLUFunction(Function):
    """ReLU gradient function"""
    
    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def backward(self, grad_output):
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors[0]
        
        # ReLU gradient: 1 where input > 0, 0 elsewhere
        relu_grad = (input_tensor.data > 0).astype(np.float32) * grad_output.data
        
        if input_tensor.grad is None:
            input_tensor.grad = TernaryTensor(np.zeros_like(input_tensor.data))
        input_tensor.grad.data += relu_grad


class SigmoidFunction(Function):
    """Sigmoid gradient function"""
    
    def __init__(self, input_tensor, output_tensor):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def backward(self, grad_output):
        if not self.saved_tensors:
            return
        
        input_tensor = self.saved_tensors
