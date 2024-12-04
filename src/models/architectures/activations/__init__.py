
from .relu_clipped import relu_clipped
from .sigmoid import sigmoid
from .softplus_clipped import softplus_clipped
from .tanh_normalized import tanh_normalized

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_activation_function(activation_name: str):
    """
    Returns the activation function corresponding to the given name.

    Args:
        activation_name (str): The name of the activation function.

    Returns:
        Callable: The corresponding activation function.
    """
    if activation_name == "relu_clipped":
        return relu_clipped
    elif activation_name == "sigmoid":
        return sigmoid
    elif activation_name == "softplus_clipped":
        return softplus_clipped
    elif activation_name == "tanh_normalized":
        return tanh_normalized
    elif activation_name == "":
        return lambda x: x
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}, avaliable options: relu_clipped, sigmoid, softplus_clipped, tanh_normalized")
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------