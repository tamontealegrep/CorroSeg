
import tkinter as tk

from src.models.manager import ModelManager
from src.models.architectures.networks.unet import Unet

from src.utils.files import load_config
from src.gui.utils.utils import toggle_multiple_widgets

#-----------------------------------------------------------------------------------------------------------------------------------------------------

config = load_config("./src/config/config.yaml")
manager_default = config["manager"]

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def loss_options_manager(
        loss_class: tk.StringVar,
        alpha: tk.Entry,
        beta: tk.Entry,
        gamma: tk.Entry,
        base_weight: tk.Entry,
        focal_weight: tk.Entry,
        ) -> None:
    """
    Manages the visibility and enabled state of loss function-related widgets
    based on the selected loss function.
    
    Parameters:
        loss_class (tk.StringVar): A Tkinter StringVar that holds the selected loss function type (e.g., "DICELoss", "FocalLoss", etc.).
        alpha (tk.Entry): Entry widget for the alpha parameter in certain loss functions.
        beta (tk.Entry): Entry widget for the beta parameter in certain loss functions.
        gamma (tk.Entry): Entry widget for the gamma parameter in certain loss functions.
        base_weight (tk.Entry): Entry widget for the base weight parameter in certain loss functions.
        focal_weight (tk.Entry): Entry widget for the focal weight parameter in certain loss functions.
    
    Notes:
        - Depending on the selected loss function, different entry fields (widgets) are enabled or disabled.
        - The loss function options and their corresponding widgets:
            - "DICELoss": Disables all widgets.
            - "FocalLoss": Enables alpha and gamma, disables others.
            - "IoULoss": Disables all widgets.
            - "IoUFocalLoss": Enables alpha, gamma, base_weight, and focal_weight, disables beta.
            - "TverskyLoss": Enables alpha and beta, disables others.
    """
    state = loss_class.get()
    if state == "DICELoss":
        toggle_multiple_widgets([],[alpha, beta, gamma, base_weight, focal_weight])
    elif state == "FocalLoss":
        toggle_multiple_widgets([alpha, gamma],[beta, base_weight, focal_weight])
    elif state == "IoULoss":
        toggle_multiple_widgets([],[alpha, beta, gamma, base_weight, focal_weight])
    elif state == "IoUFocalLoss":
        toggle_multiple_widgets([alpha, gamma, base_weight, focal_weight],[beta])
    elif state == "TverskyLoss":
        toggle_multiple_widgets([alpha, beta],[gamma, base_weight, focal_weight])

def get_block_type(num_recurrences: int, residual: bool, state: bool = True) -> str:
    """
    Determines the type of convolutional block based on the recurrence and residual parameters.
    
    Parameters:
        num_recurrences (int): The number of recurrences. A positive value indicates a recurrent block.
        residual (bool): A flag indicating whether the block has a residual connection.
        state (bool, optional): A flag indicating whether the block type should be returned or not. Default is True.
    
    Returns:
        str: The type of the block:
            - "RRConvBlock" if both recurrent and residual are True.
            - "RecConvBlock" if recurrent is True but residual is False.
            - "ResConvBlock" if residual is True but recurrent is False.
            - "ConvBlock" if neither recurrent nor residual are True.
            - "None" if `state` is False.
    
    Notes:
        - The function combines the `num_recurrences` and `residual` arguments to determine the block type.
        - If `state` is set to `False`, the function will return "None" regardless of the other arguments.
    """
    recurrent = True if num_recurrences > 0 else False
    if state:
        if recurrent and residual:
            block_type = "RRConvBlock"
        elif recurrent:
            block_type = "RecConvBlock"
        elif residual:
            block_type = "ResConvBlock"
        else:
            block_type = "ConvBlock"
    else:
        block_type = "None"

    return block_type

def get_network_dictionary(
    input_channels: tk.IntVar,
    output_channels: tk.IntVar,
    base_channels: tk.IntVar,
    num_layers: tk.IntVar,
    e_num_recurrences: tk.IntVar,
    e_residual: tk.BooleanVar,
    b_num_recurrences: tk.IntVar,
    b_residual: tk.BooleanVar,
    d_num_recurrences: tk.IntVar,
    d_residual: tk.BooleanVar,
    s_num_recurrences: tk.IntVar,
    s_residual: tk.BooleanVar,
    skip_connections: tk.BooleanVar,
    e_activation: tk.StringVar,
    e_dropout_prob: tk.StringVar,
    e_cbam: tk.BooleanVar,
    e_cbam_reduction: tk.IntVar,
    e_cbam_activation: tk.StringVar,
    b_activation: tk.StringVar,
    b_dropout_prob: tk.StringVar,
    b_cbam: tk.BooleanVar,
    b_cbam_reduction: tk.IntVar,
    b_cbam_activation: tk.StringVar,
    d_activation: tk.StringVar,
    d_dropout_prob: tk.StringVar,
    d_cbam: tk.BooleanVar,
    d_cbam_reduction: tk.IntVar,
    d_cbam_activation: tk.StringVar,
    s_activation: tk.StringVar,
    s_dropout_prob: tk.StringVar,
    s_cbam: tk.BooleanVar,
    s_cbam_reduction: tk.IntVar,
    s_cbam_activation: tk.StringVar,
    attention_gates: tk.BooleanVar,
    output_activation: tk.StringVar
) -> dict:
    """
    Retrieves the network configuration dictionary, adjusting parameters 
    based on the selected block types and other configurations.
    
    This function first constructs a complete network configuration dictionary 
    by calling `build_network_dictionary`. Then, it removes the `num_recurrences` 
    parameter from the encoder, bottleneck, decoder, or skip connections block 
    configurations if the respective block type is not of type "RRConvBlock" or "RecConvBlock".
    
    Parameters:
        input_channels (tk.IntVar): Tkinter variable holding the number of input channels.
        output_channels (tk.IntVar): Tkinter variable holding the number of output channels.
        base_channels (tk.IntVar): Tkinter variable holding the number of base channels in the network.
        num_layers (tk.IntVar): Tkinter variable holding the number of layers in the network.
        e_num_recurrences (tk.IntVar): Tkinter variable holding the number of recurrences in the encoder block.
        e_residual (tk.BooleanVar): Tkinter variable indicating whether the encoder block has a residual connection.
        b_num_recurrences (tk.IntVar): Tkinter variable holding the number of recurrences in the bottleneck block.
        b_residual (tk.BooleanVar): Tkinter variable indicating whether the bottleneck block has a residual connection.
        d_num_recurrences (tk.IntVar): Tkinter variable holding the number of recurrences in the decoder block.
        d_residual (tk.BooleanVar): Tkinter variable indicating whether the decoder block has a residual connection.
        s_num_recurrences (tk.IntVar): Tkinter variable holding the number of recurrences in the skip connections block.
        s_residual (tk.BooleanVar): Tkinter variable indicating whether the skip connections block has a residual connection.
        skip_connections (tk.BooleanVar): Tkinter variable indicating whether skip connections are used in the network.
        e_activation (tk.StringVar): Tkinter variable holding the activation function for the encoder block.
        e_dropout_prob (tk.StringVar): Tkinter variable holding the dropout probability for the encoder block.
        e_cbam (tk.BooleanVar): Tkinter variable indicating whether to use CBAM in the encoder block.
        e_cbam_reduction (tk.IntVar): Tkinter variable holding the reduction factor for CBAM in the encoder block.
        e_cbam_activation (tk.StringVar): Tkinter variable holding the activation function for CBAM in the encoder block.
        b_activation (tk.StringVar): Tkinter variable holding the activation function for the bottleneck block.
        b_dropout_prob (tk.StringVar): Tkinter variable holding the dropout probability for the bottleneck block.
        b_cbam (tk.BooleanVar): Tkinter variable indicating whether to use CBAM in the bottleneck block.
        b_cbam_reduction (tk.IntVar): Tkinter variable holding the reduction factor for CBAM in the bottleneck block.
        b_cbam_activation (tk.StringVar): Tkinter variable holding the activation function for CBAM in the bottleneck block.
        d_activation (tk.StringVar): Tkinter variable holding the activation function for the decoder block.
        d_dropout_prob (tk.StringVar): Tkinter variable holding the dropout probability for the decoder block.
        d_cbam (tk.BooleanVar): Tkinter variable indicating whether to use CBAM in the decoder block.
        d_cbam_reduction (tk.IntVar): Tkinter variable holding the reduction factor for CBAM in the decoder block.
        d_cbam_activation (tk.StringVar): Tkinter variable holding the activation function for CBAM in the decoder block.
        s_activation (tk.StringVar): Tkinter variable holding the activation function for the skip connections block.
        s_dropout_prob (tk.StringVar): Tkinter variable holding the dropout probability for the skip connections block.
        s_cbam (tk.BooleanVar): Tkinter variable indicating whether to use CBAM in the skip connections block.
        s_cbam_reduction (tk.IntVar): Tkinter variable holding the reduction factor for CBAM in the skip connections block.
        s_cbam_activation (tk.StringVar): Tkinter variable holding the activation function for CBAM in the skip connections block.
        attention_gates (tk.BooleanVar): Tkinter variable indicating whether to use attention gates in the network.
        output_activation (tk.StringVar): Tkinter variable holding the activation function for the output layer.
    
    Returns:
        dict: A network configuration dictionary with adjusted parameters for block types and other configurations.
    """
    dictionary = {
        "input_channels": int(input_channels.get()),
        "output_channels": int(output_channels.get()),
        "base_channels": int(base_channels.get()),
        "num_layers": int(num_layers.get()),
        "encoder_block_type": get_block_type(e_num_recurrences.get(),e_residual.get()),
        "bottleneck_block_type": get_block_type(b_num_recurrences.get(),b_residual.get()),
        "decoder_block_type": get_block_type(d_num_recurrences.get(),d_residual.get()),
        "skip_connections_block_type": get_block_type(s_num_recurrences.get(),s_residual.get(),skip_connections.get()),
        "encoder_kwargs":{
            "activation": e_activation.get(),
            "dropout_prob": float(e_dropout_prob.get()),
            "num_recurrences": int(e_num_recurrences.get()),
            "cbam": e_cbam.get(),
            "cbam_reduction": e_cbam_reduction.get(),
            "cbam_activation": e_cbam_activation.get(),
        },
        "bottleneck_kwargs":{
            "activation": b_activation.get(),
            "dropout_prob": float(b_dropout_prob.get()),
            "num_recurrences": int(b_num_recurrences.get()),
            "cbam": b_cbam.get(),
            "cbam_reduction": b_cbam_reduction.get(),
            "cbam_activation": b_cbam_activation.get(),
        },
        "decoder_kwargs":{
            "activation": d_activation.get(),
            "dropout_prob": float(d_dropout_prob.get()),
            "num_recurrences": int(d_num_recurrences.get()),
            "cbam": d_cbam.get(),
            "cbam_reduction": d_cbam_reduction.get(),
            "cbam_activation": d_cbam_activation.get(),
        },
        "skip_connections_kwargs":{
            "activation": s_activation.get(),
            "dropout_prob": float(s_dropout_prob.get()),
            "num_recurrences": int(s_num_recurrences.get()),
            "cbam": s_cbam.get(),
            "cbam_reduction": s_cbam_reduction.get(),
            "cbam_activation": s_cbam_activation.get(),
        },
        "attention_gates": attention_gates.get(),
        "output_activation": output_activation.get(),
    }

    if dictionary["encoder_block_type"] not in ["RRConvBlock", "RecConvBlock"]:
        dictionary["encoder_kwargs"].pop("num_recurrences", None)

    if dictionary["bottleneck_block_type"] not in ["RRConvBlock", "RecConvBlock"]:
        dictionary["bottleneck_kwargs"].pop("num_recurrences", None)

    if dictionary["decoder_block_type"] not in ["RRConvBlock", "RecConvBlock"]:
        dictionary["decoder_kwargs"].pop("num_recurrences", None)

    if dictionary.get("skip_connections_block_type") == "None":
        dictionary["skip_connections_kwargs"] = {}
    elif dictionary["skip_connections_block_type"] not in ["RRConvBlock", "RecConvBlock"]:
        dictionary["skip_connections_kwargs"].pop("num_recurrences", None)

    return dictionary

def get_manager_dictionary(
    loss_class: tk.StringVar,
    alpha: tk.StringVar,
    beta: tk.StringVar,
    gamma: tk.StringVar,
    base_weight: tk.StringVar,
    focal_weight: tk.StringVar,
    learning_rate: tk.StringVar,
    weight_decay: tk.StringVar
) -> dict:
    """
    Retrieves the manager configuration dictionary, adjusting parameters for different loss functions.
    
    This function calls `build_manager_dictionary()` to get the initial configuration and then
    adjusts the `loss_params` based on the selected `loss_class`. Certain parameters are removed
    depending on the chosen loss type.
    
    Parameters:
        loss_class (tk.StringVar): Tkinter variable for the loss class type (e.g., 'DICELoss', 'FocalLoss').
        alpha (tk.StringVar): Tkinter variable for the alpha parameter in the loss function.
        beta (tk.StringVar): Tkinter variable for the beta parameter in the loss function.
        gamma (tk.StringVar): Tkinter variable for the gamma parameter in the loss function.
        base_weight (tk.StringVar): Tkinter variable for the base weight parameter in the loss function.
        focal_weight (tk.StringVar): Tkinter variable for the focal weight parameter in the loss function.
        learning_rate (tk.StringVar): Tkinter variable for the learning rate parameter in the optimizer.
        weight_decay (tk.StringVar): Tkinter variable for the weight decay parameter in the optimizer.
    
    Returns:
        dict: The manager configuration dictionary with adjusted loss parameters.
    """
    dictionary = {
        "dataset_class": manager_default["dataset_class"],
        "loss_class": str(loss_class.get()),
        "optimizer_class": manager_default["optimizer_class"],
        "scaler_type": manager_default["scaler_type"],
        "input_shape": manager_default["input_shape"],
        "placeholders": manager_default["placeholders"],
        "value_range": manager_default["value_range"],
        "default_value": manager_default["default_value"],
        "loss_params": {
            "alpha": float(alpha.get()),
            "beta": float(beta.get()),
            "gamma": float(gamma.get()),
            "base_weight": float(base_weight.get()),
            "focal_weight": float(focal_weight.get()),
        },
        "optimizer_params": {
            "lr": float(learning_rate.get()),
            "weight_decay": float(weight_decay.get())
        },
        "scaler_params": manager_default["scaler_params"],
    }

    if dictionary["loss_class"] in ["DICELoss", "IoULoss"]:
        dictionary["loss_params"] = {}

    if dictionary["loss_class"] == "FocalLoss":
        for i in ["beta", "base_weight", "focal_weight"]:
            if i in dictionary["loss_params"]:
                del dictionary["loss_params"][i]

    if dictionary["loss_class"] == "IoUFocalLoss":
        if "beta" in dictionary["loss_params"]:
            del dictionary["loss_params"]["beta"]
        if "base_weight" in dictionary["loss_params"]:
            dictionary["loss_params"]["iou_weight"] = dictionary["loss_params"].pop("base_weight")

    if dictionary["loss_class"] == "TverskyLoss":
        for i in ["gamma", "base_weight", "focal_weight"]:
            if i in dictionary["loss_params"]:
                del dictionary["loss_params"][i]

    return dictionary

def make_model(input_channels: tk.IntVar,
    output_channels: tk.IntVar,
    base_channels: tk.IntVar,
    num_layers: tk.IntVar,
    e_num_recurrences: tk.IntVar,
    e_residual: tk.BooleanVar,
    b_num_recurrences: tk.IntVar,
    b_residual: tk.BooleanVar,
    d_num_recurrences: tk.IntVar,
    d_residual: tk.BooleanVar,
    s_num_recurrences: tk.IntVar,
    s_residual: tk.BooleanVar,
    skip_connections: tk.BooleanVar,
    e_activation: tk.StringVar,
    e_dropout_prob: tk.StringVar,
    e_cbam: tk.BooleanVar,
    e_cbam_reduction: tk.IntVar,
    e_cbam_activation: tk.StringVar,
    b_activation: tk.StringVar,
    b_dropout_prob: tk.StringVar,
    b_cbam: tk.BooleanVar,
    b_cbam_reduction: tk.IntVar,
    b_cbam_activation: tk.StringVar,
    d_activation: tk.StringVar,
    d_dropout_prob: tk.StringVar,
    d_cbam: tk.BooleanVar,
    d_cbam_reduction: tk.IntVar,
    d_cbam_activation: tk.StringVar,
    s_activation: tk.StringVar,
    s_dropout_prob: tk.StringVar,
    s_cbam: tk.BooleanVar,
    s_cbam_reduction: tk.IntVar,
    s_cbam_activation: tk.StringVar,
    attention_gates: tk.BooleanVar,
    output_activation: tk.StringVar,
    loss_class: tk.StringVar,
    alpha: tk.StringVar,
    beta: tk.StringVar,
    gamma: tk.StringVar,
    base_weight: tk.StringVar,
    focal_weight: tk.StringVar,
    learning_rate: tk.StringVar,
    weight_decay: tk.StringVar):
    """
    Makes the model
    """
    network_dict = get_network_dictionary(
    input_channels, output_channels, base_channels, num_layers,
    e_num_recurrences, e_residual,
    b_num_recurrences, b_residual,
    d_num_recurrences, d_residual,
    s_num_recurrences, s_residual,
    skip_connections,
    e_activation, e_dropout_prob, e_cbam, e_cbam_reduction, e_cbam_activation,
    b_activation, b_dropout_prob, b_cbam, b_cbam_reduction, b_cbam_activation,
    d_activation, d_dropout_prob, d_cbam, d_cbam_reduction, d_cbam_activation,
    s_activation, s_dropout_prob, s_cbam, s_cbam_reduction, s_cbam_activation,
    attention_gates, output_activation
    )

    manager_dict = get_manager_dictionary(
    loss_class, alpha, beta, gamma, base_weight, focal_weight, learning_rate, weight_decay
    )

    network = Unet.from_dict(network_dict)
    model = ModelManager.from_dict(network, manager_dict)

    return model

#-----------------------------------------------------------------------------------------------------------------------------------------------------