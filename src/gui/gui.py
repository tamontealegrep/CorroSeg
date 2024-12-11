
import tkinter as tk
from tkinter import filedialog, messagebox


import tkinter as tk
from tkinter import Toplevel, filedialog, messagebox
from PIL import Image, ImageTk

from src.utils.plots import np_array_to_pil

from src.models.manager import ModelManager
from src.models.architectures.networks.unet import Unet
from src.models.dataset.transformations import random_transformation
from src.utils.files import load_config, load_arrays_from_folders, npy_file_to_dict

from src.gui.utils.load import load_model
from src.gui.utils.save import save_model
from src.gui.utils.train import train_model
from src.gui.utils.new import loss_options_manager, make_model
from src.gui.utils.predict import load_file, predict_model
from src.gui.utils.plot import update_canvas
from src.gui.utils.widgets import create_checkbox, create_entry, create_option_menu, create_slider
from src.gui.utils.utils import toggle_widgets_by_bool, scrollbar_command

#-----------------------------------------------------------------------------------------------------------------------------------------------------

config = load_config("./src/config/config.yaml")
network_default = config["networks"]["unet"]
manager_default = config["manager"]
train_default = config["train_validation"]
predict_default = config["predict"]
threshold_default = config["threshold"]

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class Gui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CorroSeg")

        self.model:ModelManager = None
        self.file = None
        self.file_name = None
        self.mask = None
        self.prediction = None

        self.root_widgets()

    def run(self):
        self.root.mainloop()

    def root_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack()

        model_frame = tk.LabelFrame(frame, text="Model")
        model_frame.grid(row=0, column=0, padx=20, pady=10)

        new_model_button = tk.Button(model_frame, text="New Model", command=self.model_new)
        new_model_button.grid(row=0,column=0,padx=5, pady=10)

        load_model_button = tk.Button(model_frame, text="Load Model", command=self.model_load)
        load_model_button.grid(row=0,column=1,padx=5, pady=10)

        load_model_button = tk.Button(model_frame, text="Save Model", command=self.model_save)
        load_model_button.grid(row=0,column=2,padx=5, pady=10)

        operations_frame = tk.LabelFrame(frame, text="Operations")
        operations_frame.grid(row=1, column=0, padx=20, pady=10)

        train_button = tk.Button(operations_frame, text="Train", command=self.model_train)
        train_button.grid(row=0,column=0,padx=5, pady=10)

        predict_button = tk.Button(operations_frame, text="Predict", command=self.model_predict)
        predict_button.grid(row=0,column=1,padx=5, pady=10)

        plot_frame = tk.LabelFrame(frame, text="Plot")
        plot_frame.grid(row=2, column=0, padx=20, pady=10)

        plot_button = tk.Button(plot_frame, text="Plot", command=self.plot)
        plot_button.grid(row=0,column=0,padx=5, pady=10)

        save_plot_button = tk.Button(plot_frame, text="Save Plot", command=self.save_plot)
        save_plot_button.grid(row=0,column=1,padx=5, pady=10)

    def test_function(self):
        print("OK")

    def model_load(self):
        load_model(self)

    def model_save(self):
        save_model(self)

    def model_new(self):
        window = Toplevel(self.root)
        window.title("New Model")

        frame = tk.Frame(window)
        frame.pack()

        arch_model_frame = tk.LabelFrame(frame, text="Architecture")
        arch_model_frame.grid(row=0, column=0, padx=20, pady=20)

        input_channels = create_slider(arch_model_frame,0,0,"Input Channels:",1,16,network_default["input_channels"],padx=5)
        output_channels = create_slider(arch_model_frame,1,0,"Output Channels:",1,16,network_default["output_channels"],padx=5)
        base_channels = create_slider(arch_model_frame,2,0,"Base Channels:",1,64,network_default["base_channels"],padx=5)
        num_layers = create_slider(arch_model_frame,3,0,"Number of Layers:",1,5,network_default["num_layers"],padx=5)
        output_activation, _ = create_option_menu(arch_model_frame,4,0,"Output Activation:",["relu_clipped","sigmoid", "softplus_clipped", "tanh_normalized", ""],network_default["output_activation"],padx=5, pady=10)
        attention_gates, _ = create_checkbox(arch_model_frame,5,0,"Attention Gates",network_default["attention_gates"],padx=5,pady=5)
        skip_connections, _ = create_checkbox(arch_model_frame,6,0,"Skip Connections",True,padx=5,pady=5)
    
        blocks_model_frame = tk.LabelFrame(frame, text="Blocks")
        blocks_model_frame.grid(row=0, column=1, padx=10, pady=20)

        encoder_frame = tk.LabelFrame(blocks_model_frame, text="Encoder")
        encoder_frame.grid(row=0, column=0, padx=5, pady=5)

        e_activation, _ = create_option_menu(encoder_frame,0,0,"Activation:",["ReLU", "LeakyReLU", "None"],network_default["encoder_kwargs"]["activation"],pady=5)
        e_dropout_prob = create_slider(encoder_frame,1,0,"Dropout:",0,1,network_default["encoder_kwargs"]["dropout_prob"],resolution=0.05,pady=5)
        e_num_recurrences = create_slider(encoder_frame,2,0,"Number of recurrences:",0,5,network_default["encoder_kwargs"]["num_recurrences"],pady=5)
        e_residual, _ = create_checkbox(encoder_frame,3,0,"Residual conection",True,pady= 5)
        e_cbam, _ = create_checkbox(encoder_frame,4,0,"CBAM",network_default["encoder_kwargs"]["cbam"],pady= 5)
        e_cbam_reduction = create_slider(encoder_frame,5,0,"CBAM Reduction:",1,32,network_default["encoder_kwargs"]["cbam_reduction"], pady=5)
        e_cbam_activation, e_cbam_activation_menu = create_option_menu(encoder_frame,6,0,"CBAM Activation:",["ReLU", "LeakyReLU", "None"],network_default["encoder_kwargs"]["cbam_activation"],padx=5, pady=5)

        toggle_e_cbam_dependences =  lambda: toggle_widgets_by_bool(e_cbam, [e_cbam_reduction, e_cbam_activation_menu])
        e_cbam.trace("w", lambda *args: toggle_e_cbam_dependences())

        bottleneck_frame = tk.LabelFrame(blocks_model_frame, text="Bottleneck")
        bottleneck_frame.grid(row=0, column=1, padx=5, pady=5)

        b_activation, _ = create_option_menu(bottleneck_frame,0,0,"Activation:",["ReLU", "LeakyReLU", "None"],network_default["bottleneck_kwargs"]["activation"],pady=5)
        b_dropout_prob = create_slider(bottleneck_frame,1,0,"Dropout:",0,1,network_default["bottleneck_kwargs"]["dropout_prob"],resolution=0.05,pady=5)
        b_num_recurrences = create_slider(bottleneck_frame,2,0,"Number of recurrences:",0,5,network_default["bottleneck_kwargs"]["num_recurrences"],pady=5)
        b_residual, _ = create_checkbox(bottleneck_frame,3,0,"Residual conection",True,pady= 5)
        b_cbam, _ = create_checkbox(bottleneck_frame,4,0,"CBAM",network_default["bottleneck_kwargs"]["cbam"],pady=5)
        b_cbam_reduction = create_slider(bottleneck_frame,5,0,"CBAM Reduction:",1,32,network_default["bottleneck_kwargs"]["cbam_reduction"], pady=5)
        b_cbam_activation, b_cbam_activation_menu = create_option_menu(bottleneck_frame,6,0,"CBAM Activation:",["ReLU", "LeakyReLU", "None"],network_default["bottleneck_kwargs"]["cbam_activation"],padx=5,pady=5)

        toggle_b_cbam_dependences =  lambda: toggle_widgets_by_bool(b_cbam, [b_cbam_reduction, b_cbam_activation_menu])
        b_cbam.trace("w", lambda *args: toggle_b_cbam_dependences())

        decoder_frame = tk.LabelFrame(blocks_model_frame, text="Decoder")
        decoder_frame.grid(row=0, column=2, padx=5, pady=5)

        d_activation, _ = create_option_menu(decoder_frame,0,0,"Activation:",["ReLU", "LeakyReLU", "None"],network_default["decoder_kwargs"]["activation"],pady=5)
        d_dropout_prob = create_slider(decoder_frame,1,0,"Dropout:",0,1,network_default["decoder_kwargs"]["dropout_prob"],resolution=0.05,pady=5)
        d_num_recurrences = create_slider(decoder_frame,2,0,"Number of recurrences:",0,5,network_default["decoder_kwargs"]["num_recurrences"],pady=5)
        d_residual, _ = create_checkbox(decoder_frame,3,0,"Residual conection",True,pady= 5)
        d_cbam, _ = create_checkbox(decoder_frame,4,0,"CBAM",network_default["decoder_kwargs"]["cbam"],pady=5)
        d_cbam_reduction = create_slider(decoder_frame,5,0,"CBAM Reduction:",1,32,network_default["decoder_kwargs"]["cbam_reduction"], pady=5)
        d_cbam_activation, d_cbam_activation_menu = create_option_menu(decoder_frame,6,0,"CBAM Activation:",["ReLU", "LeakyReLU", "None"],network_default["decoder_kwargs"]["cbam_activation"],padx=5,pady=5)

        toggle_d_cbam_dependences =  lambda: toggle_widgets_by_bool(d_cbam, [d_cbam_reduction, d_cbam_activation_menu])
        d_cbam.trace("w", lambda *args: toggle_d_cbam_dependences())

        connections_frame = tk.LabelFrame(blocks_model_frame, text="Skip Connections")
        connections_frame.grid(row=0, column=3, padx=5, pady=5)

        s_activation, s_activation_menu = create_option_menu(connections_frame,0,0,"Activation:",["ReLU", "LeakyReLU", "None"],network_default["skip_connections_kwargs"]["activation"],pady= 5)
        s_dropout_prob = create_slider(connections_frame,1,0,"Dropout:",0,1,network_default["skip_connections_kwargs"]["dropout_prob"],resolution=0.05,pady=5)
        s_num_recurrences = create_slider(connections_frame,2,0,"Number of recurrences:",0,5,network_default["skip_connections_kwargs"]["num_recurrences"],pady=5)
        s_residual, s_residual_check = create_checkbox(connections_frame,3,0,"Residual conection",True,pady= 5)
        s_cbam, s_cbam_check = create_checkbox(connections_frame,4,0,"CBAM",network_default["skip_connections_kwargs"]["cbam"],pady= 5)
        s_cbam_reduction = create_slider(connections_frame,5,0,"CBAM Reduction:",1,32,network_default["skip_connections_kwargs"]["cbam_reduction"], pady= 5)
        s_cbam_activation, s_cbam_activation_menu = create_option_menu(connections_frame,6,0,"CBAM Activation:",["ReLU", "LeakyReLU", "None"],network_default["skip_connections_kwargs"]["cbam_activation"],pady= 5)

        toggle_s_cbam_dependences =  lambda: toggle_widgets_by_bool(s_cbam, [s_cbam_reduction, s_cbam_activation_menu])
        s_cbam.trace("w", lambda *args: toggle_s_cbam_dependences())

        toggle_skip_dependences =  lambda: toggle_widgets_by_bool(
            skip_connections,
            [s_activation_menu, s_dropout_prob, s_num_recurrences, s_residual_check, s_cbam_check, s_cbam_reduction, s_cbam_activation_menu]
            )
        
        skip_connections.trace("w", lambda *args: toggle_skip_dependences())

        parameters_frame = tk.Frame(frame)
        parameters_frame.grid(row=0, column=2, padx=10, pady=20)

        loss_frame = tk.LabelFrame(parameters_frame, text="Loss")
        loss_frame.grid(row=0, column=0, padx=5, pady=5)

        loss_class, loss_class_menu = create_option_menu(
            loss_frame,0,0,"Function:",
            ["DICELoss","FocalLoss", "IoULoss", "IoUFocalLoss", "TverskyLoss"],
            manager_default["loss_class"],
            padx=5,
            pady=5
            )
        alpha = create_entry(loss_frame,1,0,"Alpha: ","0.5", pady=5)
        beta = create_entry(loss_frame,2,0,"Beta: ","0.5", pady=5)
        gamma = create_entry(loss_frame,3,0,"Gamma: ","2", pady=5)
        base_weight = create_entry(loss_frame,4,0,"Base weight: ","1", pady=5)
        focal_weight = create_entry(loss_frame,5,0,"Focal weight: ","10", pady=5)
        
        optim_frame = tk.LabelFrame(parameters_frame, text="Optimizer")
        optim_frame.grid(row=1, column=0, padx=5, pady=5)

        learning_rate = create_entry(optim_frame,0,0,"Learning rate: ",str(manager_default["optimizer_params"]["lr"]),pady=5)
        weight_decay = create_entry(optim_frame,1,0,"Weight decay: ",str(manager_default["optimizer_params"]["weight_decay"]),width=7,padx=5,pady=5)

        loss_class.trace("w", lambda *args: loss_options_manager(loss_class, alpha, beta, gamma, base_weight, focal_weight))

        def new_model_confirmation():
            answer = messagebox.askyesno("Confirmation", "Are you sure?")
            if answer:
                self.model = make_model(
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
                    attention_gates, output_activation,
                    loss_class, alpha, beta, gamma, base_weight, focal_weight, learning_rate, weight_decay)
                window.destroy()
            else:
                pass

        save_button = tk.Button(frame, text="Make model", command=new_model_confirmation)
        save_button.grid(row=1, columnspan=3, pady=10)

    def model_train(self):
        window = tk.Toplevel(self.root)
        window.title("Train")

        frame = tk.Frame(window)
        frame.pack()

        train_frame = tk.LabelFrame(frame, text="Options")
        train_frame.grid(row=0, column=0, padx=20, pady=10)

        num_epochs = create_entry(train_frame,0,0,"Epochs: ","10", pady=5)
        batch_size = create_entry(train_frame,1,0,"Batch Size: ","128", pady= 5)
        fraction = create_slider(train_frame,2,0,"Validation Fraction:",0,0.3,train_default["fraction"],resolution=0.01,pady=5)
        seed = create_entry(train_frame,3,0,"Seed: ",str(train_default["seed"]), pady= 5)
        expand, _ = create_checkbox(train_frame,4,0,"Horizontal Expansion",train_default["expand"],pady= 5)
        augmented_ratio = create_slider(train_frame,5,0,"Augmented Ratio:",0,1,train_default["augmented_ratio"],resolution=0.1,pady=5)

        save_button = tk.Button(frame, text="Train model", command=lambda: train_model(
            self, fraction, seed, expand, augmented_ratio, batch_size, num_epochs
        ))
        save_button.grid(row=1, column=0, pady=10)

    def model_predict(self):
        window = tk.Toplevel(self.root)
        window.title("Predict")
        
        frame = tk.Frame(window)
        frame.pack()

        file_frame = tk.LabelFrame(frame, text="File")
        file_frame.grid(row=0, column=0, padx=20, pady=10)

        name_label_var = tk.StringVar()
        if self.file is None:
            name_label_var.set("No file loaded")
        else:
            name_label_var.set(self.file_name)

        name_label = tk.Label(file_frame, textvariable=name_label_var)
        name_label.grid(row=0,column=0,pady=5, padx=10)

        def load_file_and_update(self):
            load_file(self)
            
            if self.file_name:
                name_label_var.set(self.file_name)

            if self.prediction:
                self.prediction = None
                
        load_button = tk.Button(file_frame, text="Load File", command=lambda: load_file_and_update(self))
        load_button.grid(row=1,column=0, pady=5, padx=5)
        
        predict_button = tk.Button(frame, text="Predict", command=lambda: predict_model(self))
        predict_button.grid(row=1,column=0,pady=10)

    def plot(self):
        window = tk.Toplevel(self.root)
        window.title("Plot")

        frame = tk.Frame(window)
        frame.pack(fill=tk.BOTH, expand=True)

        if self.file is None and self.prediction is None:
            messagebox.showerror("Error", "A file must be loaded and a prediction made.")
            window.destroy()

        canvas_size = (500, 72)  # (height, width)

        data_img = np_array_to_pil(
            self.file,
            manager_default["value_range"],
            cmap_name="seismic",
            output_width=canvas_size[1]
        )

        mask_img = np_array_to_pil(
            self.prediction,
            (0, 1),
            cmap_name="viridis",
            output_width=canvas_size[1]
        )

        scroll_region = data_img.size  # (width, height)
        total_height = scroll_region[1]
        section_height = 10000

        num_sections = total_height // section_height
        if total_height % section_height != 0:
            num_sections += 1 

        plots_frame = tk.LabelFrame(frame, text="Images")
        plots_frame.grid(row=0, column=0, padx=20, pady=20)

        data_frame = tk.LabelFrame(plots_frame, text="Data")
        data_frame.grid(row=0, column=0, padx=20, pady=20)

        mask_frame = tk.LabelFrame(plots_frame, text="Mask")
        mask_frame.grid(row=0, column=1, padx=20, pady=20)

        data_canvas = tk.Canvas(data_frame, height=canvas_size[0], width=canvas_size[1])
        data_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        mask_canvas = tk.Canvas(mask_frame, height=canvas_size[0], width=canvas_size[1])
        mask_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(plots_frame, orient=tk.VERTICAL)
        scrollbar.grid(row=0, column=2, sticky='ns', padx=20, pady=20)
        
        scrollbar.config(command=lambda *args: scrollbar_command([data_canvas, mask_canvas], *args))

        data_canvas.config(yscrollcommand=scrollbar.set)
        mask_canvas.config(yscrollcommand=scrollbar.set)

        section_index = 0

        def next_section():
            nonlocal section_index
            if section_index < num_sections - 1:
                section_index += 1
                update_canvas(data_img, mask_img, data_canvas, mask_canvas, section_height, section_index)
                slider.set(section_index)
                update_buttons()

        def prev_section():
            nonlocal section_index
            if section_index > 0:
                section_index -= 1
                update_canvas(data_img, mask_img, data_canvas, mask_canvas, section_height, section_index)
                slider.set(section_index)
                update_buttons()

        def update_slider(value):
            nonlocal section_index
            section_index = int(value)
            update_canvas(data_img, mask_img, data_canvas, mask_canvas, section_height, section_index)
            slider.set(section_index)
            update_buttons()

        def update_buttons():
            if section_index == 0:
                prev_button.config(state=tk.DISABLED)
            else:
                prev_button.config(state=tk.NORMAL)

            if section_index == num_sections:
                next_button.config(state=tk.DISABLED)
            else:
                next_button.config(state=tk.NORMAL)

        button_frame = tk.Frame(window)
        button_frame.pack(expand=True, padx=20, pady=10)

        prev_button = tk.Button(button_frame, text="Previous", command=prev_section)
        prev_button.grid(row=0, column=0, padx=5)

        slider = tk.Scale(button_frame, from_=0, to= num_sections - 1, orient="horizontal", command=update_slider)
        slider.grid(row=0, column=1, padx=5, pady=5) 

        next_button = tk.Button(button_frame, text="Next", command=next_section)
        next_button.grid(row=0, column=2, padx=5) 

        update_canvas(data_img, mask_img, data_canvas, mask_canvas, section_height, section_index)
        update_buttons()

    def save_plot(self):
        if self.prediction is None:
            messagebox.showwarning("Error", "There is no prediction to save.")
            return

        model_path = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            filetypes=[("Image file", "*.png")]
        )

        data_img = np_array_to_pil(
            self.file,
            manager_default["value_range"],
            cmap_name="seismic",
        )

        mask_img = np_array_to_pil(
            self.prediction,
            (0, 1),
            cmap_name="viridis",
        )

        if data_img.size != mask_img.size :
            messagebox.showwarning("Error", "File and prediction must have the same size.")
            return
        
        img_size = data_img.size # (width, height)
        new_img_size = (2 * img_size[0], img_size[1])

        new_image = Image.new('RGB', new_img_size)

        new_image.paste(data_img, (0, 0))
        new_image.paste(mask_img, (img_size[0], 0))

        if model_path:
            new_image.save(model_path)

#-----------------------------------------------------------------------------------------------------------------------------------------------------