
import platform

import numpy as np
import tkinter as tk
from tkinter import Toplevel, filedialog, messagebox
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.models.manager import ModelManager
from src.utils.files import load_config
from src.utils.plots import np_array_to_pil

from src.gui.utils.load import load_model
from src.gui.utils.save import save_model
from src.gui.utils.train import train_model
from src.gui.utils.new import loss_options_manager, make_model
from src.gui.utils.files import load_file
from src.gui.utils.plot import update_canvas, plot_training
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
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.y_pred: np.ndarray = None
        self.well_name: str = None
        self.mask_name: str = None
        self.data_cmap: tk.StringVar = None
        self.mask_cmap: tk.StringVar = None
        self.canvas_height: tk.StringVar = None
        self.canvas_width : tk.StringVar = None

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

        self.save_model_button = tk.Button(model_frame, text="Save Model", command=self.model_save)
        self.save_model_button.grid(row=0,column=2,padx=5, pady=10)

        files_frame = tk.LabelFrame(frame, text="Files")
        files_frame.grid(row=1, column=0, padx=20, pady=10)

        self.file_label = tk.Label(files_frame, text="None")
        self.file_label.grid(row=0, column=0, padx=5, pady=0)

        load_file_button = tk.Button(files_frame, text="Load File", command=self.load_X)
        load_file_button.grid(row=1,column=0,padx=5, pady=10)

        self.mask_label = tk.Label(files_frame, text="None")
        self.mask_label.grid(row=0, column=1, padx=5, pady=0)
        
        self.load_mask_button = tk.Button(files_frame, text="Load Mask", command=self.load_y)
        self.load_mask_button.grid(row=1,column=1,padx=5, pady=10)

        operations_frame = tk.LabelFrame(frame, text="Operations")
        operations_frame.grid(row=2, column=0, padx=20, pady=10)

        self.train_button = tk.Button(operations_frame, text="Train", command=self.model_train)
        self.train_button.grid(row=0,column=0,padx=5, pady=10)

        self.predict_button = tk.Button(operations_frame, text="Predict", command=self.model_predict)
        self.predict_button.grid(row=0,column=1,padx=5, pady=10)

        plot_frame = tk.LabelFrame(frame, text="Plot")
        plot_frame.grid(row=3, column=0, padx=20, pady=10)

        self.data_cmap = tk.StringVar(value="seismic")
        self.mask_cmap = tk.StringVar(value="viridis") 

        data_cmap_label = tk.Label(plot_frame, text="Data Colormap:")
        data_cmap_label.grid(row=0, column=0, padx=5, pady=0)

        self.data_cmap_menu = tk.OptionMenu(plot_frame, self.data_cmap, *["seismic", "gray", "viridis", "plasma", "inferno", "magma", "cividis"])
        self.data_cmap_menu.grid(row=0, column=1, padx=5, pady=0)

        mask_cmap_label = tk.Label(plot_frame, text="Mask Colormap:")
        mask_cmap_label.grid(row=1, column=0, padx=5, pady=5)

        self.mask_cmap_menu = tk.OptionMenu(plot_frame, self.mask_cmap, *["seismic", "gray", "viridis", "plasma", "inferno", "magma", "cividis"])
        self.mask_cmap_menu.grid(row=1, column=1, padx=5, pady=10)

        self.canvas_height = tk.StringVar(value="500")
        self.canvas_width = tk.StringVar(value="72")

        self.height_label = tk.Label(plot_frame, text="Height:")
        self.height_label.grid(row=2, column=0, padx=5, pady=5)

        self.height_entry = tk.Entry(plot_frame, textvariable=self.canvas_height, width=5)
        self.height_entry.grid(row=2, column=1, padx=5, pady=5)

        self.width_label = tk.Label(plot_frame, text="Width:")
        self.width_label.grid(row=2, column=2, padx=5, pady=5)

        self.width_entry = tk.Entry(plot_frame, textvariable=self.canvas_width, width=5)
        self.width_entry.grid(row=2, column=3, padx=5, pady=5)

        self.plot_button = tk.Button(plot_frame, text="Plot", command=self.plot)
        self.plot_button.grid(row=0,column=2, rowspan=3, padx=10, pady=10)

        predict_frame = tk.LabelFrame(frame, text="Prediction")
        predict_frame.grid(row=4, column=0, padx=20, pady=10)

        predict_subframe_up = tk.Frame(predict_frame)
        predict_subframe_up.pack()

        self.save_cmap = tk.StringVar(value="viridis")

        save_cmap_label = tk.Label(predict_subframe_up, text="Colormap:")
        save_cmap_label.grid(row=0, column=0, padx=5, pady=0)

        self.save_cmap_menu = tk.OptionMenu(predict_subframe_up, self.save_cmap, *["seismic", "gray", "viridis", "plasma", "inferno", "magma", "cividis"])
        self.save_cmap_menu.grid(row=0, column=1, padx=5, pady=10)

        self.save_plot_button = tk.Button(predict_subframe_up, text="Save PNG", command=self.save_png)
        self.save_plot_button.grid(row=0,column=2, padx=10, pady=10)

        predict_subframe_down = tk.Frame(predict_frame)
        predict_subframe_down.pack()

        self.save_csv_button = tk.Button(predict_subframe_down, text="Save CSV", command=self.save_csv)
        self.save_csv_button.grid(row=0,column=0, padx=10, pady=10)

        self.save_npy_button = tk.Button(predict_subframe_down, text="Save NPY", command=self.save_npy)
        self.save_npy_button.grid(row=0,column=1, padx=10, pady=10)

        self.update_root_widgets()

    def update_root_widgets(self):
        if self.model is None:
            self.save_model_button.config(state=tk.DISABLED)
            self.train_button.config(state=tk.DISABLED)
        else:
            self.save_model_button.config(state=tk.NORMAL)
            self.train_button.config(state=tk.NORMAL)

        if self.X is None:
            self.load_mask_button.config(state=tk.DISABLED)
            self.plot_button.config(state=tk.DISABLED)
            self.data_cmap_menu.config(state=tk.DISABLED)
            self.mask_cmap_menu.config(state=tk.DISABLED)
            self.height_entry.config(state=tk.DISABLED)
            self.mask_cmap_menu.config(state=tk.DISABLED)
        else:
            self.load_mask_button.config(state=tk.NORMAL)
            self.plot_button.config(state=tk.NORMAL)
            self.data_cmap_menu.config(state=tk.NORMAL)
            self.mask_cmap_menu.config(state=tk.NORMAL)
            self.mask_cmap_menu.config(state=tk.NORMAL)
            self.height_entry.config(state=tk.NORMAL)

        if self.model is None or self.X is None:
            self.predict_button.config(state=tk.DISABLED)
        else:
            self.predict_button.config(state=tk.NORMAL)

        if self.y_pred is None:
            self.save_plot_button.config(state=tk.DISABLED)
            self.save_csv_button.config(state=tk.DISABLED)
            self.save_npy_button.config(state=tk.DISABLED)
            self.width_entry.config(state=tk.DISABLED)
        else:
            self.save_plot_button.config(state=tk.NORMAL)
            self.save_csv_button.config(state=tk.NORMAL)
            self.save_npy_button.config(state=tk.NORMAL)
            self.width_entry.config(state=tk.NORMAL)

    def test_function(self):
        print("OK")

    def model_load(self):
        load_model(self)
        self.update_root_widgets()

    def model_save(self):
        save_model(self)
        self.update_root_widgets()

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
                self.update_root_widgets()
            else:
                pass

        save_button = tk.Button(frame, text="Make model", command=new_model_confirmation)
        save_button.grid(row=1, columnspan=3, pady=10)

        loss_options_manager(loss_class, alpha, beta, gamma, base_weight, focal_weight)

    def model_train(self):
        window = tk.Toplevel(self.root)
        window.title("Train")

        frame = tk.Frame(window)
        frame.grid(row=0, column=0, padx=10, pady=10)

        train_frame = tk.LabelFrame(frame, text="Options")
        train_frame.grid(row=0, column=0, padx=20, pady=10)

        num_epochs = create_entry(train_frame,0,0,"Epochs: ","10", pady=5)
        batch_size = create_entry(train_frame,1,0,"Batch Size: ","128", pady= 5)
        fraction = create_slider(train_frame,2,0,"Validation Fraction:",0,0.3,train_default["fraction"],resolution=0.01,pady=5)
        seed = create_entry(train_frame,3,0,"Seed: ",str(train_default["seed"]), pady= 5)
        expand, _ = create_checkbox(train_frame,4,0,"Horizontal Expansion",train_default["expand"],pady= 5)
        augmented_ratio = create_slider(train_frame,5,0,"Augmented Ratio:",0,1,train_default["augmented_ratio"],resolution=0.1,pady=5)

        def update_seed_state(value):
            if float(value) == 0.0:
                seed.config(state=tk.DISABLED)
            else:
                seed.config(state=tk.NORMAL)  

        fraction.bind("<Motion>", lambda event: update_seed_state(fraction.get()))

        info_frame = tk.LabelFrame(frame, text="Information")
        info_frame.grid(row=1, column=0, padx=20, pady=10)

        epoch_label = tk.Label(info_frame, text="Epoch: 0/0")
        epoch_label.grid(row=2, column=0, pady=5)

        time_label = tk.Label(info_frame, text="Epoch Time: 0s")
        time_label.grid(row=3, column=0, pady=5)

        train_loss_label = tk.Label(info_frame, text="Train Loss: 0.0")
        train_loss_label.grid(row=4, column=0, pady=5)

        val_loss_label = tk.Label(info_frame, text="Val Loss: 0.0")
        val_loss_label.grid(row=5, column=0, pady=5)

        plot_frame = tk.Frame(window)
        plot_frame.grid(row=0, column=1, padx=10, pady=10)

        plot_sub_frame = tk.LabelFrame(plot_frame, text="Training Plot")
        plot_sub_frame.grid(row=0, column=0, padx=20, pady=10)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.clear()
        ax.set_axis_off() 

        canvas = FigureCanvasTkAgg(fig, master=plot_sub_frame)
        canvas.get_tk_widget().pack(padx=5, pady=5)

        if len(self.model.network.results.get("train_loss", [])) > 0:
            plot_training(self.model.network, canvas)
            train_loss = self.model.network.results['train_loss'][-1]
            val_loss = self.model.network.results['val_loss'][-1]

            train_loss_label.config(text=f"Train Loss: {train_loss:.5f}")
            if val_loss is None:
                val_loss_label.config(text=f"Val Loss: NaN")
            else:
                val_loss_label.config(text=f"Val Loss: {val_loss:.5f}")

        save_button = tk.Button(frame, text="Train model", command=lambda: train_model(
            self.model, fraction, seed, expand, augmented_ratio, batch_size, num_epochs, window, canvas,
            epoch_label, time_label, train_loss_label, val_loss_label
        ))
        save_button.grid(row=2, column=0, pady=10)

    def load_X(self):
        X, X_name = load_file()

        self.X = X
        self.well_name = X_name
        self.y = None
        self.mask_name = None
        self.y_pred = None

        self.file_label.config(text=str(X_name))
        self.mask_label.config(text=str("None"))

        self.update_root_widgets()

    def load_y(self):
        y, y_name = load_file()

        if self.X is not None and y.shape != self.X.shape:
            messagebox.showerror("Error", "The File and the Mask must have the same shape.")
            return

        self.y = y
        self.mask_name = y_name
        self.mask_label.config(text=str(y_name))

    def model_predict(self):
        try:
            prediction = self.model.predict_well(self.X, **predict_default)
            binary_mask = ((prediction) > threshold_default).astype(int)
            self.y_pred = binary_mask
            messagebox.showinfo("Success", "File processed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"The file could not be processed. Error: {e}")
            return
        self.update_root_widgets()

    def plot(self):
        #canvas_height, canvas_width = (500, 72)
        canvas_height = int(self.canvas_height.get())
        canvas_width = int(self.canvas_width.get())
        section_height = 10000

        window = tk.Toplevel(self.root)
        window.title("Plot")

        frame = tk.Frame(window)
        frame.pack(fill=tk.BOTH, expand=True)

        if self.X is None:
            messagebox.showerror("Error", "A file must be loaded.")
            window.destroy()
            return 

        data_img = np_array_to_pil(
            self.X,
            manager_default["value_range"],
            cmap_name=self.data_cmap.get(),
            output_width=canvas_width
        )

        images = {"data": data_img }

        img_width, img_height = data_img.size # (width, height)

        num_sections = img_height // section_height
        if img_height % section_height != 0:
            num_sections += 1 

        plots_frame = tk.LabelFrame(frame, text="Images")
        plots_frame.grid(row=0, column=0, padx=20, pady=20,sticky="ew")

        data_frame = tk.LabelFrame(plots_frame, text="Data")
        data_frame.grid(row=0, column=0, padx=20, pady=20)

        data_canvas = tk.Canvas(data_frame, height=canvas_height, width=canvas_width)
        data_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvases = {"data": data_canvas}

        canvas_list = [data_canvas]
        
        column = 1

        if self.y is not None:
            mask_img = np_array_to_pil(
                self.y,
                (0, 1),
                cmap_name=self.mask_cmap.get(),
                output_width=canvas_width
            )

            images["mask"] = mask_img
            
            mask_frame = tk.LabelFrame(plots_frame, text="Mask")
            mask_frame.grid(row=0, column=column, padx=20, pady=20)

            mask_canvas = tk.Canvas(mask_frame, height=canvas_height, width=canvas_width)
            mask_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            canvases["mask"] = mask_canvas

            canvas_list.append(mask_canvas)

            column += 1

        if self.y_pred is not None:
            pred_img = np_array_to_pil(
                self.y_pred,
                (0, 1),
                cmap_name=self.mask_cmap.get(),
                output_width=canvas_width
            )

            images["pred"] = pred_img

            pred_frame = tk.LabelFrame(plots_frame, text="Prediction")
            pred_frame.grid(row=0, column=column, padx=20, pady=20)

            pred_canvas = tk.Canvas(pred_frame, height=canvas_height, width=canvas_width)
            pred_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            canvases["pred"] = pred_canvas

            canvas_list.append(pred_canvas)

            column += 1
        
        scrollbar = tk.Scrollbar(plots_frame, orient=tk.VERTICAL)
        scrollbar.grid(row=0, column=column, sticky='ns', padx=20, pady=20)
        
        scrollbar.config(command=lambda *args: scrollbar_command(canvas_list, *args))

        data_canvas.config(yscrollcommand=scrollbar.set)

        if self.y is not None:
            mask_canvas.config(yscrollcommand=scrollbar.set)
        if self.y_pred is not None:
            pred_canvas.config(yscrollcommand=scrollbar.set)


        def on_mouse_wheel(event, canvas_list, scrollbar):
            current_os = platform.system()
            if current_os == "Linux":
                delta = event.delta / 120 
            else:
                delta = event.delta

            if delta > 0: 
                for canvas in canvas_list:
                    canvas.yview_scroll(-1, "units") 
            else:  
                for canvas in canvas_list:
                    canvas.yview_scroll(1, "units")

            scrollbar.set(*canvas_list[0].yview())

        for c in canvas_list:
            c.bind("<MouseWheel>", lambda event, canvas_list=canvas_list, scrollbar=scrollbar: on_mouse_wheel(event, canvas_list, scrollbar))
            c.bind("<Button-4>", lambda event, canvas_list=canvas_list, scrollbar=scrollbar: on_mouse_wheel(event, canvas_list, scrollbar))
            c.bind("<Button-5>", lambda event, canvas_list=canvas_list, scrollbar=scrollbar: on_mouse_wheel(event, canvas_list, scrollbar))
                
        section_index = 0

        def next_section():
            nonlocal section_index
            if section_index < num_sections - 1:
                section_index += 1
                update_canvas(images, canvases, section_height, section_index)
                slider.set(section_index)
                update_buttons()

        def prev_section():
            nonlocal section_index
            if section_index > 0:
                section_index -= 1
                update_canvas(images, canvases, section_height, section_index)
                slider.set(section_index)
                update_buttons()

        def update_slider(value):
            nonlocal section_index
            section_index = int(value)
            update_canvas(images, canvases, section_height, section_index)
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

        update_canvas(images, canvases, section_height, section_index)
        update_buttons()

    def save_csv(self):
        if self.y_pred is None:
            messagebox.showwarning("Error", "There is no prediction to save.")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
        )

        if save_path and not save_path.endswith(".csv"):
            save_path += ".csv"

        if save_path:
            try:
                np.savetxt(save_path, self.y_pred, delimiter=',', fmt='%d')
                messagebox.showinfo("Success", f"File saved successfully at {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

    def save_npy(self):
        if self.y_pred is None:
            messagebox.showwarning("Error", "There is no prediction to save.")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save NPY",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy")]
        )

        if save_path and not save_path.endswith(".npy"):
            save_path += ".npy"

        if save_path:
            try:
                np.save(save_path, self.y_pred)
                messagebox.showinfo("Success", f"File saved successfully at {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

    def save_png(self):
        if self.y_pred is None:
            messagebox.showwarning("Error", "There is no prediction to save.")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save PNG",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )

        if save_path and not save_path.endswith(".png"):
            save_path += ".png"

        if save_path:
            img = np_array_to_pil(
                self.y_pred,
                (0, 1),
                cmap_name=self.save_cmap.get(),)
            
            try:
                img.save(save_path)
                messagebox.showinfo("Success", f"File saved successfully at {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

    def save_plot(self):
        if self.y_pred is None:
            messagebox.showwarning("Error", "There is no prediction to save.")
            return

        model_path = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            filetypes=[("Image file", "*.png")]
        )

        data_img = np_array_to_pil(
            self.X,
            manager_default["value_range"],
            cmap_name="seismic",
        )

        mask_img = np_array_to_pil(
            self.y_pred,
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