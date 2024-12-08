
import tkinter as tk
from tkinter import filedialog, messagebox

from src.models.manager import ModelManager
from src.models.architectures.networks.unet import Unet
from src.models.dataset.transformations import random_transformation
from src.utils.files import load_config, load_arrays_from_folders, npy_file_to_dict

#-----------------------------------------------------------------------------------------------------------------------------------------------------

config = load_config("./src/config/config.yaml")
network_default = config["networks"]["unet"]
manager_default = config["manager"]
train_default = config["train_validation"]
predict_default = config["predict"]
threshold_default = config["threshold"]

class Gui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CorroSeg")
        self.root.geometry("400x300")
        
        self.model = None
        self.file = None
        self.prediction = None

        self.create_menu_bar()
        self.create_widgets()

    def create_menu_bar(self):
        # Crear la barra de menú
        menubar = tk.Menu(self.root)
        
        # Crear el menú "Archivo"
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open model", command=self.load_model)
        file_menu.add_command(label="Save model", command=self.save_model, state=tk.NORMAL)  # tk.DISABLED
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Crear el menú "Model"
        actions_menu = tk.Menu(menubar, tearoff=0)
        actions_menu.add_command(label="New", command=self.model_new)
        actions_menu.add_command(label="Train", command=self.model_train, state=tk.NORMAL)  # Desactivado inicialmente
        actions_menu.add_command(label="Predict", command=self.model_predict, state=tk.NORMAL)  # Desactivado inicialmente
        
        # Agregar menús a la barra de menú
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Model", menu=actions_menu)
        
        # Configurar la barra de menú
        self.root.config(menu=menubar)

    def create_widgets(self):
        self.load_button = tk.Button(self.root, text="Load Model", command=self.load_model)
        self.load_button.pack(pady=10)

    def test_function(self):
        print("OK")

    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Load model",
            filetypes=[("Model file", "*.pt;*.pth")])
        
        if model_path:
            try:
                model = ModelManager.load_model(Unet, model_path)
                self.model = model
                self.root.config(menu=self.enable_save_model_option())
                messagebox.showinfo("Éxito", "Modelo cargado correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"The model could not be loaded. Error: {e}")

    def save_model(self):
        if self.model is None:
            messagebox.showwarning("Error", "There is no model to save.")
            return

        model_path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".pt",
            filetypes=[("Model file", "*.pt;*.pth")]
        )

        if model_path:
            try:
                self.model.save_model(model_path)
                messagebox.showinfo("Éxito", "Modelo guardado correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"The model could not be saved. Error: {e}")

    def enable_save_model_option(self):
        """ Habilitar la opción de guardar el modelo en el menú """
        menubar = self.root.config()['menu']
        file_menu = menubar.entrycget(0, "menu")  # Obtiene el submenú de "Archivo"
        file_menu.entryconfig(1, state=tk.NORMAL)  # Habilita "Guardar Modelo"
        return menubar
    
    def model_new(self):
        window = tk.Toplevel(self.root)
        window.title("New Model")
        window.geometry("1200x550")

        window.grid_columnconfigure(0, weight=1, minsize=100)

        input_channels = self.create_slider(window,1,0,"Input Channels:",1,16,network_default["input_channels"],pady= 5)
        output_channels = self.create_slider(window,2,0,"Output Channels:",1,16,network_default["output_channels"],pady= 5)
        base_channels = self.create_slider(window,3,0,"Base Channels:",1,64,network_default["base_channels"],pady= 5)
        num_layers = self.create_slider(window,4,0,"Number of Layers:",1,5,network_default["num_layers"],pady=5)
        output_activation, _ = self.create_option_menu(window,5,0,"Output Activation:",["relu_clipped","sigmoid", "softplus_clipped", "tanh_normalized", ""],network_default["output_activation"],pady= 5)
        attention_gates, _ = self.create_checkbox(window,6,0,"Attention Gates",network_default["attention_gates"],pady= 5)
        skip_connections, _ = self.create_checkbox(window,7,0,"Skip Connections",True,pady= 5)
    
        window.grid_columnconfigure(1, weight=1, minsize=100)

        tk.Label(window, text="Encoder").grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        e_activation, _ = self.create_option_menu(window,1,1,"Activation:",["ReLU", "LeakyReLU", "None"],network_default["encoder_kwargs"]["activation"],pady= 5)
        e_dropout_prob = self.create_slider(window,2,1,"Dropout:",0,1,network_default["encoder_kwargs"]["dropout_prob"],resolution=0.05,pady=5)
        e_num_recurrences = self.create_slider(window,3,1,"Number of recurrences:",0,5,network_default["encoder_kwargs"]["num_recurrences"],pady=5)
        e_residual, _ = self.create_checkbox(window,4,1,"Residual conection",True,pady= 5)
        e_cbam, _ = self.create_checkbox(window,5,1,"CBAM",network_default["encoder_kwargs"]["cbam"],pady= 5)
        e_cbam_reduction = self.create_slider(window,6,1,"CBAM Reduction:",1,32,network_default["encoder_kwargs"]["cbam_reduction"], pady= 5)
        e_cbam_activation, e_cbam_activation_menu = self.create_option_menu(window,7,1,"CBAM Activation:",["ReLU", "LeakyReLU", "None"],network_default["encoder_kwargs"]["cbam_activation"],pady= 5)

        toggle_e_cbam_dependences =  lambda: self.toggle_widgets_by_var(e_cbam, [e_cbam_reduction, e_cbam_activation_menu])
        e_cbam.trace("w", lambda *args: toggle_e_cbam_dependences())

        window.grid_columnconfigure(2, weight=1, minsize=100)

        tk.Label(window, text="Bottleneck").grid(row=0, column=2, padx=10, pady=5, sticky="nsew")
        b_activation, _ = self.create_option_menu(window,1,2,"Activation:",["ReLU", "LeakyReLU", "None"],network_default["bottleneck_kwargs"]["activation"],pady= 5)
        b_dropout_prob = self.create_slider(window,2,2,"Dropout:",0,1,network_default["bottleneck_kwargs"]["dropout_prob"],resolution=0.05,pady=5)
        b_num_recurrences = self.create_slider(window,3,2,"Number of recurrences:",0,5,network_default["bottleneck_kwargs"]["num_recurrences"],pady=5)
        b_residual, _ = self.create_checkbox(window,4,2,"Residual conection",True,pady= 5)
        b_cbam, _ = self.create_checkbox(window,5,2,"CBAM",network_default["bottleneck_kwargs"]["cbam"],pady= 5)
        b_cbam_reduction = self.create_slider(window,6,2,"CBAM Reduction:",1,32,network_default["bottleneck_kwargs"]["cbam_reduction"], pady= 5)
        b_cbam_activation, b_cbam_activation_menu = self.create_option_menu(window,7,2,"CBAM Activation:",["ReLU", "LeakyReLU", "None"],network_default["bottleneck_kwargs"]["cbam_activation"],pady= 5)

        toggle_b_cbam_dependences =  lambda: self.toggle_widgets_by_var(b_cbam, [b_cbam_reduction, b_cbam_activation_menu])
        b_cbam.trace("w", lambda *args: toggle_b_cbam_dependences())

        window.grid_columnconfigure(3, weight=1, minsize=100)

        tk.Label(window, text="Decoder").grid(row=0, column=3, padx=10, pady=5, sticky="nsew")
        d_activation, _ = self.create_option_menu(window,1,3,"Activation:",["ReLU", "LeakyReLU", "None"],network_default["decoder_kwargs"]["activation"],pady= 5)
        d_dropout_prob = self.create_slider(window,2,3,"Dropout:",0,1,network_default["decoder_kwargs"]["dropout_prob"],resolution=0.05,pady=5)
        d_num_recurrences = self.create_slider(window,3,3,"Number of recurrences:",0,5,network_default["decoder_kwargs"]["num_recurrences"],pady=5)
        d_residual, _ = self.create_checkbox(window,4,3,"Residual conection",True,pady= 5)
        d_cbam, _ = self.create_checkbox(window,5,3,"CBAM",network_default["decoder_kwargs"]["cbam"],pady= 5)
        d_cbam_reduction = self.create_slider(window,6,3,"CBAM Reduction:",1,32,network_default["decoder_kwargs"]["cbam_reduction"], pady= 5)
        d_cbam_activation, d_cbam_activation_menu = self.create_option_menu(window,7,3,"CBAM Activation:",["ReLU", "LeakyReLU", "None"],network_default["decoder_kwargs"]["cbam_activation"],pady= 5)

        toggle_d_cbam_dependences =  lambda: self.toggle_widgets_by_var(d_cbam, [d_cbam_reduction, d_cbam_activation_menu])
        d_cbam.trace("w", lambda *args: toggle_d_cbam_dependences())

        window.grid_columnconfigure(4, weight=1, minsize=100)

        tk.Label(window, text="Skip Connections").grid(row=0, column=4, padx=10, pady=5, sticky="nsew")
        s_activation, s_activation_menu = self.create_option_menu(window,1,4,"Activation:",["ReLU", "LeakyReLU", "None"],network_default["skip_connections_kwargs"]["activation"],pady= 5)
        s_dropout_prob = self.create_slider(window,2,4,"Dropout:",0,1,network_default["skip_connections_kwargs"]["dropout_prob"],resolution=0.05,pady=5)
        s_num_recurrences = self.create_slider(window,3,4,"Number of recurrences:",0,5,network_default["skip_connections_kwargs"]["num_recurrences"],pady=5)
        s_residual, s_residual_check = self.create_checkbox(window,4,4,"Residual conection",True,pady= 5)
        s_cbam, s_cbam_check = self.create_checkbox(window,5,4,"CBAM",network_default["skip_connections_kwargs"]["cbam"],pady= 5)
        s_cbam_reduction = self.create_slider(window,6,4,"CBAM Reduction:",1,32,network_default["skip_connections_kwargs"]["cbam_reduction"], pady= 5)
        s_cbam_activation, s_cbam_activation_menu = self.create_option_menu(window,7,4,"CBAM Activation:",["ReLU", "LeakyReLU", "None"],network_default["skip_connections_kwargs"]["cbam_activation"],pady= 5)

        toggle_s_cbam_dependences =  lambda: self.toggle_widgets_by_var(s_cbam, [s_cbam_reduction, s_cbam_activation_menu])
        s_cbam.trace("w", lambda *args: toggle_s_cbam_dependences())

        toggle_skip_dependences =  lambda: self.toggle_widgets_by_var(skip_connections,
                                                               [s_activation_menu,
                                                                s_dropout_prob,
                                                                s_num_recurrences,
                                                                s_residual_check,
                                                                s_cbam_check,
                                                                s_cbam_reduction,
                                                                s_cbam_activation_menu])
        skip_connections.trace("w", lambda *args: toggle_skip_dependences())

        window.grid_columnconfigure(5, weight=1, minsize=100)

        loss_class, loss_class_menu = self.create_option_menu(window,1,5,"Loss:",
                                                              ["DICELoss","FocalLoss", "IoULoss", "IoUFocalLoss", "TverskyLoss"],
                                                              manager_default["loss_class"],pady= 5)
        alpha = self.create_entry(window,2,5,"alpha","0.5", pady= 5)
        beta = self.create_entry(window,3,5,"beta","0.5", pady= 5)
        gamma = self.create_entry(window,4,5,"gamma","2", pady= 5)
        base_weight = self.create_entry(window,5,5,"base weight","1", pady= 5)
        focal_weight = self.create_entry(window,6,5,"focal weight","10", pady= 5)
        learning_rate = self.create_entry(window,7,5,"learning rate",str(manager_default["optimizer_params"]["lr"]), pady= 5)
        weight_decay = self.create_entry(window,8,5,"weight decay",str(manager_default["optimizer_params"]["weight_decay"]), pady= 5)

        def loss_options_manager():
            state = loss_class.get()
            if state == "DICELoss":
                self.toggle_multiple_widgets([],[alpha, beta, gamma, base_weight, focal_weight])
            elif state == "FocalLoss":
                self.toggle_multiple_widgets([alpha, gamma],[beta, base_weight, focal_weight])
            elif state == "IoULoss":
                self.toggle_multiple_widgets([],[alpha, beta, gamma, base_weight, focal_weight])
            elif state == "IoUFocalLoss":
                self.toggle_multiple_widgets([alpha, gamma, base_weight, focal_weight],[beta])
            elif state == "TverskyLoss":
                self.toggle_multiple_widgets([alpha, beta],[gamma, base_weight, focal_weight])

        loss_class.trace("w", lambda *args: loss_options_manager())

        def get_block_type(num_recurrences, residual, state=True):
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

        def build_network_dictionary():
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
            return dictionary
        
        def get_network_dictionary():
            dictionary = build_network_dictionary()

            if dictionary["encoder_block_type"] not in ["RRConvBlock", "RecConvBlock"]:
                if "num_recurrences" in dictionary["encoder_kwargs"]:
                    del dictionary["encoder_kwargs"]["num_recurrences"]

            if dictionary["bottleneck_block_type"] not in ["RRConvBlock", "RecConvBlock"]:
                if "num_recurrences" in dictionary["bottleneck_kwargs"]:
                    del dictionary["bottleneck_kwargs"]["num_recurrences"]
            
            if dictionary["decoder_block_type"] not in ["RRConvBlock", "RecConvBlock"]:
                if "num_recurrences" in dictionary["decoder_kwargs"]:
                    del dictionary["decoder_kwargs"]["num_recurrences"]
            
            if dictionary.get("skip_connections_block_type") == "None":
                dictionary["skip_connections_kwargs"] = {}
            elif dictionary["skip_connections_block_type"] not in ["RRConvBlock", "RecConvBlock"]:
                if "num_recurrences" in dictionary["skip_connections_kwargs"]:
                    del dictionary["skip_connections_kwargs"]["num_recurrences"]

            return dictionary
        
        def build_manager_dictionary():
            dictionary = {
                "dataset_class": manager_default["dataset_class"],
                "loss_class": str(loss_class.get()),
                "optimizer_class": manager_default["optimizer_class"],
                "scaler_type": manager_default["scaler_type"],
                "input_shape": manager_default["input_shape"],
                "placeholders": manager_default["placeholders"],
                "value_range": manager_default["value_range"],
                "default_value": manager_default["default_value"],
                "loss_params":{
                    "alpha": float(alpha.get()),
                    "beta": float(beta.get()),
                    "gamma": float(gamma.get()),
                    "base_weight": float(base_weight.get()),
                    "focal_weight": float(focal_weight.get()),
                },
                "optimizer_params":{
                    "lr": float(learning_rate.get()),
                    "weight_decay": float(weight_decay.get())
                },
                "scaler_params": manager_default["scaler_params"],
            }
            
            return dictionary
        
        def get_manager_dictionary():
            dictionary = build_manager_dictionary()

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
                for i in [gamma, base_weight, focal_weight]:
                    if i in dictionary["loss_params"]:
                        del dictionary["loss_params"][i]

            return dictionary

        def make_model():
            network_dict = get_network_dictionary()
            manager_dict = get_manager_dictionary()

            network = Unet.from_dict(network_dict)
            model = ModelManager.from_dict(network, manager_dict)

            return model
        
        def new_model_confirmation():
            answer = messagebox.askyesno("Confirmation", "Are you sure?")
            if answer:
                self.model = make_model()
                window.destroy()
            else:
                pass

        save_button = tk.Button(window, text="Make model", command=new_model_confirmation)
        save_button.grid(row=9, columnspan=6, pady=5)
    
    def model_train(self):
        window = tk.Toplevel(self.root)
        window.title("Train")
        window.geometry("200x350")

        window.grid_columnconfigure(0, weight=1, minsize=100)

        tk.Label(window, text="Training Config").grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        num_epochs = self.create_entry(window,1,0,"Epochs:","10", pady= 5)
        batch_size = self.create_entry(window,2,0,"Batch Size:","128", pady= 5)
        fraction = self.create_slider(window,3,0,"Validation Fraction:",0,0.3,train_default["fraction"],resolution=0.01,pady=5)
        seed = self.create_entry(window,4,0,"Seed:",str(train_default["seed"]), pady= 5)
        expand, expand_check = self.create_checkbox(window,5,0,"Horizontal Expansion",train_default["expand"],pady= 5)
        augmented_ratio = self.create_slider(window,6,0,"Augmented Ratio:",0,1,train_default["augmented_ratio"],resolution=0.1,pady=5)

        def get_train_dictionary():
            dictionary = {
                "height_stride": train_default["height_stride"],
                "width_stride": train_default["width_stride"],
                "fraction": float(fraction.get()),
                "seed": int(seed.get()),
                "expand": expand.get(),
                "augmented_ratio": float(augmented_ratio.get()),
            }

            return dictionary
        
        def train_model():
            train_dictionary = get_train_dictionary()

            X, y = load_arrays_from_folders(config["folders"]["train"]) # ERROR
            X_train, y_train, X_val, y_val = self.model.setup_train_val_data(X, y, **train_dictionary)

            validate = True if len(X_val) > 0 else False

            train_dataset = self.model.build_dataset(X_train, y_train, random_transformation, config["random_transformation"])
            if validate:
                val_dataset = self.model.build_dataset(X_val,y_val)

            train_loader = self.model.build_dataloader(train_dataset, int(batch_size.get()), True)
            if validate:
                val_loader = self.model.build_dataloader(val_dataset, int(batch_size.get()), True)
        
            if validate:
                self.model.train(train_loader, val_loader, num_epochs=int(num_epochs.get()))
            else:
                self.model.train(train_loader, num_epochs=int(num_epochs.get()))
        
        save_button = tk.Button(window, text="Train model", command=train_model)
        save_button.grid(row=9, columnspan=6, pady=5)

    def model_predict(self):
        window = tk.Toplevel(self.root)
        window.title("Predict")
        window.geometry("200x150")

        key_label_var = tk.StringVar()
        if self.file is None:
            key_label_var.set("No file loaded")

        # Crear el Label debajo del botón que mostrará el nombre de la clave
        key_label = tk.Label(window, textvariable=key_label_var)
        key_label.pack(pady=10)

        def load_file(self):
            file_path = filedialog.askopenfilename(
                title="Load file",
                filetypes=[("Data file", "*.npy")])
            
            if file_path:
                try:
                    self.file = npy_file_to_dict(file_path)
                    key_label_var.set(str(list(self.file.keys())[0]))
                    messagebox.showinfo("Exit", "File loaded correctly.")
                except Exception as e:
                    messagebox.showerror("Error", f"The File could not be loaded. Error: {e}")
                    return

        load_button = tk.Button(window, text="Load File", command=lambda: load_file(self))
        load_button.pack(pady=10)

        def predict_model(self):
            try:
                prediction = self.model.predict_well(list(self.file.values())[0], **predict_default)
                self.prediction = ((prediction) > threshold_default).astype(int)
                messagebox.showinfo("Success", "File processed successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"The file could not be processed. Error: {e}")
                return
        
        predict_button = tk.Button(window, text="Predict", command=lambda: predict_model(self))
        predict_button.pack(pady=10)

        
    @staticmethod
    def create_entry(parent, row, column, label_text, default_value, validate_input=None, width=5, padx=0, pady=0):
        """Crea un campo de entrada con validación opcional"""
        frame = tk.Frame(parent)
        tk.Label(frame, text=label_text).pack(side="left")

        if validate_input is None:
            entry = tk.Entry(frame, width=width)
        else:
            entry = tk.Entry(frame, validate="key", validatecommand=(parent.register(validate_input), '%P'), width=width)
        
        entry.insert(0, default_value)
        entry.pack(side="left")
        frame.grid(row=row, column=column, padx=padx, pady=pady)
        
        return entry
    
    @staticmethod
    def create_slider(parent, row, column, label_text, from_, to_, default_value, resolution=1, padx=0, pady=0):
        """Crea un control deslizante"""
        if not (from_ <= default_value <= to_):
            raise ValueError(f"default_value '{default_value}' must be in range [{from_}, {to_}].")
        
        frame = tk.Frame(parent)
        tk.Label(frame, text=label_text).pack(side="top")
        slider = tk.Scale(frame, from_=from_, to=to_, orient="horizontal", resolution=resolution)
        slider.set(default_value)
        slider.pack(side="top")
        frame.grid(row=row, column=column, padx=padx, pady=pady)
        return slider
    
    @staticmethod
    def create_checkbox(parent, row, column, label_text, default_value, padx=0, pady=0):
        """Crea una casilla de verificación"""
        if not isinstance(default_value, bool):
            raise ValueError(f"default_value '{default_value}' must be True or False.")
        
        frame = tk.Frame(parent)
        var = tk.BooleanVar(value=default_value)
        checkbox = tk.Checkbutton(frame, text=label_text, variable=var)
        checkbox.pack(side="top")
        frame.grid(row=row, column=column, padx=padx, pady=pady)
        return var, checkbox
    
    @staticmethod
    def create_option_menu(parent, row, column, label_text, options, default_value, width=10, padx=0, pady=0):
        """Crea un menú desplegable con un valor predeterminado válido y tamaño personalizable"""
        if default_value not in options:
            raise ValueError(f"default_value '{default_value}' is not in options: {options}.")
        
        var = tk.StringVar(value=default_value)
        frame = tk.Frame(parent)
        tk.Label(frame, text=label_text).pack(side="left")
        menu = tk.OptionMenu(frame, var, *options)
        menu.config(width=width) 
        menu.pack(side="left")
        frame.grid(row=row, column=column, padx=padx, pady=pady)
        return var, menu
    
    @staticmethod
    def toggle_widgets_by_var(control_var, widgets):
            state = tk.NORMAL if control_var.get() else tk.DISABLED
            for widget in widgets:
                widget.config(state=state)

    @staticmethod
    def toggle_multiple_widgets(widgets_enabled, widgets_disabled):
        for widget in widgets_enabled:
            widget.config(state=tk.NORMAL)

        for widget in widgets_disabled:
            widget.config(state=tk.DISABLED)

#-----------------------------------------------------------------------------------------------------------------------------------------------------