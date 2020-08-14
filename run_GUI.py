import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from utils import make_seed, to_rgb, load_emoji_dict, pad_target
from NeuralCellularAutomata import GrowingNCA, ConditionalNCA

global device 

# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     torch.cuda.set_device(device)
# else:
#     device = torch.device('cpu')

device = torch.device('cpu')
class Model(object):
    def __init__(self, conditional = False, target = None, enable_vae = False):
        if conditional:
            if enable_vae:
                self.model = ConditionalNCA(device = device, enable_vae = True)
                self.model.load_state_dict(torch.load(f'{os.getcwd()}/vNCA_Models/evolution.pth'))
            else:
                self.model = ConditionalNCA(device = device)
                self.model.load_state_dict(torch.load(f'{os.getcwd()}/models/nca_1.pth'))
        else:
            self.model = GrowingNCA(device = device)
            self.model.load_state_dict(torch.load(f'{os.getcwd()}/models/nca_{target}.pth'))
        self.x = make_seed((56, 56), 1, 16).to(device)

class GUI(tk.Frame):
    def __init__(self, *args, **kwargs):

        # Initialise frame
        tk.Frame.__init__(self, *args, **kwargs)
        
        # Some necessary variables
        self.interval_speed = tk.IntVar() # Varable for interval speed, initial speed = 1x
        self.interval_speed.set(1)
        self.mouse_clicked = False

        self.vae_slider_value = tk.IntVar()
        self.vae_slider_value.set(0)

        self.modelType = tk.StringVar() # Variable for model type
        self.modelType.set("conditional")

        self.conditional_target = {k: pad_target(v) for k, v in load_emoji_dict('data/train').items()}
        self.conditional_target_names = {'Shower head' : 'emoji_u1f6bf.png', 
                                         'Ok sign': 'emoji_u1f44c.png', 
                                         'Soldier' : 'emoji_u1f482_200d_2642.png', 
                                         'Frying Pan':'emoji_u1f373.png', 
                                         'Eggplant' : 'emoji_u1f346.png', 
                                         'Curse Word': 'emoji_u1f92c.png', 
                                         'A little bit' : 'emoji_u1f90f_1f3fb.png', 
                                         'Shocked Face' : 'emoji_u1f62f.png'}

        self.vae_target = {k: pad_target(v) for k, v in load_emoji_dict('data/train_vae').items()}
        self.vae_target_names = {'Germ' : 'emoji_u1f9a0.png', 
                                 'Caterpillar': 'emoji_u1f41b.png', 
                                 'Gorilla' : 'emoji_u1f98d.png', 
                                 'Man':'emoji_u1f468_1f3fb_200d_1f9b1.png'}

        self.modelTarget = tk.StringVar()
        self.modelTarget.set("Select Model Target")

        self.modelTarget_two = tk.StringVar()
        self.modelTarget_two.set("Select Second Emoji")
        
        self.current_modelType = "conditional"
        self.current_modelTarget = "Select Model Target"

        self.scale_labels = {}
        for i in range(1,6):
            self.scale_labels[i] = f"Speed: {i}x ({i} iterations/s)" 
        
        self.vae_scale_labels = {}
        for i in range(101):
            self.vae_scale_labels[i] = f"{i}% first emoji, {100-i}% second emoji" 
        
        # Add widgets

        # Radio Button to choose type of model
        tk.Label(master= root, text = 'Please choose the type of model:').grid(row=0, column=0, sticky = 'NW')        
        self.conditional_button = ttk.Radiobutton(master = root, text="Conditional Model", variable=self.modelType,
                            value="conditional", command=self.change_conditional)
        self.conditional_button.grid(row=1, column=0, sticky = 'NW')
        self.normal_button = ttk.Radiobutton(master = root, text="Normal Model", variable=self.modelType,
                                    value="normal", command=self.change_normal)
        self.normal_button.grid(row=2, column=0, sticky = 'NW')
        self.vae_button = ttk.Radiobutton(master = root, text="VAE Model", variable=self.modelType,
                                    value="vae", command=self.change_vae)
        self.vae_button.grid(row=3, column=0, sticky = 'NW')

        # Dropdown box to choose target for corresponding model
        self.select_model_target_label = tk.Label(master= root, text = 'Please choose the model target:')
        self.select_model_target_label.grid(row=4, column=0, sticky = 'NW')

        self.select_model_target = ttk.Combobox(master= root, justify = tk.LEFT, width = 35, textvariable = self.modelTarget, state = 'readonly')
        self.select_model_target.grid(row=5, column=0, sticky = 'SW')
        
        self.select_model_target_label_two = tk.Label(master= root, text = 'Please choose the second emoji:')
        self.select_model_target_label_two.grid(row=6, column=0, sticky = 'NW')

        self.select_emoji_two = ttk.Combobox(master= root, justify = tk.LEFT, width = 35, textvariable = self.modelTarget_two, state = 'readonly')
        self.select_emoji_two.grid(row=7, column=0, sticky = 'SW')
        
        self.vae_scale = tk.Scale(master= root, from_=0, to= 100, command=self.onScale_vae, orient = tk.HORIZONTAL, tickinterval= 100, length=400, showvalue = 0, label = self.vae_scale_labels[0])
        self.vae_scale.grid(row=8, column = 0)

        # Generate Button
        self.generate_button = tk.Button(master=root, text="Generate", command=self.generate_model)
        self.generate_button.grid(row=9, column= 0, sticky = 'W')

        # Refresh Button
        self.refresh_button = tk.Button(master=root, text = 'Refresh', command = self.refresh_x, state=tk.DISABLED)
        self.refresh_button.grid(row = 9, column = 1, sticky = 'W')

        # Speed Scale
        self.scale = tk.Scale(master= root, from_=1, to= 5, command=self.onScale, orient = tk.HORIZONTAL, tickinterval= 1, length=400, showvalue = 0, label = self.scale_labels[1])
        self.scale.grid(row=11, column = 0)

        # Quit Button
        self.quit_button = tk.Button(master=root, text="Quit", command=self._quit)
        self.quit_button.grid(row=11, column= 1)

        # Plot initial README canvas and set up options
        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.fig.add_subplot(111)
        image = Image.open(f'{os.getcwd()}/assets/readme.png').convert('RGB') # Replace with instructions
        plt.imshow(image)
        plt.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=10, columnspan = 2)
        self.change_conditional()

    def change_conditional(self):
        self.select_model_target.grid(row=5, column=0, sticky = 'SW')
        self.select_model_target_label['text'] = 'Please choose the model target:'
        self.select_model_target_label_two.grid_remove()
        self.select_emoji_two.grid_remove()
        self.vae_scale.grid_remove()
        # Changes the dropdown box options to show the relevant options
        self.modelTarget.set("Select Model Target")
        self.select_model_target['values'] = list(self.conditional_target_names.keys())

    def change_normal(self):
        self.select_model_target.grid(row=5, column=0, sticky = 'SW')
        self.select_model_target_label['text'] = 'Please choose the model target:'
        self.select_model_target_label_two.grid_remove()
        self.select_emoji_two.grid_remove()
        self.vae_scale.grid_remove()
        # Changes the dropdown box options to show the relevant options
        self.modelTarget.set("Select Model Target")
        self.select_model_target['values'] = ['Smiley Face', 'Lizard', 'Explosion']

    def change_vae(self):
        # Changes the dropdown box options to show the relevant options
        self.modelTarget.set("Select First Emoji")
        self.select_model_target_label['text'] = 'Please select the first emoji:'
        self.select_model_target['values'] = list(self.vae_target_names.keys())
        
        self.modelTarget_two.set('Select Second Emoji')
        self.select_model_target_label_two.grid(row=6, column=0, sticky = 'NW')
        
        self.select_emoji_two.grid(row=7, column=0, sticky = 'SW')
        self.select_emoji_two['values'] = list(self.vae_target_names.keys())
        self.vae_scale.grid(row=8, column = 0)

    def onScale(self, val):
        # Set speed of scale
        v = int(val)
        self.interval_speed.set(v)
        self.scale.config(label=self.scale_labels[v])

    def onScale_vae(self, val):
        v = int(val)
        self.vae_slider_value.set(v)
        self.vae_scale.config(label=self.vae_scale_labels[v])

    def update_plot(self):
        # Updates plot by passing through the current image (x) into the NCA model
        self.fig.clear()
        self.nca.model.eval()
        with torch.no_grad():
            if self.current_modelType == 'conditional':
                self.nca.x = self.nca.model(self.nca.x, None, self.conditional_encoding[self.conditional_target_names[self.current_modelTarget]])
                out = np.transpose(to_rgb(self.nca.x).detach().cpu().numpy()[0], (1, 2, 0))
            elif self.current_modelType == 'vae':
                self.nca.x = self.nca.model(self.nca.x, None, self.interpolated_encoding)
                out = np.transpose(to_rgb(self.nca.x).detach().cpu().numpy()[0], (1, 2, 0))
            else:
                self.nca.x = self.nca.model(self.nca.x)
                out = np.transpose(to_rgb(self.nca.x).detach().cpu().numpy()[0], (1, 2, 0))
        plt.imshow((out * 255).astype(np.uint8))
        plt.axis('off')
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=10, columnspan=3)
        self.after(int(1000/self.interval_speed.get()), self.update_plot) # how often you want to refresh

    def generate_model(self):
        # Function called when "generate" button is pressed

        # Certain values that we want to save
        model_type = self.modelType.get()
        model_target = self.modelTarget.get()

        self.current_modelType = model_type
        self.current_modelTarget = model_target

        # Will only initialise if we are using VAE
        try:
            model_target_two = self.modelTarget_two.get()
            vae_slider_val = self.vae_slider_value.get()
            self.current_modelTarget_two = model_target_two
            self.current_vae_slider_val = vae_slider_val
        except:
            pass

        # If invalid selection, show error message, prompt user to choose correct option and regenerate
        if model_target == 'Select Model Target':
            self.fig.clear()
            image = Image.open(f'{os.getcwd()}/assets/error.png').convert('RGB')
            plt.axis('off')
            plt.imshow(image)
            self.canvas.draw()
            self.refresh_button['state'] = tk.DISABLED

        # If correct option, present loading screen while model is loading, subsequently, show model.
        else:
            self.fig.clear()
            image = Image.open(f'{os.getcwd()}/assets/loading.png').convert('RGB')
            plt.axis('off')
            plt.imshow(image)
            self.canvas.draw()
            
            if model_type == 'conditional':
                self.nca = Model(conditional = True)
                self.conditional_encoding = {}
                for k, v in self.conditional_target.items():
                    self.conditional_encoding[k] = self.nca.model.get_encoding(v.unsqueeze(0).to(device))
            elif model_type == 'vae':
                self.nca = Model(conditional = True, enable_vae=True)
                first_emoji_encoding = self.vae_target[self.vae_target_names[self.current_modelTarget]]
                second_emoji_encoding = self.vae_target[self.vae_target_names[self.current_modelTarget_two]]
                self.interpolated_encoding = self.nca.model.interpolate(torch.stack([first_emoji_encoding, second_emoji_encoding]).to(device), r = self.current_vae_slider_val/100.0)
                
            else:
                normal_model_target = {'Smiley Face': '2', 'Lizard' : '0', 'Explosion' : '3'}
                self.nca = Model(target = normal_model_target[model_target])

            self.refresh_button['state'] = tk.NORMAL
            self.canvas.callbacks.connect('button_press_event', self.on_click)
            self.canvas.callbacks.connect('motion_notify_event', self.drag)
            self.canvas.callbacks.connect('button_release_event', self.release)
            self.canvas.draw()
            self.update_plot()

    def refresh_x(self):
        # Tries to refresh the x, otherwise show an error message
        try:
            self.nca.x = make_seed((56, 56), 1, 16).to(device)
        except:
            self.fig.clear()
            image = Image.open(f'{os.getcwd()}/assets/error.png').convert('RGB')
            plt.axis('off')
            plt.imshow(image)
            self.canvas.draw()
            self.refresh_button['state'] = tk.DISABLED

    # Some functions to white out a small circle of the image when clicked on
    def on_click(self, event):
        if event.inaxes is not None:
            self.mouse_clicked = True
            y,x = np.ogrid[-int(event.ydata):56-int(event.ydata), -int(event.xdata):56-int(event.xdata)]
            mask = x*x + y*y <= 5*5
            self.nca.x[:, :, mask] = 0

    def drag(self, event):
        if event.inaxes is not None and self.mouse_clicked == True:
            y,x = np.ogrid[-int(event.ydata):56-int(event.ydata), -int(event.xdata):56-int(event.xdata)]
            mask = x*x + y*y <= 5*5
            self.nca.x[:, :, mask] = 0

    def release(self,event):
        self.mouse_clicked = False

    # Fully stops function
    def _quit(self):
        root.quit()
        root.destroy()

# Fully stops function
def on_closing():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.quit()
        root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    GUI(root)
    root.geometry("500x825")
    root.wm_title("Neural Cellular Automata")
    root.iconbitmap(f"{os.getcwd()}/assets/emoji_u1f44c.ico")
    root.resizable(width=False, height=False)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()