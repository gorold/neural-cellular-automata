import tkinter as tk
from tkinter import ttk
import random
import os
import numpy as np
import torch
from PIL import Image

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import matplotlib.pyplot as plt

from utils import make_seed, to_rgb, to_rgba
from NeuralCellularAutomata import GrowingNCA
class Model(object):
    def __init__(self, *args, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        self.model = GrowingNCA(device = device)
        self.model.load_state_dict(torch.load(f'{os.getcwd()}/models/nca_0.pth'))
        self.x = make_seed((56, 56), 1, 16).to(device)

class GUI(tk.Frame):
    def __init__(self, *args, **kwargs):

        # Initialise frame
        tk.Frame.__init__(self, *args, **kwargs)
        
        # Some global variables
        self.interval_speed = tk.IntVar() # Varable for interval speed, initial speed = 1x
        self.interval_speed.set(1)
        self.mouse_clicked = False

        self.modelType = tk.StringVar() # Variable for model type
        self.modelType.set("conditional")

        self.modelTarget = tk.StringVar()
        self.modelTarget.set("Select Model Target")
        
        self.scale_labels = {}
        for i in range(1,6):
            self.scale_labels[i] = f"Speed: {i}x ({i * 2} iterations/s)" 
        
        # Add widgets
        # Radio Button to choose type of model
        tk.Label(master= root, text = 'Please choose the type of model:').grid(row=0, column=0, sticky = 'NW')        
        self.conditional_button = ttk.Radiobutton(master = root, text="Conditional Model", variable=self.modelType,
                            value="conditional", command=self.change_conditional)
        self.conditional_button.grid(row=1, column=0, sticky = 'NW')
        self.normal_button = ttk.Radiobutton(master = root, text="Normal Model", variable=self.modelType,
                                    value="normal", command=self.change_normal)
        self.normal_button.grid(row=2, column=0, sticky = 'NW')

        # Dropdown box to choose target for corresponding model
        tk.Label(master= root, text = 'Please choose the model target:').grid(row=3, column=0, sticky = 'NW')
        self.select_model_target = ttk.Combobox(master= root, justify = tk.LEFT, width = 35, textvariable = self.modelTarget, state = 'readonly')
        self.select_model_target.grid(row=4, column=0, sticky = 'SW')

        # Generate Button
        self.generate_button = tk.Button(master=root, text="Generate", command=self.generate_model)
        self.generate_button.grid(row=4, column= 1)

        # Speed Scale
        self.scale = tk.Scale(master= root, from_=1, to= 5, command=self.onScale, orient = tk.HORIZONTAL, tickinterval= 1, length=400, showvalue = 0, label = self.scale_labels[1])
        self.scale.grid(row=6, column = 0)

        # Quit Button
        self.quit_button = tk.Button(master=root, text="Quit", command=self._quit)
        self.quit_button.grid(row=6, column= 1)

        # Plot matplotlib canvas
        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.fig.add_subplot(111)
        image = Image.open(r'C:\Users\Dell\Desktop\neural-cellular-automata\data\train\emoji_u1f62f.png').convert('RGB') # Replace with instructions
        plt.imshow(image)
        plt.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=5, columnspan = 2)
        
        # Update plot every x interval, depending on the user's speed
        self.change_conditional()

    def change_conditional(self):
        self.modelTarget.set("Select Model Target")
        self.select_model_target['values'] = ['1', '2']
    def change_normal(self):
        self.modelTarget.set("Select Model Target")
        self.select_model_target['values'] = ['3', '4']    

    def onScale(self, val):
        v = int(val)
        self.interval_speed.set(v)
        self.scale.config(label=self.scale_labels[v])

    def update_plot(self):
        self.fig.clear()
        self.nca.model.eval()
        with torch.no_grad():
            self.nca.x = self.nca.model(self.nca.x)
        out = np.transpose(to_rgb(self.nca.x).detach().cpu().numpy()[0], (1, 2, 0))
        plt.imshow((out * 255).astype(np.uint8))
        plt.axis('off')
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=5, columnspan=3)
        self.after(int(100/self.interval_speed.get()), self.update_plot) # how often you want to refresh

    def generate_model(self):
        model_type = self.modelType.get()
        model_target = self.modelTarget.get()
        
        self.nca = Model()
        self.canvas.callbacks.connect('button_press_event', self.on_click)
        self.canvas.callbacks.connect('motion_notify_event', self.drag)
        self.canvas.callbacks.connect('button_release_event', self.release)
        self.canvas.draw()
        self.update_plot()

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

    def _quit(self):
        root.quit()
        root.destroy()

def on_closing():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.quit()
        root.destroy()

if __name__ == "__main__":
    import time
    start = time.time()
    
    root = tk.Tk()
    GUI(root)
    root.geometry("500x650")
    root.resizable(width=False, height=False)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    end = time.time()
    print(end - start)
    root.mainloop()
