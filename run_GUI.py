import tkinter as tk
import random
import os
import numpy as np
import torch

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import matplotlib.pyplot as plt

from utils import make_seed, to_rgba
from NeuralCellularAutomata import GrowingNCA

class Model(object):
    def __init__(self, *args, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GrowingNCA(device = device)
        self.model.load_state_dict(torch.load(r'C:\Users\Dell\Desktop\neural-cellular-automata\models\nca_0.pth'))
        self.x = make_seed((56, 56), 1, 16).to(device)

    def get_img1_weight(self): return random.uniform(0,1)

class GUI(tk.Frame):
    def __init__(self, *args, **kwargs):

        # Initialise frame
        tk.Frame.__init__(self, *args, **kwargs)
        
        # Some global variables
        self.interval_speed = tk.IntVar() # Varable for interval speed, initial speed = 1x
        self.interval_speed.set(1)
        self.mouse_clicked = False

        self.scale_labels = {}
        for i in range(1,6):
            self.scale_labels[i] = f"Speed: {i}x ({i * 2} iterations/s)" 
        
        # Add widgets
        self.scale = tk.Scale(master= root, from_=1, to= 5, command=self.onScale, orient = tk.HORIZONTAL, tickinterval= 1, length=400, showvalue = 0, label = self.scale_labels[1])
        self.scale.grid(row=1, column = 0)

        self.quit_button = tk.Button(master=root, text="Quit", command=self._quit)
        self.quit_button.grid(row=1, column= 1)

        # Plot matplotlib canvas
        self.nca = Model()
        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.fig.add_subplot(111)
        out = np.transpose(to_rgba(self.nca.x).detach().cpu().numpy()[0], (1, 2, 0))
        plt.imshow((out * 255).astype(np.uint8))
        plt.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.callbacks.connect('button_press_event', self.on_click)
        self.canvas.callbacks.connect('motion_notify_event', self.drag)
        self.canvas.callbacks.connect('button_release_event', self.release)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, columnspan = 2)
        
        # Update plot every x interval, depending on the user's speed
        self.update_plot()

    def onScale(self, val):
        v = int(val)
        self.interval_speed.set(v)
        self.scale.config(label=self.scale_labels[v])

    def update_plot(self):
        self.fig.clear()
        self.nca.x = self.nca.model(self.nca.x)
        out = np.transpose(to_rgba(self.nca.x).detach().cpu().numpy()[0], (1, 2, 0))
        plt.imshow((out * 255).astype(np.uint8))
        plt.axis('off')
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, columnspan=3)
        self.after(int(100/self.interval_speed.get()), self.update_plot) # how often you want to refresh

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

if __name__ == "__main__":
    root = tk.Tk()
    GUI(root)
    root.geometry("500x575")
    root.resizable(width=False, height=False)
    root.mainloop()