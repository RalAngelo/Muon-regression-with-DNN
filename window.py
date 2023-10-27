import tkinter as tk
import numpy as np

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        group_elem = tk.LabelFrame(self, padx=15, pady=10, text="Event Information")
        group_elem.pack(padx=10, pady=5)

        tk.Label(group_elem, text="pt").grid(row=0)
        tk.Label(group_elem, text="eta").grid(row=1)
        tk.Label(group_elem, text="phi").grid(row=2)
        tk.Label(group_elem, text="Q").grid(row=3)
        tk.Label(group_elem, text="chiSq").grid(row=4)
        tk.Label(group_elem, text="dxy").grid(row=5)
        tk.Label(group_elem, text="iso").grid(row=6)
        tk.Label(group_elem, text="MET").grid(row=7)
        tk.Label(group_elem, text="phiMET").grid(row=8)

        self.pt = tk.Entry(group_elem, width = 10)
        self.eta = tk.Entry(group_elem, width = 10)
        self.phi = tk.Entry(group_elem, width = 10)
        self.Q = tk.Entry(group_elem, width = 10)
        self.chiSq = tk.Entry(group_elem, width = 10)
        self.dxy = tk.Entry(group_elem, width = 10)
        self.iso = tk.Entry(group_elem, width = 10)
        self.MET = tk.Entry(group_elem, width = 10)
        self.phiMET = tk.Entry(group_elem, width = 10)
        
        self.pt.grid(row=0, column=1, sticky=tk.W)
        self.eta.grid(row=1, column=1, sticky=tk.W)
        self.phi.grid(row=2, column=1, sticky=tk.W)
        self.Q.grid(row=3, column=1, sticky=tk.W)
        self.chiSq.grid(row=4, column=1, sticky=tk.W)
        self.dxy.grid(row=5, column=1, sticky=tk.W)
        self.iso.grid(row=6, column=1, sticky=tk.W)
        self.MET.grid(row=7, column=1, sticky=tk.W)
        self.phiMET.grid(row=8, column=1, sticky=tk.W)
    
        self.btn_predict = tk.Button(self, text="predict")
        self.btn_predict.pack(padx=10, pady=10, side=tk.RIGHT)


if __name__ == "__main__":
    app = App()
    app.title("Title in progress")
    app.mainloop()