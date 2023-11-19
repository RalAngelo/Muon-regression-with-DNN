import tkinter as tk
from tkinter import *
from tkinter import messagebox
import numpy as np
import tensorflow as tf

from TrainModel import*
from model import*

class ptPredictor(tk.Tk):
    def __init__(self, master = None):
        super().__init__(master)

        self.title("pt Predictor")
        self.geometry("400x500")

        self.instruction_label = tk.Label(self, text='Enter the data:')
        self.instruction_label.place(x=10, y=10)

        self.label_pt = tk.LabelFrame(self, text = "pt")
        self.label_pt.place(x = 210, y = 100, width = 160 , height = 80)

        self.label_pt_p = tk.LabelFrame(self, text = "Predictpt")
        self.label_pt_p.place(x = 210, y = 200, width = 160 , height = 80)

        self.pt_entries = []
        titles = ["pt: ","eta: " , "phi: ","Q: " , "chiSq: ","dxy: " , "iso: ","MET: " , "phiMET: "]
        for i in range(9):
            entry = tk.Entry(self, width=10)
            tk.Label(self, text=titles[i]).place(x=10, y=80 * ((i * (30/100)) + 1))
            self.pt_entries.append(entry)
            entry.place(x=75 , y=80 * ((i * (30/100)) + 1))
        
        self.predict_button = tk.Button(self, text='Predict', command=self.predict_pt)
        self.predict_button.place(x=10, y=300)

        self.btn_predict = tk.Button(self, text="show accuracy", command=accuracyPlot)
        self.btn_predict.place(x=10, y=330)

        self.btn_predict2 = tk.Button(self, text="Show loss function", command=lossFunctionPlot)
        self.btn_predict2.place(x=10, y=360)

        self.btn_predict3 = tk.Button(self, text="DistributionOfPredictVsActual", command=PredictVSactual)
        self.btn_predict3.place(x=10, y=390)

        self.btn_predict4 = tk.Button(self, text="Model", command=showModelWeights)
        self.btn_predict4.place(x=10, y=420)



    def predict_pt(self):
        # Check if all entries are filled
        for entry in self.pt_entries:
            value = entry.get()
            if value == '':
                show_error_message("Prediction Error", "Please enter all data.")
                return

        # Extract pt values from entries
        pt_values = []
        for entry in self.pt_entries:
            value = entry.get()
            pt_values.append(float(value))
        
        print(pt_values)
        if pt_values[0] < 25.0001 or pt_values[0] > 49712.4:
            show_error_message("Prediction Error", "25.0001 < pt < 49712.4")
        
        if pt_values[1] < -2.1 or pt_values[1] > 2.0999:
            show_error_message("Prediction Error", "-2.1 < eta < 2.0999")
        
        if pt_values[2] < -3.1413 or pt_values[2] > 3.1416:
            show_error_message("Prediction Error", "-3.1413 < phi < 3.1416")
        
        if pt_values[3] != -1 and pt_values[3] != 1:
            show_error_message("Prediction Error", "Q = -1 or 1")
        
        if pt_values[4] < 0.0049 or pt_values[4] > 5328.73:
            show_error_message("Prediction Error", "0.0049 < chiSq < 5328.73")

        if pt_values[5] < -340.023 or pt_values[5] > 291.479:
            show_error_message("Prediction Error", "-340.023 < dxy < 291.479")
        
        if pt_values[6] < 0 or pt_values[6] > 24918.3:
            show_error_message("Prediction Error", "0 < iso < 24918.3")
        
        if pt_values[7] < 0.0345 or pt_values[7] > 332.55:
            show_error_message("Prediction Error", "0.0345 < MET < 332.55")
        
        if pt_values[8] < -3.1413 or pt_values[8] > 3.1414:
            show_error_message("Prediction Error", "-3.1413 < phiMET < 3.1414")
        
        result_label = tk.Label(self.label_pt, text = pt_values[0])
        result_label.pack()

        model = tf.keras.models.load_model('PHEmodel.h5')
        prediction = model.predict(np.array(pt_values).reshape(1, 9))

        result_label_p = tk.Label(self.label_pt_p, text = prediction)
        result_label_p.pack()
        


def show_error_message(title, message):
    """Displays an error message in an alert window."""
    root = tk.Tk()
    root.withdraw()
    tk.messagebox.showerror(title, message)
    root.mainloop()

if __name__ == '__main__':
    app = ptPredictor()
    app.mainloop()