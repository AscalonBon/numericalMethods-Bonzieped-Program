from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from sympy import symbols, Eq, solve, sin, sqrt
from scipy.optimize import fsolve

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master=None, borderwidth=24)
        self.pack(anchor=W) 
        
        self.formula_frame = tk.Frame(self)
        self.formula_frame.pack(anchor=W, pady=2)
        
        self.method_frame = tk.Frame(self)
        self.method_frame.pack(anchor=W, pady=2)
        
        self.ab_frame = tk.Frame(self)
        self.ab_frame.pack(anchor=W)
        
        self.takeFormula()
        self.takeMethod()
        self.takeA0()
        self.takeB0()

    def takeFormula(self):
        self.inputExpres_txt = Label(self.formula_frame, text='Insert Expression: ')
        self.inputExpres_txt.pack(side=LEFT)
        self.inputExpres = Entry(self.formula_frame, width=40)
        self.inputExpres.pack(side=LEFT)

    def takeMethod(self):
        self.selMethod_txt = Label(self.method_frame, text='Choose method:')
        self.selMethod_txt.pack(side=LEFT)
        self.methBisect = Radiobutton(self.method_frame, text='Bisect', variable=vars, value=1)
        self.methBisect.pack(side=LEFT, padx=(0, 4))
        self.methNewton = Radiobutton(self.method_frame, text='Newton', variable=vars, value=1)
        self.methNewton.pack(side=LEFT, padx=(0, 4))
        self.methSecant = Radiobutton(self.method_frame, text='Secant', variable=vars, value=1)
        self.methSecant.pack(side=LEFT, padx=(0, 4))

    def takeA0(self):
        self.inputA0_txt = Label(self.ab_frame, text='Insert initial [a0]: ')
        self.inputA0_txt.pack(side=LEFT)
        self.inputA0 = Entry(self.ab_frame, width=11)
        self.inputA0.pack(side=LEFT, padx=(0, 4))

    def takeB0(self):
        self.inputB0_txt = Label(self.ab_frame, text='Insert initial [b0]: ')
        self.inputB0_txt.pack(side=LEFT)
        self.inputB0 = Entry(self.ab_frame, width=11)
        self.inputB0.pack(side=LEFT)

app = Application()
app.master.geometry('800x400')
app.master.title('Bonzieped')
app.mainloop()