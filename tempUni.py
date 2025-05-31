from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from sympy import symbols, sympify, lambdify, diff
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, 
                                       implicit_multiplication, convert_xor)
import numpy as np
from math import isnan

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master=None, borderwidth=24)
        self.pack(fill=BOTH, expand=True)

app = Application()
app.master.iconbitmap('knife.ico')
app.master.geometry('1000x600')
app.master.title('Bonzieped')
app.mainloop()