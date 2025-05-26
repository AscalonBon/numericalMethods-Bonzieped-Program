from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from sympy import symbols, sympify, lambdify, diff
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, 
                                       implicit_multiplication, convert_xor)
import numpy as np
from math import isnan

transformations = (standard_transformations + 
                  (implicit_multiplication, convert_xor))

class RootFinder:
    def __init__(self):
        self.x = symbols('x')
        
    def parse_function(self, equation_str):
        try:
            #symbol handler
            expr = parse_expr(equation_str, transformations=transformations)
            return expr
        except Exception as e:
            print(f"Error parsing function: {e}")
            return None

    def bisection(self, f, a, b, tol=1e-6, max_iter=100):
        steps = []
        f_lambda = lambdify(self.x, f)
        for i in range(max_iter):
            c = (a + b) / 2
            fc = f_lambda(c)
            steps.append((i+1, a, b, c, fc, abs(b-a)/2))
            if fc == 0 or (b - a)/2 < tol:
                return c, steps
            if np.sign(fc) == np.sign(f_lambda(a)):
                a = c
            else:
                b = c
        return (a + b) / 2, steps

    def newton(self, f, x0, tol=1e-6, max_iter=100):
        steps = []
        df = diff(f, self.x)  # Calculate derivative symbolically
        f_lambda = lambdify(self.x, f)
        df_lambda = lambdify(self.x, df)
        x = x0
        for i in range(max_iter):
            fx = f_lambda(x)
            dfx = df_lambda(x)
            steps.append((i+1, x, fx, dfx, abs(fx)))
            if abs(fx) < tol:
                return x, steps
            if dfx == 0 or isnan(dfx):
                return None, steps
            x = x - fx/dfx
        return x, steps

    def secant(self, f, x0, x1, tol=1e-6, max_iter=100):
        steps = []
        f_lambda = lambdify(self.x, f)
        fx0 = f_lambda(x0)
        fx1 = f_lambda(x1)
        steps.append((1, x0, fx0, None, abs(x1-x0)))
        steps.append((2, x1, fx1, None, abs(x1-x0)))
        
        for i in range(2, max_iter):
            if fx1 - fx0 == 0:
                return None, steps
            x2 = x1 - fx1*(x1 - x0)/(fx1 - fx0)
            fx2 = f_lambda(x2)
            steps.append((i+1, x2, fx2, None, abs(x2-x1)))
            if abs(x2 - x1) < tol:
                return x2, steps
            x0, x1, fx0, fx1 = x1, x2, fx1, fx2
        return x1, steps

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master=None, borderwidth=24)
        self.pack(fill=BOTH, expand=True)
        self.root_finder = RootFinder()
        self.method_var = tk.IntVar(value=1)
        
        self.create_widgets()
        self.load_example("x^3 - 2x - 5")
        
    def create_widgets(self):
       
        left_panel = Frame(self)
        left_panel.pack(side=LEFT, fill=Y, padx=10, pady=10)
        
       
        equation_frame = Frame(left_panel)
        equation_frame.pack(anchor=W, pady=2, fill=X)
        
        Label(equation_frame, text='Enter f(x) = 0 (e.g., x^3 - 2x - 5):').pack(side=TOP, anchor=W)
        self.inputExpr = Entry(equation_frame, width=40)
        self.inputExpr.pack(side=TOP, fill=X, expand=True)
      
        method_frame = Frame(left_panel)
        method_frame.pack(anchor=W, pady=2, fill=X)
        
        Label(method_frame, text='Method:').pack(side=LEFT)
        Radiobutton(method_frame, text='Bisection', variable=self.method_var, 
                   value=1, command=self.update_input_labels).pack(side=LEFT, padx=(0, 4))
        Radiobutton(method_frame, text='Newton', variable=self.method_var, 
                   value=2, command=self.update_input_labels).pack(side=LEFT, padx=(0, 4))
        Radiobutton(method_frame, text='Secant', variable=self.method_var, 
                   value=3, command=self.update_input_labels).pack(side=LEFT, padx=(0, 4))
        
        
        self.ab_frame = Frame(left_panel)
        self.ab_frame.pack(anchor=W, pady=2, fill=X)
        
        self.a0_label = Label(self.ab_frame, text='a:')
        self.a0_label.pack(side=LEFT)
        self.inputA0 = Entry(self.ab_frame, width=15)
        self.inputA0.pack(side=LEFT, padx=(0, 10))
        
        self.b0_label = Label(self.ab_frame, text='b:')
        self.b0_label.pack(side=LEFT)
        self.inputB0 = Entry(self.ab_frame, width=15)
        self.inputB0.pack(side=LEFT)
        
        # Buttons
        button_frame = Frame(left_panel)
        button_frame.pack(anchor=W, pady=10, fill=X)
        
        Button(button_frame, text="Calculate Root", command=self.compute_root).pack(side=LEFT)
        
        self.result_label = Label(left_panel, text="Results will appear here", foreground='green')
        self.result_label.pack(anchor=W, pady=5)
        
       
        right_panel = Frame(self)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)
        
       
        tree_frame = Frame(right_panel)
        tree_frame.pack(fill=BOTH, expand=True)
        
       
        hscroll = Scrollbar(tree_frame, orient="horizontal")
        hscroll.pack(side=BOTTOM, fill=X)
        
        
        vscroll = Scrollbar(tree_frame)
        vscroll.pack(side=RIGHT, fill=Y)
        
        
        self.tree = Treeview(tree_frame, 
                           xscrollcommand=hscroll.set, 
                           yscrollcommand=vscroll.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Configure the scrollbars
        hscroll.config(command=self.tree.xview)
        vscroll.config(command=self.tree.yview)
        
        # Configure tree columns for all methods upfront
        self.tree["columns"] = ("iter", "col1", "col2", "col3", "col4")
        self.tree.heading("#0", text="Method")
        self.tree.heading("iter", text="Iteration")
        self.tree.heading("col1", text="Value 1")
        self.tree.heading("col2", text="Value 2")
        self.tree.heading("col3", text="Value 3")
        self.tree.heading("col4", text="Error")
        
        # Set column widths
        self.tree.column("#0", width=120)
        self.tree.column("iter", width=80)
        self.tree.column("col1", width=120)
        self.tree.column("col2", width=120)
        self.tree.column("col3", width=120)
        self.tree.column("col4", width=120)
        
        self.update_input_labels()
        
        Label(left_panel, text="Example equations:").pack(anchor=W)
        self.example_equations = [
            "x^3 - 2x - 5",          # Using ^ and implicit multiplication
            "x^2 - 2",               # Square root of 2
            "3x^4 - 76x^2 + 448",    # Your polynomial with ^
            "exp(x) - 2",            # Natural log of 2
            "(x+2)(x-1)^2"           # Parentheses and exponents
        ]
        self.example_var = tk.StringVar(value=self.example_equations[0])
        example_menu = OptionMenu(left_panel, self.example_var, *self.example_equations, command=self.load_example)
        example_menu.pack(anchor=W, pady=5, fill=X)
        
    def load_example(self, equation):
        self.inputExpr.delete(0, END)
        self.inputExpr.insert(0, equation)
        
    def update_input_labels(self):
        method = self.method_var.get()
        if method == 1:  # Bisection
            self.a0_label.config(text='a0:')
            self.b0_label.config(text='b0:')
        elif method == 2:  # Newton
            self.a0_label.config(text='x0:')
            self.b0_label.config(text='(n/a)')
            self.inputB0.delete(0, END)
            self.inputB0.insert(0, '')
        elif method == 3:  # Secant
            self.a0_label.config(text='x0:')
            self.b0_label.config(text='x1:')
    
    def compute_root(self):
        try:
            equation_str = self.inputExpr.get()
            if not equation_str:
                self.result_label.config(text="Please enter an equation", foreground='red')
                return
                
            f = self.root_finder.parse_function(equation_str)
            if f is None:
                self.result_label.config(text="Error parsing equation", foreground='red')
                return
                
            method = self.method_var.get()
            self.tree.delete(*self.tree.get_children())
            
            if method == 1:  # Bisection
                self.tree.heading("#0", text="Bisection Method")
                self.tree.heading("col1", text="a")
                self.tree.heading("col2", text="b")
                self.tree.heading("col3", text="c")
                
                a = float(self.inputA0.get())
                b = float(self.inputB0.get())
                root, steps = self.root_finder.bisection(f, a, b)
                
                for step in steps:
                    self.tree.insert("", "end", text="", values=step)
                
            elif method == 2:  # Newton
                self.tree.heading("#0", text="Newton's Method")
                self.tree.heading("col1", text="x")
                self.tree.heading("col2", text="f(x)")
                self.tree.heading("col3", text="f'(x)")
                
                x0 = float(self.inputA0.get())
                root, steps = self.root_finder.newton(f, x0)
                
                for step in steps:
                    self.tree.insert("", "end", text="", values=step)
                
            elif method == 3:  # Secant
                self.tree.heading("#0", text="Secant Method")
                self.tree.heading("col1", text="x")
                self.tree.heading("col2", text="f(x)")
                self.tree.heading("col3", text="Ratio")
                
                x0 = float(self.inputA0.get())
                x1 = float(self.inputB0.get())
                root, steps = self.root_finder.secant(f, x0, x1)
                
                for step in steps:
                    self.tree.insert("", "end", text="", values=step)
            
            if root is not None:
                self.result_label.config(text=f"Approximate root: {root:.8f}", foreground='green')
            else:
                self.result_label.config(text="Method failed to converge", foreground='orange')
                
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", foreground='red')

app = Application()
app.master.iconbitmap('knife.ico')
app.master.geometry('1000x600')
app.master.title('Bonzieped')
app.mainloop()