from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from sympy import symbols, sympify, lambdify, diff, solve
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication, convert_xor)
import numpy as np
from math import isnan
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

transformations = (standard_transformations +
                   (implicit_multiplication, convert_xor))

class RootFinder:
    def __init__(self):
        self.x = symbols('x')
        self.x_sym, self.y_sym = symbols('x y')  # For 2D plotting

    def _prepare_equation_string(self, equation_str):
        """
        Helper to preprocess the equation string, handling '=' sign.
        Converts 'LHS = RHS' to 'LHS - RHS'.
        """
        if '=' in equation_str:
            parts = equation_str.split('=')
            if len(parts) != 2:
                return None, "Invalid equation format. Only one '=' sign is allowed."
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            # Ensure both sides are not empty after stripping
            if not lhs and not rhs:
                return None, "Equation cannot be empty on both sides of '='."
            if not lhs: # e.g., "= x**2 - 4" -> "-(x**2 - 4)"
                return f"-({rhs})", None
            if not rhs: # e.g., "x**2 - 4 =" -> "x**2 - 4"
                return lhs, None
            return f"({lhs}) - ({rhs})", None
        return equation_str, None

    def parse_function(self, equation_str):
        """
        Parses a string into a SymPy expression, assuming it's a 1D function of 'x'.
        Returns (expr, None) on success, or (None, error_message) on failure.
        """
        processed_str, error_msg = self._prepare_equation_string(equation_str)
        if error_msg:
            return None, error_msg
        if processed_str is None: # Should not happen if error_msg is handled
            return None, "Internal error during equation processing."

        try:
            expr = parse_expr(processed_str, transformations=transformations)
            if self.y_sym in expr.free_symbols:
                # Indicate that it's a 2D function, not a 1D function of x only
                return None, "Function contains 'y'." # Specific error to differentiate from parsing failure
            return expr, None
        except SyntaxError as e:
            return None, f"Syntax Error: '{e}'. Ensure correct Python-like syntax (e.g., use '**' for powers)."
        except ValueError as e:
            return None, f"Value Error: '{e}'. Check your expression for valid mathematical operations."
        except Exception as e:
            return None, f"Error parsing 1D function: {e}"

    def parse_2d_function(self, equation_str):
        """
        Parses a string into a SymPy expression, assuming it's a 2D function of 'x' and 'y'.
        Returns (expr, None) on success, or (None, error_message) on failure.
        """
        processed_str, error_msg = self._prepare_equation_string(equation_str)
        if error_msg:
            return None, error_msg
        if processed_str is None: # Should not happen if error_msg is handled
            return None, "Internal error during equation processing."

        try:
            expr = parse_expr(processed_str, transformations=transformations)
            return expr, None
        except SyntaxError as e:
            return None, f"Syntax Error: '{e}'. Ensure correct Python-like syntax (e.g., use '**' for powers)."
        except ValueError as e:
            return None, f"Value Error: '{e}'. Check your expression for valid mathematical operations."
        except Exception as e:
            return None, f"Error parsing 2D function: {e}"

    def bisection(self, f, a, b, tol=1e-6, max_iter=100):
        steps = []
        f_lambda = lambdify(self.x, f, modules=['numpy']) # Ensure numpy module is used
        try:
            val_fa = f_lambda(a)
            val_fb = f_lambda(b)
        except (TypeError, ValueError) as e:
            return None, [] # Return None if evaluation fails (e.g., domain error)

        if np.sign(val_fa) == np.sign(val_fb):
            return None, [] # Condition not met for bisection

        for i in range(max_iter):
            c = (a + b) / 2
            try:
                fc = f_lambda(c)
            except (TypeError, ValueError) as e:
                steps.append((i+1, a, b, c, float('nan'), float('nan'))) # Log NaN if evaluation fails
                return None, steps

            steps.append((i+1, a, b, c, fc, abs(b-a)/2))
            if fc == 0 or abs(b - a)/2 < tol:
                return c, steps
            if np.sign(fc) == np.sign(f_lambda(a)):
                a = c
            else:
                b = c
        return (a + b) / 2, steps

    def newton(self, f, x0, tol=1e-6, max_iter=100):
        steps = []
        df = diff(f, self.x)
        f_lambda = lambdify(self.x, f, modules=['numpy']) # Ensure numpy module is used
        df_lambda = lambdify(self.x, df, modules=['numpy']) # Ensure numpy module is used
        x = x0
        for i in range(max_iter):
            try:
                fx = f_lambda(x)
                dfx = df_lambda(x)
            except (TypeError, ValueError) as e:
                steps.append((i+1, x, float('nan'), float('nan'), float('nan')))
                return None, steps

            steps.append((i+1, x, fx, dfx, abs(fx)))
            if abs(fx) < tol:
                return x, steps
            if dfx == 0 or isnan(dfx):
                return None, steps
            x = x - fx/dfx
        return x, steps

    def secant(self, f, x0, x1, tol=1e-6, max_iter=100):
        steps = []
        f_lambda = lambdify(self.x, f, modules=['numpy']) # Ensure numpy module is used
        try:
            fx0 = f_lambda(x0)
            fx1 = f_lambda(x1)
        except (TypeError, ValueError) as e:
            return None, []

        steps.append((1, x0, fx0, None, abs(x1-x0)))
        if abs(x1-x0) < tol: # Initial error check for secant
             return x1, steps
        steps.append((2, x1, fx1, None, abs(x1-x0))) # Re-add absolute difference as error for consistency with bisection

        for i in range(2, max_iter):
            if fx1 - fx0 == 0:
                return None, steps
            x2 = x1 - fx1*(x1 - x0)/(fx1 - fx0)
            try:
                fx2 = f_lambda(x2)
            except (TypeError, ValueError) as e:
                steps.append((i+1, x2, float('nan'), None, float('nan')))
                return None, steps

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

        # Create main panels
        self.left_panel = Frame(self)
        self.left_panel.pack(side=LEFT, fill=Y, padx=10, pady=10)

        self.right_panel = Frame(self)
        self.right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

        # Setup the notebook and tabs first, as other widgets will go into them
        self.notebook = tk.ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=BOTH, expand=True)

        # First tab for results table
        self.results_tab = Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")

        # Second tab for plotting
        self.plot_tab = Frame(self.notebook)
        self.notebook.add(self.plot_tab, text="Plot")

        self.create_widgets()
        self.setup_plot_area() # This will now only configure the plot area
        self.load_example("x^3 - 2x - 5")

    def update_input_labels(self):
        selected_method = self.method_var.get()
        if selected_method == 1:  # Bisection
            self.a0_label.config(text='a:')
            self.b0_label.config(text='b:')
            self.inputA0.config(state='normal')
            self.inputB0.config(state='normal')
        elif selected_method == 2:  # Newton
            self.a0_label.config(text='x0:')
            self.b0_label.config(text='')
            self.inputA0.config(state='normal')
            self.inputB0.config(state='disabled')
        elif selected_method == 3:  # Secant
            self.a0_label.config(text='x0:')
            self.b0_label.config(text='x1:')
            self.inputA0.config(state='normal')
            self.inputB0.config(state='normal')

    def create_widgets(self):
        # Left panel widgets (original functionality)
        equation_frame = Frame(self.left_panel)
        equation_frame.pack(anchor=W, pady=2, fill=X)

        # Updated label to indicate '=' is now allowed
        Label(equation_frame, text='Enter f(x) or f(x,y) (e.g., x^3 - 2^x - 5 or 3(x^2+y^2)^2 = 100^(x^2-y^2)):').pack(side=TOP, anchor=W)
        self.inputExpr = Entry(equation_frame, width=40)
        self.inputExpr.pack(side=TOP, fill=X, expand=True)

        method_frame = Frame(self.left_panel)
        method_frame.pack(anchor=W, pady=2, fill=X)

        Label(method_frame, text='Method:').pack(side=LEFT)
        Radiobutton(method_frame, text='Bisection', variable=self.method_var,
                            value=1, command=self.update_input_labels).pack(side=LEFT, padx=(0, 4))
        Radiobutton(method_frame, text='Newton', variable=self.method_var,
                            value=2, command=self.update_input_labels).pack(side=LEFT, padx=(0, 4))
        Radiobutton(method_frame, text='Secant', variable=self.method_var,
                            value=3, command=self.update_input_labels).pack(side=LEFT, padx=(0, 4))

        self.ab_frame = Frame(self.left_panel)
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
        button_frame = Frame(self.left_panel)
        button_frame.pack(anchor=W, pady=10, fill=X)

        Button(button_frame, text="Calculate Root", command=self.compute_root).pack(side=LEFT)
        Button(button_frame, text="Plot Function", command=self.plot_function).pack(side=LEFT, padx=5)

        self.result_label = Label(self.left_panel, text="Results will appear here", foreground='green')
        self.result_label.pack(anchor=W, pady=5)

        # Right panel widgets (treeview) - Now directly packed into results_tab
        hscroll = Scrollbar(self.results_tab, orient="horizontal")
        hscroll.pack(side=BOTTOM, fill=X)

        vscroll = Scrollbar(self.results_tab)
        vscroll.pack(side=RIGHT, fill=Y)

        self.tree = Treeview(self.results_tab,
                                xscrollcommand=hscroll.set,
                                yscrollcommand=vscroll.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True) # Pack directly into results_tab

        hscroll.config(command=self.tree.xview)
        vscroll.config(command=self.tree.yview)

        self.tree["columns"] = ("iter", "col1", "col2", "col3", "col4")
        self.tree.heading("#0", text="Method")
        self.tree.heading("iter", text="Iteration")
        self.tree.heading("col1", text="Value 1")
        self.tree.heading("col2", text="Value 2")
        self.tree.heading("col3", text="Value 3")
        self.tree.heading("col4", text="Error")

        # Increased column widths for better readability
        self.tree.column("#0", width=180, minwidth=120) # Increased width for Method column
        self.tree.column("iter", width=120, minwidth=100) # Increased width for Iteration column
        self.tree.column("col1", width=180, minwidth=120)
        self.tree.column("col2", width=180, minwidth=120)
        self.tree.column("col3", width=180, minwidth=120)
        self.tree.column("col4", width=180, minwidth=120)

        # Call update_input_labels once to set initial state based on default method
        self.update_input_labels()

    def setup_plot_area(self):
        # Increased figsize for a larger plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8)) # Changed from (8, 6) to (10, 8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_tab)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        # Add navigation toolbar (optional)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_tab)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill=BOTH, expand=True)

    def plot_function(self):
        equation_str = self.inputExpr.get()
        if not equation_str:
            self.result_label.config(text="Please enter an equation", foreground='red')
            return

        try:
            self.ax.clear()

            # Attempt to parse as a 1D function first
            f_1d, error_1d = self.root_finder.parse_function(equation_str)

            if f_1d is not None:
                # Plot 1D function
                x_vals = np.linspace(-10, 10, 400)
                f_lambda = lambdify(self.root_finder.x, f_1d, modules=['numpy'])
                y_vals = f_lambda(x_vals)

                self.ax.plot(x_vals, y_vals, label='f(x)')
                self.ax.axhline(0, color='black', linewidth=0.5)
                self.ax.axvline(0, color='black', linewidth=0.5)
                self.ax.set_xlabel('x')
                self.ax.set_ylabel('f(x)')
                self.ax.set_title('Function Plot: f(x)')
                self.ax.grid(True)
                self.ax.legend()

                # Mark any roots found
                roots = solve(f_1d, self.root_finder.x)
                for root in roots:
                    if root.is_real:
                        root_val = float(root)
                        self.ax.scatter([root_val], [0], color='red', s=50, label='Roots')
                        # Display exact root value below the graph
                        y_offset = self.ax.get_ylim()[0] - 0.1 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
                        self.ax.text(root_val, y_offset, f"x={root}",
                                     horizontalalignment='center', color='darkgreen', fontsize=9,
                                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

                        # Update legend to show 'Roots' only once
                        handles, labels = self.ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        self.ax.legend(by_label.values(), by_label.keys())

                self.result_label.config(text="1D function plotted", foreground='green')
            else:
                # If not a 1D function, try to parse as 2D implicit function
                f_2d, error_2d = self.root_finder.parse_2d_function(equation_str)
                if f_2d is not None:
                    # Plot 2D implicit function
                    x = np.linspace(-6, 6, 500)
                    y = np.linspace(-4, 4, 500)
                    X, Y = np.meshgrid(x, y)

                    f_lambda = lambdify((self.root_finder.x_sym, self.root_finder.y_sym), f_2d, modules=['numpy'])
                    Z = f_lambda(X, Y)

                    self.ax.contour(X, Y, Z, levels=[0], colors='blue')
                    self.ax.axhline(0, color='black', linewidth=0.5)
                    self.ax.axvline(0, color='black', linewidth=0.5)
                    self.ax.set_xlabel('x')
                    self.ax.set_ylabel('y')
                    self.ax.set_title('Implicit Function Plot: f(x,y)=0')
                    self.ax.grid(True)

                    # Find and mark roots (intercepts with axes)
                    # For x-axis roots (y=0)
                    y_zero_expr = f_2d.subs(self.root_finder.y_sym, 0)
                    x_roots = solve(y_zero_expr, self.root_finder.x_sym, real=True)
                    for root in x_roots:
                        if root.is_real:
                            root_val = float(root)
                            self.ax.scatter([root_val], [0], color='red', s=50, label='Intercepts')
                            # Display exact x-intercept value below the graph
                            y_offset = self.ax.get_ylim()[0] - 0.1 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
                            self.ax.text(root_val, y_offset,
                                         f"x={root}", horizontalalignment='center', color='darkgreen', fontsize=9,
                                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))


                    # For y-axis roots (x=0)
                    x_zero_expr = f_2d.subs(self.root_finder.x_sym, 0)
                    y_roots = solve(x_zero_expr, self.root_finder.y_sym, real=True)
                    for root in y_roots:
                        if root.is_real:
                            self.ax.scatter([0], [float(root)], color='red', s=50)
                            # Display exact y-intercept value next to the graph
                            x_offset = self.ax.get_xlim()[1] + 0.1 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
                            self.ax.text(x_offset, float(root),
                                         f"y={root}", verticalalignment='center', color='darkgreen', fontsize=9,
                                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))


                    # Update legend to show 'Intercepts' only once
                    handles, labels = self.ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    self.ax.legend(by_label.values(), by_label.keys())

                    self.result_label.config(text="2D implicit function plotted. For f(x,y)=0, the 'roots' are the entire curve. Red dots indicate intercepts with the x and y axes.", foreground='green')
                else:
                    # If neither 1D nor 2D parsing succeeded, show the most relevant error
                    if error_1d and "Function contains 'y'." in error_1d: # Specific message from parse_function
                        self.result_label.config(text="Could not plot. Equation contains 'y' but could not be parsed as a 2D function.", foreground='red')
                    elif error_2d:
                        self.result_label.config(text=f"Plotting error: {error_2d}", foreground='red')
                    else:
                        self.result_label.config(text="Could not parse equation. Please check syntax.", foreground='red')


            self.canvas.draw()
            self.notebook.select(1)  # Switch to plot tab

        except Exception as e:
            self.result_label.config(text=f"An unexpected error occurred during plotting: {str(e)}", foreground='red')
            print(f"Plotting error: {e}")

    def compute_root(self):
        # Get the equation string from the input field
        equation_str = self.inputExpr.get()
        if not equation_str:
            self.result_label.config(text="Please enter an equation", foreground='red')
            return

        # Attempt to parse as a 1D function first
        f_expr, parse_error_message = self.root_finder.parse_function(equation_str)

        is_2d_expression = False
        if f_expr is None and "Function contains 'y'." in str(parse_error_message):
            # It's a 2D expression, so we'll try to solve for x by setting y=0
            f_2d_original, error_2d_parse = self.root_finder.parse_2d_function(equation_str)
            if f_2d_original is None:
                self.result_label.config(text=f"Error parsing 2D expression: {error_2d_parse}", foreground='red')
                for item in self.tree.get_children(): self.tree.delete(item)
                return

            f_expr = f_2d_original.subs(self.root_finder.y_sym, 0)
            is_2d_expression = True
            if self.root_finder.y_sym in f_expr.free_symbols: # Should not happen after subs(y,0)
                 self.result_label.config(text="Internal error: 'y' still present after substitution.", foreground='red')
                 for item in self.tree.get_children(): self.tree.delete(item)
                 return
        elif f_expr is None: # A general parsing error for 1D function
            self.result_label.config(text=f"Error parsing function: {parse_error_message}", foreground='red')
            for item in self.tree.get_children(): self.tree.delete(item)
            return

        # If we reach here, f_expr is a valid 1D function of x (either original or derived)
        f = f_expr
        method = self.method_var.get()
        root = None
        steps = []
        method_name = ""

        try:
            if method == 1:  # Bisection
                method_name = "Bisection Method"
                a_str = self.inputA0.get()
                b_str = self.inputB0.get()
                if not a_str or not b_str:
                    self.result_label.config(text="Please enter values for 'a' and 'b' for Bisection method.", foreground='red')
                    return
                a = float(a_str)
                b = float(b_str)
                root, steps = self.root_finder.bisection(f, a, b)
                if not steps and root is None: # Bisection returned no steps, likely f(a) and f(b) same sign
                    self.result_label.config(text="Bisection requires f(a) and f(b) to have opposite signs, or root not found within interval. Try different 'a' and 'b'.", foreground='red')
                    return

            elif method == 2:  # Newton
                method_name = "Newton's Method"
                x0_str = self.inputA0.get()
                if not x0_str:
                    self.result_label.config(text="Please enter a value for 'x0' for Newton's method.", foreground='red')
                    return
                x0 = float(x0_str)
                root, steps = self.root_finder.newton(f, x0)
            elif method == 3:  # Secant
                method_name = "Secant Method"
                x0_str = self.inputA0.get()
                x1_str = self.inputB0.get()
                if not x0_str or not x1_str:
                    self.result_label.config(text="Please enter values for 'x0' and 'x1' for Secant method.", foreground='red')
                    return
                x0 = float(x0_str)
                x1 = float(x1_str)
                root, steps = self.root_finder.secant(f, x0, x1)

            # Clear previous results in the treeview
            for item in self.tree.get_children():
                self.tree.delete(item)

            result_prefix = "Numerical Root found: "
            if is_2d_expression:
                result_prefix = "Numerical X-intercept (Y=0) found: "

            if root is not None:
                numerical_root_text = f"{result_prefix}{root:.6f}"
                exact_roots_text = ""
                try:
                    # Attempt to find symbolic roots for the 1D function (original or derived)
                    exact_roots = solve(f, self.root_finder.x)
                    if exact_roots:
                        real_exact_roots = [r for r in exact_roots if r.is_real]
                        if real_exact_roots:
                            exact_roots_text = "\nExact Roots (from SymPy): " + ", ".join([str(r) for r in real_exact_roots])
                        else:
                            exact_roots_text = "\nNo real exact roots found by SymPy."
                    else:
                        exact_roots_text = "\nNo exact roots found by SymPy."
                except Exception as ex_solve:
                    exact_roots_text = f"\nError finding exact roots: {ex_solve}"

                self.result_label.config(text=f"{numerical_root_text}{exact_roots_text}", foreground='green')
                # Populate the treeview with steps
                for step in steps:
                    if method == 1: # Bisection
                        self.tree.insert("", "end", text=method_name, values=(step[0], f"{step[1]:.6f}", f"{step[2]:.6f}", f"{step[3]:.6f}", f"{step[5]:.6e}"))
                    elif method == 2: # Newton
                        self.tree.insert("", "end", text=method_name, values=(step[0], f"{step[1]:.6f}", f"{step[2]:.6e}", f"{step[3]:.6e}", f"{step[4]:.6e}"))
                    elif method == 3: # Secant
                        self.tree.insert("", "end", text=method_name, values=(step[0], f"{step[1]:.6f}", f"{step[2]:.6e}", "", f"{step[4]:.6e}"))
            else:
                self.result_label.config(text="Could not find a root within max iterations or conditions not met for the derived f(x) (where y=0).", foreground='red')
                # Populate the treeview with steps even if no root found
                for step in steps:
                    if method == 1: # Bisection
                        self.tree.insert("", "end", text=method_name, values=(step[0], f"{step[1]:.6f}", f"{step[2]:.6f}", f"{step[3]:.6f}", f"{step[5]:.6e}"))
                    elif method == 2: # Newton
                        self.tree.insert("", "end", text=method_name, values=(step[0], f"{step[1]:.6f}", f"{step[2]:.6e}", f"{step[3]:.6e}", f"{step[4]:.6e}"))
                    elif method == 3: # Secant
                        self.tree.insert("", "end", text=method_name, values=(step[0], f"{step[1]:.6f}", f"{step[2]:.6e}", "", f"{step[4]:.6e}"))


            self.notebook.select(0) # Switch to results tab

        except ValueError:
            self.result_label.config(text="Please enter valid numerical inputs for a, b, x0, x1.", foreground='red')
        except Exception as e:
            self.result_label.config(text=f"An error occurred during computation: {str(e)}", foreground='red')
            print(f"Computation error: {e}")

    def load_example(self, equation):
        self.inputExpr.delete(0, tk.END)
        self.inputExpr.insert(0, equation)
        self.update_input_labels() # Ensure labels are updated after loading example
        self.result_label.config(text="Results will appear here", foreground='green') # Reset result label


app = Application()
# Increased overall window size
app.master.geometry('1200x800')
app.master.iconbitmap('knife.ico')
app.master.title('Bonzieped')
app.mainloop()
