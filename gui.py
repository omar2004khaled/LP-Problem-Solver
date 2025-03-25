import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from simplex import SimplexSolver
from bigmtest import BigMSolver
from twophase import TwoPhaseSimplexSolver
from Goal import GoalSolver
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from graphical import LPPlotter

class LPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Programming Solver")
        self.root.geometry("1000x700")
        self.style = ttk.Style(theme="cosmo")

        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Notebook to hold the three pages
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create pages
        self.page1 = ttk.Frame(self.notebook)
        self.page2 = ttk.Frame(self.notebook)
        self.page3 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.page1, text="Simplex/Big M/Two-Phase")
        self.notebook.add(self.page2, text="Goal Programming")
        self.notebook.add(self.page3, text="Graphical Solution")

        # Variables
        self.method_var = tk.StringVar(value="simplex")
        self.problem_type_var = tk.StringVar(value="max")
        self.num_vars = tk.IntVar(value=2)
        self.num_constraints = tk.IntVar(value=2)
        self.num_goals = tk.IntVar(value=1)

        # Create widgets
        self.create_page1_widgets()
        self.create_page2_widgets()
        self.create_page3_widgets()

    def create_page1_widgets(self):
        # Input section
        input_frame = ttk.Labelframe(self.page1, text="Input Parameters", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Method selection
        ttk.Label(input_frame, text="Select Method:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Simplex", variable=self.method_var, value="simplex").grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Big M", variable=self.method_var, value="bigm").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Two-Phase", variable=self.method_var, value="two_phase").grid(row=0, column=3, padx=10, pady=10, sticky="w")

        # Problem type
        ttk.Label(input_frame, text="Problem Type:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Maximization", variable=self.problem_type_var, value="max").grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Minimization", variable=self.problem_type_var, value="min").grid(row=1, column=2, padx=10, pady=10, sticky="w")

        # Number inputs
        ttk.Label(input_frame, text="Number of Variables:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        ttk.Entry(input_frame, textvariable=self.num_vars, width=10).grid(row=2, column=1, padx=10, pady=10, sticky="w")

        ttk.Label(input_frame, text="Number of Constraints:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        ttk.Entry(input_frame, textvariable=self.num_constraints, width=10).grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # Update button
        ttk.Button(input_frame, text="Update Input Fields", command=self.update_input_fields).grid(row=4, column=0, columnspan=3, padx=10, pady=10)

        # Coefficients frame
        self.coeff_frame = ttk.Labelframe(self.page1, text="Coefficients", padding=10)
        self.coeff_frame.pack(fill=tk.X, padx=10, pady=10)

        # Solve button
        ttk.Button(self.page1, text="Solve", command=self.solve).pack(pady=10)

        # Initialize
        self.update_input_fields()

    def update_input_fields(self):
        # Clear previous widgets
        for widget in self.coeff_frame.winfo_children():
            widget.destroy()

        num_vars = self.num_vars.get()
        num_constraints = self.num_constraints.get()

        # Objective function
        ttk.Label(self.coeff_frame, text="Objective Function Coefficients:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.obj_coeff_entries = []
        for i in range(num_vars):
            entry = ttk.Entry(self.coeff_frame, width=10)
            entry.grid(row=0, column=i+1, padx=5, pady=5, sticky="w")
            self.obj_coeff_entries.append(entry)

        # Variable restrictions
        ttk.Label(self.coeff_frame, text="Variable Restrictions:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.var_restriction_vars = []
        for i in range(num_vars):
            var = tk.StringVar(value="non-negative")
            ttk.OptionMenu(self.coeff_frame, var, "non-negative", "unrestricted").grid(row=1, column=i+1, padx=5, pady=5, sticky="w")
            self.var_restriction_vars.append(var)

        # Constraints
        self.constraint_entries = []
        self.rhs_entries = []
        self.constraint_type_vars = []
        for i in range(num_constraints):
            ttk.Label(self.coeff_frame, text=f"Constraint {i+1}:").grid(row=i+2, column=0, padx=10, pady=5, sticky="w")
            
            constraint_entries_row = []
            for j in range(num_vars):
                entry = ttk.Entry(self.coeff_frame, width=10)
                entry.grid(row=i+2, column=j+1, padx=5, pady=5, sticky="w")
                constraint_entries_row.append(entry)
            self.constraint_entries.append(constraint_entries_row)

            # RHS
            rhs_entry = ttk.Entry(self.coeff_frame, width=10)
            rhs_entry.grid(row=i+2, column=num_vars+1, padx=5, pady=5, sticky="w")
            self.rhs_entries.append(rhs_entry)

            # Constraint type
            constraint_type_var = tk.StringVar(value="<=")
            ttk.OptionMenu(self.coeff_frame, constraint_type_var, "<=", ">=", "=").grid(row=i+2, column=num_vars+2, padx=5, pady=5, sticky="w")
            self.constraint_type_vars.append(constraint_type_var)

    def solve(self):
        try:
            num_vars = self.num_vars.get()
            num_constraints = self.num_constraints.get()

            c = [float(entry.get()) for entry in self.obj_coeff_entries]
            A = [[float(entry.get()) for entry in row] for row in self.constraint_entries]
            b = [float(entry.get()) for entry in self.rhs_entries]
            constraint_types = [var.get() for var in self.constraint_type_vars]
            variable_restrictions = [var.get() for var in self.var_restriction_vars]
            problem_type = self.problem_type_var.get()

            if self.method_var.get() == "simplex":
                solver = SimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
                steps_file = "simplex.txt"
            elif self.method_var.get() == "bigm":
                solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
                steps_file = "bigmtest.txt"
            elif self.method_var.get() == "two_phase":
                solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
                steps_file = "twophase.txt"

            solver.solve(display_steps=True)
            results = solver.get_results()

            if results['status'].lower() == 'optimal':
                messagebox.showinfo("Results", 
                                f"Optimal Solution: {results['optimal_solution']}\n"
                                f"Optimal Value: {results['optimal_value']}\n"
                                f"Status: {results['status']}")
            else:
                messagebox.showinfo("Results", f"Status: {results['status']}")

            self.display_steps(steps_file)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_steps(self, steps_file):
        try:
            with open(steps_file, "r") as f:
                steps = f.read()

            steps_window = ttk.Toplevel(self.root)
            steps_window.title("Solution Steps")
            steps_window.geometry("1000x800")

            # Configure text widget with monospace font
            text_area = scrolledtext.ScrolledText(
                steps_window, 
                wrap=tk.NONE,  # Disable word wrap for tables
                font=("Courier New", 10),  # Monospace font
                tabs=('0.5in', '1in'),  # Set tab stops
                tabstyle='wordprocessor',  # Preserve tabs
                width=120, 
                height=40
            )
            text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Insert text and make read-only
            text_area.insert(tk.INSERT, steps)
            text_area.configure(state="disabled")

            # Add copy button
            copy_btn = ttk.Button(
                steps_window, 
                text="Copy to Clipboard", 
                command=lambda: self.copy_to_clipboard(steps)
            )
            copy_btn.pack(pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read steps: {str(e)}")

    def copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Steps copied to clipboard!")

    def create_page2_widgets(self):
        # Input section
        input_frame = ttk.Labelframe(self.page2, text="Goal Programming Parameters", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Number inputs
        ttk.Label(input_frame, text="Number of Variables:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.num_vars, width=10).grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(input_frame, text="Number of Constraints:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.num_constraints, width=10).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(input_frame, text="Number of Goals:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.num_goals, width=10).grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Update button
        ttk.Button(input_frame, text="Update Input Fields", command=self.update_goal_input_fields).grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # Frames
        self.normal_constraints_frame = ttk.Labelframe(self.page2, text="Normal Constraints", padding=10)
        self.normal_constraints_frame.pack(fill=tk.X, padx=10, pady=10)

        self.goals_frame = ttk.Labelframe(self.page2, text="Goals", padding=10)
        self.goals_frame.pack(fill=tk.X, padx=10, pady=10)

        # Solve button
        ttk.Button(self.page2, text="Solve Goal Programming", command=self.solve_goal).pack(pady=10)

        # Initialize
        self.update_goal_input_fields()

    def update_goal_input_fields(self):
        # Clear previous widgets
        for widget in self.normal_constraints_frame.winfo_children():
            widget.destroy()
        for widget in self.goals_frame.winfo_children():
            widget.destroy()

        num_vars = self.num_vars.get()
        num_constraints = self.num_constraints.get()
        num_goals = self.num_goals.get()

        # Normal constraints
        ttk.Label(self.normal_constraints_frame, text="Objective Coefficients:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.obj_coeff_entries_goal = []
        for i in range(num_vars):
            entry = ttk.Entry(self.normal_constraints_frame, width=10)
            entry.insert(0, "0")
            entry.grid(row=0, column=i+1, padx=5, pady=5, sticky="w")
            self.obj_coeff_entries_goal.append(entry)

        # Variable restrictions
        ttk.Label(self.normal_constraints_frame, text="Variable Restrictions:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.var_restriction_vars_goal = []
        for i in range(num_vars):
            var = tk.StringVar(value="non-negative")
            ttk.OptionMenu(self.normal_constraints_frame, var, "non-negative", "unrestricted").grid(row=1, column=i+1, padx=5, pady=5, sticky="w")
            self.var_restriction_vars_goal.append(var)

        # Constraints
        self.normal_constraint_entries = []
        self.normal_rhs_entries = []
        self.normal_constraint_type_vars = []
        for i in range(num_constraints):
            ttk.Label(self.normal_constraints_frame, text=f"Constraint {i+1}:").grid(row=i+2, column=0, padx=10, pady=5, sticky="w")
            
            constraint_entries_row = []
            for j in range(num_vars):
                entry = ttk.Entry(self.normal_constraints_frame, width=10)
                entry.grid(row=i+2, column=j+1, padx=5, pady=5, sticky="w")
                constraint_entries_row.append(entry)
            self.normal_constraint_entries.append(constraint_entries_row)

            # RHS
            rhs_entry = ttk.Entry(self.normal_constraints_frame, width=10)
            rhs_entry.grid(row=i+2, column=num_vars+1, padx=5, pady=5, sticky="w")
            self.normal_rhs_entries.append(rhs_entry)

            # Constraint type
            constraint_type_var = tk.StringVar(value="<=")
            ttk.OptionMenu(self.normal_constraints_frame, constraint_type_var, "<=", ">=", "=").grid(row=i+2, column=num_vars+2, padx=5, pady=5, sticky="w")
            self.normal_constraint_type_vars.append(constraint_type_var)

        # Goals
        self.goal_constraint_entries = []
        self.goal_rhs_entries = []
        self.goal_constraint_type_vars = []
        self.goal_priority_entries = []
        for i in range(num_goals):
            ttk.Label(self.goals_frame, text=f"Goal {i+1}:").grid(row=i, column=0, padx=10, pady=5, sticky="w")
            
            goal_entries_row = []
            for j in range(num_vars):
                entry = ttk.Entry(self.goals_frame, width=10)
                entry.grid(row=i, column=j+1, padx=5, pady=5, sticky="w")
                goal_entries_row.append(entry)
            self.goal_constraint_entries.append(goal_entries_row)

            # RHS
            rhs_entry = ttk.Entry(self.goals_frame, width=10)
            rhs_entry.grid(row=i, column=num_vars+1, padx=5, pady=5, sticky="w")
            self.goal_rhs_entries.append(rhs_entry)

            # Constraint type
            constraint_type_var = tk.StringVar(value="<=")
            ttk.OptionMenu(self.goals_frame, constraint_type_var, "<=", ">=", "=").grid(row=i, column=num_vars+2, padx=5, pady=5, sticky="w")
            self.goal_constraint_type_vars.append(constraint_type_var)

            # Priority
            ttk.Label(self.goals_frame, text="Priority:").grid(row=i, column=num_vars+3, padx=5, pady=5, sticky="w")
            priority_entry = ttk.Entry(self.goals_frame, width=10)
            priority_entry.grid(row=i, column=num_vars+4, padx=5, pady=5, sticky="w")
            self.goal_priority_entries.append(priority_entry)

    def solve_goal(self):
        try:
            num_vars = self.num_vars.get()
            num_constraints = self.num_constraints.get()
            num_goals = self.num_goals.get()

            c = [float(entry.get()) for entry in self.obj_coeff_entries_goal]
            A = [[float(entry.get()) for entry in row] for row in self.normal_constraint_entries]
            b = [float(entry.get()) for entry in self.normal_rhs_entries]
            normal_constraint_types = [var.get() for var in self.normal_constraint_type_vars]
            
            A_goals = [[float(entry.get()) for entry in row] for row in self.goal_constraint_entries]
            goals = [float(entry.get()) for entry in self.goal_rhs_entries]
            goal_constraint_types = [var.get() for var in self.goal_constraint_type_vars]
            priority = [float(entry.get()) for entry in self.goal_priority_entries]
            
            variable_restrictions = [var.get() for var in self.var_restriction_vars_goal]

            solver = GoalSolver(c, A, A_goals, b, goals, priority, 
                              normal_constraint_types, goal_constraint_types,
                              variable_restrictions)
            
            solver.solve(display_steps=True)
            results = solver.get_results()

            result_text = f"Optimal Solution: {results['optimal_solution']}\n"
            result_text += f"Optimal Value: {results['optimal_value']}\n"
            result_text += f"Status: {results['status']}\n"
            result_text += "Goal Status:\n"
            for status in results['goal_status']:
                result_text += f"- {status}\n"

            self.display_steps("Goal.txt")
            messagebox.showinfo("Results", result_text)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_page3_widgets(self):
        """Create widgets for the graphical solution page"""
        # Input frame
        input_frame = ttk.Labelframe(self.page3, text="Problem Input", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Problem type
        ttk.Label(input_frame, text="Problem Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.graph_problem_type = tk.StringVar(value="max")
        ttk.Radiobutton(input_frame, text="Maximize", variable=self.graph_problem_type, value="max").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(input_frame, text="Minimize", variable=self.graph_problem_type, value="min").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Objective function
        ttk.Label(input_frame, text="Objective Coefficients:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.obj_x1 = ttk.Entry(input_frame, width=8)
        self.obj_x1.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(input_frame, text="x₁ +").grid(row=1, column=2, padx=0, pady=5)
        self.obj_x2 = ttk.Entry(input_frame, width=8)
        self.obj_x2.grid(row=1, column=3, padx=5, pady=5)
        ttk.Label(input_frame, text="x₂").grid(row=1, column=4, padx=0, pady=5)

        # Constraints frame
        constraints_frame = ttk.Labelframe(self.page3, text="Constraints", padding=10)
        constraints_frame.pack(fill=tk.X, padx=10, pady=10)

        # Initial constraints
        self.constraint_entries = []
        for i in range(2):
            self.add_constraint_field()

        # Add constraint button
        ttk.Button(constraints_frame, text="Add Constraint", command=self.add_constraint_field).pack(pady=5)

        # Plot button
        ttk.Button(self.page3, text="Plot Solution", command=self.plot_solution).pack(pady=10)

        # Graph frame
        self.graph_frame = ttk.Frame(self.page3)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize figure
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_constraint_field(self):
        """Add another constraint input field"""
        frame = ttk.Frame(self.page3.winfo_children()[1])  # Get constraints_frame
        frame.pack(fill=tk.X, pady=5)
        
        entries = []
        # x1 coefficient
        entry = ttk.Entry(frame, width=5)
        entry.pack(side=tk.LEFT, padx=2)
        entries.append(entry)
        
        ttk.Label(frame, text="x₁ +").pack(side=tk.LEFT, padx=2)
        
        # x2 coefficient
        entry = ttk.Entry(frame, width=5)
        entry.pack(side=tk.LEFT, padx=2)
        entries.append(entry)
        
        ttk.Label(frame, text="x₂").pack(side=tk.LEFT, padx=2)
        
        # Constraint type
        constr_type = ttk.Combobox(frame, values=["≤", "≥", "="], width=3)
        constr_type.current(0)
        constr_type.pack(side=tk.LEFT, padx=2)
        
        # RHS
        entry = ttk.Entry(frame, width=5)
        entry.pack(side=tk.LEFT, padx=2)
        entries.append(entry)
        
        self.constraint_entries.append((entries, constr_type))

    def plot_solution(self):
        """Plot the graphical solution"""
        try:
            # Get objective function
            c = [float(self.obj_x1.get()), float(self.obj_x2.get())]
            
            # Get constraints
            A = []
            b = []
            constraint_types = []
            
            for entries, constr_type in self.constraint_entries:
                try:
                    a1 = float(entries[0].get())
                    a2 = float(entries[1].get())
                    rhs = float(entries[2].get())
                    
                    A.append([a1, a2])
                    b.append(rhs)
                    
                    # Convert constraint type
                    ct = constr_type.get()
                    if ct == "≤":
                        constraint_types.append("<=")
                    elif ct == "≥":
                        constraint_types.append(">=")
                    else:
                        constraint_types.append("=")
                except ValueError:
                    continue
            
            if not A:
                raise ValueError("Please enter at least one valid constraint")
            
            # Problem parameters
            variable_restrictions = ['non-negative', 'non-negative']
            problem_type = self.graph_problem_type.get()
            
            # Clear previous plot
            self.ax.clear()
            
            # Create and plot
            plotter = LPPlotter(c, A, b, constraint_types, variable_restrictions, problem_type)
            plotter.ax = self.ax
            plotter.plot()
            
            # Redraw
            self.canvas.draw()
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")
    app = LPApp(root)
    root.mainloop()
