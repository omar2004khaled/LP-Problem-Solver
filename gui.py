import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from simplex import SimplexSolver  # Import SimplexSolver from simplex.py
from bigmtest import BigMSolver  # Import BigMSolver from bigmtest.py
from twophase import TwoPhaseSimplexSolver  # Import TwoPhaseSimplexSolver from two_phase.py
from Goal import GoalSolver  # Import GoalSolver from Goal.py
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from graphical import LPPlotter  # Import LPPlotter from graphical.py
class LPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Programming Solver")
        self.root.geometry("1000x700")
        self.style = ttk.Style(theme="cosmo")  # Use a modern theme (e.g., "cosmo", "minty", "flatly")

        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Notebook to hold the two pages
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Page 1: Simplex, Big M, Two-Phase
        self.page1 = ttk.Frame(self.notebook)
        self.notebook.add(self.page1, text="Simplex/Big M/Two-Phase")

        # Page 2: Goal Programming
        self.page2 = ttk.Frame(self.notebook)
        self.notebook.add(self.page2, text="Goal Programming")
        self.page3 = ttk.Frame(self.notebook)
        self.notebook.add(self.page3, text="Graphical Solution")
        # Method selection
        self.method_var = tk.StringVar(value="simplex")
        self.problem_type_var = tk.StringVar(value="max")
        self.num_vars = tk.IntVar(value=2)
        self.num_constraints = tk.IntVar(value=2)
        self.num_goals = tk.IntVar(value=1)  # New variable for number of goals

        self.create_page1_widgets()
        self.create_page2_widgets()
        self.create_page3_widgets()
    def create_page1_widgets(self):
        # Input section
        input_frame = ttk.Labelframe(self.page1, text="Input Parameters", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Method selection
        ttk.Label(input_frame, text="Select Method:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Simplex", variable=self.method_var, value="simplex", bootstyle="info").grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Big M", variable=self.method_var, value="bigm", bootstyle="info").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Two-Phase Simplex", variable=self.method_var, value="two_phase", bootstyle="info").grid(row=0, column=3, padx=10, pady=10, sticky="w")

        # Problem type selection
        ttk.Label(input_frame, text="Problem Type:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Maximization", variable=self.problem_type_var, value="max", bootstyle="info").grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(input_frame, text="Minimization", variable=self.problem_type_var, value="min", bootstyle="info").grid(row=1, column=2, padx=10, pady=10, sticky="w")

        # Number of variables and constraints
        ttk.Label(input_frame, text="Number of Variables:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.num_vars_entry = ttk.Entry(input_frame, textvariable=self.num_vars, width=10, bootstyle="primary")
        self.num_vars_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        ttk.Label(input_frame, text="Number of Constraints:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.num_constraints_entry = ttk.Entry(input_frame, textvariable=self.num_constraints, width=10, bootstyle="primary")
        self.num_constraints_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # Button to update input fields
        ttk.Button(input_frame, text="Update Input Fields", command=self.update_input_fields, bootstyle="success").grid(row=4, column=0, columnspan=3, padx=10, pady=10)

        # Input fields for coefficients
        self.coeff_frame = ttk.Labelframe(self.page1, text="Coefficients", padding=10)
        self.coeff_frame.pack(fill=tk.X, padx=10, pady=10)

        # Solve button
        ttk.Button(self.page1, text="Solve", command=self.solve, bootstyle="primary").pack(pady=10)

        # Initialize input fields
        self.update_input_fields()

    def update_input_fields(self):
        # Clear previous input fields
        for widget in self.coeff_frame.winfo_children():
            widget.destroy()

        num_vars = self.num_vars.get()
        num_constraints = self.num_constraints.get()

        # Objective function coefficients
        ttk.Label(self.coeff_frame, text="Objective Function Coefficients:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.obj_coeff_entries = []
        for i in range(num_vars):
            entry = ttk.Entry(self.coeff_frame, width=10, bootstyle="primary")
            entry.grid(row=0, column=i+1, padx=5, pady=5, sticky="w")
            self.obj_coeff_entries.append(entry)

        # Variable restrictions
        ttk.Label(self.coeff_frame, text="Variable Restrictions:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.var_restriction_vars = []
        for i in range(num_vars):
            var = tk.StringVar(value="non-negative")
            ttk.OptionMenu(self.coeff_frame, var, "non-negative", "unrestricted", bootstyle="info").grid(row=1, column=i+1, padx=5, pady=5, sticky="w")
            self.var_restriction_vars.append(var)

        # Constraint coefficients
        self.constraint_entries = []
        self.rhs_entries = []
        self.constraint_type_vars = []
        for i in range(num_constraints):
            ttk.Label(self.coeff_frame, text=f"Constraint {i+1}:").grid(row=i+2, column=0, padx=10, pady=10, sticky="w")
            constraint_entries_row = []
            for j in range(num_vars):
                entry = ttk.Entry(self.coeff_frame, width=10, bootstyle="primary")
                entry.grid(row=i+2, column=j+1, padx=5, pady=5, sticky="w")
                constraint_entries_row.append(entry)
            self.constraint_entries.append(constraint_entries_row)

            # RHS
            rhs_entry = ttk.Entry(self.coeff_frame, width=10, bootstyle="primary")
            rhs_entry.grid(row=i+2, column=num_vars+1, padx=5, pady=5, sticky="w")
            self.rhs_entries.append(rhs_entry)

            # Constraint type
            constraint_type_var = tk.StringVar(value="<=")
            ttk.OptionMenu(self.coeff_frame, constraint_type_var, "<=", ">=", "=", bootstyle="info").grid(row=i+2, column=num_vars+2, padx=5, pady=5, sticky="w")
            self.constraint_type_vars.append(constraint_type_var)

    def solve(self):
        try:
            # Get input values
            num_vars = self.num_vars.get()
            num_constraints = self.num_constraints.get()

            c = [float(entry.get()) for entry in self.obj_coeff_entries]
            A = [[float(entry.get()) for entry in row] for row in self.constraint_entries]
            b = [float(entry.get()) for entry in self.rhs_entries]
            constraint_types = [var.get() for var in self.constraint_type_vars]
            variable_restrictions = [var.get() for var in self.var_restriction_vars]  # Get variable restrictions
            problem_type = self.problem_type_var.get()

            # Solve using the selected method
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

            # Display results based on status
            if results['status'].lower() == 'optimal':
                messagebox.showinfo("Results", 
                                f"Optimal Solution: {results['optimal_solution']}\n"
                                f"Optimal Value: {results['optimal_value']}\n"
                                f"Status: {results['status']}")
            else:
                messagebox.showinfo("Results", f"Status: {results['status']}")

            # Read and display steps from the file
            self.display_steps(steps_file)

        except Exception as e:
            messagebox.showerror("Error", str(e))
    def display_steps(self, steps_file):
        try:
            with open(steps_file, "r") as f:
                steps = f.read()

            # Create a new window to display steps
            steps_window = ttk.Toplevel(self.root)
            steps_window.title("Steps")
            steps_window.geometry("800x600")

            # Add a scrolled text widget to display steps
            text_area = scrolledtext.ScrolledText(steps_window, wrap=tk.WORD, width=100, height=30)
            text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_area.insert(tk.INSERT, steps)
            text_area.configure(state="disabled")  # Make the text area read-only

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read steps from {steps_file}: {str(e)}")

    def create_page2_widgets(self):
        # Input section
        input_frame = ttk.Labelframe(self.page2, text="Goal.txt", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Number of variables and constraints
        ttk.Label(input_frame, text="Number of Variables:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.num_vars_goal = ttk.Entry(input_frame, textvariable=self.num_vars, width=10, bootstyle="primary")
        self.num_vars_goal.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        ttk.Label(input_frame, text="Number of Normal Constraints:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.num_constraints_goal = ttk.Entry(input_frame, textvariable=self.num_constraints, width=10, bootstyle="primary")
        self.num_constraints_goal.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        ttk.Label(input_frame, text="Number of Goals:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.num_goals_entry = ttk.Entry(input_frame, textvariable=self.num_goals, width=10, bootstyle="primary")
        self.num_goals_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # Button to update input fields
        ttk.Button(input_frame, text="Update Input Fields", command=self.update_goal_input_fields, bootstyle="success").grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # Input frames for coefficients
        self.normal_constraints_frame = ttk.Labelframe(self.page2, text="Normal Constraints", padding=10)
        self.normal_constraints_frame.pack(fill=tk.X, padx=10, pady=10)

        self.goals_frame = ttk.Labelframe(self.page2, text="Goals", padding=10)
        self.goals_frame.pack(fill=tk.X, padx=10, pady=10)

        # Solve button
        ttk.Button(self.page2, text="Solve Goal Programming", command=self.solve_goal, bootstyle="primary").pack(pady=10)

        # Initialize input fields
        self.update_goal_input_fields()

    def update_goal_input_fields(self):
        # Clear previous input fields
        for widget in self.normal_constraints_frame.winfo_children():
            widget.destroy()
        for widget in self.goals_frame.winfo_children():
            widget.destroy()

        num_vars = self.num_vars.get()
        num_constraints = self.num_constraints.get()
        num_goals = self.num_goals.get()

        # Normal constraints coefficients
        ttk.Label(self.normal_constraints_frame, text="Objective Function Coefficients (if any):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.obj_coeff_entries_goal = []
        for i in range(num_vars):
            entry = ttk.Entry(self.normal_constraints_frame, width=10, bootstyle="primary")
            entry.grid(row=0, column=i+1, padx=5, pady=5, sticky="w")
            entry.insert(0, "0")  # Default to 0 for goal programming
            self.obj_coeff_entries_goal.append(entry)

        # Variable restrictions
        ttk.Label(self.normal_constraints_frame, text="Variable Restrictions:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.var_restriction_vars_goal = []
        for i in range(num_vars):
            var = tk.StringVar(value="non-negative")
            ttk.OptionMenu(self.normal_constraints_frame, var, "non-negative", "unrestricted", bootstyle="info").grid(row=1, column=i+1, padx=5, pady=5, sticky="w")
            self.var_restriction_vars_goal.append(var)

        # Normal constraint coefficients
        self.normal_constraint_entries = []
        self.normal_rhs_entries = []
        self.normal_constraint_type_vars = []
        for i in range(num_constraints):
            ttk.Label(self.normal_constraints_frame, text=f"Normal Constraint {i+1}:").grid(row=i+2, column=0, padx=10, pady=5, sticky="w")
            constraint_entries_row = []
            for j in range(num_vars):
                entry = ttk.Entry(self.normal_constraints_frame, width=10, bootstyle="primary")
                entry.grid(row=i+2, column=j+1, padx=5, pady=5, sticky="w")
                constraint_entries_row.append(entry)
            self.normal_constraint_entries.append(constraint_entries_row)

            # RHS
            rhs_entry = ttk.Entry(self.normal_constraints_frame, width=10, bootstyle="primary")
            rhs_entry.grid(row=i+2, column=num_vars+1, padx=5, pady=5, sticky="w")
            self.normal_rhs_entries.append(rhs_entry)

            # Constraint type
            constraint_type_var = tk.StringVar(value="<=")
            ttk.OptionMenu(self.normal_constraints_frame, constraint_type_var, "<=", ">=", "=", bootstyle="info").grid(row=i+2, column=num_vars+2, padx=5, pady=5, sticky="w")
            self.normal_constraint_type_vars.append(constraint_type_var)

        # Goal constraints
        self.goal_constraint_entries = []
        self.goal_rhs_entries = []
        self.goal_constraint_type_vars = []
        self.goal_priority_entries = []
        for i in range(num_goals):
            ttk.Label(self.goals_frame, text=f"Goal {i+1}:").grid(row=i, column=0, padx=10, pady=5, sticky="w")
            goal_entries_row = []
            for j in range(num_vars):
                entry = ttk.Entry(self.goals_frame, width=10, bootstyle="primary")
                entry.grid(row=i, column=j+1, padx=5, pady=5, sticky="w")
                goal_entries_row.append(entry)
            self.goal_constraint_entries.append(goal_entries_row)

            # RHS (goal target)
            rhs_entry = ttk.Entry(self.goals_frame, width=10, bootstyle="primary")
            rhs_entry.grid(row=i, column=num_vars+1, padx=5, pady=5, sticky="w")
            self.goal_rhs_entries.append(rhs_entry)

            # Constraint type
            constraint_type_var = tk.StringVar(value="<=")
            ttk.OptionMenu(self.goals_frame, constraint_type_var, "<=", ">=", "=", bootstyle="info").grid(row=i, column=num_vars+2, padx=5, pady=5, sticky="w")
            self.goal_constraint_type_vars.append(constraint_type_var)

            # Priority
            ttk.Label(self.goals_frame, text="Priority:").grid(row=i, column=num_vars+3, padx=5, pady=5, sticky="w")
            priority_entry = ttk.Entry(self.goals_frame, width=10, bootstyle="primary")
            priority_entry.grid(row=i, column=num_vars+4, padx=5, pady=5, sticky="w")
            self.goal_priority_entries.append(priority_entry)

    def solve_goal(self):
        try:
            # Get input values
            num_vars = self.num_vars.get()
            num_constraints = self.num_constraints.get()
            num_goals = self.num_goals.get()

            # Objective function coefficients (usually zeros in goal programming)
            c = [float(entry.get()) for entry in self.obj_coeff_entries_goal]
            
            # Normal constraints
            A = [[float(entry.get()) for entry in row] for row in self.normal_constraint_entries]
            b = [float(entry.get()) for entry in self.normal_rhs_entries]
            normal_constraint_types = [var.get() for var in self.normal_constraint_type_vars]
            
            # Goal constraints
            A_goals = [[float(entry.get()) for entry in row] for row in self.goal_constraint_entries]
            goals = [float(entry.get()) for entry in self.goal_rhs_entries]
            goal_constraint_types = [var.get() for var in self.goal_constraint_type_vars]
            priority = [float(entry.get()) for entry in self.goal_priority_entries]
            
            # Variable restrictions
            variable_restrictions = [var.get() for var in self.var_restriction_vars_goal]

            # Solve using GoalSolver
            solver = GoalSolver(c, A, A_goals, b, goals, priority, 
                            normal_constraint_types, goal_constraint_types,
                            variable_restrictions)
            
            solver.solve(display_steps=True)
            results = solver.get_results()

            # Display results
            result_text = f"Optimal Solution: {results['optimal_solution']}\n"
            result_text += f"Optimal Value: {results['optimal_value']}\n"
            result_text += f"Status: {results['status']}\n"
            result_text += "Goal Status:\n"
            for status in results['goal_status']:
                result_text += f"- {status}\n"

            # Read and display steps from the file
            self.display_steps("Goal.txt")
            
            # Show results in messagebox
            messagebox.showinfo("Goal Programming Results", result_text)

        except Exception as e:
            messagebox.showerror("Error", str(e))
    


    def create_page3_widgets(self):
        """Create widgets for the graphical solution page"""
        # Input frame
        input_frame = ttk.Labelframe(self.page3, text="Problem Input", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Problem type selection
        ttk.Label(input_frame, text="Problem Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.graph_problem_type = tk.StringVar(value="max")
        ttk.Radiobutton(input_frame, text="Maximize", variable=self.graph_problem_type, value="max").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(input_frame, text="Minimize", variable=self.graph_problem_type, value="min").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Objective function coefficients
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

        # Constraint entries (we'll start with 2 constraints)
        self.constraint_entries = []
        for i in range(2):
            frame = ttk.Frame(constraints_frame)
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
            ttk.Label(frame, text="").pack(side=tk.LEFT, padx=2)
            entry = ttk.Entry(frame, width=5)
            entry.pack(side=tk.LEFT, padx=2)
            entries.append(entry)
            
            self.constraint_entries.append((entries, constr_type))

        # Button to add more constraints
        ttk.Button(constraints_frame, text="Add Constraint", command=self.add_constraint_field).pack(pady=5)

        # Plot button
        ttk.Button(self.page3, text="Plot Solution", command=self.plot_solution, bootstyle="primary").pack(pady=10)

        # Canvas for matplotlib figure
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
        ttk.Label(frame, text="").pack(side=tk.LEFT, padx=2)
        entry = ttk.Entry(frame, width=5)
        entry.pack(side=tk.LEFT, padx=2)
        entries.append(entry)
        
        self.constraint_entries.append((entries, constr_type))

    def plot_solution(self):
        """Plot the graphical solution"""
        try:
            # Get objective function coefficients
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
                    
                    # Convert constraint type symbol to standard form
                    ct = constr_type.get()
                    if ct == "≤":
                        constraint_types.append("<=")
                    elif ct == "≥":
                        constraint_types.append(">=")
                    else:
                        constraint_types.append("=")
                except ValueError:
                    continue  # Skip incomplete constraints
            
            if not A:
                raise ValueError("Please enter at least one valid constraint")
            
            # Get problem parameters
            variable_restrictions = ['non-negative', 'non-negative']
            problem_type = self.graph_problem_type.get()
            
            # Clear previous plot
            self.ax.clear()
            
            # Create and plot the LP problem
            plotter = LPPlotter(c, A, b, constraint_types, variable_restrictions, problem_type)
            plotter.ax = self.ax  # Use our axis
            plotter.plot()
            
            # Redraw canvas
            self.canvas.draw()
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")  # Use a modern theme
    app = LPApp(root)
    root.mainloop()
