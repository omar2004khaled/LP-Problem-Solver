import tkinter as tk
from tkinter import messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from simplex import SimplexSolver  # Import SimplexSolver from simplex.py
from bigM import BigMSolver  # Import BigMSolver from bigM.py
from twophase import TwoPhaseSimplexSolver  # Import TwoPhaseSimplexSolver from two_phase.py
import numpy as np
import pandas as pd
from tabulate import tabulate

class LPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Programming Solver")
        self.root.geometry("1000x700")
        self.style = ttk.Style(theme="cosmo")  # Use a modern theme (e.g., "cosmo", "minty", "flatly")

        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Method selection
        self.method_var = tk.StringVar(value="simplex")
        self.problem_type_var = tk.StringVar(value="max")
        self.num_vars = tk.IntVar(value=2)
        self.num_constraints = tk.IntVar(value=2)

        self.create_widgets()

    def create_widgets(self):
        # Input section
        input_frame = ttk.Labelframe(self.main_frame, text="Input Parameters", padding=10)
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
        self.coeff_frame = ttk.Labelframe(self.main_frame, text="Coefficients", padding=10)
        self.coeff_frame.pack(fill=tk.X, padx=10, pady=10)

        # Solve button
        ttk.Button(self.main_frame, text="Solve", command=self.solve, bootstyle="primary").pack(pady=10)

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

        # Constraint coefficients
        self.constraint_entries = []
        self.rhs_entries = []
        self.constraint_type_vars = []
        for i in range(num_constraints):
            ttk.Label(self.coeff_frame, text=f"Constraint {i+1}:").grid(row=i+1, column=0, padx=10, pady=10, sticky="w")
            constraint_entries_row = []
            for j in range(num_vars):
                entry = ttk.Entry(self.coeff_frame, width=10, bootstyle="primary")
                entry.grid(row=i+1, column=j+1, padx=5, pady=5, sticky="w")
                constraint_entries_row.append(entry)
            self.constraint_entries.append(constraint_entries_row)

            # RHS
            rhs_entry = ttk.Entry(self.coeff_frame, width=10, bootstyle="primary")
            rhs_entry.grid(row=i+1, column=num_vars+1, padx=5, pady=5, sticky="w")
            self.rhs_entries.append(rhs_entry)

            # Constraint type
            constraint_type_var = tk.StringVar(value="<=")
            ttk.OptionMenu(self.coeff_frame, constraint_type_var, "<=", ">=", "=", bootstyle="info").grid(row=i+1, column=num_vars+2, padx=5, pady=5, sticky="w")
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
            variable_restrictions = ['non-negative'] * num_vars
            problem_type = self.problem_type_var.get()

            # Solve using the selected method
            if self.method_var.get() == "simplex":
                solver = SimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
            elif self.method_var.get() == "bigm":
                solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
            elif self.method_var.get() == "two_phase":
                solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)

            solver.solve(display_steps=True)
            results = solver.get_results()

            # Display results
            messagebox.showinfo("Results", f"Optimal Solution: {results['optimal_solution']}\nOptimal Value: {results['optimal_value']}\nStatus: {results['status']}")

            # Save steps to a file
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, "w") as f:
                    f.write(f"Optimal Solution: {results['optimal_solution']}\n")
                    f.write(f"Optimal Value: {results['optimal_value']}\n")
                    f.write(f"Status: {results['status']}\n")
                    f.write("\nSteps:\n")
                    for i in range(solver.num_constraints + 1):
                        f.write(tabulate(solver.tableau, headers=[f'x{j+1}' for j in range(solver.num_vars)] + [f's{j+1}' for j in range(solver.tableau.shape[1] - solver.num_vars - 1)] + ['RHS'], tablefmt='grid', floatfmt='.2f') + "\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")  # Use a modern theme
    app = LPApp(root)
    root.mainloop()
