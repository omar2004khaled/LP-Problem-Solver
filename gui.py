import tkinter as tk
from tkinter import messagebox, filedialog
from simplex import SimplexSolver  # Import SimplexSolver from simplex.py
from bigM import BigMSolver  # Import BigMSolver from bigM.py
import numpy as np
import pandas as pd
from tabulate import tabulate

class LPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Programming Solver")
        self.root.geometry("800x600")

        self.method_var = tk.StringVar(value="simplex")
        self.problem_type_var = tk.StringVar(value="max")
        self.num_vars = tk.IntVar(value=2)
        self.num_constraints = tk.IntVar(value=2)

        self.create_widgets()

    def create_widgets(self):
        # Method selection
        tk.Label(self.root, text="Select Method:").grid(row=0, column=0, padx=10, pady=10)
        tk.Radiobutton(self.root, text="Simplex", variable=self.method_var, value="simplex").grid(row=0, column=1, padx=10, pady=10)
        tk.Radiobutton(self.root, text="Big M", variable=self.method_var, value="bigm").grid(row=0, column=2, padx=10, pady=10)

        # Problem type selection
        tk.Label(self.root, text="Problem Type:").grid(row=1, column=0, padx=10, pady=10)
        tk.Radiobutton(self.root, text="Maximization", variable=self.problem_type_var, value="max").grid(row=1, column=1, padx=10, pady=10)
        tk.Radiobutton(self.root, text="Minimization", variable=self.problem_type_var, value="min").grid(row=1, column=2, padx=10, pady=10)

        # Number of variables and constraints
        tk.Label(self.root, text="Number of Variables:").grid(row=2, column=0, padx=10, pady=10)
        self.num_vars_entry = tk.Entry(self.root, textvariable=self.num_vars)
        self.num_vars_entry.grid(row=2, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Number of Constraints:").grid(row=3, column=0, padx=10, pady=10)
        self.num_constraints_entry = tk.Entry(self.root, textvariable=self.num_constraints)
        self.num_constraints_entry.grid(row=3, column=1, padx=10, pady=10)

        # Button to update input fields
        tk.Button(self.root, text="Update Input Fields", command=self.update_input_fields).grid(row=4, column=0, columnspan=3, padx=10, pady=10)

        # Input fields for coefficients
        self.coeff_frame = tk.Frame(self.root)
        self.coeff_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

        # Solve button
        tk.Button(self.root, text="Solve", command=self.solve).grid(row=6, column=0, columnspan=3, padx=10, pady=10)

        # Initialize input fields
        self.update_input_fields()

    def update_input_fields(self):
        # Clear previous input fields
        for widget in self.coeff_frame.winfo_children():
            widget.destroy()

        num_vars = self.num_vars.get()
        num_constraints = self.num_constraints.get()

        # Objective function coefficients
        tk.Label(self.coeff_frame, text="Objective Function Coefficients:").grid(row=0, column=0, padx=10, pady=10)
        self.obj_coeff_entries = []
        for i in range(num_vars):
            entry = tk.Entry(self.coeff_frame, width=10)
            entry.grid(row=0, column=i+1, padx=5, pady=5)
            self.obj_coeff_entries.append(entry)

        # Constraint coefficients
        self.constraint_entries = []
        self.rhs_entries = []
        self.constraint_type_vars = []
        for i in range(num_constraints):
            tk.Label(self.coeff_frame, text=f"Constraint {i+1}:").grid(row=i+1, column=0, padx=10, pady=10)
            constraint_entries_row = []
            for j in range(num_vars):
                entry = tk.Entry(self.coeff_frame, width=10)
                entry.grid(row=i+1, column=j+1, padx=5, pady=5)
                constraint_entries_row.append(entry)
            self.constraint_entries.append(constraint_entries_row)

            # RHS
            rhs_entry = tk.Entry(self.coeff_frame, width=10)
            rhs_entry.grid(row=i+1, column=num_vars+1, padx=5, pady=5)
            self.rhs_entries.append(rhs_entry)

            # Constraint type
            constraint_type_var = tk.StringVar(value="<=")
            tk.OptionMenu(self.coeff_frame, constraint_type_var, "<=", ">=", "=").grid(row=i+1, column=num_vars+2, padx=5, pady=5)
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
            else:
                solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)

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
    root = tk.Tk()
    app = LPApp(root)
    root.mainloop()
