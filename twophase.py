import numpy as np
import pandas as pd
from tabulate import tabulate

class TwoPhaseSimplexSolver:
    def __init__(self, c, A, b, constraint_types, variable_restrictions, problem_type='max', M=1e6):
        """
        Initialize the Two-Phase Simplex Solver.
        
        :param c: Coefficients of the objective function (1D array).
        :param A: Constraint coefficients (2D array).
        :param b: Right-hand side values (1D array).
        :param constraint_types: List of constraint types ('<=', '>=', '=').
        :param variable_restrictions: List of variable restrictions ('non-negative', 'unrestricted').
        :param problem_type: Type of problem ('max' for maximization, 'min' for minimization).
        :param M: Large penalty value for artificial variables (default: 1e6).
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.constraint_types = constraint_types
        self.variable_restrictions = variable_restrictions
        self.problem_type = problem_type
        self.M = M
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self.tableau = None
        self.basis = None
        self.optimal_solution = None
        self.optimal_value = None
        self.status = None

    def initialize_tableau(self):
        """
        Initialize the tableau for the Two-Phase Simplex method.
        """
        # Convert minimization problem to maximization
        if self.problem_type == 'min':
            self.c = -self.c

        # Add slack, surplus, and artificial variables based on constraint types
        slack_vars = 0
        artificial_vars = 0
        for constraint in self.constraint_types:
            if constraint == '<=':
                slack_vars += 1
            elif constraint == '>=':
                slack_vars += 1
                artificial_vars += 1
            elif constraint == '=':
                artificial_vars += 1

        # Initialize the tableau
        total_vars = self.num_vars + slack_vars + artificial_vars
        self.tableau = np.zeros((self.num_constraints + 1, total_vars + 1))

        # Fill the tableau with constraint coefficients
        slack_index = self.num_vars
        artificial_index = self.num_vars + slack_vars
        for i in range(self.num_constraints):
            self.tableau[i, :self.num_vars] = self.A[i]
            if self.constraint_types[i] == '<=':
                self.tableau[i, slack_index] = 1
                slack_index += 1
            elif self.constraint_types[i] == '>=':
                self.tableau[i, slack_index] = -1
                self.tableau[i, artificial_index] = 1
                slack_index += 1
                artificial_index += 1
            elif self.constraint_types[i] == '=':
                self.tableau[i, artificial_index] = 1
                artificial_index += 1
            self.tableau[i, -1] = self.b[i]

        # Initialize basis
        self.basis = list(range(self.num_vars, self.num_vars + self.num_constraints))

    def phase1(self):
        """
        Perform Phase 1 of the Two-Phase Simplex method.
        """
        # Create a new objective function for Phase 1: minimize the sum of artificial variables
        phase1_obj = np.zeros(self.tableau.shape[1])
        for i in range(self.num_vars, self.tableau.shape[1] - 1):
            if i >= self.num_vars + (self.tableau.shape[1] - self.num_vars - 1 - self.num_constraints):
                phase1_obj[i] = -self.M  # Use -M for minimization

        # Replace the original objective function with the Phase 1 objective
        original_obj = self.tableau[-1, :].copy()
        self.tableau[-1, :] = phase1_obj

        # Perform row operations to eliminate artificial variables from the Z-row
        for i in range(self.num_constraints):
            if self.basis[i] >= self.num_vars:  # If the basis variable is an artificial variable
                self.tableau[-1, :] -= self.tableau[-1, self.basis[i]] * self.tableau[i, :]

        # Solve the Phase 1 problem (minimization)
        self.solve_simplex(phase='Phase 1', is_minimization=True)

        # Check if the optimal value of Phase 1 is zero (feasible solution found)
        if not np.isclose(self.tableau[-1, -1], 0):
            self.status = 'infeasible'
            return False

        # Restore the original objective function
        self.tableau[-1, :] = original_obj

        # Remove artificial variables from the tableau and basis
        self.remove_artificial_variables()
        return True

    def remove_artificial_variables(self):
        """
        Remove artificial variables from the tableau and basis.
        """
        # Identify columns corresponding to artificial variables
        artificial_cols = [col for col in range(self.num_vars, self.tableau.shape[1] - 1) if col >= self.num_vars + (self.tableau.shape[1] - self.num_vars - 1 - self.num_constraints)]

        # Remove artificial variable columns from the tableau
        self.tableau = np.delete(self.tableau, artificial_cols, axis=1)

        # Update basis to remove artificial variables
        self.basis = [var for var in self.basis if var < self.num_vars + (self.tableau.shape[1] - self.num_vars - 1)]

    def update_z_row_for_phase2(self):
        """
        Update the Z-row for Phase 2 to reflect the original objective function.
        """
        # Set the Z-row to the original objective function coefficients
        self.tableau[-1, :self.num_vars] = -self.c  # Negative because we are maximizing

        # Perform row operations to eliminate basic variables from the Z-row
        for i in range(self.num_constraints):
            if self.basis[i] < self.num_vars:  # If the basis variable is a decision variable
                self.tableau[-1, :] -= self.tableau[-1, self.basis[i]] * self.tableau[i, :]

    def solve_simplex(self, phase='Phase 2', is_minimization=False):
        """
        Solve the LP problem using the Simplex method.
        
        :param phase: The current phase ('Phase 1' or 'Phase 2').
        :param is_minimization: Whether the problem is a minimization problem.
        """
        iteration = 0

        # Update the Z-row for Phase 2 if we are in Phase 2
        if phase == 'Phase 2':
            self.update_z_row_for_phase2()

        # Display the initial tableau (Iteration 0)
        print(f"\n{phase} - Iteration {iteration}:")
        self.display_tableau()

        while True:
            # Check for optimality
            if is_minimization:
                if all(self.tableau[-1, :-1] <= 0):  # Minimization: all coefficients in Z-row should be <= 0
                    self.status = 'optimal'
                    print(f"\n{phase} - Optimal solution reached.")
                    break
            else:
                if all(self.tableau[-1, :-1] >= 0):  # Maximization: all coefficients in Z-row should be >= 0
                    self.status = 'optimal'
                    print(f"\n{phase} - Optimal solution reached.")
                    break

            # Select entering variable
            if is_minimization:
                entering_var = np.argmax(self.tableau[-1, :-1])  # Minimization: choose the most positive coefficient
            else:
                entering_var = np.argmin(self.tableau[-1, :-1])  # Maximization: choose the most negative coefficient
            print(f"\nEntering variable: x{entering_var + 1}, because it has the most {'positive' if is_minimization else 'negative'} coefficient {self.tableau[-1, entering_var]:.2f} in the Z-row.")

            # Check for unboundedness
            if all(self.tableau[:-1, entering_var] <= 0):
                self.status = 'unbounded'
                print("\nThe problem is unbounded.")
                break

            # Select leaving variable (minimum ratio test)
            ratios = []
            for i in range(self.num_constraints):
                if self.tableau[i, entering_var] > 0:
                    ratios.append(self.tableau[i, -1] / self.tableau[i, entering_var])
                else:
                    ratios.append(np.inf)
            leaving_var = np.argmin(ratios)
            print(f"Leaving variable: s{leaving_var + 1}, because it has the smallest ratio {ratios[leaving_var]:.2f}.")

            # Pivot
            pivot_element = self.tableau[leaving_var, entering_var]
            print(f"\nPivot element: {pivot_element:.2f} at row {leaving_var + 1}, column {entering_var + 1}.")
            self.tableau[leaving_var, :] /= pivot_element
            print(f"Divide row {leaving_var + 1} by {pivot_element:.2f} to make the pivot element 1.")
            for i in range(self.num_constraints + 1):
                if i != leaving_var:
                    factor = self.tableau[i, entering_var]
                    self.tableau[i, :] -= factor * self.tableau[leaving_var, :]
                    print(f"Subtract {factor:.2f} times row {leaving_var + 1} from row {i + 1} to eliminate the entering variable in other rows.")

            # Update basis
            self.basis[leaving_var] = entering_var
            print(f"Update basis: x{entering_var + 1} enters the basis, s{leaving_var + 1} leaves the basis.")

            iteration += 1

            # Display the updated tableau
            print(f"\n{phase} - Iteration {iteration}:")
            self.display_tableau()

    def solve(self, display_steps=False):
        """
        Solve the LP problem using the Two-Phase Simplex method.
        
        :param display_steps: If True, display the tableau at each iteration in a fancy way.
        """
        self.initialize_tableau()

        # Phase 1
        if not self.phase1():
            print("\nPhase 1 - No feasible solution found.")
            return

        # Phase 2
        self.solve_simplex(phase='Phase 2', is_minimization=(self.problem_type == 'min'))

        # Extract solution
        self.optimal_solution = np.zeros(self.num_vars)
        for i in range(self.num_constraints):
            if self.basis[i] < self.num_vars:
                self.optimal_solution[self.basis[i]] = self.tableau[i, -1]
        self.optimal_value = self.tableau[-1, -1]

        # Adjust the optimal value for minimization problems
        if self.problem_type == 'min':
            self.optimal_value = -self.optimal_value

    def display_tableau(self):
        """
        Display the current tableau in a fancy way using tabulate.
        """
        headers = [f'x{i + 1}' for i in range(self.num_vars)] + [f's{i + 1}' for i in range(self.tableau.shape[1] - self.num_vars - 1)] + ['RHS']
        print(tabulate(self.tableau, headers=headers, tablefmt='grid', floatfmt='.2f'))

    def get_results(self):
        """
        Return the results of the optimization.
        """
        return {
            'optimal_solution': self.optimal_solution,
            'optimal_value': self.optimal_value,
            'status': self.status
        }


if __name__ == '__main__':
    # Problem setup
    c = [2, 3, 4]
    A = [[3, 2, 1], [2, 3, 3], [1, 1, -1]]
    b = [10, 15, 4]
    constraint_types = ['<=', '<=', '>=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
    problem_type = 'max'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])
'''

    
if __name__ == '__main__':
    # Problem setup
    c = [1, 2, 1]
    A = [[1, 1, 1], [2, -5, 1]]
    b = [7, 10]
    constraint_types = ['=', '>=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
    problem_type = 'max'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])'
    '''
