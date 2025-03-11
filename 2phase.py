import numpy as np
import pandas as pd
from tabulate import tabulate

class TwoPhaseSolver:
    def __init__(self, c, A, b, constraint_types, variable_restrictions, problem_type='max'):
        """
        Initialize the Two-Phase Solver.
        
        :param c: Coefficients of the objective function (1D array).
        :param A: Constraint coefficients (2D array).
        :param b: Right-hand side values (1D array).
        :param constraint_types: List of constraint types ('<=', '>=', '=').
        :param variable_restrictions: List of variable restrictions ('non-negative', 'unrestricted').
        :param problem_type: Type of problem ('max' for maximization, 'min' for minimization).
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.constraint_types = constraint_types
        self.variable_restrictions = variable_restrictions
        self.problem_type = problem_type
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self.tableau = None
        self.basis = None
        self.optimal_solution = None
        self.optimal_value = None
        self.status = None

    def initialize_tableau_phase1(self):
        """
        Initialize the tableau for Phase 1 of the Two-Phase Method.
        """
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

        # Fill the objective function row for Phase 1 (minimize sum of artificial variables)
        self.tableau[-1, self.num_vars + slack_vars:total_vars] = 1

        # Initialize basis
        self.basis = list(range(self.num_vars + slack_vars, total_vars))

    def initialize_tableau_phase2(self, phase1_tableau):
        """
        Initialize the tableau for Phase 2 using the results from Phase 1.
        """
        # Remove artificial variables and restore the original objective function
        self.tableau = np.delete(phase1_tableau, self.basis, axis=1)
        self.tableau[-1, :self.num_vars] = -self.c if self.problem_type == 'max' else self.c
        self.tableau[-1, -1] = 0

    def solve_phase1(self, display_steps=False):
        """
        Solve Phase 1 of the Two-Phase Method.
        """
        self.initialize_tableau_phase1()
        iteration = 0

        while True:
            if display_steps:
                print(f"\nPhase 1 - Iteration {iteration}:")
                self.display_tableau()

            # Check for optimality
            if all(self.tableau[-1, :-1] >= 0):
                if np.isclose(self.tableau[-1, -1], 0):
                    self.status = 'feasible'
                    print("\nFeasible solution found in Phase 1.")
                    break
                else:
                    self.status = 'infeasible'
                    print("\nThe problem is infeasible.")
                    return False

            # Select entering variable (most negative coefficient in the objective row)
            entering_var = np.argmin(self.tableau[-1, :-1])

            # Check for unboundedness
            if all(self.tableau[:-1, entering_var] <= 0):
                self.status = 'unbounded'
                print("\nThe problem is unbounded.")
                return False

            # Select leaving variable (minimum ratio test)
            ratios = []
            for i in range(self.num_constraints):
                if self.tableau[i, entering_var] > 0:
                    ratios.append(self.tableau[i, -1] / self.tableau[i, entering_var])
                else:
                    ratios.append(np.inf)
            leaving_var = np.argmin(ratios)

            # Pivot
            pivot_element = self.tableau[leaving_var, entering_var]
            self.tableau[leaving_var, :] /= pivot_element
            for i in range(self.num_constraints + 1):
                if i != leaving_var:
                    factor = self.tableau[i, entering_var]
                    self.tableau[i, :] -= factor * self.tableau[leaving_var, :]

            # Update basis
            self.basis[leaving_var] = entering_var
            iteration += 1

        return True

    def solve_phase2(self, display_steps=False):
        """
        Solve Phase 2 of the Two-Phase Method.
        """
        self.initialize_tableau_phase2(self.tableau)
        iteration = 0

        while True:
            if display_steps:
                print(f"\nPhase 2 - Iteration {iteration}:")
                self.display_tableau()

            # Check for optimality
            if all(self.tableau[-1, :-1] >= 0):
                self.status = 'optimal'
                print("\nOptimal solution reached in Phase 2.")
                break

            # Select entering variable (most negative coefficient in the objective row)
            entering_var = np.argmin(self.tableau[-1, :-1])

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

            # Pivot
            pivot_element = self.tableau[leaving_var, entering_var]
            self.tableau[leaving_var, :] /= pivot_element
            for i in range(self.num_constraints + 1):
                if i != leaving_var:
                    factor = self.tableau[i, entering_var]
                    self.tableau[i, :] -= factor * self.tableau[leaving_var, :]

            # Update basis
            self.basis[leaving_var] = entering_var
            iteration += 1

        # Extract solution
        self.optimal_solution = np.zeros(self.num_vars)
        for i in range(self.num_constraints):
            if self.basis[i] < self.num_vars:
                self.optimal_solution[self.basis[i]] = self.tableau[i, -1]
        self.optimal_value = self.tableau[-1, -1]

        # Adjust the optimal value for minimization problems
        if self.problem_type == 'min':
            self.optimal_value = -self.optimal_value

    def solve(self, display_steps=False):
        """
        Solve the LP problem using the Two-Phase Method.
        """
        if not self.solve_phase1(display_steps):
            return

        self.solve_phase2(display_steps)

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
    c = [1, 2, 1]
    A = [[1, 1, 1], [2, -5, 1]]
    b = [7, 10]
    constraint_types = ['=', '>=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
    problem_type = 'max'

    solver = TwoPhaseSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])
