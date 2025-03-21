import numpy as np
import pandas as pd
from tabulate import tabulate

class TwoPhaseSimplexSolver:
    def __init__(self, c, A, b, constraint_types, variable_restrictions, problem_type='max', M=1e6):
        
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
        self.num_slack = 0
        self.num_artificial = 0
        self.num_surplus = 0
        self.tableau_rows = None
        self.headers = None
        self.temp = 0

    def initialize_tableau(self):
        
        for constraint in self.constraint_types:
            if constraint == '<=':
                self.num_slack += 1
            elif constraint == '>=':
                self.num_surplus += 1
                self.num_artificial += 1
            elif constraint == '=':
                self.num_artificial += 1

        total_vars = self.num_vars + self.num_slack + self.num_artificial+self.num_surplus
        self.tableau = np.zeros((self.num_artificial + self.num_slack + 1, total_vars + 1))

        slack_index = self.num_vars
        artificial_index = self.num_vars + self.num_slack 
        surplus_index = self.num_vars + self.num_slack +self.num_artificial
        

        for i in range(self.num_constraints):
            self.tableau[i, :self.num_vars] = self.A[i]
            if self.constraint_types[i] == '<=':
                self.tableau[i, slack_index] = 1
                slack_index += 1
            elif self.constraint_types[i] == '>=':
                self.tableau[i, surplus_index] = -1
                self.tableau[i, artificial_index] = 1
                surplus_index += 1
                artificial_index += 1
            elif self.constraint_types[i] == '=':
                self.tableau[i, artificial_index] = 1
                artificial_index += 1

            self.tableau[i, -1] = self.b[i]
        self.temp = self.num_artificial
        self.basis = list(range(self.num_vars, self.num_vars + self.num_slack+self.num_artificial))

    def phase1(self):                    #Always minmization
        phase1_obj = np.zeros(self.tableau.shape[1])
        artificial_start = self.num_vars + self.num_slack
        artificial_end = artificial_start + self.num_artificial
        
        for i in range(artificial_start, artificial_end):
            phase1_obj[i] = -1 
    
        self.tableau[-1, :] = phase1_obj
        
        self.make_consistent()
        self.solve_simplex(phase='Phase 1')
        

        for i in range(self.num_slack+self.num_artificial):
            if self.basis[i] >= artificial_start and self.basis[i] <= artificial_end :
                return False
            
        for i in range(self.num_slack+self.num_artificial):
            if self.basis[i] >= artificial_start and self.basis[i] <= artificial_end :
                if self.tableau[i,-1] != 0 :
                    return False
        """ original_obj = np.zeros_like(self.tableau[-1, :])
        original_obj[:len(self.c)] = self.c 
        self.tableau[-1, :] = original_obj"""
        
        self.remove_artificial_variables()
        self.update_z_row_for_phase2()
        return True

    def remove_artificial_variables(self):
        artificial_start = self.num_vars + self.num_slack
        artificial_end = artificial_start + self.num_artificial
        artificial_cols = list(range(artificial_start, artificial_end))
        self.tableau = np.delete(self.tableau, artificial_cols, axis=1)
        self.basis = [var - self.num_artificial if var >= artificial_start else var for var in self.basis]
        self.num_artificial = 0

    def update_z_row_for_phase2(self):
        self.tableau[-1, :self.num_vars] = -self.c 
        print('Due to inconsistency :')
        self.make_consistent2()

    def make_consistent(self):  
            obj_row = self.num_artificial + self.num_slack
            for i in range(self.num_artificial):
              self.tableau[-1,] = self.tableau[-1,] + 1*self.tableau[obj_row-i-1]    

    def make_consistent2(self) :
        for i in range(self.temp + self.num_slack):
            basic_var = self.basis[i]  # Get the basic variable index
            coeff_in_obj = self.tableau[-1, basic_var]  # Get its coefficient in the objective function

            if coeff_in_obj != 0:  # If the variable appears in Z with a nonzero coefficient
                self.tableau[-1, :] -= coeff_in_obj * self.tableau[i, :]

    def solve_simplex(self, phase):
        iteration = 0

        if phase == 'Phase 1' :
            problemtype = 'min'
        else:
            problemtype = self.problem_type      

        print(f"\n{phase} - Iteration {iteration}:")
        self.display_tableau()

        while True:
            if problemtype == 'min':
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
            if problemtype == 'min':
                entering_var = np.argmax(self.tableau[-1, :-1])  # Minimization: choose the most positive coefficient
            else:
                entering_var = np.argmin(self.tableau[-1, :-1])  # Maximization: choose the most negative coefficient
            print(f"\nEntering variable: x{entering_var + 1}, because it has the most {'positive' if problemtype == 'min' else 'negative'} coefficient {self.tableau[-1, entering_var]:.2f} in the Z-row.")

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
        self.initialize_tableau()

        # Phase 1
        if not self.phase1():
            print("\nPhase 1 - No feasible solution found.")
            return

        # Phase 2
        self.solve_simplex(phase='Phase 2')

        # Extract solution
        self.optimal_solution = np.zeros(self.num_vars)
        for i in range(self.num_constraints):
            if self.basis[i] < self.num_vars:
                self.optimal_solution[self.basis[i]] = self.tableau[i, -1]
        self.optimal_value = self.tableau[-1, -1]

        

    def display_tableau(self):
        self.headers = ['Basic'] + [f'x{i + 1}' for i in range(self.num_vars)]
        self.headers += [f's{i + 1}' for i in range(self.num_slack)]
        self.headers += [f'A{i + 1}' for i in range(self.num_artificial)]
        self.headers += [f'e{i + 1}' for i in range(self.num_surplus)]
        self.headers.append('RHS')

        self.tableau_rows = []
        for i in range(self.num_slack+self.temp):
            if self.basis[i] < self.num_vars:
                basic_var = f'x{self.basis[i] + 1}'  
            elif self.basis[i] < self.num_vars + self.num_slack:
                basic_var = f's{self.basis[i] - self.num_vars + 1}'  
            elif self.basis[i] < self.num_vars + self.num_slack + self.num_artificial:
                basic_var = f'A{self.basis[i] - self.num_vars - self.num_slack + 1}'  
            else:
                basic_var = f'e{self.basis[i] - self.num_vars - self.num_slack - self.num_artificial + 1}'  

            row = [basic_var] + list(self.tableau[i, :])
            self.tableau_rows.append(row)    

        objective_row = ['Z'] + list(self.tableau[-1, :])
        self.tableau_rows.append(objective_row)
            
        print(tabulate(self.tableau_rows, headers=self.headers, tablefmt='grid', floatfmt='.2f'))   
        

    def get_results(self):
        return {
            'optimal_solution': self.optimal_solution,
            'optimal_value': self.optimal_value,
            'status': self.status
        }


"""if __name__ == '__main__':
    # Problem setup
    c = [1,2,1]
    A = [[1,1,1], [2,-5,1]]
    b = [7,10]
    constraint_types = ['=', '>=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
    problem_type = 'max'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])"""


    
if __name__ == '__main__':
    # Problem setup
    c = [1,1]
    A = [[2,1], [1,7]]
    b = [4,7]
    constraint_types = ['>=', '>=']
    variable_restrictions = ['non-negative', 'non-negative']
    problem_type = 'min'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])
    
