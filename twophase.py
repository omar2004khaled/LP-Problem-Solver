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
        slack_rows = []
        artificial_rows = []
        artificial_with_surplus_rows =[]
        

        for constraint in self.constraint_types:
            if constraint == '<=':
                self.num_slack += 1
            elif constraint == '>=':
                self.num_surplus += 1
                self.num_artificial += 1
            elif constraint == '=':
                self.num_artificial += 1

        for i in range(self.num_constraints):
            if self.constraint_types[i] == '<=':
                slack_rows.append(i)               
            elif self.constraint_types[i] == '>=':
                artificial_with_surplus_rows.append(i)
            elif self.constraint_types[i] == '=':
                artificial_rows.append(i)
                            
        
        total_vars = self.num_vars + self.num_slack + self.num_artificial + self.num_surplus
        self.tableau = np.zeros((self.num_artificial + self.num_slack + 1, total_vars + 1))

        slack_index = self.num_vars
        artificial_index = self.num_vars + self.num_slack
        surplus_index = self.num_vars + self.num_slack + self.num_artificial
        index = 0
        """
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
            """
        for i in slack_rows:
            self.tableau[index, :self.num_vars] = self.A[i]
            self.tableau[index, slack_index] = 1  # Slack variable
            slack_index += 1
            self.tableau[index, -1] = self.b[i]
            index+=1
        for i in artificial_rows:
            self.tableau[index, :self.num_vars] = self.A[i]
            self.tableau[index, artificial_index] = 1  # Artificial variable
            self.tableau[index, -1] = self.b[i]
            artificial_index += 1
            index+=1  
        for i in artificial_with_surplus_rows:
            self.tableau[index, :self.num_vars] = self.A[i]
            self.tableau[index , surplus_index] = -1
            self.tableau[index, artificial_index] = 1
            self.tableau[index, -1] = self.b[i]
            surplus_index += 1
            artificial_index += 1
            index+=1      

        self.temp = self.num_artificial
        self.basis = list(range(self.num_vars, self.num_vars + self.num_slack + self.num_artificial))

    def phase1(self, file):
        # Phase 1: Minimize the sum of artificial variables
        phase1_obj = np.zeros(self.tableau.shape[1])
        artificial_start = self.num_vars + self.num_slack
        artificial_end = artificial_start + self.num_artificial

        for i in range(artificial_start, artificial_end):
            phase1_obj[i] = -1  # Coefficients for artificial variables in Phase 1 objective

        self.tableau[-1, :] = phase1_obj

        file.write("\nPhase 1: Initial Tableau\n")
        self.display_tableau(file)

        self.make_consistent(file)
        file.write("\nPhase 1: After Consistency Adjustments\n")
        self.display_tableau(file)

        self.solve_simplex(phase='Phase 1', file=file)


        for i in range(len(self.basis)):  # Iterate over all basis elements
         if artificial_start <= self.basis[i] < artificial_end:
          if self.tableau[i, -1] != 0:  # Check if the RHS is nonzero
            return False
        
        self.remove_artificial_variables()
        self.update_z_row_for_phase2(file)
        return True

    def remove_artificial_variables(self):
        artificial_start = self.num_vars + self.num_slack
        artificial_end = artificial_start + self.num_artificial
        artificial_cols = [col for col in range(artificial_start, artificial_end) if col not in self.basis]

        if artificial_cols:
            self.tableau = np.delete(self.tableau, artificial_cols, axis=1)

        self.basis = [var - sum(1 for col in artificial_cols if col < var) if var >= artificial_start else var for var in self.basis]

        self.num_artificial -= len(artificial_cols)

    def update_z_row_for_phase2(self, file):
        self.tableau[-1, :self.num_vars] = -self.c
        file.write("\nPhase 2: Updated Z-row for Original Objective\n")
        self.display_tableau(file)
        self.make_consistent2(file)

    def make_consistent(self, file):
        obj_row = self.num_artificial + self.num_slack
        for i in range(self.num_artificial):
            self.tableau[-1, :] = self.tableau[-1, :] + 1 * self.tableau[obj_row - i - 1, :]
        file.write("\nPhase 1: Consistency Adjustments Applied\n")

    def make_consistent2(self, file):
        for i in range(self.temp + self.num_slack):
            basic_var = self.basis[i]
            coeff_in_obj = self.tableau[-1, basic_var]

            if coeff_in_obj != 0:
                self.tableau[-1, :] -= coeff_in_obj * self.tableau[i, :]
        file.write("\nPhase 2: Consistency Adjustments Applied\n")

    def solve_simplex(self, phase, file):
        iteration = 0
        print(f"\nIteration {iteration}:")
        self.display_tableau()   
        if phase == 'Phase 1':
            problemtype = 'min'
        else:
            problemtype = self.problem_type

        file.write(f"\n{phase} - Iteration {iteration}:\n")
        self.display_tableau(file)

        while True:
            if problemtype == 'min':
                if all(self.tableau[-1, :-1] <= 0):  # Minimization: all coefficients in Z-row should be <= 0
                    self.status = 'optimal'
                    file.write(f"\n{phase} - Optimal solution reached.\n")
                    print(f"\n{phase} - Optimal solution reached.")
                    break
            else:
                if all(self.tableau[-1, :-1] >= 0):  # Maximization: all coefficients in Z-row should be >= 0
                    self.status = 'optimal'
                    file.write(f"\n{phase} - Optimal solution reached.\n")
                    print(f"\n{phase} - Optimal solution reached.")
                    break

            # Select entering variable
            if problemtype == 'min':
                entering_var = np.argmax(self.tableau[-1, :-1])  # Minimization: choose the most positive coefficient
            else:
                entering_var = np.argmin(self.tableau[-1, :-1])  # Maximization: choose the most negative coefficient

            # Determine the variable type
            if entering_var < self.num_vars:
                variable_type = f'x{entering_var + 1}'  # Decision variable
            elif entering_var < self.num_vars + self.num_slack:
                variable_type = f's{entering_var - self.num_vars + 1}'  # Slack variable
            elif entering_var < self.num_vars + self.num_slack + self.num_artificial:
                variable_type = f'A{entering_var - self.num_vars - self.num_slack + 1}'  # Artificial variable
            else:
                variable_type = f'e{entering_var - self.num_vars - self.num_slack - self.num_artificial + 1}'  # Surplus variable

            file.write(f"\nEntering variable: {variable_type}, because it has the most {'positive' if problemtype == 'min' else 'negative'} coefficient {self.tableau[-1, entering_var]:.2f} in the Z-row.\n")
            print(f"\nEntering variable: {variable_type}, because it has the most {'positive' if problemtype == 'min' else 'negative'} coefficient {self.tableau[-1, entering_var]:.2f} in the Z-row.")

            # Check for unboundedness
            if all(self.tableau[:-1, entering_var] <= 0):
                self.status = 'unbounded'
                file.write("\nThe problem is unbounded.\n")
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

            # Determine the variable type for the leaving variable
            if self.basis[leaving_var] < self.num_vars:
                leaving_variable_type = f'x{self.basis[leaving_var] + 1}'  # Decision variable
            elif self.basis[leaving_var] < self.num_vars + self.num_slack:
                leaving_variable_type = f's{self.basis[leaving_var] - self.num_vars + 1}'  # Slack variable
            elif self.basis[leaving_var] < self.num_vars + self.num_slack + self.num_artificial:
                leaving_variable_type = f'A{self.basis[leaving_var] - self.num_vars - self.num_slack + 1}'  # Artificial variable
            else:
                leaving_variable_type = f'e{self.basis[leaving_var] - self.num_vars - self.num_slack - self.num_artificial + 1}'  # Surplus variable

            file.write(f"Leaving variable: {leaving_variable_type}, because it has the smallest ratio {ratios[leaving_var]:.2f}.\n")
            print(f"Leaving variable: {leaving_variable_type}, because it has the smallest ratio {ratios[leaving_var]:.2f}.")

            # Pivot
            pivot_element = self.tableau[leaving_var, entering_var]
            file.write(f"\nPivot element: {pivot_element:.2f} at row {leaving_var + 1}, column {entering_var + 1}.\n")
            file.write(f"Divide row {leaving_var + 1} by {pivot_element:.2f} to make the pivot element 1.\n")
            print(f"\nPivot element: {pivot_element:.2f} at row {leaving_var + 1}, column {entering_var + 1}.")
            print(f"Divide row {leaving_var + 1} by {pivot_element:.2f} to make the pivot element 1.")
            self.tableau[leaving_var, :] /= pivot_element
            for i in range(self.num_constraints + 1):
                if i != leaving_var:
                    factor = self.tableau[i, entering_var]
                    self.tableau[i, :] -= factor * self.tableau[leaving_var, :]
                    file.write(f"Subtract {factor:.2f} times row {leaving_var + 1} from row {i + 1} to eliminate the entering variable in other rows.\n")
                    print(f"Subtract {factor:.2f} times row {leaving_var + 1} from row {i + 1} to eliminate the entering variable in other rows.")

            # Update basis
            self.basis[leaving_var] = entering_var
            file.write(f"Update basis: {variable_type} enters the basis, {leaving_variable_type} leaves the basis.\n")
            print(f"Update basis: {variable_type} enters the basis, {leaving_variable_type} leaves the basis.")

            iteration += 1
            file.write(f"\n{phase} - Iteration {iteration}:\n")
            self.display_tableau(file)
            print(f"\n{phase} - Iteration {iteration}:")
            self.display_tableau()

    def solve(self, display_steps=False): 
        self.initialize_tableau()

        with open('twophase.txt', 'w') as file:
            # Phase 1
            if not self.phase1(file):
                self.status = 'NON-optimal'
                file.write("\nPhase 1 - No feasible solution found because Artificial variable appears in the basic column.\n")
                print("\nPhase 1 - No feasible solution found because Artificial variable appears in the basic column.")
                return

            # Phase 2
            self.solve_simplex(phase='Phase 2', file=file)

            # Extract solution
            self.optimal_solution = np.zeros(self.num_vars)
            for i in range(self.num_constraints):
                if self.basis[i] < self.num_vars:
                    self.optimal_solution[self.basis[i]] = self.tableau[i, -1]
            self.optimal_value = self.tableau[-1, -1]

            file.write("\nFinal Results:\n")
            file.write(f"Optimal Solution: {self.optimal_solution}\n")
            file.write(f"Optimal Value: {self.optimal_value}\n")
            file.write(f"Status: {self.status}\n")

    def display_tableau(self, file=None):
        self.headers = ['Basic'] + [f'x{i + 1}' for i in range(self.num_vars)]
        self.headers += [f's{i + 1}' for i in range(self.num_slack)]
        self.headers += [f'A{i + 1}' for i in range(self.num_artificial)]
        self.headers += [f'e{i + 1}' for i in range(self.num_surplus)]
        self.headers.append('RHS')

        self.tableau_rows = []
        for i in range(self.num_slack + self.temp):
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

        tableau_str = tabulate(self.tableau_rows, headers=self.headers, tablefmt='grid', floatfmt='.2f')
        if file:
            file.write(tableau_str + '\n')
        else:
            print(tableau_str)

    def get_results(self):
        return {
            'optimal_solution': self.optimal_solution if self.optimal_solution is not None else "No solution found",
            'optimal_value': self.optimal_value if self.optimal_value is not None else "No value found",
            'status': self.status
        }

if __name__ == '__main__':
    c = [2, 3]
    A = [[0.5, 0.25], [1, 3], [1, 1]]
    b = [4, 36, 10]
    constraint_types = ['<=', '>=', '=']
    variable_restrictions = ['non-negative', 'non-negative']
    problem_type = 'min'
    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])

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

"""if __name__ == '__main__':
    # Problem setup
    c = [5,8]
    A = [[1,1] ,[3,2], [1,4]]
    b = [5,3,4]
    constraint_types = ['<=','>=','>=']
    variable_restrictions = ['non-negative', 'non-negative']
    problem_type = 'max'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])"""


    
"""if __name__ == '__main__':
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
    print("Status:", results['status'])"""

"""if __name__ == '__main__':
    # Problem setup
    c = [-3,1,-2]
    A = [[1,3,1], [2,-1,1] , [4,3,-2]]
    b = [5,2,5]
    constraint_types = ['<=', '>=' ,'=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
    problem_type = 'min'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])"""

"""if __name__ == '__main__':
    # Problem setup
    c = [5,2,10]
    A = [[1,0,-1], [0,1,1]]
    b = [10,10]
    constraint_types = ['<=', '>=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
    problem_type = 'min'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status']) """   
    
"""if __name__ == '__main__':
    # Problem setup
    c = [1,-2,-3]
    A = [[-2,1,3], [2,3,4]]
    b = [2,1]
    constraint_types = ['=', '=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
    problem_type = 'min'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])"""

"""if __name__ == '__main__':
    # Problem setup
    c = [2,3]
    A = [[0.5,0.25], [1,3] ,[1,1]]
    b = [4,36,10]
    constraint_types = ['<=', '>=' , '=']
    variable_restrictions = ['non-negative', 'non-negative']
    problem_type = 'min'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])"""

"""if __name__ == '__main__':
    # Problem setup
    c = [40,10,0,0,7,14]
    A = [[1,-1,0,0,2,0], [-2,1,0,0,-2,0], [1,0,1,0,1,-1] ,[0,1,1,1,2,1]]
    b = [0,0,3,4]
    constraint_types = ['=', '=' , '=' ,'=']
    variable_restrictions = ['non-negative', 'non-negative','non-negative','non-negative','non-negative','non-negative']
    problem_type = 'max'

    solver = TwoPhaseSimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])"""

     
