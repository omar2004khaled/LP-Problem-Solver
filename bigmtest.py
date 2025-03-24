import numpy as np
import pandas as pd
from tabulate import tabulate

class BigMSolver:
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
                artificial_rows.append(i)
            elif self.constraint_types[i] == '=':
                artificial_with_surplus_rows.append(i)        
        
        total_vars = self.num_vars + self.num_slack + self.num_artificial + self.num_surplus
        self.tableau = np.zeros((self.num_artificial + self.num_slack + 1, total_vars + 1))
        
        slack_index = self.num_vars
        artificial_index = self.num_vars + self.num_slack
        surplus_index = self.num_vars + self.num_slack + self.num_artificial
        index=0

        """for i in range(self.num_constraints):
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
                self.tableau[i, artificial_index] = +1
                artificial_index += 1
            self.tableau[i, -1] = self.b[i]"""
        
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
        self.tableau[-1, :self.num_vars] = -self.c
        if self.problem_type == 'max':
            self.tableau[-1, self.num_vars + self.num_slack : self.num_vars + self.num_slack + self.num_artificial] = [self.M] * self.num_artificial
        else:
            self.tableau[-1, self.num_vars + self.num_slack : self.num_vars + self.num_slack + self.num_artificial] = [-self.M] * self.num_artificial

        self.basis = list(range(self.num_vars, self.num_vars + self.num_slack + self.num_artificial))

    def make_consistent(self, file):
        self.display_tableau(file)
        obj_row = self.num_artificial + self.num_slack
        for i in range(self.num_artificial):
            if self.problem_type == 'max':
                self.tableau[-1, :] = self.tableau[-1, :] - self.M * self.tableau[obj_row - i - 1, :]
            else:
                self.tableau[-1, :] = self.tableau[-1, :] + self.M * self.tableau[obj_row - i - 1, :]

    def solve_simplex(self, file):
        iteration = 0
        print(f"\nIteration {iteration}:")
        self.display_tableau()
        file.write(f"\nIteration {iteration}:\n")
        self.display_tableau(file)
        while True:
            if self.problem_type == 'min':
                if all(self.tableau[-1, :-1] <= 0):  # Minimization: all coefficients in Z-row should be <= 0
                    self.status = 'optimal'
                    file.write("\nOptimal solution reached.\n")
                    print("\nOptimal solution reached.")
                    break
            else:
                if all(self.tableau[-1, :-1] >= 0):  # Maximization: all coefficients in Z-row should be >= 0
                    self.status = 'optimal'
                    file.write("\nOptimal solution reached.\n")
                    print("\nOptimal solution reached.")
                    break

            if self.problem_type == 'min':
                entering_var = np.argmax(self.tableau[-1, :-1])  # Minimization: choose the most positive coefficient
            else:
                entering_var = np.argmin(self.tableau[-1, :-1])  # Maximization: choose the most negative coefficient

            if entering_var < self.num_vars:
                variable_type = f'x{entering_var + 1}'  # Decision variable
            elif entering_var < self.num_vars + self.num_slack:
                variable_type = f's{entering_var - self.num_vars + 1}'  # Slack variable
            elif entering_var < self.num_vars + self.num_slack + self.num_artificial:
                variable_type = f'A{entering_var - self.num_vars - self.num_slack + 1}'  # Artificial variable
            else:
                variable_type = f'e{entering_var - self.num_vars - self.num_slack - self.num_artificial + 1}'  # Surplus variable

            file.write(f"\nEntering variable: {variable_type}, because it has the most {'positive' if self.problem_type == 'min' else 'negative'} coefficient {self.tableau[-1, entering_var]:.2f} in the Z-row.\n")
            print(f"\nEntering variable: {variable_type}, because it has the most {'positive' if self.problem_type == 'min' else 'negative'} coefficient {self.tableau[-1, entering_var]:.2f} in the Z-row.")

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
            self.basis[leaving_var] = entering_var
            file.write(f"Update basis: {variable_type} enters the basis, {leaving_variable_type} leaves the basis.\n")
            print(f"Update basis: {variable_type} enters the basis, {leaving_variable_type} leaves the basis.")

            iteration += 1
            file.write(f"\nIteration {iteration}:\n")
            self.display_tableau(file)
            print(f"\nIteration {iteration}:")
            self.display_tableau()

        """artificial_start = self.num_vars + self.num_slack
        artificial_end = artificial_start + self.num_artificial
        for i in range(self.num_slack + self.num_artificial):
            if self.basis[i] >= artificial_start and self.basis[i] <= artificial_end:
                if self.tableau[i, -1] != 0:
                    return False
        return True"""
        artificial_start = self.num_vars + self.num_slack
        artificial_end = artificial_start + self.num_artificial
        for i in range(len(self.basis)):  # Iterate over all basis elements
         if artificial_start <= self.basis[i] < artificial_end:
          if self.tableau[i, -1] != 0:  # Check if the RHS is nonzero
            return False
        return True

    def solve(self, display_steps=False):
        self.initialize_tableau()

        with open('bigmtest.txt', 'w') as file:
            file.write("Initial Tableau:\n")
            self.make_consistent(file)

            if display_steps:
                file.write("\nAfter Consistency Adjustments:\n")
                self.display_tableau(file)

            if not self.solve_simplex(file):
                self.status = 'NON-optimal'
                file.write("\nNo feasible solution found because Artificial variable appears in the basic column.\n")
                print("\nNo feasible solution found because Artificial variable appears in the basic column.")
                return

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

"""if __name__ == '__main__':
    c = [1, 2, 1]
    A = [[1, 1, 1], [2, -5, 1]]
    b = [7, 10]
    constraint_types = ['=', '>=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
    problem_type = 'min'

    solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])"""


# if __name__ == '__main__':
#     c = [3,2,1]
#     A = [
#         [1,1,1],  
#         [0,1,-1],  
#         [1,1,2]   
#     ]
#     b = [4,2,6]
#     constraint_types = ['>=','<=','=']  # All are <= constraints
#     variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
#     problem_type = 'min'

#     solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
#     solver.solve(display_steps=True)

#     results = solver.get_results()
#     print("\nOptimal Solution:", results['optimal_solution'])
#     print("Optimal Value:", results['optimal_value'])
#     print("Status:", results['status'])
    
# if __name__ == '__main__':
#     c = [2000,1500]
#     A = [
#         [6,2],  
#         [2,4],  
#         [4,12]   
#     ]
#     b = [8,12,24]
#     constraint_types = ['>=','>=','>=']  # All are <= constraints
#     variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
#     problem_type = 'min'

#     solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
#     solver.solve(display_steps=True)

#     results = solver.get_results()
#     print("\nOptimal Solution:", results['optimal_solution'])
#     print("Optimal Value:", results['optimal_value'])
#     print("Status:", results['status'])


# if __name__ == '__main__':
#     c = [3, 5, 4]
#     A = [
#         [2, 3, 0],  
#         [0, 2, 5],  
#         [3, 2, 4]   
#     ]
#     b = [8, 10, 15]
#     constraint_types = ['<=', '<=', '<=']  # All are <= constraints
#     variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
#     problem_type = 'max'

#     solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
#     solver.solve(display_steps=True)

#     results = solver.get_results()
#     print("\nOptimal Solution:", results['optimal_solution'])
#     print("Optimal Value:", results['optimal_value'])
#     print("Status:", results['status'])

            

# if __name__ == '__main__':
#     c = [1,2,1]
#     A = [[1,1,1], [2,-5,1]]
#     b = [7,10]
#     constraint_types = ['=', '>=']
#     variable_restrictions = ['non-negative', 'non-negative', 'non-negative']
#     problem_type = 'min'

#     solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
#     solver.solve(display_steps=True)

#     results = solver.get_results()
#     print("\nOptimal Solution:", results['optimal_solution'])
#     print("Optimal Value:", results['optimal_value'])
#     print("Status:", results['status'])

# if __name__ == '__main__':
#     c = [6,4]
#     A = [[1,1], [0,1]]
#     b = [5,8]
#     constraint_types = ['<=', '>=']
#     variable_restrictions = ['non-negative', 'non-negative']
#     problem_type = 'max'

#     solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
#     solver.solve(display_steps=True)

#     results = solver.get_results()
#     print("\nOptimal Solution:", results['optimal_solution'])
#     print("Optimal Value:", results['optimal_value'])
#     print("Status:", results['status'])
if __name__ == '__main__':
    # Problem setup
    c = [3, 2,1]
    A = [[0,1, -1],[1, 1, 2]]
    b = [4, 2]
    constraint_types = ['<=','=']
    variable_restrictions = ['non-negative', 'non-negative','non-negative']
    problem_type = 'min'

    solver = BigMSolver(c, A, b, constraint_types, variable_restrictions, problem_type)
    solver.solve(display_steps=True)

    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])

            
