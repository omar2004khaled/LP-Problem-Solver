
"""
import numpy as np
import pandas as pd
from tabulate import tabulate

class GoalSolver:
    def __init__(self, c, A, b, goals, priority, constraint_types, variable_restrictions, problem_type='min'):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.goals = np.array(goals, dtype=float)
        self.priority = np.array(priority, dtype=float)
        self.constraint_types = constraint_types
        self.variable_restrictions = variable_restrictions
        self.problem_type = problem_type
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self.num_goals = len(goals)
        self.tableau = None
        self.basis = None
        self.optimal_solution = None
        self.optimal_value = None
        self.status = None

    def initialize_tableau(self):
        num_slack = sum(1 for t in self.constraint_types if t == '<=')  # Slack variables
        num_deviation = sum(2 if t in ('>=', '=') else 1 for t in self.constraint_types)  # Deviations
        total_vars = self.num_vars + num_slack + num_deviation

        # Create tableau with space for all variables + RHS
        self.tableau = np.zeros((self.num_constraints + self.num_goals , total_vars))

        slack_index = self.num_vars
        deviation_index = self.num_vars + num_slack

        for i in range(self.num_constraints):
            self.tableau[i, :self.num_vars] = self.A[i]  # Copy constraint coefficients
            
            if self.constraint_types[i] == '=':
                self.tableau[i, deviation_index] = 1   # d+
                self.tableau[i, deviation_index + 1] = -1  # d-
                deviation_index += 2
            elif self.constraint_types[i] == '<=':
                self.tableau[i, slack_index] = 1  # Slack variable
                slack_index += 1
            elif self.constraint_types[i] == '>=':
                self.tableau[i, deviation_index] = -1  # d+
                self.tableau[i, deviation_index + self.num_goals] = 1  # d-
                deviation_index += 1

            self.tableau[i, -1] = self.b[i]  # RHS values

        
        # Add goal programming objective functions
        for i in range(self.num_goals):
            #goalpr = -1 * self.priority[i] 
            goal_row = self.num_constraints + i
            self.tableau[goal_row, self.num_vars + self.num_goals + num_slack + i] = self.priority[i] #goalpr  # Only negative deviation (d_i^-)

        # Initialize basis with slack and deviation variables
        self.basis = list(range(self.num_vars, total_vars,2))
       

    def solve(self, display_steps=False):
       
        self.initialize_tableau()
        iteration = 0
        
        while True:
            if display_steps:
                print(f"\nIteration {iteration}:")
                self.display_tableau()


            
            # Optimality check
            if all(self.tableau[-1, :-1] >= 0):
                self.status = 'optimal'
                print("\nOptimal solution reached.")
                break
            
            # Find entering variable (most negative value in last row)
            entering_var = np.argmin(self.tableau[-1, :-1])
            if all(self.tableau[:-1, entering_var] <= 0):
                self.status = 'unbounded'
                print("\nThe problem is unbounded.")
                break
            
            # Find leaving variable (minimum positive ratio rule)
            ratios = [self.tableau[i, -1] / self.tableau[i, entering_var] if self.tableau[i, entering_var] > 0 else np.inf for i in range(self.num_constraints)]
            leaving_var = np.argmin(ratios)
            
            # Perform pivoting
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

    def display_tableau(self):
        
        headers = ['Basic'] + [f'x{i + 1}' for i in range(self.num_vars)]
        
        slack_count = sum(1 for t in self.constraint_types if t == '<=')
        deviation_count = sum(2 if t in ('>=', '=') else 0 for t in self.constraint_types)

        headers += [f's{i + 1}' for i in range(slack_count)]
        headers += [f'd+{i + 1}' for i in range(deviation_count // 2)]
        headers += [f'd-{i + 1}' for i in range(deviation_count // 2)]
        headers.append('RHS')

        tableau_rows = []

        for i in range(self.num_constraints):
            if self.basis[i] < self.num_vars:
                basic_var = f'x{self.basis[i] + 1}'  # Decision variable
            elif self.basis[i] < self.num_vars + slack_count:
                basic_var = f's{self.basis[i] - self.num_vars + 1}'  # Slack variable
            else:
                deviation_index = self.basis[i] - self.num_vars - slack_count
                if deviation_index % 2 == 0:
                    basic_var = f'd+{(deviation_index // 2) + 1}'  # Positive deviation
                else:
                    basic_var = f'd-{(deviation_index // 2) + 1}'  # Negative deviation
            
            # Add the row to the tableau
            row = [basic_var] + list(self.tableau[i, :])
            tableau_rows.append(row)

        for i in range(self.num_goals):
            row = [f'Z{i + 1}'] + list(self.tableau[self.num_constraints + i, :])
            tableau_rows.append(row)

        print(tabulate(tableau_rows, headers=headers, tablefmt='grid', floatfmt='.2f'))

    def get_results(self):
        
        return {
            'optimal_solution': self.optimal_solution,
            'optimal_value': self.optimal_value,
            'status': self.status
        }


if __name__ == '__main__':
    # Example Goal Programming Problem
    c = [5, -4, 6, -3]  
    A = [
        [1, 2, 2, 4],  
        [2, -1, 1, 2],  
        [4, -2, 1, -1]  
    ]
    b = [40, 18, 10]  
    goals = [30, 15]  
    priority = [1, 2]  
    constraint_types = ['>=', '>=', '<=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative', 'non-negative']
    
    solver = GoalSolver(c, A, b, goals, priority, constraint_types, variable_restrictions)
    solver.solve(display_steps=True)
    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])
"""
import numpy as np
import pandas as pd
from tabulate import tabulate

class GoalSolver:
    def __init__(self, c, A, b, goals, priority, constraint_types, variable_restrictions, problem_type='min'):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.goals = np.array(goals, dtype=float)
        self.priority = np.array(priority, dtype=float)
        self.constraint_types = constraint_types
        self.variable_restrictions = variable_restrictions
        self.problem_type = problem_type
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self.num_goals = len(goals)
        self.tableau = None
        self.basis = None
        self.optimal_solution = None
        self.optimal_value = None
        self.status = None

    def initialize_tableau(self):
        num_slack =0
        num_deviation =0
        for i in range(self.num_constraints):
         if self.constraint_types[i] == '<=' :
            num_slack += 1
         elif self.constraint_types[i] == '>=' :
            num_deviation +=1

        total_vars = self.num_vars + 2*num_deviation + num_slack
        self.tableau = np.zeros((self.num_constraints + self.num_goals , total_vars + 1)) 

        slack_index = self.num_vars 
        deviation_index = self.num_vars + num_slack

        for i in range(self.num_constraints):
            if self.constraint_types[i] == '<=':
                self.tableau[i, :self.num_vars] = self.A[i]  
                self.tableau[i, slack_index] = 1  # Slack variable
                slack_index += 1
            elif self.constraint_types[i] == '>=':
                self.tableau[i, :self.num_vars] = self.A[i]  
                self.tableau[i, deviation_index] = 1  # d-
                self.tableau[i, deviation_index + self.num_goals] = -1  # d+
                deviation_index += 1

            self.tableau[i, -1] = self.b[i]    

        for i in range(self.num_goals): 
            goal_row = self.num_constraints + i
            self.tableau[goal_row, self.num_vars + self.num_goals + i ] = self.priority[i] ## -self.priority[i]

        self.basis = list(range(self.num_vars, total_vars))  
          

    def display_tableau(self):
         
        headers = ['Basic'] + [f'x{i + 1}' for i in range(self.num_vars)]

        slack_count = sum(1 for t in self.constraint_types if t == '<=')
        deviation_count = sum(2 if t in ('>=') else 0 for t in self.constraint_types)

        headers += [f's{i + 1}' for i in range(slack_count)]
        for i in range (deviation_count//2):
           headers += [f'd-{i + 1}']
        for i in range (deviation_count//2):
           headers += [f'd+{i + 1}']   
        headers.append('RHS')

        tableau_rows = []
        for i in range(self.num_constraints):
            if self.basis[i] < self.num_vars:
                basic_var = f'x{self.basis[i] + 1}'  # Decision variable
            elif self.basis[i] < self.num_vars + slack_count:
                basic_var = f's{self.basis[i] - self.num_vars + 1}'  # Slack variable
            else:
                deviation_index = self.basis[i] - self.num_vars - slack_count
                if deviation_index < 2:  # First two deviation variables are d1- and d2-
                    basic_var = f'd{deviation_index + 1}-'  # d1-, d2-
                else:
                    if deviation_index % 2 == 0:
                        basic_var = f'd{(deviation_index // 2) + 1}+'  # Positive deviation
                    else:
                        basic_var = f'd{(deviation_index // 2) + 1}-'
                    

        # Add the row to the tableau
            row = [basic_var] + list(self.tableau[i, :])
            tableau_rows.append(row)

        for i in range(self.num_goals):
            row = [f'Z{i + 1}'] + list(self.tableau[self.num_constraints + i, :])
            tableau_rows.append(row)

        print(tabulate(tableau_rows, headers=headers, tablefmt='grid', floatfmt='.2f'))   

    def solve(self, display_steps=False):
       
        self.initialize_tableau()
        iteration = 0
        
        while True:
            if display_steps:
                print(f"\nIteration {iteration}:")
                self.display_tableau()


            
            # Optimality check
            if all(self.tableau[-1, :-1] >= 0):
                self.status = 'optimal'
                print("\nOptimal solution reached.")
                break
            
            # Find entering variable (most negative value in last row)
            entering_var = np.argmin(self.tableau[-1, :-1])
            if all(self.tableau[:-1, entering_var] <= 0):
                self.status = 'unbounded'
                print("\nThe problem is unbounded.")
                break
            
            # Find leaving variable (minimum positive ratio rule)
            ratios = [self.tableau[i, -1] / self.tableau[i, entering_var] if self.tableau[i, entering_var] > 0 else np.inf for i in range(self.num_constraints)]
            leaving_var = np.argmin(ratios)
            
            # Perform pivoting
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

    def get_results(self):
        
        return {
            'optimal_solution': self.optimal_solution,
            'optimal_value': self.optimal_value,
            'status': self.status
        }            

if __name__ == '__main__':
    # Example Goal Programming Problem
    c = [5, -4, 6, -3]  
    A = [
        [1, 2, 2, 4],  
        [2, -1, 1, 2],  
        [4, -2, 1, -1],
        [2,1,1,1]  
    ]
    b = [40, 18, 10 ,11]  
    goals = [30, 15]  
    priority = [1, 2]  
    constraint_types = ['<=','<=', '>=', '>=']
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative', 'non-negative']
    
    solver = GoalSolver(c, A, b, goals, priority, constraint_types, variable_restrictions)
    solver.solve(display_steps=True)
    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])
       

