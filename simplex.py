import numpy as np
import pandas as pd
from tabulate import tabulate

class SimplexSolver:
    def __init__(self, c, A, b, constraint_types, variable_restrictions, problem_type='max'):
        """
        Initialize the Simplex Solver.
        
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

    def initialize_tableau(self):
        """
        Initialize the tableau for the Simplex method.
        """
        # Convert minimization problem to maximization
        if self.problem_type == 'min':
            self.c = -self.c

        # Handle unrestricted variables
        self.unrestricted_map = {}
        new_c = []
        new_A = []
        new_var_count = 0

        for i in range(self.num_vars):
            if self.variable_restrictions[i] == 'unrestricted':
                # Replace unrestricted variable x_i with x_i^+ - x_i^-
                new_c.append(self.c[i])  # x_i^+
                new_c.append(-self.c[i])  # x_i^-
                self.unrestricted_map[i] = (new_var_count, new_var_count + 1)
                new_var_count += 2
            else:
                new_c.append(self.c[i])
                new_var_count += 1

        # Update the number of variables
        self.num_vars = len(new_c)
        self.c = np.array(new_c)

        # Construct new_A with the correct number of columns
        for row in self.A:
            new_row = []
            new_var_count = 0
            for i in range(len(row)):
                if self.variable_restrictions[i] == 'unrestricted':
                    # Add x_i^+ and x_i^-
                    new_row.append(row[i])  # x_i^+
                    new_row.append(-row[i])  # x_i^-
                else:
                    new_row.append(row[i])
            new_A.append(new_row)

        self.A = np.array(new_A)

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
        self.tableau = np.zeros((self.num_constraints + 1, total_vars +1))

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

        # Fill the objective function row
        self.tableau[-1, :self.num_vars] = -self.c

        # Initialize basis
        self.basis = list(range(self.num_vars, total_vars))

    def get_variable_name(self, index):
        """ Returns the correct name for variables, accounting for unrestricted variables. """
        adjusted_index = 0  # Adjusted index for correct x numbering
        
        for i in range(len(self.variable_restrictions)):
            if self.variable_restrictions[i] == 'unrestricted':
                pos_idx, neg_idx = self.unrestricted_map[i]
                if index == pos_idx:
                    return f"x{i + 1}'"
                elif index == neg_idx:
                    return f"x{i + 1}''"
                # Since unrestricted variables take two indices, increase the counter accordingly
                adjusted_index += 2
            else:
                if adjusted_index == index:
                    return f"x{i + 1}"
                adjusted_index += 1

        # If it's a slack variable
        return f"s{index - self.num_vars + 1}"
    
    def solve(self, display_steps=False):
        
        self.initialize_tableau()
        iteration = 0

        # Open a file to write the steps
        with open('simplex.txt', 'w') as file:
            while True:
                # Display the current tableau if requested
                if display_steps:
                    print(f"\nIteration {iteration}:")
                    self.display_tableau()

                # Write the current tableau to the file
                file.write(f"\nIteration {iteration}:\n")
                file.write(self.get_tableau_as_string())

                # Check for optimality
                if all(self.tableau[-1, :-1] >= 0):
                    self.status = 'optimal'
                    file.write("\nOptimal solution reached.\n")
                    print("\nOptimal solution reached.")
                    break

                # Select entering variable (most negative coefficient in the objective row)
                entering_var = np.argmin(self.tableau[-1, :-1])
                entering_var_name = self.get_variable_name(entering_var)
                file.write(f"\nEntering variable: {entering_var_name}, because it has the most negative coefficient {self.tableau[-1, entering_var]:.2f} in the Z-row.\n")
                print(f"\nEntering variable: {entering_var_name}, because it has the most negative coefficient {self.tableau[-1, entering_var]:.2f} in the Z-row.")

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
                leaving_var_name = self.get_variable_name(self.basis[leaving_var])
                file.write(f"Leaving variable: {leaving_var_name}, because it has the smallest ratio {ratios[leaving_var]:.2f}.\n")
                print(f"Leaving variable: {leaving_var_name}, because it has the smallest ratio {ratios[leaving_var]:.2f}.")

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
                file.write(f"Update basis: {entering_var_name} enters the basis, {leaving_var_name} leaves the basis.\n")
                print(f"Update basis: {entering_var_name} enters the basis, {leaving_var_name} leaves the basis.")

                # Save tableau for tracking
                self.save_tableau(iteration)
                iteration += 1

            # Extract solution
            self.optimal_solution = np.zeros(self.num_vars)
            for i in range(self.num_constraints):
                if self.basis[i] < self.num_vars:
                    self.optimal_solution[self.basis[i]] = self.tableau[i, -1]

            # Handle unrestricted variables
            final_solution = []
            for i in range(len(self.variable_restrictions)):
                if self.variable_restrictions[i] == 'unrestricted':
                    pos_idx, neg_idx = self.unrestricted_map[i]
                    final_solution.append(self.optimal_solution[pos_idx] - self.optimal_solution[neg_idx])
                else:
                    final_solution.append(self.optimal_solution[i])

            self.optimal_solution = np.array(final_solution)
            self.optimal_value = self.tableau[-1, -1]

            # Adjust the optimal value for minimization problems
            if self.problem_type == 'min':
                self.optimal_value = -self.optimal_value

            # Write final results to the file
            file.write("\nFinal Results:\n")
            file.write(f"Optimal Solution: {self.optimal_solution}\n")
            file.write(f"Optimal Value: {self.optimal_value}\n")
            file.write(f"Status: {self.status}\n")

    def save_tableau(self, iteration):
        """
        Save the current tableau to a file for tracking.
        """
        df = pd.DataFrame(self.tableau, columns=[f'x{i + 1}' for i in range(self.num_vars)] + [f's{i + 1}' for i in range(self.tableau.shape[1] - self.num_vars - 1)] + ['RHS'])
        df.to_csv(f'tableau_iteration_{iteration}.csv', index=False)

    def get_tableau_as_string(self):

        #headers = ['Basic'] + [f'x{i + 1}' for i in range(self.num_vars)] + [f's{i + 1}' for i in range(self.tableau.shape[1] - self.num_vars - 1)] + ['RHS']
        headers = ['Basic']
        num_decision_vars = 0  # Count actual decision variable columns

        for i in range(len(self.variable_restrictions)):
            if self.variable_restrictions[i] == 'unrestricted':
                headers.append(f"x{i + 1}'")
                headers.append(f"x{i + 1}''")
                num_decision_vars += 2  # Each unrestricted variable counts as two columns
            else:
                headers.append(f"x{i + 1}")
                num_decision_vars += 1  # Normal variables count as one column

        # Compute slack variables correctly
        num_slack_vars = self.tableau.shape[1] - (num_decision_vars + 1)  # Exclude RHS column
        headers += [f's{i + 1}' for i in range(num_slack_vars)] + ['RHS']
    
        tableau_rows = []
        for i in range(self.num_constraints):
            basic_var = self.get_variable_name(self.basis[i])
            row = [basic_var] + list(self.tableau[i, :])
            tableau_rows.append(row)
            
        objective_row = ['Z'] + list(self.tableau[-1, :])
        tableau_rows.append(objective_row)

        return tabulate(tableau_rows, headers, tablefmt='grid', floatfmt='.2f')
    
    def display_tableau(self):
        """
        Display the current tableau in a fancy way using tabulate.
        """
        print(self.get_tableau_as_string())

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
    # Given LP problem:
    # MAX z = 0.2x1 + 0.15x2 - 0.25x3
    # Subject to:
    # 0.25x1 + 0.2x2 - x3 <= 200
    # x1 + x2 <= 900
    # x1 >= 0, x2 >= 0, x3 is unrestricted

    # Objective function coefficients
    c = [-4,30]

    # Constraint coefficients
    A = [
        [-1, 5],  # Constraint 1
        [0, 1],        # Constraint 2
    ]

    # Right-hand side values
    b = [30,5]

    # Constraint types (all are <=)
    constraint_types = ['<=', '<=']

    # Variable restrictions (x1 and x2 are non-negative, x3 is unrestricted)
    variable_restrictions = [ 'unrestricted' ,'non-negative' ]

    # Problem type (maximization)
    problem_type = 'max'

    # Initialize the solver
    solver = SimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)

    # Solve the problem and display detailed steps
    solver.solve(display_steps=True)

    # Get and print the results
    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])
    
"""
if __name__ == '__main__':
    # Given LP problem:
    # Maximize z = 5x1 + 4x2
    # Subject to:
    # 6x1 + 4x2 <= 24   (1)
    # x1 + 2x2 <= 6     (2)
    # -x1 + x2 <= 1     (3)
    # x2 <= 2           (4)
    # x1 >= 0, x2 >= 0  (5)

    # Objective function coefficients
    c = [5, 4]

    # Constraint coefficients
    A = [
        [6, 4],  # Constraint (1)
        [1, 2],  # Constraint (2)
        [-1, 1], # Constraint (3)
        [0, 1]   # Constraint (4)
    ]

    # Right-hand side values
    b = [24, 6, 1, 2]

    # Constraint types (all are <=)
    constraint_types = ['<=', '<=', '<=', '<=']

    # Variable restrictions (both are non-negative)
    variable_restrictions = ['non-negative', 'non-negative']

    # Problem type (maximization)
    problem_type = 'max'

    # Initialize the solver
    solver = SimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)

    # Solve the problem and display detailed steps
    solver.solve(display_steps=True)

    # Get and print the results
    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])"""    

"""
 if __name__ == '__main__':
    # Given LP problem:
    # Minimize z = 5x1 - 4x2 + 6x3 - 8x4
    # Subject to:
    # x1 + 2x2 + 2x3 + 4x4 <= 40
    # 2x1 - x2 + x3 + 2x4 <= 8
    # 4x1 - 2x2 + x3 - x4 <= 10
    # x1 >= 0, x2 >= 0, x3 >= 0, x4 >= 0

    # Objective function coefficients
    c = [5, -4, 6, -8]

    # Constraint coefficients
    A = [
        [1, 2, 2, 4],  # Constraint 1
        [2, -1, 1, 2],  # Constraint 2
        [4, -2, 1, -1]  # Constraint 3
    ]

    # Right-hand side values
    b = [40, 8, 10]

    # Constraint types (all are <=)
    constraint_types = ['<=', '<=', '<=']

    # Variable restrictions (all are non-negative)
    variable_restrictions = ['non-negative', 'non-negative', 'non-negative', 'non-negative']

    # Problem type (minimization)
    problem_type = 'min'

    # Initialize the solver
    solver = SimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)

    # Solve the problem and display detailed steps
    solver.solve(display_steps=True)

    # Get and print the results
    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])
"""

"""if __name__ == '__main__':
    
    c = [2,3]

    A = [
        [-1,2],  
        [1,1], 
        [1,3] 
    ]

    # Right-hand side values
    b = [4,6,9]

    # Constraint types (all are <=)
    constraint_types = ['<=', '<=' , '<=']

    # Variable restrictions (all are non-negative)
    variable_restrictions = ['unrestricted' , 'unrestricted']

    # Problem type (minimization)
    problem_type = 'max'

    # Initialize the solver
    solver = SimplexSolver(c, A, b, constraint_types, variable_restrictions, problem_type)

    # Solve the problem and display detailed steps
    solver.solve(display_steps=True)

    # Get and print the results
    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])"""

