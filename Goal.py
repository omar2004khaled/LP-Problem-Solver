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
        self.num_slack = 0
        self.tableau = None
        self.basis = None
        self.optimal_solution = None
        self.optimal_value = None
        self.status = np.array(['unsatisfied'] * self.num_goals)
        self.headers = None
        self.tableau_rows = None
        self.goal_status = None

    def initialize_tableau(self):
        num_deviation = 0
        for i in range(self.num_constraints):
            if self.constraint_types[i] == '<=':
                self.num_slack += 1
            elif self.constraint_types[i] == '>=':
                num_deviation += 1

        total_vars = self.num_vars + 2 * num_deviation + self.num_slack
        self.tableau = np.zeros((self.num_constraints + self.num_goals, total_vars + 1))

        slack_index = self.num_vars
        deviation_index = self.num_vars + self.num_slack

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
            self.tableau[goal_row, self.num_vars + self.num_slack + i] = -self.priority[i]

        self.basis = list(range(self.num_vars, self.num_vars + self.num_goals + self.num_slack))

    def display_tableau(self, file=None):
        self.headers = ['Basic'] + [f'x{i + 1}' for i in range(self.num_vars)]

        slack_count = sum(1 for t in self.constraint_types if t == '<=')
        deviation_count = sum(2 if t in ('>=') else 0 for t in self.constraint_types)

        self.headers += [f's{i + 1}' for i in range(slack_count)]
        for i in range(deviation_count // 2):
            self.headers += [f'd-{i + 1}']
        for i in range(deviation_count // 2):
            self.headers += [f'd+{i + 1}']
        self.headers.append('RHS')

        self.tableau_rows = []
        for i in range(self.num_constraints):
            if self.basis[i] < self.num_vars:
                basic_var = f'x{self.basis[i] + 1}'  # Decision variable
            elif self.basis[i] < self.num_vars + slack_count:
                basic_var = f's{self.basis[i] - self.num_vars + 1}'  # Slack variable
            else:
                deviation_index = self.basis[i] - self.num_vars - slack_count
                if deviation_index < self.num_goals:  # First two deviation variables are d1- and d2-
                    basic_var = f'd-{deviation_index + 1}'  # d1-, d2-
                else:
                    if deviation_index % 2 == 0:
                        basic_var = f'd+{(deviation_index // 2)}'  # Positive deviation
                    else:
                        basic_var = f'd-{(deviation_index // 2)}'

            # Add the row to the tableau
            row = [basic_var] + list(self.tableau[i, :])
            self.tableau_rows.append(row)

        for i in range(self.num_goals):
            row = [f'Z{i + 1}'] + list(self.tableau[self.num_constraints + i, :])
            self.tableau_rows.append(row)

        tableau_str = tabulate(self.tableau_rows, headers=self.headers, tablefmt='grid', floatfmt='.2f')
        if file:
            file.write(tableau_str + '\n')
        else:
            print(tableau_str)

    def make_consistent(self, file):
        for i in range(self.num_goals):
            goal_row = self.num_constraints + i
            self.tableau[goal_row, :] = self.tableau[goal_row, :] + self.priority[i] * self.tableau[goal_row - self.num_goals, :]
        file.write("\nConsistency Adjustments Applied\n")

    def solve_priority(self, priority_level, index, file, display_steps=False):
        iteration = 0
        while True:
            if display_steps:
                file.write(f"\nIteration {iteration} for goal {index + 1} of priority = {priority_level}:\n")
                self.display_tableau(file)

            # Check for optimality
            if all(self.tableau[self.num_constraints + index, :-1] <= 0):
                self.status[index] = 'satisfied'
                file.write(f"\nGoal {index + 1} is satisfied.\n")
                print(f"\nGoal {index + 1} is satisfied.")
                break

            # Find entering variable (most positive coefficient in goal row)
            entering_var = np.argmax(self.tableau[self.num_constraints + index, :-1])
            pivot_column_values = self.tableau[:, entering_var]
            satisfied_goal_indices = [i for i in range(len(self.goals)) if self.status[i] == 'satisfied']
            goal_rows = self.tableau[[self.num_constraints + i for i in satisfied_goal_indices], entering_var]

            if any(goal_rows != 0):
                file.write("\nPivoting aborted: Non-zero values detected in other satisfied goal rows for pivot column.\n")
                print("\nPivoting aborted: Non-zero values detected in other satisfied goal rows for pivot column.")
                break

            file.write(f"\nEntering variable: {self.headers[entering_var + 1]}, because it has the most positive coefficient {self.tableau[self.num_constraints + index, entering_var]:.2f} in the Z{index + 1}-row.\n")
            print(f"\nEntering variable: {self.headers[entering_var + 1]}, because it has the most positive coefficient {self.tableau[self.num_constraints + index, entering_var]:.2f} in the Z{index + 1}-row.")

            # Find leaving variable (minimum positive ratio rule)
            ratios = []
            for i in range(self.num_constraints):
                if self.tableau[i, entering_var] > 0:
                    ratio = self.tableau[i, -1] / self.tableau[i, entering_var]
                    if ratio >= 0:
                        ratios.append(ratio)
                else:
                    ratios.append(np.inf)
            leaving_var = np.argmin(ratios)
            file.write(f"Leaving variable: {self.tableau_rows[leaving_var][0]}, because it has the smallest ratio {ratios[leaving_var]:.2f}.\n")
            print(f"Leaving variable: {self.tableau_rows[leaving_var][0]}, because it has the smallest ratio {ratios[leaving_var]:.2f}.")

            if all(r == np.inf for r in ratios):
                file.write("\nNo valid leaving variable found. The problem may be unbounded.\n")
                print("\nNo valid leaving variable found. The problem may be unbounded.")
                self.status = 'unbounded'
                break

            # Perform pivoting
            pivot_element = self.tableau[leaving_var, entering_var]
            file.write(f"\nPivot element: {pivot_element:.2f} at row {leaving_var + 1}, column {entering_var + 1}.\n")
            file.write(f"Divide row {leaving_var + 1} by {pivot_element:.2f} to make the pivot element 1.\n")
            print(f"\nPivot element: {pivot_element:.2f} at row {leaving_var + 1}, column {entering_var + 1}.")
            print(f"Divide row {leaving_var + 1} by {pivot_element:.2f} to make the pivot element 1.")
            self.tableau[leaving_var, :] /= pivot_element
            for i in range(self.num_constraints + self.num_goals):
                if i != leaving_var:
                    factor = self.tableau[i, entering_var]
                    self.tableau[i, :] -= factor * self.tableau[leaving_var, :]
                    file.write(f"Subtract {factor:.2f} times row {leaving_var + 1} from row {i + 1} to eliminate the entering variable in other rows.\n")
                    print(f"Subtract {factor:.2f} times row {leaving_var + 1} from row {i + 1} to eliminate the entering variable in other rows.")

            # Update basis
            self.basis[leaving_var] = entering_var
            iteration += 1

    def solve(self, display_steps=False):
        self.initialize_tableau()

        with open('Goal.txt', 'w') as file:
            file.write("Initial Tableau:\n")
            self.display_tableau(file)

            self.make_consistent(file)
            file.write("\nAfter Consistency Adjustments:\n")
            self.display_tableau(file)

            # Create a hashmap (dictionary) of priorities with their indices
            priority_map = {}
            for index, priority in enumerate(self.priority):
                if priority not in priority_map:
                    priority_map[priority] = []
                priority_map[priority].append(index)

            sorted_priorities = sorted(priority_map.items(), key=lambda item: item[0], reverse=True)

            # Iterate over the sorted priorities and call solve_priority
            for priority, indices in sorted_priorities:
                for index in indices:  # Ensure all goals with the same priority are processed
                    self.solve_priority(priority, index, file, display_steps)

            # Extract solution
            self.optimal_solution = np.zeros(self.num_vars)
            for i in range(self.num_constraints):
                if self.basis[i] < self.num_vars:
                    self.optimal_solution[self.basis[i]] = self.tableau[i, -1]
            self.optimal_value = self.tableau[-1, -1]

            self.goal_status = []
            for i in range(self.num_constraints):
                if self.basis[i] > self.num_vars + self.num_slack and self.basis[i] < self.num_vars + self.num_slack + self.num_goals:  # for negative d's
                    res = self.tableau[i, -1]
                    index = self.basis[i] - self.num_vars - self.num_slack
                    self.goal_status.append(f"Goal {index + 1} not satisfied, penalty = {res}")
                elif self.basis[i] > self.num_vars + self.num_slack + self.num_goals:
                    res = self.tableau[i, -1]
                    index = self.basis[i] - self.num_vars - self.num_slack - self.num_goals
                    self.goal_status.append(f"Goal {index + 1} is satisfied with excess = {res}")

            file.write("\nFinal Results:\n")
            file.write(f"Optimal Solution: {self.optimal_solution}\n")
            file.write(f"Optimal Value: {self.optimal_value}\n")
            file.write(f"Status: {self.status.tolist()}\n")
            file.write(f"Goal Status: {self.goal_status}\n")

    def get_results(self):
        return {
            'optimal_solution': self.optimal_solution,
            'optimal_value': self.optimal_value,
            'status': self.status.tolist(),
            'goal_status': self.goal_status,
        }

if __name__ == '__main__':
    # Example Goal Programming Problem
    c = [5, -4]
    A = [
        [1.5, 3],
        [200, 0],
        [100, 400],
        [0, 250]
    ]
    b = [15, 1000, 1200, 800]
    goals = [30, 15, 100]
    priority = [1, 2, 1]
    constraint_types = ['<=', '>=', '>=', '>=']
    variable_restrictions = ['non-negative', 'non-negative']

    solver = GoalSolver(c, A, b, goals, priority, constraint_types, variable_restrictions)
    solver.solve(display_steps=True)
    results = solver.get_results()
    print("\nOptimal Solution:", results['optimal_solution'])
    print("Optimal Value:", results['optimal_value'])
    print("Status:", results['status'])
    print("Goal Status:", results['goal_status'])
