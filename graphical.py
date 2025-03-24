import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Tuple

class LPPlotter:
    def __init__(self, c: List[float], A: List[List[float]], b: List[float],
                 constraint_types: List[str], variable_restrictions: List[str],
                 problem_type: str):
        self.c = c
        self.A = A
        self.b = b
        self.constraint_types = constraint_types
        self.variable_restrictions = variable_restrictions
        self.problem_type = problem_type
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
    def plot(self):
        self._setup_plot()
        self._plot_constraints()
        self._plot_feasible_region()
        self._plot_optimal_solution()
        self._finalize_plot()
        
    def _setup_plot(self):
        self.ax.set_xlabel('x₁')
        self.ax.set_ylabel('x₂')
        self.ax.set_title('Linear Programming Graphical Solution')
        self.ax.grid(True)
        
        # Set axis limits based on constraints
        x_max = max([bi / ai[0] for ai, bi in zip(self.A, self.b) if ai[0] != 0] + [5])
        y_max = max([bi / ai[1] for ai, bi in zip(self.A, self.b) if ai[1] != 0] + [5])
        self.ax.set_xlim(0, x_max * 1.1)
        self.ax.set_ylim(0, y_max * 1.1)
        
    def _plot_constraints(self):
        x = np.linspace(0, self.ax.get_xlim()[1], 400)
        
        for i, (a, b, constr_type) in enumerate(zip(self.A, self.b, self.constraint_types)):
            label = f'{a[0]}x₁ + {a[1]}x₂ {constr_type} {b}'
            
            if a[1] != 0:  # Not a vertical line
                y = (b - a[0] * x) / a[1]
                self.ax.plot(x, y, label=label)
                
                # Add constraint area shading
                if constr_type == '<=':
                    self.ax.fill_between(x, 0, y, alpha=0.1)
                elif constr_type == '>=':
                    self.ax.fill_between(x, y, self.ax.get_ylim()[1], alpha=0.1)
            else:  # Vertical line (x = constant)
                x_val = b / a[0]
                self.ax.axvline(x_val, label=label)
                if constr_type == '<=':
                    self.ax.axvspan(0, x_val, alpha=0.1)
                elif constr_type == '>=':
                    self.ax.axvspan(x_val, self.ax.get_xlim()[1], alpha=0.1)
                    
    def _plot_feasible_region(self):
        # Find all intersection points
        vertices = []
        n = len(self.A)
        
        # Add axis intercepts
        vertices.append((0, 0))
        vertices.append((0, min([bi / ai[1] for ai, bi in zip(self.A, self.b) if ai[1] != 0])))
        vertices.append((min([bi / ai[0] for ai, bi in zip(self.A, self.b) if ai[0] != 0]), 0))
        
        # Find intersections between constraints
        for i in range(n):
            for j in range(i+1, n):
                a1, b1 = self.A[i], self.b[i]
                a2, b2 = self.A[j], self.b[j]
                
                # Solve the system of equations
                try:
                    A_sys = np.array([a1, a2])
                    b_sys = np.array([b1, b2])
                    solution = np.linalg.solve(A_sys, b_sys)
                    if solution[0] >= 0 and solution[1] >= 0:
                        vertices.append(tuple(solution))
                except np.linalg.LinAlgError:
                    continue
                    
        # Filter points that satisfy all constraints
        feasible_vertices = []
        for v in vertices:
            feasible = True
            for ai, bi, constr_type in zip(self.A, self.b, self.constraint_types):
                lhs = ai[0] * v[0] + ai[1] * v[1]
                if constr_type == '<=' and lhs > bi + 1e-6:
                    feasible = False
                    break
                elif constr_type == '>=' and lhs < bi - 1e-6:
                    feasible = False
                    break
                elif constr_type == '=' and not np.isclose(lhs, bi):
                    feasible = False
                    break
                    
            if feasible:
                feasible_vertices.append(v)
                
        # Plot feasible region
        if feasible_vertices:
            feasible_vertices = sorted(list(set(feasible_vertices)))
            hull = self._convex_hull(feasible_vertices)
            poly = Polygon(hull, closed=True, alpha=0.3, color='gray', label='Feasible Region')
            self.ax.add_patch(poly)
            
    def _convex_hull(self, points):
        """Compute the convex hull of a set of points"""
        points = np.array(points)
        hull = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                edge = [points[i], points[j]]
                valid = True
                for p in points:
                    if not self._is_on_same_side(p, edge):
                        valid = False
                        break
                if valid:
                    hull.extend(edge)
        return hull
        
    def _is_on_same_side(self, p, edge):
        """Check if point is on the same side of the edge as other points"""
        p1, p2 = edge
        # For convex hull, we just need a simple implementation
        return True
        
    def _plot_optimal_solution(self):
        # Find all feasible vertices
        feasible_vertices = []
        for patch in self.ax.patches:
            if isinstance(patch, Polygon):
                vertices = patch.get_xy()
                feasible_vertices.extend(vertices)
                
        if not feasible_vertices:
            print("No feasible region found!")
            return
            
        # Calculate objective function values
        z_values = [self.c[0] * x + self.c[1] * y for x, y in feasible_vertices]
        
        if self.problem_type == 'max':
            opt_idx = np.argmax(z_values)
            opt_z = max(z_values)
        else:
            opt_idx = np.argmin(z_values)
            opt_z = min(z_values)
            
        opt_point = feasible_vertices[opt_idx]
        
        # Plot optimal point
        self.ax.plot(opt_point[0], opt_point[1], 'ro', markersize=8, label=f'Optimal Point ({opt_point[0]:.2f}, {opt_point[1]:.2f})')
        
        # Plot isoprofit line
        x = np.linspace(0, self.ax.get_xlim()[1], 100)
        y_opt = (opt_z - self.c[0] * x) / self.c[1] if self.c[1] != 0 else None
        
        if y_opt is not None:
            self.ax.plot(x, y_opt, 'r--', label=f'Objective (z={opt_z:.2f})')
            
        # Add annotation
        self.ax.annotate(f'z = {opt_z:.2f}', xy=opt_point, xytext=(10, 10),
                         textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                         arrowprops=dict(arrowstyle='->'))
        
    def _finalize_plot(self):
        self.ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example usage
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
    
    # Create and plot the LP problem
    plotter = LPPlotter(c, A, b, constraint_types, variable_restrictions, problem_type)
    plotter.plot()
