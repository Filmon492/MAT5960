import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from local_conservation_laws import Local_Conservation_Laws
from nonlocal_conservation_laws import Nonlocal_Conservation_laws

class Convergence_Rates(Nonlocal_Conservation_laws):
    """Class for computing the order of convergence rate as both h and epsilon go to zero"""
    
    def __init__(self, T, x_L, x_R, K, N, epsilon, alpha):
        super().__init__(T, x_L, x_R, K, N, epsilon, alpha)
    
    def l1_error(self, bx, T, numerical_solution, K, h):
        self.K = K
        entropy_solution = np.zeros(self.K)
        error_values = np.zeros(self.K)
        for i in range(self.K):
            entropy_solution[i] = self.entropy_solution(bx[i], T)
            error_values[i] = entropy_solution[i] - numerical_solution[i]

        return (h * np.sum(np.abs(error_values))), entropy_solution
    
    def convergence_rates(self, m, T, x_L, x_R, K, N, epsilon, alpha, u0):
        h_values = []
        E_values = []
        T = 0.5  
        alpha = 2.0  
        r = np.zeros(m-1)

        for i in range(m):
            epsilon = epsilon/2
            K = 2*K
            N = 2*N
            
            h = 1 / K  # define mesh size
            a = Convergence_Rates(T, x_L, x_R, K, N, epsilon, alpha)
            bx, bt = a.create_mesh() 
            U0 = a.initial_value(u0, bx)  # initial condition
            plot_data_nonlocal = a.nonlocal_solver(U0, "dirichlet", "nonlocal")
            numerical_solution = plot_data_nonlocal[-1]

            E, _ = self.l1_error(bx, T, numerical_solution, K, h)
            h_values.append(h)
            E_values.append(E)

        # Compute convergence rates
        for i in range(1, m):
            error_prev = E_values[i-1]  # Previous error
            error_curr = E_values[i]    # Current error

            if error_prev <= 0 or error_curr <= 0:
                raise ValueError(f"Invalid error values: E_prev={error_prev}, E_curr={error_curr}")

            r[i-1] = (np.log(error_prev / error_curr)) / (np.log(h_values[i-1] / h_values[i]))

        return r, E_values, h_values



# Initial conditions and parameters
u0 = lambda x: 0.6 if x < 0 else 0.1  # Simple piecewise constant initial condition
alpha = 2.0
epsilon = 0.1
K = 50
N = 50
x_L = -1
x_R = 1
T = 0.5
m = 8


b = Convergence_Rates(T, x_L, x_R, K, N, epsilon, alpha)
r, E, h = b.convergence_rates(m, T, x_L, x_R, K, N, epsilon, alpha, u0)

print("Convergence Rates:", r)
plt.plot(h, E)
plt.xlabel('Mesh Size (h)')
plt.ylabel('l1 error ')
plt.title('Convergence Analysis')
#plt.xscale('log')
#plt.yscale('log')
plt.show()
