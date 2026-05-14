import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from nonlocal_conservation_laws import Nonlocal_Conservation_laws
from second_order_nonlocal import Second_Order_Nonlocal

class Convergence_Rates_Second_Order(Second_Order_Nonlocal):
    """Class for computing the order of convergence rate as both h and epsilon go to zero"""
    def __init__(self, T, x_L, x_R, K, N, epsilon, alpha):
        super().__init__(T, x_L, x_R, K, N, epsilon, alpha)
    
    def l1_error_analytic(self, bx, T, numerical_sol, K , h, reference_solution):  
        #self.K = K
   
        entropy_solution = np.zeros(K)
        error_values = np.zeros(K)
        for i in range(K):
            entropy_solution[i] = reference_solution(bx[i], T)
            error_values[i] = entropy_solution[i] - numerical_sol[i]

        return (h * np.sum(np.abs(error_values)))
    
    def l1_error_reference(self, bx, numerical_sol, h, reference_solution, bx_ref):
        # Interpolate fine-grid reference to current grid
        reference_sol_coarse = np.interp(bx, bx_ref, reference_solution)

        # Compute L1 error 
        error_values = reference_sol_coarse - numerical_sol
        return h * np.sum(np.abs(error_values))
    
    def convergence_second_order_scheme(self, m, T, u0 , reference_solution, bx_ref, limit, bc, entropy):
        
        epsilon = self.epsilon
        K = self.K
        N = self.N

        h_values = []
        E_values = []
        number_of_cells = []
        r = np.zeros(m-1)

        for i in range(m):
            if limit == "nonlocal":
                epsilon = epsilon
            if limit == "local":
                epsilon = epsilon/2
            K = 2*K
            N = 2*N
            h = (self.x_R - self.x_L) / K # define mesh size
            
            #a = Convergence_Rates(T, x_L, x_R, K, N, epsilon, alpha)
            a = Convergence_Rates_Second_Order(self.T, self.x_L, self.x_R, K, N, epsilon, self.alpha)
            bx, bt = a.create_mesh() 
            U0 = a.initial_value(u0, bx)  # initial condition
            numerical_sol = a.second_order_solver(U0, bc, store_history=False)
            if entropy == "reference_sol":
                E = self.l1_error_reference(bx, numerical_sol, h , reference_solution, bx_ref)
            if entropy == "analytic":
                E = self.l1_error_analytic(bx,T, numerical_sol,K, h , reference_solution)
            h_values.append(h)
            E_values.append(E)
            number_of_cells.append(K)

        # Compute convergence rates
        for i in range(1, m):
            error_prev = E_values[i-1]  # Previous error
            error_curr = E_values[i]    # Current error

            if error_prev <= 0 or error_curr <= 0:
                raise ValueError(f"Invalid error values: E_prev={error_prev}, E_curr={error_curr}")
            print(error_prev / error_curr)
            r[i-1] = (np.log(error_prev / error_curr)) / (np.log(h_values[i-1] / h_values[i]))
        return r, E_values, number_of_cells

