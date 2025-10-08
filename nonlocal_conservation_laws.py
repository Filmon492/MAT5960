import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from local_conservation_laws import Local_Conservation_Laws

class Nonlocal_Conservation_laws(Local_Conservation_Laws):
  
    """Class for solving the nonlocal scalar conservation laws using the Lax_Friedrichs scheme"""
    
    def __init__(self, T, x_L, x_R, K, N, epsilon, alpha):
        """Initialize the parameters that include M, epsilon and alpha, for nonlocal interactions"""
        super().__init__(T, x_L, x_R, K, N)
        self.epsilon = epsilon 
        self.alpha = alpha 
        self.M = int(self.epsilon / self.h + 1.0)
        self.q_w = self.quadrature_weights()
    
    def nonlocal_kernel_w(self, y):
        epsilon= self.epsilon
        return 2*(epsilon -y)/epsilon**2
        

    def quadrature_weights(self):
        kh= np.arange(self.M) * self.h
        q_w = np.zeros(self.M)
        q_w[:]= self.nonlocal_kernel_w(kh) # quadrature weights given by left endpoint
        return q_w / np.sum(q_w) # / self.h) * self.h cancels out
    
    def approx_velocity(self,U0, model):
        Vj= np.zeros(self.K)
        I = np.zeros(self.M)
        if model == "local":
            I = U0.copy()
            Vj= 1-I
        elif model == "nonlocal":
            #Create a 2D array of indices for U0
            idx = np.arange(self.K)[:, None] + np.arange(self.M)  # shape (K, M)
            idx = np.clip(idx, 0, self.K - 1) # make sure no out of bounds 
            I = U0[idx] @ self.q_w  
            Vj= 1 - I
        return Vj

    def nonlocal_solver(self, U0, boundary_condition, model):
        """Solving the nonlocal conservation laws at every time step:
        Input: array vector with the solution at t = 0
        Returns: Plot_data with the solution vector at each time step t"""
        self.boundary_condition = boundary_condition
        h = self.h
        dt = self.dt
        U1 = np.zeros(len(U0))  # Vector solution at next time step
        plot_data = [U0.copy()]  # Store initial data and the updated data

        for i in range(1, self.N + 1):  

            # Handle boundary conditions
            if boundary_condition == "dirichlet":
                U1[0] = U0[0]
                U1[self.K - 1] = U0[self.K - 1]
            
            elif boundary_condition == "periodic":
                pass 
            
            elif boundary_condition == "artificial":
                pass

            Vj = self.approx_velocity(U0, model)

            if np.any(np.isnan(Vj)):
                print("nan")

            print(100 * i / self.N)

            # Create index arrays for right and left fluxes
            idR = np.arange(1, self.K - 1)[:, None] + [0, 1]  # shape (K-2, 2)
            idL = np.arange(1, self.K - 1)[:, None] + [0, -1]      # shape (K-2, 2)

            U0R = U0[idR]  # shape (K-2, 2)
            VjR = Vj[idR]  # shape (K-2, 2)

            U0L = U0[idL]  # shape (K-2, 2)
            VjL = Vj[idL]  

            # Calculate the fluxes F_R and F_L for the whole array
            FR = (U0R[:, 0] * VjR[:, 0] + U0R[:, 1] * VjR[:, 1]) / 2  \
            + self.alpha / 2 * (U0R[:, 0] - U0R[:, 1])  #numerical flux on the right interface
            FL = (U0L[:, 0] * VjL[:, 0] + U0L[:, 1] * VjL[:, 1]) / 2  \
            + self.alpha / 2 * (U0L[:, 1] - U0L[:, 0]) #numerical flux on the left interface

            # Vectorized update of U1
            U1[1:self.K - 1] = U0[1:self.K - 1] - dt / h * (FR - FL)

            plot_data.append(U1.copy())  # Store the result at each time step
            U0 = U1.copy()  # Prepare for the next iteration
        return plot_data 
    
    def entropy_solution(self, x ,t):
        if x <= -0.2*t:
            return 0.6
        if -0.2*t < x <= 0.8*t:
            return (t - x)/(2*t)
        if x > 0.8*t:
            return 0.1 
        
        #return 0.1 if x < 0.8*t else 0.6
    
    def l1_error(self, bx, t,numerical_solution):
        self.t = t
        h = self.h
        entropy_solution = np.zeros(self.K)
        en = np.zeros(self.K)
        for i in range(self.K):
            entropy_solution[i] = self.entropy_solution(bx[i],t)
            en[i] = entropy_solution[i]  - numerical_solution[i]
        return (h*np.sum(abs(en))), entropy_solution    
   
