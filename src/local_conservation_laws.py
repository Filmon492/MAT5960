import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp


class Local_Conservation_Laws:
    """Class for solving the scalar conservation laws using the Lax_Friedrichs scheme"""
    def __init__(self, T, x_L, x_R , K, N,):
        """Initialize the parameters

        Parameters
        ----------

        T : The end point of the time.
        x_L : The left end point of the domain [x_L, x_R]
        x_R : The right end point of the domain [x_L, x_R]  
        K : Number of mesh points in space
        N : Number of mesh points in time
        """
        self.K = K
        self.N = N
        self.T= T
        self.x_L = x_L
        self.x_R = x_R
        self.h = (self.x_R - self.x_L)/(self.K+1)
        self.dt = self.T/(self.N+1)
        

    def create_mesh(self):
        """Create mesh points for both space and time: the method returns vectors bx and bt 
        containing midpoint values and time steps, respectively"""
        h = self.h
        dt = self.dt
        
        bx = np.zeros(self.K)  # Midpoint values of each cell C_j in space 
        for j in range(self.K):
            bx[j] = self.x_L + (j + 0.5) * h  
        
        bt = np.zeros(self.N + 1)  # Mesh points in time 
        for i in range(self.N + 1):
            bt[i] = dt * i
        
        return bx, bt
    
    def cell_averages(self, u0, c, d):
        """ Input: an intial function u(x,0) that will be intgrated from xj-h to xj+h.
        This method computes the average value at each cell C_j to evaluate the value at xj,
        hence it returns a vector of cell averages"""
        
        C_j = quad(u0, c, d)[0] / self.h # quad is used to integrate numerically
        return C_j
    
    def initial_value(self, u0, bx):
        """Computes the cell averages at t = 0"""
        h = self.h
        U0_values = np.zeros(len(bx))
        for j in range(len(bx)):
            U0_values[j] = self.cell_averages(u0, bx[j] - h / 2, bx[j] + h / 2)
        return U0_values
    
    def f(self,U):
        "local flux of the local conservation laws"
        return U*(1-U)
        

    def local_solver(self, U0, boundary_condition):
        """Solving the local conservation laws at every time step:
        Input: array vector with the solution at t = 0
        Returns: Plot_data with the solution vector at each time step t"""
        
        h = self.h
        dt = self.dt
        U1 = np.zeros(len(U0)) 
        plot_data = [U0.copy()]
      
        
        for i in range(1, self.N + 1):  

            if boundary_condition == "dirichlet":
                U1[0] = U0[0]
                U1[self.K - 1] = U0[self.K - 1]

            elif boundary_condition == "periodic":
                pass

            elif boundary_condition == "artificial":
                pass
            
            idR = np.arange(1, self.K - 1)[:, None] + [0, 1]  # shape (K-2, 2)
            idL = np.arange(1, self.K - 1)[:, None] + [0, -1]
            U0R = U0[idR] 
            U0L = U0[idL]
            # Lax-Friedrichs-type scheme
            
            FR = (self.f(U0R[:,0]) + self.f(U0R[:,1]))/2 - h / (2 * dt) * ( U0R[:,1] - U0R[:,0])
            FL = (self.f(U0L[:,0]) + self.f(U0L[:,1]))/2 - h / (2 * dt) * ( U0L[:,0] - U0L[:,1])
            
            U1[1: self.K - 1] = U0[1:self.K - 1] - dt / h * (FR - FL)

            plot_data.append(U1.copy())  # Store the result at each time step
            U0 = U1.copy()  

        return plot_data

    
    def entropy_solution(self, x ,t):
         
        if x <= -0.2*t:
            return 0.6
        if -0.2*t < x <= 0.8*t:
            return (t - x)/(2*t)
        if x > 0.8*t:
            return 0.1 
        
        
    
    def l1_error(self, bx, t,numerical_solution):
        self.t = t
        h = self.h
        entropy_solution = np.zeros(self.K)
        en = np.zeros(self.K)
        for i in range(self.K):
            entropy_solution[i] = self.entropy_solution(bx[i],t)
            en[i] = entropy_solution[i]  - numerical_solution[i]
        return (h*np.sum(abs(en))), entropy_solution 
    













   