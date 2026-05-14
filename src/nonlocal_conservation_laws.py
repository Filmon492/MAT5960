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
        self.M = int(self.epsilon / self.h + 1.0) # number of celles involved in the nonlocal integral
        self._first_order_quadrature_weights_cache = {}
        self._first_order_velocity_index_cache = {}
        self._first_order_flux_index_cache = {}
    
    #def nonlocal_kernel_w(self, y):
        #epsilon= self.epsilon
        #return 2*(epsilon -y)/epsilon**2
    def nonlocal_kernel_w(self, y):
        epsilon = self.epsilon
        y = np.asarray(y)

        return np.where(
            (0.0 <= y) & (y <= epsilon),
            2.0 * (epsilon - y) / epsilon**2,
            0.0
        )

    def quadrature_weights(self, quadrature_weights):
        cached_weights = self._first_order_quadrature_weights_cache.get(quadrature_weights)
        if cached_weights is not None:
            return cached_weights

        kh= np.arange(self.M) * self.h
        q_w = np.zeros(self.M)
        q_w[:]= self.nonlocal_kernel_w(kh)*self.h # quadrature weights given by left endpoint
        
        if quadrature_weights == "left endpoints":
            weights = q_w
        if quadrature_weights == "Normalized left endpoints" : 
            weights = q_w / np.sum(q_w) # normalized left endponit

        self._first_order_quadrature_weights_cache[quadrature_weights] = weights
        return weights

    def _velocity_indices(self, bc):
        cached_indices = self._first_order_velocity_index_cache.get(bc)
        if cached_indices is not None:
            return cached_indices

        idx = np.arange(self.K)[:, None] + np.arange(self.M)[None, :]

        if bc == "periodic":
            idx = idx % self.K
        else:
            idx = np.clip(idx, 0, self.K - 1)

        self._first_order_velocity_index_cache[bc] = idx
        return idx

    def _flux_indices(self, bc):
        cached_indices = self._first_order_flux_index_cache.get(bc)
        if cached_indices is not None:
            return cached_indices

        if bc == "periodic":
            idR = (np.arange(self.K)[:, None] + [0, 1]) % self.K
            idL = (np.arange(self.K)[:, None] + [-1, 0]) % self.K
        else:
            interior = np.arange(1, self.K - 1)[:, None]
            idR = interior + [0, 1]
            idL = interior + [-1, 0]

        cached_indices = (idR, idL)
        self._first_order_flux_index_cache[bc] = cached_indices
        return cached_indices
    
   
    def approx_velocity(self, U0 , model, quadrature_weights):
        if model == "local": # implement the local model
            return 1 - U0
        elif model == "nonlocal":
            q_w = self.quadrature_weights(quadrature_weights)
            idx = self._velocity_indices(getattr(self, "bc", None))
            I = U0[idx] @ q_w
            return 1 - I

        return np.zeros(self.K)
            
            
        """ 
            #Create a 2D array of indices for U0
            idx = (np.arange(self.K)[:, None] + np.arange(self.M)[None, :])
            #idx = np.arange(self.K)[:, None] + np.arange(self.M)  # shape (K, M)
            idx = np.clip(idx, 0, self.K - 1) # make sure no out of bounds 
            I = U0[idx] @ q_w
            Vj= 1 - I
        return Vj """
    

    def nonlocal_solver(self, U0, bc, model, quadrature_weights, numerical_fluxes, store_history=True):
        """Solving the nonlocal conservation laws at every time step:
        Input: array vector with the solution at t = 0
        Returns: Plot_data with the solution vector at each time step t.
        If store_history is False, only the final state is returned."""
        self.bc = bc
        h = self.h
        dt = self.dt
        U1 = np.zeros(len(U0))  # Vector solution at next time step
        plot_data = [U0.copy()] if store_history else None
        idR, idL = self._flux_indices(bc)
       

        for i in range(1, self.N + 1):    
            # Handle boundary conditions
            if bc == "absorbing":
                
                U1[0] = U0[0]
                U1[self.K - 1] = U0[self.K - 1]
            
            if bc == "dirichlet":
                pass
            
           
            Vj = self.approx_velocity(U0, model, quadrature_weights)
            if np.any(np.isnan(Vj)):
                print("nan")

            print(100 * i / self.N)
            
            if bc == "periodic":
                U0R = U0[idR]  # shape (K, 2)
                VjR = Vj[idR]
                U0L = U0[idL]
                VjL = Vj[idL]

                    # Numerical fluxes
                if numerical_fluxes == "Lax-F":
                    FR = (U0R[:, 0] * VjR[:, 0] + U0R[:, 1] * VjR[:, 1]) / 2 \
                        + self.alpha / 2 * (U0R[:, 0] - U0R[:, 1])
                    FL = (U0L[:, 0] * VjL[:, 0] + U0L[:, 1] * VjL[:, 1]) / 2 \
                        + self.alpha / 2 * (U0L[:, 0] - U0L[:, 1])
                elif numerical_fluxes == "Godunov":
                    FR = U0R[:, 0] * VjR[:, 1]
                    FL = U0L[:, 0] * VjL[:, 1]
                
                #periodic update
                U1[:] = U0[:] - dt / h * (FR - FL)

            
            
            else:
                U0R = U0[idR]; VjR = Vj[idR]
                U0L = U0[idL]; VjL = Vj[idL]

                if numerical_fluxes == "Lax-F":
                    FR = (U0R[:, 0] * VjR[:, 0] + U0R[:, 1] * VjR[:, 1]) / 2 \
                        + self.alpha / 2 * (U0R[:, 0] - U0R[:, 1])
                    FL = (U0L[:, 0] * VjL[:, 0] + U0L[:, 1] * VjL[:, 1]) / 2 \
                        + self.alpha / 2 * (U0L[:, 0] - U0L[:, 1])
                elif numerical_fluxes == "Godunov":
                    FR = U0R[:, 0] * VjR[:, 1]
                    FL = U0L[:, 0] * VjL[:, 1]

                U1[1:self.K - 1] = U0[1:self.K - 1] - dt / h * (FR - FL)

            if store_history:
                plot_data.append(U1.copy())
            U0 = U1.copy()

        if store_history:
            return plot_data
        return U0.copy()

    
    def l1_error(self, bx, t,numerical_solution, entropy_solution):
        h = self.h
        entropy_solution_values = np.zeros(self.K)
        en = np.zeros(self.K)
        for i in range(self.K):
            entropy_solution_values[i] = entropy_solution(bx[i],t)
            en[i] = entropy_solution_values[i]  - numerical_solution[i]
        return (h*np.sum(abs(en))), entropy_solution_values    
   
