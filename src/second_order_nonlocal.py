import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from nonlocal_conservation_laws import Nonlocal_Conservation_laws 

class Second_Order_Nonlocal(Nonlocal_Conservation_laws):    
    """Class for solving the nonlocal model using second order numerical scheme"""
    def __init__(self, T, x_L, x_R, K, N, epsilon, alpha):
        super().__init__(T, x_L, x_R, K, N, epsilon, alpha)

    def nonlocal_kernel_w(self, y):
        epsilon= self.epsilon
        return 2*np.maximum(epsilon - y, 0.0) / epsilon**2
        

    def quadrature_weights(self):
        kh= np.arange(self.M+1) * self.h
        q_w = np.zeros(self.M+1)
        q_w[:]= self.nonlocal_kernel_w(kh)
        Q_w = q_w.copy()
        Q_w[0]= q_w[0]* 0.5 
        Q_w[-1] = q_w[-1] *0.5
        return q_w / (self.h*np.sum(Q_w)) # Normalized quadrature weights
    
    def minmod(self, a, b, c):
        """Vectorized minmod limiter for arrays of the same shape."""
        same_sign = (np.sign(a) == np.sign(b)) & (np.sign(b) == np.sign(c))
        min_abs = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
        return np.where(same_sign, np.sign(a) * min_abs, 0.0)

    def ghost_extension(self, U0, g):
        """ adding ghost celles on the left and right"""

        K  = self.K
        U0e = np.empty(K + 2 * g, dtype=U0.dtype)
        U0e[g:g+K] = U0
        
        # extension of U0 using artificial bc
        U0e[:g] = U0[0]
        U0e[K+g:] = U0[-1]

        return U0e 
 
    def approx_left_right_values(self, U0):
        """ Computes the reconstructed values at xj+1/2 and xj-1/2"""
    
       
        K  = self.K
        g= 2
        U0e = self.ghost_extension(U0,g)


        # Build convenient stencil slices for vectorized slopes
        Ujm2 = U0e[g-2 : K+g-2 ]   # u_{j-2}
        Ujm1 = U0e[g-1 : K+g-1]   # u_{j-1}
        Uj = U0e[g: K+g]           # u_{j}
        Ujp1 = U0e[g+1 : K+g+1]   # u_{j+1}

        
        # Slopes at cell j: minmod(u_j - u_{j-1}, 0.5*(u_{j+1}-u_{j-1}), u_{j+1}-u_j)
        a  = Uj - Ujm1
        b  = 0.5 * (Ujp1 - Ujm1)
        c  = Ujp1 - Uj
        sigma = self.minmod(a, b, c)

        
        # Slopes at cell (j-1): needed for left face x_{j-1/2} from the left side
        aL = Ujm1 - Ujm2
        bL = 0.5 * (Uj - Ujm2)
        cL = Uj - Ujm1
        sigma_L = self.minmod(aL, bL, cL)

        # Reconstruction value u(xj+1/2) approching xj+1/2 from left
        U_R = (Uj + 0.5 * sigma)
        # Recontraction value u(xj-1/2) approching xj-1/2 from left
        U_L = Ujm1 + 0.5 * sigma_L
        
        return U_L, U_R
        
   

    def approx_velocity(self, U0):
        """
        Compute the nonlocal convolution at all interfaces and return
        the left/right interface values per cell:
            R_left[j]  = R_{j-1/2}
            R_right[j] = R_{j+1/2}

        Parameters
        ----------
        U0 : ndarray, shape (K,)
            Cell-average density values rho_j.

        Returns
        -------
        R_left  : ndarray, shape (K,)
        R_right : ndarray, shape (K,)
        """
        h  = self.h
        K  = self.K
        M  = self.M
        q = self.quadrature_weights()     # shape (M+1,)
        qL = q[:-1]                       # w_k
        qR = q[1:]                        # w_{k+1}

        qLw = qL[None, :]   # shape (1, M)
        qRw = qR[None, :]   # shape (1, M)


        # Indices for j,k
        j = np.arange(K)[:, None]        # shape (K,1)
        k = np.arange(M)[None, :]      # shape (1,M)
        
    
        G = max(M+2, 3)
        U0e = self.ghost_extension(U0,G)
        c_idx = G + j + k
         # Build convenient stencil slices for vectorized slopes

        
        Ujm1 = U0e[c_idx - 1]     #u_{j+m-1}
        Uj= U0e[c_idx]            #u_{j+m}
        Ujp1 = U0e[c_idx + 1]    #u_{j+m+1}
        Ujp2 =  U0e[c_idx + 2]    # u_{j+m+2}

        #slopes sigma[j+k] 
        a = Uj - Ujm1
        b = 0.5 * (Ujp1- Ujm1)
        c = Ujp1- Uj
        sigma = self.minmod(a, b, c)

        # slopes sigma[j+k+1] 
        aL = Ujp1 - Uj
        bL = 0.5 * (Ujp2 - Uj)
        cL = Ujp2- Ujp1
        sigma_L = self.minmod(aL, bL, cL)

         # interface values at xj+1/2
        Aj = Ujp1 - 0.5 * sigma_L
        Bj = Ujp1 + 0.5 * sigma_L

        # interface values at xj-1/2
        Cj = Uj - 0.5 * sigma
        Dj=  Uj + 0.5 * sigma
        
         # the convolution terms
        
        R_right = 0.5 * h * np.sum(Aj * qLw + Bj * qRw, axis=1)   # for x_{j+1/2}
        R_left  = 0.5 * h * np.sum(Cj * qLw + Dj * qRw, axis=1)   # for x_{j-1/2}


        Vj_R = 1 - R_right
        Vj_L = 1 - R_left
        
        return Vj_L, Vj_R


    def second_order_solver(self, U0):
        """Solving the nonlocal conservation laws at every time step"""
        
        h = self.h
        dt = self.dt
        U1 = np.zeros_like(U0) # Vector solution at next time step
        plot_data = [U0.copy()]  # Store initial data and the updated data

        for i in range(1, self.N + 1):  

            #Stage 1 
            U_L, U_R = self.approx_left_right_values(U0)
            Vj_L, Vj_R = self.approx_velocity(U0)

            F_right = U_R * Vj_R
            F_left  = U_L * Vj_L

            U_first = U0 - dt/h * (F_right - F_left)

            # Stage 2 
            U_L1, U_R1 = self.approx_left_right_values(U_first)
            Vj_L1, Vj_R1 = self.approx_velocity(U_first)

            F_right1 = U_R1 * Vj_R1
            F_left1  = U_L1 * Vj_L1

            U_second = U_first - dt/h * (F_right1 - F_left1)
            if np.any(np.isnan(U_second)):
                print("nan")

            print(100 * i / self.N)
            
            
            #Final update (RK2 averaging)
            U_new = 0.5 * (U0+ U_second)

            plot_data.append(U_new.copy())  # Store the result at each time step
            U0 = U_new.copy()  # Prepare for the next iteration
        return plot_data

    