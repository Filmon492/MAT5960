import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from local_conservation_laws import Local_Conservation_Laws

class Nonlocal_Conservation_laws(Local_Conservation_Laws):
    """Class for solving the nonlocal scalar conservation laws using the Lax_Friedrichs scheme"""
    def __init__(self, T, x_L, x_R, K, N, epslon, alpha):
        """Initialize the parameters that include M, epsilon and alpha, for nonlocal interactions"""
        super().__init__(T, x_L, x_R, K, N)
        self.epslon = epslon  
        self.alpha = alpha 
        self.M = int(self.epslon / self.h)
    
    def nonlocal_kernel_w(self, y):
        epslon= self.epslon
        return   2*(epslon-y)/ epslon**2 #np.exp(-y/epslon)/ epslon 
    
    def quadrature_weights(self):
        q_w= np.zeros(self.M) # quadrature weights given by left endpoint
        for k in range(self.M):
            q_w[k] = self.nonlocal_kernel_w(k*self.h) * self.h
        return q_w
    
    def approx_velocity(self,U0):
        q_w= self.quadrature_weights()
        Vj= np.zeros(self.K)
        
        for j in range(self.M):
            vj= 0
            for k in range(self.M):
                vj +=  q_w[k] *U0[j+k]
            Vj[j]= 1-vj
        return Vj 

    def nonlocal_solver(self, U0, bt, boundary_condition):
        """Solving the nonlocal conservation laws at every time step:
        Input: array vector with the solution at t = 0
        Returns: Plot_data with the solution vector at each time step t"""
        
        h = self.h
        dt = self.dt
        U1 = np.zeros(len(U0))  # Vector solution at next time step
        plot_data = [U0.copy()]  # Store initial data and the updated data
        Vj= self.approx_velocity(U0)
        
        for i in range(1, self.N + 1):  

            if boundary_condition == "dirichlet":
                U1[0] = U0[0]
                U1[self.K - 1] = U0[self.K - 1]

            elif boundary_condition == "periodic":
                pass

            elif boundary_condition == "artificial":
                pass

            for j in range(1, self.K - 1):  # Avoid index out of range
                F_R = (U0[j]* Vj[j])/2 + (U0[j+1]*Vj[j+1])/2 + alpha/2 * (U0[j+1] -U0[j]) 
                F_L = (U0[j-1]* Vj[j-1])/2 + (U0[j]*Vj[j])/2 + alpha/2 * (U0[j] -U0[j-1]) 
                U1[j] = U0[j] - dt / h * (F_R - F_L) # Correct Lax-Friedrichs formula
            
            plot_data.append(U1.copy())  # Store the result at each time step
            U0 = U1.copy()

        return plot_data

u0 = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)
#u0 = lambda x: 0.1 if x < 0.5 else 0.6  # Initial condition
alpha = 0.0005
epslon = 0.005
a = Nonlocal_Conservation_laws(1, 0, 1, 1000, 1000 ,epslon, alpha)
bx, bt = a.create_mesh()  
U0 = a.initial_value(u0, bx)
plot_data = a.nonlocal_solver(U0, bt, "dirichlet")  
    #print(bt)

    # Plotting the result at t = 1
final_time = 1
plt.plot(bx, plot_data[-1], label=f'Time={final_time}')  # Plot the final time state
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Lax-Friedrichs Method')
plt.legend()
plt.show()

""" Neste gang bør jeg sjekke om jeg har riktig scheme for nonlocal flux. 
sjekke nonlocal parameterene ,m, epslon, aplha dt/h"""  