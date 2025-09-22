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
        self.M = int(self.epslon / self.h + 1.0)
        self.q_w = self.quadrature_weights()
    
    def nonlocal_kernel_w(self, y):
        epslon= self.epslon
        return 2*(epslon -y)* epslon**2
        #return np.exp(-y/epslon)/ epslon 
        #return np.exp(-y * y / (epslon * epslon))

    def quadrature_weights(self):
        q_w= np.zeros(self.M) # quadrature weights given by left endpoint
        for k in range(self.M):
            #q_w[k] = quad(self.nonlocal_kernel_w, k*self.h, min((k+1)*self.h, self.epslon))[0] #self.nonlocal_kernel_w(k*self.h) * self.h
            q_w[k] = self.nonlocal_kernel_w(k * self.h)
        return q_w / np.sum(q_w) # / self.h) * self.h cancels out
    
    def approx_velocity(self,U0):
        Vj= np.zeros(self.K)
        I = np.zeros(self.M)

        for j in range(self.K):
            for k in range(self.M):
                I[k] = self.q_w[k] * U0[min(j + k, self.K - 1)]

            Vj[j]= 1 - np.sum(I)

        return Vj 

    def nonlocal_solver(self, U0, bt, boundary_condition):
        """Solving the nonlocal conservation laws at every time step:
        Input: array vector with the solution at t = 0
        Returns: Plot_data with the solution vector at each time step t"""
        
        h = self.h
        dt = self.dt
        U1 = np.zeros(len(U0))  # Vector solution at next time step
        plot_data = [U0.copy()]  # Store initial data and the updated data
        
        for i in range(1, self.N + 1):  

            if boundary_condition == "dirichlet":
                U1[0] = U0[0]
                U1[self.K - 1] = U0[self.K - 1]

            elif boundary_condition == "periodic":
                pass

            elif boundary_condition == "artificial":
                pass

            Vj= self.approx_velocity(U0)

            if np.any(np.isnan(Vj)):
                print("nan")

            print(100 * i / self.N)

            for j in range(1, self.K - 1):  # Avoid index out of range
                F_R = (U0[j]* Vj[j])/2 + (U0[j+1]*Vj[j+1])/2 + self.alpha / 2 * (U0[j] - U0[j+1]) 
                F_L = (U0[j-1]* Vj[j-1])/2 + (U0[j]*Vj[j])/2 + self.alpha / 2 * (U0[j-1] - U0[j]) 
                U1[j] = U0[j] - dt / h * (F_R - F_L) # Correct Lax-Friedrichs formula
            
            plot_data.append(U1.copy())  # Store the result at each time step
            U0 = U1.copy()

        return plot_data

#u0 = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)
u0 = lambda x: 0.1 if x < 0.5 else 0.6  # Initial condition
#u0 = lambda x: 1 if x < 0 else 0
alpha = 2.0
epslon_1 = 0.01
#epslon_2 = 0.000001
a = Nonlocal_Conservation_laws(1, 0, 1, 1000, 4000 ,epslon_1, alpha)
b= Local_Conservation_Laws(1, 0, 1, 1000, 4000)
bx_a, bt_a = a.create_mesh()
bx_b , bt_b = b.create_mesh()  
U0_a = a.initial_value(u0, bx_a)
U0_b = b.initial_value(u0,bx_b)
plot_data_1 = a.nonlocal_solver(U0_a, bt_a, "dirichlet")  
plot_data_2 = b.local_solver(U0_b, bt_b, "dirichlet") 




    #print(bt)

    # Plotting the result at t = 1
final_time = 1
plt.plot(bx_a, plot_data_1[-1], label=f'nonlocal at t ={final_time}')
plt.plot(bx_b, plot_data_2[-1], label=f'local at time={final_time}')   # Plot the final time state
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Lax-Friedrichs Method')
plt.legend()
plt.show()
#print(a.M)
""" Neste gang bør jeg sjekke om jeg har riktig scheme for nonlocal flux. 
sjekke nonlocal parameterene ,m, epslon, aplha dt/h"""  