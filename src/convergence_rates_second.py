import numpy as np 
import matplotlib.pyplot as plt
#from scipy.integrate import quad
import sympy as sp
from second_order_nonlocal import Second_Order_Nonlocal

class Convergence_Rates(Second_Order_Nonlocal):
    """Class for computing the order of convergence rate as both h and epsilon go to zero"""
    def __init__(self, T, x_L, x_R, K, N, epsilon, alpha):
        super().__init__(T, x_L, x_R, K, N, epsilon, alpha)
    
    def l1_error(self, bx, numerical_sol, K , h, entropy_sol, bx_ref):
        # Interpolate fine-grid reference to current grid
        entropy_sol_coarse = np.interp(bx, bx_ref, entropy_sol)

        # Compute L1 error 
        error_values = entropy_sol_coarse - numerical_sol
        return h * np.sum(np.abs(error_values))
    
    def convergence_rates(self, m, u0 , entropy_sol, bx_ref):
        
        epsilon = self.epsilon
        K = self.K
        N = self.N

        h_values = []
        E_values = []
        number_of_cells = []
        r = np.zeros(m-1)

        for i in range(m):
            #epsilon = epsilon/2
           
            K = 2*K
            N = 2*N
            h = (self.x_R - self.x_L) / K # define mesh size
            
            #a = Convergence_Rates(T, x_L, x_R, K, N, epsilon, alpha)
            a = Convergence_Rates(self.T, self.x_L, self.x_R, K, N, epsilon, self.alpha)
            bx, bt = a.create_mesh() 
            U0 = a.initial_value(u0, bx)  # initial condition
            numerical_solutions = a.second_order_solver(U0)
            numerical_sol = numerical_solutions[-1]
            E = self.l1_error(bx, numerical_sol, K, h, entropy_sol, bx_ref)
            #E, _ = self.l1_error(bx, T, numerical_sol, K, h, entropy_sol)
            h_values.append(h)
            E_values.append(E)
            number_of_cells.append(K)

        # Compute convergence rates
        for i in range(1, m):
            error_prev = E_values[i-1]  # Previous error
            error_curr = E_values[i]    # Current error

            if error_prev <= 0 or error_curr <= 0:
                raise ValueError(f"Invalid error values: E_prev={error_prev}, E_curr={error_curr}")

            r[i-1] = (np.log(error_prev / error_curr)) / (np.log(h_values[i-1] / h_values[i]))
        return r, E_values, number_of_cells

# Initial conditions and parameters
u0_l = lambda x: 0.6 if x < 0 else 0.1  # Initial condition with UL > UR
en_l = lambda x, t: 0.6 if x <= -0.2*t else (t - x)/(2*t) if x <= 0.8*t else 0.1 # entropy solution


# u_l < u_r
u0_r = lambda x: 0.1 if x < 0 else 0.6  # Initial condition Reimann with UR > UL
en_r = lambda x,t: 0.1 if x < 0.3*t else 0.6 # entropy solution 
    
alpha = 2
epsilon = 0.1
K = 10
N =  K + 20
x_L = -1
x_R = 1
T = 0.15
m = 7

epsilon_nonlocal = 0.1
K_Nonlocal = 2560
N_nonlocal= 4*K_Nonlocal
c = Convergence_Rates(T, x_L, x_R, K_Nonlocal, N_nonlocal, epsilon_nonlocal, alpha)
bx_ref, bt = c.create_mesh() 

u0_paper = lambda x: 0.8 if -0.5 < x < 0.1 else 0               # discontinuous initial data
u0_smooth = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)         # smooth intitial data

U0 = c.initial_value(u0_r, bx_ref)  # initial condition
numerical_solutions = c.second_order_solver(U0)
entropy_sol = numerical_solutions[-1]

#etropy = c.second_order_solver()

b = Convergence_Rates(T, x_L, x_R, K, N, epsilon, alpha)
r_G, E_G, nx_G = b.convergence_rates(m, u0_r, entropy_sol, bx_ref)

#r_G, E_G, nx_G = b.convergence_rates(m, T, u0_r , en_r)

def latex_table(nx, E, r, fmt_r="{:.4f}"):
    # Pair and sort by nx
    pairs = sorted(zip(nx,E, r), key=lambda t: t[0])
    lines = []
    for nx_i, E_i, r_i in pairs:
        lines.append(f"{int(nx_i)} & {fmt_r.format(E_i)} & {fmt_r.format(r_i)} \\\\")
    return "\n".join(lines)

print(latex_table(nx_G, E_G, r_G, fmt_r="{:.4f}"))

plt.plot(nx_G, E_G, label=f'Godunov')
plt.xlabel('number of cells')
plt.ylabel('$\ell^1$ error ')
#plt.title('Convergence Analysis')
plt.xscale('log')
plt.yscale('log')
plt.legend()
#plt.savefig("Conver_2.pdf", dpi=400, bbox_inches="tight")
plt.show()