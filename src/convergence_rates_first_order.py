import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from local_conservation_laws import Local_Conservation_Laws
from nonlocal_conservation_laws import Nonlocal_Conservation_laws

class Convergence_Rates_First_Order(Nonlocal_Conservation_laws):
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
 
    
    def convergence_first_order_scheme(self, m, T, u0 , reference_solution, bx_ref, numerical_fluxes, limit, bc, entropy):
        epsilon = self.epsilon
        K = self.K
        N = self.N

        h_values = []
        E_values = []
        number_of_cells = []
        #T = 1 
        #alpha = 2 
        r = np.zeros(m-1)

        for i in range(m):
            if limit == "nonlocal":
                pass #epsilon = epsilon
            if limit == "local":
                epsilon = epsilon/2
            K = 2*K
            N = 2*N
            h = (self.x_R - self.x_L) / K # define mesh size
    
            a = Convergence_Rates_First_Order(self.T, self.x_L, self.x_R, K, N, epsilon, self.alpha)
            #a = Convergence_Rates(T, x_L, x_R, K, N, epsilon, alpha)
            bx, bt = a.create_mesh() 
            U0 = a.initial_value(u0, bx)  # initial condition
            numerical_sol = a.nonlocal_solver(
                U0,
                bc,
                "nonlocal",
                "Normalized left endpoints",
                numerical_fluxes,
                store_history=False,
            )
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

            r[i-1] = (np.log(error_prev / error_curr)) / (np.log(h_values[i-1] / h_values[i]))
        return r, E_values, number_of_cells


""" 
# Initial conditions and parameters
u0_l = lambda x: 0.6 if x < 0 else 0.1  # Initial condition with UL > UR
en_l = lambda x, t: 0.6 if x <= -0.2*t else (t - x)/(2*t) if x <= 0.8*t else 0.1 # entropy solution


# u_l < u_r
u0_r = lambda x: 0.1 if x < 0 else 0.6  # Initial condition Reimann with UR > UL
en_r = lambda x,t: 0.1 if x < 0.3*t else 0.6 # entropy solution 

u0_test = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)         # smooth intitial data
u0_paper = lambda x: 0.8 if -0.5 < x < 0.1 else 0  
u0_smooth = lambda x: 0.5+ 0.3 * np.exp(-x**2)
    
alpha = 2
x_L = -2
x_R = 2
#epsilon fixed
#epsilon = 0.1 
K = 8
#apply this epsilon when testeng both delta x and epslion tend to zero
epsilon = 3*(x_R -x_L)/K # m * delta x, that means the convolution takes 3 cells
N = 4*K
T = 1
m = 7

# computing the referce solution
epsilon_nonlocal = 0.1
K_Nonlocal = 2560
N_nonlocal=  4 * K_Nonlocal 
c = Nonlocal_Conservation_laws(T, x_L, x_R, K_Nonlocal, N_nonlocal ,epsilon_nonlocal, alpha)
bx_ref, bt = c.create_mesh() 


U0 = c.initial_value(u0_smooth, bx_ref)  # initial condition
numerical_solutions_G = c.nonlocal_solver(U0, "artificial", "local", "Normalized left endpoints", "Godunov")
entropy_sol_G = numerical_solutions_G[-1]
numerical_solutions_F = c.nonlocal_solver(U0, "artificial", "local", "Normalized left endpoints", "Lax-F")
entropy_sol_F = numerical_solutions_F[-1]

b = Convergence_Rates(T, x_L, x_R, K, N, epsilon, alpha)
r_G, E_G, nx_G = b.convergence_rates(m, u0_smooth, bx_ref, entropy_sol_G, "Godunov")
r_F, E_F, nx_F = b.convergence_rates(m, u0_smooth,bx_ref, entropy_sol_F, "Lax-F")


def latex_table(nx, E, r, fmt_r="{:.4f}"):
    def sci(x):
        #Return x in LaTeX scientific notation 'a.bcd x 10^{-k}' format.
        a = f"{x:.4e}"
        base, exp = a.split("e")
        exp = int(exp)
        return f"{base} \\times 10^{{{exp}}}"

    pairs = sorted(zip(nx, E, [None] + list(r)), key=lambda t: t[0])

    lines = []
    for nx_i, E_i, r_i in pairs:
        E_fmt = sci(E_i)

        if r_i is None:  # first row has no rate
            lines.append(f"{int(nx_i)} & {E_fmt} & -- \\\\")
        else:
            lines.append(f"{int(nx_i)} & {E_fmt} & {fmt_r.format(r_i)} \\\\")

    return "\n".join(lines)
print(latex_table(nx_F, E_F, r_F, fmt_r="{:.4f}"))
print("----")
print(latex_table(nx_G, E_G, r_G, fmt_r="{:.4f}"))

#print(len(E))


#print("Convergence Rates:", r) 
 
plt.plot(nx_G, E_G, label=f'Godunov')
plt.plot(nx_F, E_F, label=f'Lax-F')
#plt.plot(nx_Ft, E_Ft, label=f'Lax-Ft')
plt.xlabel('number of cells')
plt.ylabel('$\ell^1$ error ')
#plt.title('Convergence Analysis')
plt.xscale('log')
plt.yscale('log')
plt.legend()
#plt.savefig("Conver_1_smooth.pdf", dpi=400, bbox_inches="tight")
plt.show() 
"""
