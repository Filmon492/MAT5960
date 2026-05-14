import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
#from nonlocal_conservation_laws import Nonlocal_Conservation_laws
from convergence_rates_first_order import Convergence_Rates_First_Order
from convergence_rates_second_order import Convergence_Rates_Second_Order

# parematers
alpha = 2
x_L = -2
x_R = 2
epsilon = 0.1
K = 8
N = 2*K
T = 1
m = 7

# initial data
u0_l = lambda x: 0.6 if x < 0 else 0.1  # Reiman initial condition with UL > UR, where the solution is rarefaction wave 
u0_r = lambda x: 0.1 if x < 0 else 0.6  # Reimann initial condition with UR > UL, where the solution is discontinuous 
u0_smooth = lambda x: 0.5 + 0.3*np.cos(np.pi * x / 2)                # smooth intitial data, where the solution is smooth
#u0_test = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)         # smooth intitial data
u0_smooth_p = lambda x : np.exp(-1/(1-x**2)) if abs(x) < 1 else 0
u0_sin = lambda x : 0.5 + 0.4*np.sin(np.pi*x)
# Testing the convergence of the first order scheme to the nonlocal model 

    #compute the reference solution for the first order scheme 
epsilon_ref = 0.1 
K_ref = 4096
N_ref=  4 * K_ref
c = Convergence_Rates_First_Order(T, x_L, x_R, K_ref, N_ref, epsilon_ref, alpha)
bx_ref, bt = c.create_mesh() 
U0 = c.initial_value(u0_smooth, bx_ref)  # initial condition

# Godunov first order scheme 
reference_sol_G = c.nonlocal_solver(
    U0,
    "periodic",
    "nonlocal",
    "Normalized left endpoints",
    "Godunov",
    store_history=False,
)

# Lax--Friedrichs first order scheme
reference_sol_F = c.nonlocal_solver(
    U0,
    "periodic",
    "nonlocal",
    "Normalized left endpoints",
    "Lax-F",
    store_history=False,
)

# Outputs for first order Godunov and Lax-F
b = Convergence_Rates_First_Order(T, x_L, x_R, K, N, epsilon, alpha)
r_G, E_G, nx_G = b.convergence_first_order_scheme(m, T, u0_smooth, reference_sol_G, bx_ref, "Godunov", "nonlocal", "periodic", "reference_sol")
r_F, E_F, nx_F = b.convergence_first_order_scheme(m, T, u0_smooth, reference_sol_F, bx_ref, "Lax-F", "nonlocal",  "periodic", "reference_sol")

# Testing the convergence of the second order scheme tsmooth the nonlocal model 

c2 = Convergence_Rates_Second_Order(T, x_L, x_R, K_ref, N_ref, epsilon_ref, alpha)
bx_ref, bt = c2.create_mesh() 
U0 = c2.initial_value(u0_smooth, bx_ref)  # initial condition
reference_sol_G_second_order = c2.second_order_solver(U0, "periodic", store_history=False)

b2 = Convergence_Rates_Second_Order(T, x_L, x_R, K, N, epsilon, alpha)
r_G2, E_G2, nx_G2 = b2.convergence_second_order_scheme(m, T, u0_smooth, reference_sol_G_second_order, bx_ref, "nonlocal", "periodic", "reference_sol")




def latex_table(nx, E, r, fmt_r="{:.4f}"):
    def sci(x):
        """Return x in LaTeX scientific notation 'a.bcd x 10^{-k}' format."""
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
print("-----")
print(latex_table(nx_G2, E_G2, r_G2, fmt_r="{:.4f}"))

plt.plot(nx_G, E_G, label=f'Godunov')
#plt.plot(nx_G2, E_G2, label=f'Godunov_second')
plt.plot(nx_F, E_F, label=f'Lax-F')
plt.xlabel('number of cells')
plt.ylabel('$L^1$ error ')
#plt.title('Convergence Analysis')
plt.xscale('log')
plt.yscale('log')
plt.legend()
#plt.savefig("Conver_2l.pdf", dpi=400, bbox_inches="tight")
plt.show()
