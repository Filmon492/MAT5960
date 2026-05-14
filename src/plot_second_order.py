import numpy as np 
import matplotlib.pyplot as plt
#from scipy.integrate import quad
import sympy as sp
from second_order_nonlocal import Second_Order_Nonlocal
from convergence_rates_first_order import Convergence_Rates_First_Order

  
def plot_second_order():
            # u_l > u_r
    u0_l = lambda x: 0.6 if x < 0 else 0.1  # Initial condition Reimann with UL > UR
    en_l = lambda x, t: 0.6 if x <= -0.2*t else (t - x)/(2*t) if x <= 0.8*t else 0.1 # entropy solution
    # u_l < u_r
    u0_r = lambda x: 0.1 if x < 0 else 0.6  # Initial condition Reimann with UR > UL
    en_r = lambda x,t: 0.1 if x < 0.3*t else 0.6 # entropy solution 
    
    u0_test = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)
    u0_smooth = lambda x: 0.5 + 0.3*np.cos(np.pi * x / 2)

    u0_paper = lambda x: 0.40 + 0.08*np.exp(-6*x**2)
    u0_smooth_p = lambda x : np.exp(-1/(1-(x/2)**2)) if abs(x) < 0.5 else 0
    u0_sin = lambda x : 0.5 + 0.4*np.sin(np.pi*x)
    
    alpha = 2
    epsilon_ref = 0.00001
    K_ref = 4096
    N_ref= 4*K_ref 
    x_L= -2
    x_R= 2
    T = 1
    b = Second_Order_Nonlocal(T, x_L, x_R, K_ref, N_ref , epsilon_ref, alpha)
    #c = Local_Conservation_Laws(T, x_L, x_R, K, N)
    bx_ref, bt = b.create_mesh()

    U0 = b.initial_value(u0_smooth, bx_ref)
    plot_data_1 = b.second_order_solver(U0, "absorbing", store_history=False)
    plot_data_2 = b.second_order_solver(U0,"periodic", store_history=False)
    solution_at_1 = plot_data_1[-1]
    
    
    #epsilon_nonlocal = 0.1 
    #K_Nonlocal = 2560
    #N_nonlocal=  4 * K_Nonlocal 
    c = Convergence_Rates_First_Order(T, x_L, x_R, K_ref, N_ref, epsilon_ref, alpha)
    bx_ref, bt = c.create_mesh() 
    U0 = c.initial_value(u0_smooth, bx_ref)  # initial condition

    # Godunov first order scheme 
    reference_solutions_G = c.nonlocal_solver(U0, "periodic", "local", "Normalized left endpoints", "Godunov", store_history=False)
    #reference_sol_G = reference_solutions_G[-1]
    
    #l1, entropy_solution = b.l1_error(bx_ref, T, solution_at_1, en_r)
    #plt.plot(bx_ref,solution_at_1, label = f"second order absorbing")
    plt.plot(bx_ref,plot_data_2, label = f"second order periodic")
    plt.plot(bx_ref, reference_solutions_G, label = f"first order")

    #plt.plot(bx_ref,en_l)
    plt.legend()
    plt.show()
plot_second_order()