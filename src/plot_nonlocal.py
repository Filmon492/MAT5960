import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from nonlocal_conservation_laws import Nonlocal_Conservation_laws 
from second_order_nonlocal import Second_Order_Nonlocal

def plot_noncal():
    # u_l > u_r
    u0_l = lambda x: 0.6 if x < 0 else 0.1  # Initial condition Reimann with UL > UR
    en_l = lambda x, t: 0.6 if x <= -0.2*t else (t - x)/(2*t) if x <= 0.8*t else 0.1 # entropy solution
    # u_l < u_r
    u0_r = lambda x: 0.1 if x < 0 else 0.6  # Initial condition Reimann with UR > UL
    en_r = lambda x,t: 0.1 if x < 0.3*t else 0.6 # entropy solution 
    
    u0_e = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)
    alpha = 2
    epsilon = 0.001
   
   
    # the parameters (epsilon, K, N) in exp_1a are set to (0.1, 200, 2*K)
    # the parameters (epsilon, K, N) in exp_1b are set to (0.01, 400, 2*K)
    # the parameters (epsilon, K, N) in exp_1c are set to (0.001, 1000, 2*k)
    
    x_L= -2
    x_R= 2
    T = 1
    K = 4000
    N = 2*K 
    #epsilon = (x_R-x_L)/K
    
    bc = "absorbing"
    u0_test = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)
    u0_smooth = lambda x: 0.5+ 0.3 * np.exp(-x**2)  
    
    b = Nonlocal_Conservation_laws(T, x_L, x_R, K, N ,epsilon, alpha)
    bx, bt = b.create_mesh()
  
    U0 = b.initial_value(u0_l, bx)
   
    
    plot_data_local = b.nonlocal_solver(U0, bc, "local", "left endpoints", "Lax-F") # we use the scheme for the nolocal solver
    plot_data_nonlocal = b.nonlocal_solver(U0, bc, "nonlocal", "left endpoints", "Lax-F")
    
    plot_data_local_N = b.nonlocal_solver(U0, bc, "local", "Normalized left endpoints","Lax-F") # we use the scheme for the nolocal solver
    plot_data_nonlocal_N = b.nonlocal_solver(U0, bc, "nonlocal", "Normalized left endpoints", "Lax-F")
    
    #plot_data_local_G = b.nonlocal_solver(U0, bc, "local", "Normalized left endpoints", "Godunov")
    #plot_data_nonlocal_G = b.nonlocal_solver(U0, bc, "nonlocal", "Normalized left endpoints", "Godunov")
    
    solution_at_t = plot_data_nonlocal[-1] # last index which is used to save the solution at T = 1
    l1, entropy_solution = b.l1_error(bx, T, solution_at_t, en_l)

    epsilon_ref = 0.0001
    b = Second_Order_Nonlocal(T, x_L, x_R, K, N , epsilon_ref, alpha)
    #c = Local_Conservation_Laws(T, x_L, x_R, K, N)
    bx, bt = b.create_mesh()

    U0 = b.initial_value(u0_smooth, bx)
    

    plot_data = b.second_order_solver(U0, bc)
    solution_at_1 = plot_data[-1]
        # Plotting the result at t = 1
    final_time = 1
    
        # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    
    ax1.plot(bx, plot_data_local[-1], 'r--', label=f'local at T ={final_time}')
    ax1.plot(bx, plot_data_nonlocal[-1],'b' ,label=f'nonlocal at T ={final_time}')
    ax1.plot(bx, entropy_solution, 'black', label=f'entropy solution at T ={final_time}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    #ax1.set_title('A Lax-F scheme using the left endpoint')
    ax1.legend() 

    ax2.plot(bx, plot_data_local_N[-1], 'r--', label=f'local at T ={final_time}')
    ax2.plot(bx, plot_data_nonlocal_N[-1],'b' ,label=f'nonlocal at T ={final_time}')
    ax2.plot(bx, entropy_solution,'black',label=f'entropy solution at T={final_time}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    #ax2.set_title('A Lax-F scheme using the normalized left endpoint')
    ax2.legend()
    plt.savefig("exp_1c.pdf", dpi=400, bbox_inches="tight")
    plt.tight_layout()
    plt.show() 
    
    """ 
    
    plt.plot(bx,solution_at_1, label=f' Godunov second order')
    #plt.plot(bx, plot_data_local_N[-1], label=f'Lax-F')
    plt.plot(bx,plot_data_local_G[-1], label=f' Godunov first order')
    #plt.plot(bx, entropy_solution,label=f' entropy solution' )
    #plt.savefig("s.pdf", dpi=400, bbox_inches="tight")
    plt.legend()
    plt.show() """
plot_noncal()
