import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from local_conservation_laws import Local_Conservation_Laws
from nonlocal_conservation_laws import Nonlocal_Conservation_laws 

def plot_noncal():
    # u_l > u_r
    u0_l = lambda x: 0.6 if x < 0 else 0.1  # Initial condition Reimann with UL > UR
    en_l = lambda x, t: 0.6 if x <= -0.2*t else (t - x)/(2*t) if x <= 0.8*t else 0.1 # entropy solution
    # u_l < u_r
    u0_r = lambda x: 0.1 if x < 0 else 0.6  # Initial condition Reimann with UR > UL
    en_r = lambda x,t: 0.1 if x < 0.3*t else 0.6 # entropy solution 
    
    u0_e = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)
    alpha = 2
    epsilon = 0.1
   
   #(delta x, delta t)--- E1(400,200) E2(550,400), E3(1000, 650)
    # the parameters (epsilon, K, N) in exp_1a are set to (0.1, 400, 250)
    # the parameters (epsilon, K, N) in exp_1b are set to (0.01, 550, 400)
    # the parameters (epsilon, K, N) in exp_1c are set to (0.001, 1000, 650)
    K = 200
    N=  K + 20
    x_L= -1
    x_R= 1
    T = 0.5
    
    b = Nonlocal_Conservation_laws(T, x_L, x_R, K, N ,epsilon, alpha)
    c = Local_Conservation_Laws(T, x_L, x_R, K, N)
    bx, bt = b.create_mesh()
  
    U0 = b.initial_value(u0_l, bx)
   
    
    plot_data_local = b.nonlocal_solver(U0, "artificial", "local", "left endpoints", "Lax-F") # we use the scheme for the nolocal solver
    plot_data_nonlocal = b.nonlocal_solver(U0,"artificial","nonlocal", "left endpoints", "Lax-F")
    
    plot_data_local_N = b.nonlocal_solver(U0, "artificial", "local", "Normalized left endpoints","Lax-F") # we use the scheme for the nolocal solver
    plot_data_nonlocal_N = b.nonlocal_solver(U0,"artificial","nonlocal", "Normalized left endpoints", "Lax-F")
    #plot_data_nonlocal_G = b.nonlocal_solver(U0,"artificial","nonlocal", "Normalized left endpoints", "Godunov")
    
    plot_data_local_F = c.local_solver(U0, "artificial")  # we use the scheme for the local solver 
    
    solution_at_t = plot_data_nonlocal[-1] # last index which is used to save the solution at t = 1
    l1, entropy_solution = b.l1_error(bx, T, solution_at_t, en_l)

        # Plotting the result at t = 1
    final_time = 1
    
        # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    
    ax1.plot(bx, plot_data_local[-1], 'r--', label=f'local at t ={final_time}')
    ax1.plot(bx, plot_data_nonlocal[-1],'b' ,label=f'nonlocal at t ={final_time}')
    #ax1.plot(bx, plot_data_local_F[-1],'green', label=f'nonlocal_F at t ={final_time}')
    ax1.plot(bx, entropy_solution, 'black', label=f'entropy solution at t={final_time}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.set_title('A Lax-F scheme with left endpoint')
    ax1.legend()
    #plt.title('A Lax-F scheme with left endpoints')
    #plt.show() 

    ax2.plot(bx, plot_data_local_N[-1], 'r--', label=f'local at t ={final_time}')
    ax2.plot(bx, plot_data_nonlocal_N[-1],'b' ,label=f'nonlocal at t ={final_time}')
    #ax2.plot(bx, plot_data_nonlocal_G[-1],'green' ,label=f'nonlocal at t ={final_time}')
    #ax2.plot(bx, plot_data_local_F[-1],'green', label=f'nonlocal_F at t ={final_time}')
    ax2.plot(bx, entropy_solution,'black',label=f'entropy solution at t={final_time}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.set_title('A Lax-F scheme with normalized left endpoint')
    #plt.title('A Lax-F scheme with normalized left endpoints')
    ax2.legend()
    #plt.savefig("exp_1c.pdf", dpi=400, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    
plot_noncal()