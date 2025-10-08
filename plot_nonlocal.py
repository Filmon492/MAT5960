import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from local_conservation_laws import Local_Conservation_Laws
from nonlocal_conservation_laws import Nonlocal_Conservation_laws 

def plot_noncal():
    u0 = lambda x: 0.6 if x < 0 else 0.1  # Initial condition
    alpha = 2.0
    epsilon = 0.01
   
    K = 1000
    N = 500
    x_L= -1
    x_R= 1
    T = 0.5 
    
    
    a = Nonlocal_Conservation_laws(T, x_L, x_R, K, N ,epsilon, alpha)
    b = Nonlocal_Conservation_laws(T, x_L, x_R, K, N ,epsilon, alpha)
    c = Local_Conservation_Laws(T, x_L, x_R, K, N)
    bx, bt = a.create_mesh()
  
    U0 = a.initial_value(u0, bx)
   
    
    
    plot_data_local = a.nonlocal_solver(U0, "dirichlet", "local") # we use the scheme for the nolocal solver
    plot_data_nonlocal = b.nonlocal_solver(U0,"dirichlet","nonlocal")
    plot_data_local_F = c.local_solver(U0, "dirichlet")  # we use the scheme for the local solver 
    
    solution_at_t = plot_data_nonlocal[-1] # last index which is used to save the solution at t = 1
    l1, entropy_solution = a.l1_error(bx, T, solution_at_t)

        # Plotting the result at t = 1
    final_time = 0.5
    plt.plot(bx, plot_data_local[-1], 'r--', label=f'local at t ={final_time}')
    plt.plot(bx, plot_data_nonlocal[-1],'b' ,label=f'nonlocal at t ={final_time}')
    plt.plot(bx, plot_data_local_F[-1],'green', label=f'nonlocal_F at t ={final_time}')
    plt.plot(bx, entropy_solution, label=f'entropy solution at t={final_time}')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Lax-Friedrichs Method')
    plt.legend()
    plt.show()
plot_noncal()