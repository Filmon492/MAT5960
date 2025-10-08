import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
from local_conservation_laws import Local_Conservation_Laws
def plot_local():
    #u0 = lambda x: 0 if x < 0 else 1 # Initial condition
    u0 = lambda x: 0.6 if x < 0 else 0.1
    #u0 = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)
    K = 2000
    N = 4000
    x_L= -1
    x_R= 1
    T = 0.5 
    #t= 1
   
    a = Local_Conservation_Laws(T, x_L, x_R, K, N)
    bx, bt = a.create_mesh()  
    U0 = a.initial_value(u0, bx) 
    plot_data = a.local_solver(U0, "dirichlet") 
    numerical_solution = plot_data[-1] 

    l1, entropy_solution = a.l1_error(bx, T, numerical_solution)
    print(f'l2 error is {l1}')
    print("----")

   

    # Plotting the result at t = 1
    final_time = 0.5
    plt.plot(bx, plot_data[-1], label=f'numerical solution at time={final_time}')  # Plot the final time state
    plt.plot(bx, entropy_solution, label=f'entropy solution at time={final_time}')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Lax-Friedrichs Method')
    plt.legend()
    plt.show()
plot_local()