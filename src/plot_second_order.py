import numpy as np 
import matplotlib.pyplot as plt
#from scipy.integrate import quad
import sympy as sp
from second_order_nonlocal import Second_Order_Nonlocal

  
def plot_second_order():
            # u_l > u_r
    u0_l = lambda x: 0.6 if x < 0 else 0.1  # Initial condition Reimann with UL > UR
    en_l = lambda x, t: 0.6 if x <= -0.2*t else (t - x)/(2*t) if x <= 0.8*t else 0.1 # entropy solution
    # u_l < u_r
    u0_r = lambda x: 0.1 if x < 0 else 0.6  # Initial condition Reimann with UR > UL
    en_r = lambda x,t: 0.1 if x < 0.3*t else 0.6 # entropy solution 
    
    u0_smooth = lambda x: 0.4 + 0.4*np.exp(-100*(x-0.5)**2)
    
    u0_paper = lambda x: 0.8 if - 0.5 < x < 0 else 0
    alpha = 2
    epsilon = 0.1
    K = 200
    N= K + 20
    x_L= -1
    x_R= 1
    T = 0.5
    
    b = Second_Order_Nonlocal(T, x_L, x_R, K, N ,epsilon, alpha)
    #c = Local_Conservation_Laws(T, x_L, x_R, K, N)
    bx, bt = b.create_mesh()

    U0 = b.initial_value(u0_l, bx)
    

    plot_data = b.second_order_solver(U0)
    solution_at_1 = plot_data[-1]
    #l1, entropy_solution = b.l1_error(bx, T, solution_at_1, en_r)
    plt.plot(bx,solution_at_1)
    #plt.plot(bx,entropy_solution)
    plt.show()
plot_second_order()