import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp

x,t = sp.symbols("x,t")
class Local_Conservation_Laws:
    """Class for solving the scalar conservation laws using the Lax_Friedrichs scheme"""
    def __init__(self, T, x_L, x_R , K, N,):
        """Initialize the parameters

        Parameters
        ----------

        T : The end point of the time.
        x_L : The left end point of the domain [x_L, x_R]
        x_R : The right end point of the domain [x_L, x_R]  
        K : Number of mesh points in space
        N : Number of  mesh points in time
        """
        self.K = K
        self.N = N
        self.T= T
        self.x_L = x_L
        self.x_R = x_R
        self.h = (self.x_L - self.x_R)/(self.K+1)
        self.dt = self.T/(self.N+1)
        

    def create_mesh(self):
        """Create mesh points for both x and t"""
        h = self.h ; dt= self.dt
        X = np.zeros(self.K) #consists midpoint values of each cell C_j in space 
        Y = np.zeros(self.N) #consists mesh points in time 
        for j in range(self.N):
            X[j]= self.x_L + (j + 1/2) * h
            Y[j] = dt * j
        return X, Y
    
    def  cell_average(self, u0, c, d):
        """ Computes the averge value at each discrete point xj"""
        C_j = quad(u0, c, d)[0]
        return C_j
    
    def initial_value(self, u0,bx,bt):
        #u0_sympy = u0(x)
        self.bx = bx
        self.bt = bt
        self.U0= np.zeros(self.N)
        for i in range(len(bt)):
            for j in range(len(bx)):
                self.U0[i]= self.cell_average(u0, bx[i]-self.h/2, bx[i]+self.h/2 )
        return self.U0
    
    
    
a = Local_Conservation_Laws(1,0,1,5,5)
u0 = lambda x: x**2
bx, bt= a.create_mesh()
#print( a.cell_average( u0, 0, 1,))
U0= a.initial_value(u0,bx,bt)
print(U0)