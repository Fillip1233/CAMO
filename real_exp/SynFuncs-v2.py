import numpy as np
import matplotlib.pyplot as plt
import random
import string
import os
import subprocess

from sklearn.preprocessing import StandardScaler

from hdf5storage import loadmat
from hdf5storage import savemat

class SfFn1:
    def __init__(self):
        self.dim = 1
        self.flevels = 1
        self.maximum = 1.90
        
        self.bounds = ((0.0, 10.0))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def generate(self, N, m):
        X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
        y = self.query(X, m).reshape([-1,1])

        return X, y

    def query(self, X, m):
        f = np.sin(X) + np.sin(10*X/3)
        return -f
    
class SfFn2:
    def __init__(self):
        self.dim = 1
        self.flevels = 1
        self.maximum = 0.487
        
        self.bounds = ((0.0, 4.0))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def generate(self, N, m):
        X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
        y = self.query(X, m).reshape([-1,1])

        return X, y

    def query(self, X, m):
        f = np.exp(-X) * np.sin(2*np.pi*X)
        return -f
    
class SfFn3:
    def __init__(self):
        self.dim = 1
        self.flevels = 1
        self.maximum = 9.51
        
        self.bounds = ((0.0, 10.0))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def generate(self, N, m):
        X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
        y = self.query(X, m).reshape([-1,1])

        return X, y

    def query(self, X, m):
        f = X*np.sin(X) + X*np.cos(2*X)
        return -f


class SynFn1:
    def __init__(self, debug=False):
        self.dim = 1
        self.flevels = 3     
        self.maximum = -0.0014033987
        
        self.bounds = ((0.0, 10.0))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
            
        self.Flist = []
        self.Flist.append(self.eval_f0)
        self.Flist.append(self.eval_f1)
        self.Flist.append(self.eval_f2)
        
    def generate(self, N, m):
        X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
        y = self.query(X, m).reshape([-1,1])

        return X, y
    
    def query(self, X, m):

        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)


        return ym

    def eval_f0(self, xn):
        f = -np.square(xn-3)
        return f  
    
    def eval_f1(self, xn):
        f = -3 * np.square((xn-3)) - 4
        return f
    
    def eval_f2(self, xn):
        f = -3 * np.square((xn-3)) - 0.01*np.sin(xn)
        return f
    
#     function [mff, sff] = getMFStybTang(numFidels, numDims, costs)

#     numFidels = 2;
#     numDims = 2;

#     f2 = @(x) 0.5*sum(x.^4 - 16 * x.^2 + 5 * x);
#     f1 = @(x) 0.5*sum(0.9*x.^4 - 15 * x.^2 + 6 * x);
    
#     neg_f2 = @(x) -0.5*sum(x.^4 - 16 * x.^2 + 5 * x);
#     neg_f1 = @(x) -0.5*sum(0.9*x.^4 - 15 * x.^2 + 6 * x);
    

#     funcHs = {neg_f1; neg_f2};
#     hfMaxPt = [];

#     hfMaxVal = 39.16599*2;
#     bounds = [-5,5;-5,5];
#     costs = costs';
# %     costs = (10.^(0:(numFidels-1)))';
# %     costs = (5.^(0:(numFidels-1)))';

#     mff = mfFunction(funcHs, bounds, costs, [], hfMaxPt, hfMaxVal);
#     sff = mfFunction({funcHs{numFidels}}, bounds, costs(numFidels), [], ...
#                hfMaxPt, hfMaxVal);

# end
    
class StybTang:
    def __init__(self, debug=False):
        self.dim = 2
        self.flevels = 2
        self.maximum = 39.16599*2
        
        self.bounds = ((-5.0,5.0), (-5.0,5.0))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
   
        self.Flist = []
        self.Flist.append(self.eval_fed_L0)
        self.Flist.append(self.eval_fed_L1)
        
#     def generate(self, N, m):
# #         X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
        
#         noise = np.random.uniform(0,1,size=[N,self.dim])
#         support = (self.ub - self.lb).reshape([1,-1])

#         X = noise * support + self.lb
        
#         y = self.query(X, m).reshape([-1,1])

#         return X, y
        
        
    def query(self, X, m):

        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)


        return -ym

    def eval_fed_L0(self, xn):
        
        
        f = 0.5*np.sum(0.9*xn**4 - 15*xn**2 + 6*xn)
        
        
        
        return f    

    def eval_fed_L1(self, xn):
        
        f = 0.5*np.sum(xn**4 - 16*xn**2 + 5*xn)

        return f



class CurrinExp:
    def __init__(self, debug=False):
        self.dim = 2
        self.flevels = 2
        self.maximum = 13.798702307261388
        
        self.bounds = ((0.0,1.0), (0.0,1.0))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
   
        self.Flist = []
        self.Flist.append(self.eval_fed_L0)
        self.Flist.append(self.eval_fed_L1)
        
#     def generate(self, N, m):
# #         X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
        
#         noise = np.random.uniform(0,1,size=[N,self.dim])
#         support = (self.ub - self.lb).reshape([1,-1])

#         X = noise * support + self.lb
        
#         y = self.query(X, m).reshape([-1,1])

#         return X, y
        
        
    def query(self, X, m):

        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)


        return ym

    def eval_fed_L0(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        args1 = np.array([x1+0.05, x2+0.05])
        args2 = np.array([x1+0.05, np.max(np.array([0, x2-0.05]))])
        args3 = np.array([x1-0.05, x2+0.05])
        args4 = np.array([x1-0.05, np.max(np.array([0, x2-0.05]))])
        
        f = 0.25 * (self.eval_fed_L1(args1) + self.eval_fed_L1(args2)) +\
            0.25 * (self.eval_fed_L1(args3) + self.eval_fed_L1(args4))
        
        return f    

    def eval_fed_L1(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        nom = 2300 * x1 * x1 * x1 + 1900 * x1 * x1 + 2092 * x1 + 60
        den = 100 * x1 * x1 * x1 + 500 * x1 * x1 + 4 * x1 + 20
        
        if x2 == 0:
            f = nom / den
        else:
            f = (1 - np.exp(-1/(2*x2))) * nom / den

        return f

    
class Hartmann3D:
    """ negative harmann3D, find maximum instead of global minimum """
    def __init__(self, debug=False):
        self.dim = 3
        self.flevels = 3
        self.maximum = 3.86277979

        self.bounds = ((0,1),(0,1),(0,1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

        
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        
        self.A = np.array([[3.0, 10, 30], [0.1, 10, 35],
                      [3.0, 10, 30], [0.1, 10, 35]])
        self.P = np.array([[3689, 1170, 2673], [4699, 4387, 7470], [
                     1091, 8732, 5547], [381, 5743, 8828]])*1e-4

        
#     def generate(self, N, m):
#         X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
#         y = self.query(X, m).reshape([-1,1])

#         return X, y
    
    def query(self, X, m):
        # negate function to find maximum
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        
        
        for n in range(X.shape[0]):
            xn = X[n]
            
            if m == 2:
                ym[n] = -self.eval_fed_L2(xn)
            elif m == 1:
                ym[n] = 1 * np.sqrt(-self.eval_fed_L2(xn-0.3)) + 0.2*np.exp(-self.eval_fed_L2(xn-0.15))
            elif m == 0:
                ym[n] = 0.5*np.exp(-self.eval_fed_L2(xn-0.15))

        return ym
    
    
    def eval_fed_L2(self, xn):
        outer = 0.0
        for ii in range(4):
            inner = 0.0
            for jj in range(self.dim):
                xj = xn[jj]
                Aij = self.A[ii, jj]
                Pij = self.P[ii, jj]
                inner = inner + Aij * (xj - Pij) * (xj - Pij)
            # end for
            new = self.alpha[ii] * np.exp(-inner)
            outer = outer + new
        # end for
        
        f = -outer
        return f
    

    
# class Hartmann3D2:
#     """ negative harmann3D, find maximum instead of global minimum """
#     def __init__(self, debug=False):
#         self.dim = 3
#         self.flevels = 2
#         self.maximum = 3.86277979

#         self.bounds = ((0,1),(0,1),(0,1))
#         self.lb = np.array(self.bounds, ndmin=2)[:, 0]
#         self.ub = np.array(self.bounds, ndmin=2)[:, 1]

        
#         self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        
#         self.A = np.array([[3.0, 10, 30], [0.1, 10, 35],
#                       [3.0, 10, 30], [0.1, 10, 35]])
#         self.P = np.array([[3689, 1170, 2673], [4699, 4387, 7470], [
#                      1091, 8732, 5547], [381, 5743, 8828]])*1e-4
        
# #         self.A = np.array([[3.0, 10, 30],
# #                            [0.1, 10, 35],
# #                            [3.0, 10, 30],
# #                            [0.1, 10, 35]])
# #         self.P = 1e-4* np.array([[3689, 1170, 2673],
# #                                  [4699, 4387, 7470],
# #                                  [1091, 8732, 5547],
# #                                  [381, 5743, 8828]]) 
        
# #         self.Flist = []
# #         self.Flist.append(self.eval_fed_L0)
# #         self.Flist.append(self.eval_fed_L1)
# #         self.Flist.append(self.eval_fed_L2)
        
#     def generate(self, N, m):
#         X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
#         y = self.query(X, m).reshape([-1,1])

#         return X, y
    
#     def query(self, X, m):
#         # negate function to find maximum
#         if X.ndim == 1:
#             X = np.expand_dims(X, 0)
        
#         N = X.shape[0]
#         ym = np.zeros(N)
        
        
#         for n in range(X.shape[0]):
#             xn = X[n]
            
# #             10*sqrt(h(x-3)) + 2*log(h(x-1))
            
#             if m == 1:
#                 ym[n] = -self.eval_fed_L2(xn)
#             elif m == 0:
#                 ym[n] = 10 * np.sqrt(-self.eval_fed_L2(xn-0.3)) + 2*np.exp(-self.eval_fed_L2(xn-0.1))

#         return ym
    
    
#     def eval_fed_L2(self, xn):
#         outer = 0.0
#         for ii in range(4):
#             inner = 0.0
#             for jj in range(self.dim):
#                 xj = xn[jj]
#                 Aij = self.A[ii, jj]
#                 Pij = self.P[ii, jj]
#                 inner = inner + Aij * (xj - Pij) * (xj - Pij)
#             # end for
#             new = self.alpha[ii] * np.exp(-inner)
#             outer = outer + new
#         # end for
        
#         f = -outer
#         return f

# class Hartmann3D:
#     """ negative harmann3D, find maximum instead of global minimum """
#     def __init__(self, debug=False):
#         self.dim = 3
#         self.flevels = 3
#         self.maximum = 3.86278

#         self.bounds = ((1e-12,1-1e-12),(1e-12,1-1e-12),(1e-12,1-1e-12))
#         self.lb = np.array(self.bounds, ndmin=2)[:, 0]
#         self.ub = np.array(self.bounds, ndmin=2)[:, 1]

        
#         self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
#         self.A = np.array([[3.0, 10, 30],
#                            [0.1, 10, 35],
#                            [3.0, 10, 30],
#                            [0.1, 10, 35]])
#         self.P = 1e-4* np.array([[3689, 1170, 2673],
#                                  [4699, 4387, 7470],
#                                  [1091, 8732, 5547],
#                                  [381, 5743, 8828]]) 
#     def generate(self, N, m):
#         X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
#         y = self.query(X, m).reshape([-1,1])

#         return X, y
    
#     def query(self, X, m):
#         # negate function to find maximum
#         if X.ndim == 1:
#             X = np.expand_dims(X, 0)
        
#         N = X.shape[0]
#         ym = np.zeros(N)
#         for n in range(X.shape[0]):
#             xn = X[n]
#             ym[n] = self.meta_hartmann(self.alpha - 0.1*(self.flevels-(m+1)), xn)


#         return -ym
    
    
#     def meta_hartmann(self, alpha, xn):
#         outer = 0.0
#         for ii in range(4):
#             inner = 0.0
#             for jj in range(self.dim):
#                 xj = xn[jj]
#                 Aij = self.A[ii, jj]
#                 Pij = self.P[ii, jj]
#                 inner = inner + Aij * (xj - Pij) * (xj - Pij)
#             # end for
#             new = alpha[ii] * np.exp(-inner)
#             outer = outer + new
#         # end for
        
#         f = -outer
#         return f
    
class Hartmann6D:
    """ negative hartmann6d, find maximum """
    def __init__(self, debug=False):

        self.dim = 6
        self.flevels = 2
        self.maximum = 3.32237
        
#         self.bounds = ((0,1),(1e-12,1-1e-12),(1e-12,1-1e-12),(1e-12,1-1e-12),(1e-12,1-1e-12),(1e-12,1-1e-12))
        self.bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

        
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                           [0.05, 10, 17, 0.1, 8, 14],
                           [3, 3.5, 1.7, 10, 17, 8],
                           [17, 8, 0.05, 10, 0.1, 14]])
        self.P = 1e-4* np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]]) 
        
        self.Flist = []
        self.Flist.append(self.eval_fed_L0)
        self.Flist.append(self.eval_fed_L1)
    
#     def generate(self, N, m):
        
#         noise = np.random.uniform(0,1,size=[N,self.dim])
#         support = (self.ub - self.lb).reshape([1,-1])

#         X = noise * support + self.lb
        
#         y = self.query(X, m).reshape([-1,1])

#         return X, y
    
    def query(self, X, m):
        # negate function to find maximum
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)


        return -ym

    
    def eval_fed_L0(self, xn):
        
#         f_L(x) = sin(f(x)) * sqrt( 1 + f(x)^2)
        
        fh = self.eval_fed_L1(xn)
        
        f = np.sin(fh) * np.log(1 + np.square(fh))
        
        return f    

    def eval_fed_L1(self, xn):
        
        outer = 0.0
        for ii in range(4):
            inner = 0.0
            for jj in range(self.dim):
                xj = xn[jj]
                Aij = self.A[ii, jj]
                Pij = self.P[ii, jj]
                inner = inner + Aij * (xj - Pij) * (xj - Pij)
            # end for
            new = self.alpha[ii] * np.exp(-inner)
            outer = outer + new
        # end for
        
        f = -(2.58 + outer) / 1.94

        return f
    
# class Hartmann6D:
#     """ negative hartmann6d, find maximum """
#     def __init__(self, debug=False):

#         self.dim = 6
#         self.flevels = 3
#         self.maximum = 3.32237
        
#         self.bounds = ((1e-12,1-1e-12),(1e-12,1-1e-12),(1e-12,1-1e-12),(1e-12,1-1e-12),(1e-12,1-1e-12),(1e-12,1-1e-12))
#         self.lb = np.array(self.bounds, ndmin=2)[:, 0]
#         self.ub = np.array(self.bounds, ndmin=2)[:, 1]

        
#         self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
#         self.A = np.array([[10, 3, 17, 3.5, 1.7, 8],
#                            [0.05, 10, 17, 0.1, 8, 14],
#                            [3, 3.5, 1.7, 10, 17, 8],
#                            [17, 8, 0.05, 10, 0.1, 14]])
#         self.P = 1e-4* np.array([[1312, 1696, 5569, 124, 8283, 5886],
#                                  [2329, 4135, 8307, 3736, 1004, 9991],
#                                  [2348, 1451, 3522, 2883, 3047, 6650],
#                                  [4047, 8828, 8732, 5743, 1091, 381]]) 
    
#     def generate(self, N, m):
#         X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
#         y = self.query(X, m).reshape([-1,1])

#         return X, y
    
#     def query(self, X, m):
#         # negate function to find maximum
#         if X.ndim == 1:
#             X = np.expand_dims(X, 0)
        
#         N = X.shape[0]
#         ym = np.zeros(N)
#         for n in range(X.shape[0]):
#             xn = X[n]
#             ym[n] = self.meta_hartmann(self.alpha - 0.1*(self.flevels-(m+1)), xn)


#         return -ym
    
#     def meta_hartmann(self, alpha, xn):
#         outer = 0.0
#         for ii in range(4):
#             inner = 0.0
#             for jj in range(self.dim):
#                 xj = xn[jj]
#                 Aij = self.A[ii, jj]
#                 Pij = self.P[ii, jj]
#                 inner = inner + Aij * (xj - Pij) * (xj - Pij)
#             # end for
#             new = alpha[ii] * np.exp(-inner)
#             outer = outer + new
#         # end for
        
#         f = -(2.58 + outer) / 1.94
        
#         return f

    
class Branin:
    def __init__(self, debug=False):
        self.dim = 2
        self.flevels = 3
        self.maximum = -0.397887
        
        self.bounds = ((-5,10), (0,15))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
            
        self.Flist = []
        self.Flist.append(self.eval_fed_L0)
        self.Flist.append(self.eval_fed_L1)
        self.Flist.append(self.eval_fed_L2)
        
#     def generate(self, N, m):

# #         X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])

#         noise = np.random.uniform(0,1,size=[N,self.dim])
#         support = (self.ub - self.lb).reshape([1,-1])

#         X = noise * support + self.lb


#         y = self.query(X, m).reshape([-1,1])

#         return X, y
        
    
    def query(self, X, m):
        # negate function to find maximum
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)


        return -ym
    
    def eval_fed_L2(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        term1 = -1.275*np.square(x1)/np.square(np.pi) + 5*x1/np.pi + x2 - 6
        term2 = (10 - 5 / (4*np.pi))*np.cos(x1)
        
        f3 = np.square(term1) + term2 + 10
        
        return f3
    
    def eval_fed_L1(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        f3 = self.eval_fed_L2(xn-2)
        f2 = 10*np.sqrt(f3) + 2*(x1-0.5) - 3*(3*x2-1) - 1

        return f2

    def eval_fed_L0(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        f2 = self.eval_fed_L1(1.2*(xn+2))
        f1 = f2 - 3*x2 + 1
        
        return f1

class Shekel:
    def __init__(self, debug=False):
        self.dim = 4
        self.flevels = 3
        self.maximum = 10.5364
        
        self.bounds = ((0.0,10.0), (0.0,10.0), (0.0,10.0), (0.0,10.0))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
        self.b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        self.C = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                           [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                           [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                           [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
    
    def generate(self, N, m):
        X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
        y = self.query(X, m).reshape([-1,1])

        return X, y
    
    def query(self, X, m):
        # negate function to find maximum
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.meta_shekel(m, xn)

        return -ym
    
    def meta_shekel(self, m, xn):
        
        c = [5,7,10]
        
        outer = 0.0
        for ii in range(c[m]):
            bi = self.b[ii]
            inner = 0.0
            for jj in range(self.dim):
                xj = xn[jj]
                Cji = self.C[jj, ii]
                inner = inner + (xj - Cji) * (xj - Cji)
            # end for
            outer = outer + 1/(inner + bi)
        # end for
        
        f = -outer
        
        return f
    
class Park1:
    def __init__(self, debug=False):
        self.dim = 4
        self.flevels = 2
        self.maximum = 25.589254158606547
        
        self.bounds = ((0.0,1.0), (0.0,1.0), (0.0,1.0), (0.0,1.0))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

            
        self.Flist = []
        self.Flist.append(self.eval_fed_L0)
        self.Flist.append(self.eval_fed_L1)
        
#     def generate(self, N, m):
#         X = np.random.uniform(self.lb, self.ub, size=[N,self.dim])
#         y = self.query(X, m).reshape([-1,1])

#         return X, y

    def query(self, X, m):

        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)

        return ym

    def eval_fed_L0(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        x3 = xn[2]
        x4 = xn[3]
        
        hf = self.eval_fed_L1(xn)
        
        f = (1 + np.sin(x1) / 10) * hf - 2*x1**2 + x2**2 + x3**2 + 0.5
        
        return f    

    def eval_fed_L1(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        x3 = xn[2]
        x4 = xn[3]
        
        if x1 == 0:
            x1 = 1e-12
            
        f = (np.sqrt(1 + (x2+x3**2)*x4/(x1**2)) - 1) * x1 / 2 + (x1 + 3*x4)*np.exp(1 + np.sin(x3))

        return f
    
class Levy:
    """ negative harmann3D, find maximum instead of global minimum """
    def __init__(self, debug=False):
        self.dim = 2
        self.flevels = 3
        self.maximum = 0

        self.bounds = ((-10,10),(-10,10))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]


    
    def query(self, X, m):
        # negate function to find maximum
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        
        
        for n in range(X.shape[0]):
            xn = X[n]
            
            if m == 2:
                ym[n] = -self.eval_high_fidel(xn)
            elif m == 1:
                ym[n] = -np.exp(0.1*np.sqrt(self.eval_high_fidel(xn))) - 0.1*np.sqrt(1 + np.square(self.eval_high_fidel(xn)))
            elif m == 0:
                ym[n] = -np.sqrt(1 + np.square(self.eval_high_fidel(xn)))

        return ym
    
    
    def eval_high_fidel(self, xn):
        outer = 0.0
        x1 = xn[0]
        x2 = xn[1]
        
        term1 = np.square(np.sin(3*np.pi*x1))
        term2 = np.square(x1-1) * (1 + np.square(np.sin(3*np.pi*x2)))
        term3 = np.square(x2-1) * (1 + np.square(np.sin(2*np.pi*x2)))
        
        f = term1 + term2 + term3
        
        return f
    
class Airfoil:
    """ Airfoil """
    def __init__(self, debug=False):
        
        self.dim = 11
        self.flevels = 2
        self.maximum = 3.5

        self.bounds = ((0.02,0.023),(0.32,0.37),(0.077,0.08),(-0.65,-0.63),(0.15,0.19),(-0.05,-0.02),
                      (0.6,0.75),(0,0.01),(0,0.01),(-4.9,-4.55),(15,15.1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

    
    def query(self, X, m):
        
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
            
        matlab_input = '['
        for i in range(X.shape[0]):

            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            matlab_input += s
            if i < X.shape[0] - 1:
                matlab_input += ';'

        matlab_input += ']'
        
        matlab_cmd = 'addpath(genpath(\'airfoil\'));'
        matlab_cmd += 'airfoil_query(' + matlab_input  + ',' + str(m) + ');'

        matlab_cmd += 'quit force'
        
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        message_out = stdout.decode("utf-8")
        
        [start, end] = [i for i,ltr in enumerate(message_out) if ltr=='$']
        
        matlab_output = message_out[start+1:end]
        
        ym = np.array([float(yi) for yi in matlab_output.split(',')])

        return ym

class VibPlate:
    """ Airfoil """
    def __init__(self, debug=False):
        
        self.dim = 3
        self.flevels = 2
        self.maximum = 250

        self.bounds = ((100e9,500e9), (0.2,0.6),(6000,10000))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

    
    def query(self, X, m):
        
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
            
        matlab_input = '['
        for i in range(X.shape[0]):

            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            matlab_input += s
            if i < X.shape[0] - 1:
                matlab_input += ';'

        matlab_input += ']'
        
        matlab_cmd = 'addpath(genpath(\'VibPlate\'));'
        matlab_cmd += 'VibratePlateQuery(' + matlab_input  + ',' + str(m) + ');'

        matlab_cmd += 'quit force'
        
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
#         print(stdout)
#         print(stderr)
        
        message_out = stdout.decode("utf-8")
        message_err = stderr.decode("utf-8")
        
        print(message_out)
        print(message_err)
        
        [start, end] = [i for i,ltr in enumerate(message_out) if ltr=='$']
        
        matlab_output = message_out[start+1:end]
        
        ym = np.array([float(yi) for yi in matlab_output.split(',')])

        return ym
    
class HeatedBlock:
    """ HeatedBlock """
    def __init__(self, debug=False):
        
        self.dim = 3
        self.flevels = 2
        self.maximum = 2.0

        self.bounds = ((0.1,0.4), (0.1,0.4),(0,2*np.pi))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

    
    def query(self, X, m):
        
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
            
        matlab_input = '['
        for i in range(X.shape[0]):

            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            matlab_input += s
            if i < X.shape[0] - 1:
                matlab_input += ';'

        matlab_input += ']'
        
        matlab_cmd = 'addpath(genpath(\'HeatedBlock\'));'
        matlab_cmd += 'HeatedBlockQuery(' + matlab_input  + ',' + str(m) + ');'

        matlab_cmd += 'quit force'
        
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
#         print(stdout)
#         print(stderr)
        
        message_out = stdout.decode("utf-8")
        message_err = stderr.decode("utf-8")
        
        print(message_out)
        print(message_err)
        
        [start, end] = [i for i,ltr in enumerate(message_out) if ltr=='$']
        
        matlab_output = message_out[start+1:end]
        
        ym = np.array([float(yi) for yi in matlab_output.split(',')])

        return ym
    
class DoublePendu:
    """ Double Pendulum """
    def __init__(self, debug=False):
        
        self.dim = 1
        self.flevels = 2
        self.maximum = 15.066782742562044

        self.bounds = ((0,2*np.pi))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

    
    def query(self, X, m):
        
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
            
        matlab_input = '['
        for i in range(X.shape[0]):

            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            matlab_input += s
            if i < X.shape[0] - 1:
                matlab_input += ';'

        matlab_input += ']'
        
        matlab_cmd = 'addpath(genpath(\'doublePendu\'));'
        matlab_cmd += 'dp_query(' + matlab_input  + ',' + str(m) + ');'

        matlab_cmd += 'quit force'
        
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        print(stdout)
        print(stderr)

        
        message_out = stdout.decode("utf-8")
        
        [start, end] = [i for i,ltr in enumerate(message_out) if ltr=='$']
        
        matlab_output = message_out[start+1:end]
        
        ym = np.array([float(yi) for yi in matlab_output.split(',')])

        return ym

    
class SynMfData:
    def __init__(self, domain, Ntrain_list, Ntest_list, seed=None, perturb_scale=1e-2, perturb_thresh=1e-3):
        self.domain = domain

        Fdict = {
            'SfFn1'      : SfFn1(),
            'SfFn2'      : SfFn2(),
            'SfFn3'      : SfFn3(),
            'SynFn1'     : SynFn1(),
            'CurrinExp'  : CurrinExp(),
            'Hartmann3D' : Hartmann3D(),
            'Hartmann6D' : Hartmann6D(),
            'Shekel'     : Shekel(),
            'Branin'     : Branin(),
            'Park1'      : Park1(),
            'StybTang'   : StybTang(),
            'Airfoil'    : Airfoil(),
            'Levy'       : Levy(),
            'DoublePendu': DoublePendu(),
            'VibPlate'   : VibPlate(),
            'HeatedBlock': HeatedBlock(),
        }
        
        self.MfFn = Fdict[self.domain]

        self.Ntrain_list = Ntrain_list
        self.Ntest_list = Ntest_list
        self.dim = self.MfFn.dim
        self.Nfid = self.MfFn.flevels
        self.maximum = self.MfFn.maximum
        
        if seed == None:
            self.seed = np.random.randint(0,100000)
        else:
            self.seed = seed
        
#         self.lb = self.MfFn.lb
#         self.ub = self.MfFn.ub
        
        self.perturb_scale = perturb_scale
        self.perturb_thresh = perturb_thresh

        self.data = []
        
        self.Xscalers = []
        self.yscalers = []
        for m in range(self.Nfid):
            self.Xscalers.append(StandardScaler())
            self.yscalers.append(StandardScaler())

        for m in range(self.Nfid):
            Nm_train = self.Ntrain_list[m]
            Nm_test = self.Ntest_list[m]
            
            Dm = {}
            
#             raw_Xall, raw_yall = self.generate(Nm_train + Nm_test, m, self.seed)
            raw_Xall, yall = self.generate(Nm_train + Nm_test, m, self.seed)
    
#             print
    
            Dm['ytrain'] = yall[0:Nm_train]
            Dm['ytest'] = yall[Nm_train:Nm_train+Nm_test]
            
            Dm['raw_Xall'] = raw_Xall
            Dm['raw_Xtrain'] = raw_Xall[0:Nm_train,:]
            Dm['raw_Xtest'] = raw_Xall[Nm_train:Nm_train+Nm_test,:]
            
            self.Xscalers[m].fit(raw_Xall)

            
            Xall = self.Xscalers[m].transform(raw_Xall)

            Dm['Xtrain'] = Xall[0:Nm_train,:]
            Dm['Xtest'] = Xall[Nm_train:Nm_train+Nm_test,:]

            self.data.append(Dm)
            
        self.lb = np.squeeze(self.Xscalers[-1].transform(self.MfFn.lb.reshape([1,-1])))
        self.ub = np.squeeze(self.Xscalers[-1].transform(self.MfFn.ub.reshape([1,-1])))
            

            
    def query(self, X, m):
        
        rescale_X = self.Xscalers[m].inverse_transform(X)
        rescale_X = np.clip(np.squeeze(rescale_X), self.MfFn.lb, self.MfFn.ub).reshape([1,-1])
        
        ym = self.MfFn.query(rescale_X, m)
        return ym
            
            
    def generate(self, N, m, seed):
        
        state = np.random.get_state()
        X = None
        y = None
        try:
            np.random.seed(seed+m)

            noise = np.random.uniform(0,1,size=[N,self.dim])
            support = (self.MfFn.ub - self.MfFn.lb).reshape([1,-1])

            X = noise * support + self.MfFn.lb
            
            y = self.MfFn.query(X, m).reshape([-1,1])

        except:
            perm = np.arange(N)
        finally:
            np.random.set_state(state)

        return X, y

    
    def append(self, X, m):
        
        X = self.perturb(X, m)
        
        rescale_X = self.Xscalers[m].inverse_transform(X)
        rescale_X = np.clip(np.squeeze(rescale_X), self.MfFn.lb, self.MfFn.ub).reshape([1,-1])

        yq = self.MfFn.query(rescale_X, m)
        ystar = self.MfFn.query(rescale_X, self.Nfid - 1)


        raw_Xall = np.concatenate([rescale_X, self.data[m]['raw_Xall']], axis=0)
        self.Xscalers[m].fit(raw_Xall)
        self.data[m]['raw_Xall'] = self.Xscalers[m].transform(raw_Xall)
        
        
        raw_Xtrain = np.concatenate([rescale_X, self.data[m]['raw_Xtrain']], axis=0)
        Xtrain = self.Xscalers[m].transform(raw_Xtrain)
        perm = np.random.permutation(Xtrain.shape[0])
        
        Xtrain = Xtrain[perm]
        ytrain = self.MfFn.query(raw_Xtrain[perm], m)

        self.data[m]['Xtrain'] = Xtrain
        self.data[m]['ytrain'] = ytrain.reshape([-1,1])
        
        self.Ntrain_list[m] = Xtrain.shape[0]
        
        # updat the boundary
        self.lb = np.squeeze(self.Xscalers[-1].transform(self.MfFn.lb.reshape([1,-1])))
        self.ub = np.squeeze(self.Xscalers[-1].transform(self.MfFn.ub.reshape([1,-1])))
        
        return yq, ystar, X
    
    def perturb(self, X, m):
        # perturb X if necessary
        Xm = self.data[m]['Xtrain']
        dist = np.sqrt(np.sum(np.square(Xm - X), axis=1))
        
        if np.min(dist) < self.perturb_thresh:
            print('Perturb X!!!!!')
            bounds = self.ub - self.lb
            perturbation = bounds * self.perturb_scale * (np.random.uniform() - 0.5)
            
            X_perturb = X + perturbation.reshape([1,-1])
            return X_perturb
        
        return X
    