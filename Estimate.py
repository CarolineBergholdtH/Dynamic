# Import packages
import numpy as np
from scipy.optimize import minimize


def ll(theta, model, solver,data, pnames, out=1): # out=1 solve optimization
    
    # a. Unpack and convert to numpy array
    x = np.array(data.x)       
    d = np.array(data.d)
    t = np.array(data.t)        
    dx1 = np.array(data.dx1)

    # b. Update values
    model = updatepar(model, pnames, theta)
                               
    # c. Solve the model
    ev, pnc = solver.BackwardsInduction3(model)
    
    # d. Evaluate likelihood functionnX
    epsilon = 1e-10  # Small constant to avoid division by zero

    # e. Likelihood function
    lik_pr = pnc[x,t]  
    function = lik_pr * (1 - d) * (model.p1_list[t,1]*dx1 + model.p1_list[t,0]*(1-dx1)) + (1-lik_pr) * d *(model.p2_list[t,1]*dx1 + model.p2_list[t,0]*(1-dx1)) 
    function = np.maximum(function, epsilon)  # Add small constant to avoid zero values
    
    # f. Log-likehood
    log_lik = np.log(function)

    # g. Return objective function (negative mean log likleihood)
    if out == 1:
        return np.mean(-log_lik)

    return model, lik_pr, pnc, ev, d, x, dx1 

def updatepar(par,parnames, parvals):

    # a. Update parameters
    for i,parname in enumerate(parnames):
        # i. First two parameters are scalars
        parval = parvals[i]
        setattr(par,parname,parval)

    return par