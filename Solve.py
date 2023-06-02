# Import packages used
import numpy as np
import pandas as pd

def BackwardsInduction(model):
    
    # a. Step 1: Initialize value function and policy
    T = model.T
    n = model.n 
    V = np.nan + np.zeros([n, T])
    V[:, T - 1] = life(model)
    pnc =  np.nan + np.zeros([n, T])
    pnc[:, T-1] = 1

    # b. Step 3: Backward induction
    for t in range(T - 2, -1, -1):
        
        # i. Update transition matrix with birth probabilities at given age
        model.p1 = model.p1_list[t]
        model.p2 = model.p2_list[t] 
        model.state_transition() 

        # ii. run bellman equation to find value and choice probability for age
        ev1, pnc_t = model.bellman(ev0 = V[:, t+1])

        # iii. Store values
        V[:, t] = ev1  
        pnc[:, t] = pnc_t
 
    return V.round(3), pnc.round(3) #, dev
    
def life(model):

    # a. Initialize life value
    life_value = 0

    # b. Calculate value for infertile years a time of menopause
    for t in range(model.meno_p_years):
        life_value += (model.beta**t) * (model.eta1 * model.grid + model.eta2 * (model.grid**2))
    
    return life_value
    
def P_list(model, data):
    
    # a. Make list of birth probabilities based on birth in data
    T = model.T
    
    # B. Backwards Induction
    for t in range(T - 2, -1, -1):
        # i. Subset data 
        datad0 = data[(data['d']==0) & (data['t']==t)] 
        datad1 = data[(data['d']==1) & (data['t']==t)]
        
        # ii. Count number of observations for each dx1
        tabulate0 = datad0.dx1.value_counts() 
        tabulate1 = datad1.dx1.value_counts()
        
        # iii. Calculate birth probabilities
        for i in range(tabulate0.size-1):
            p1 = tabulate0[i]/sum(tabulate0) 
            p2 = tabulate1[i]/sum(tabulate1)

        # iv. Append to list
        model.p1  = np.append(p1,1-np.sum(p1))
        model.p2 = np.append(p2,1-np.sum(p2))

        # v. Add to list at time t 
        model.p1_list[t] = model.p1
        model.p2_list[t] = model.p2
