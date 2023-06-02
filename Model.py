#import packages
import numpy as np
import time
import pandas as pd
import copy

class child_model():
    def __init__(self,**kwargs):
        self.setup(**kwargs)

    def setup(self,**kwargs):     
  
        # a. parameters
        self.n = 5                     # 4 = Number of possible children
        self.nX = 5                    # Number of possible states/grid points
        self.max = 5                   # Max of children groups
        
        # b. number of couples for simulations
        self.N = 2748
       
        # c. Age and timespans
        self.terminal_age = 45      # Assumed age of menopause
        self.marriage_age = 18      # Minimum age of marriage 
        self.death_age = 76         # Assumed age at death
        self.meno_p_years = self.death_age - self.terminal_age  # Non-fertile years
        self.T = self.terminal_age - self.marriage_age          # Fertile years 

        # d. structual parameters
        self.p1 = np.array([0.6, 0.4])              # Transition probability when not contracepting
        self.p2 = np.array([1, 0])                  # Transition probability when contracepting
        self.p1_list = np.ones([self.T,2])*self.p1  # Transistion probabilities at each age
        self.p2_list = np.ones([self.T,2])*self.p2

        # e. utility parameters
        self.eta1 =  1.40   # Marginal utility of children
        self.eta2 = -0.35   # Marginal utility of children squared                           
        self.mu = 0.88      # Cost of contraception                                    
        self.beta = 0.90    # Discount factor

        # f. update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val) 

        # g. Create grid
        self.create_grid()

    def create_grid(self):
        
        self.grid = np.arange(0,self.n)                                 # Grid for number of children
        self.utility = self.eta1*self.grid + self.eta2*(self.grid**2)   # Utilty function without choice
        self.state_transition() 
    
    def state_transition(self):

        # a. Conditional on d=0, do not contracept
        p1 = np.append(self.p1,1-np.sum(self.p1))   # Get transition probabilities
        P1 = np.zeros((self.n,self.n))              # Initialize transition matrix
        # b. Loop over rows
        for i in range(self.n):
            # i. Check if p1 vector fits entirely
            if i <= self.n-len(p1):
                P1[i][i:i+len(p1)]=p1
            else:
                P1[i][i:] = p1[:self.n-len(p1)-i]
                P1[i][-1] = 1.0-P1[i][:-1].sum()

        # c. Conditional on d=1, contracept
        p2 = np.append(self.p2,1-np.sum(self.p2))   # Get transition probabilities
        P2 = np.zeros((self.n,self.n))              # Initialize transition matrix
        # d. Loop over rows
        for i in range(self.n):
            # i. Check if p2 vector fits entirely
            if i <= self.n-len(p2):
                P2[i][i:i+len(p2)]=p2
            else:
                P2[i][i:] = p2[:self.n-len(p2)-i]
                P2[i][-1] = 1.0-P2[i][:-1].sum()

        self.P1 = P1
        self.P2 = P2

    def bellman(self, ev0):

        # a. Value of options:
        value_0 = self.utility + self.beta * self.P1 @ ev0 # nx1 matrix
        value_1 = self.mu1 + self.utility + self.beta * self.P2 @ ev0   # nx1 matrix

        # b. Recenter Bellman by subtracting max(VK, VR)
        maxV = np.maximum(value_0, value_1) 
        logsum = (maxV + np.log(np.exp(value_0-maxV)  +  np.exp(value_1-maxV))) # Compute logsum to handle expectation over unobserved states
        ev1 = logsum # Bellman operator as integrated value

        # c. Compute choice probability of not contracepting
        pnc = 1/(1+np.exp(value_1-value_0))       
        
        return ev1, pnc

    def sim_data(self, pnc):
        
        # a. Set N (number of couples) and T (fertile years)
        N = self.N
        T = self.T

        # b. Set random seed 
        np.random.seed(2020)

        # c. Index 
        idx = np.tile(np.arange(1,N+1),(T,1))  
        time = np.tile(np.arange(self.marriage_age-self.marriage_age,self.terminal_age-self.marriage_age),(N,1)).T
            
        # d. Draw random numbers
        u_d = np.random.rand(T,N)  # Decision/choice
        u_dx = np.random.rand(T,N) # Birth               

        # e. Find states and choices
        # i. State 
        x  = np.zeros((T,N), dtype=int)
        # ii. State next period 
        x1 = np.zeros((T,N), dtype=int)
        # iii. Birth indicator
        dx1 = np.zeros((T,N), dtype=int)
        # iv. Decision/choices
        d  = np.zeros((T,N), dtype=int) # np.nan + np.zeros((T,N))
        # v. Initial condition
        x[0,:] = np.zeros((1,N)) # u_init.astype(int)

        # f. Loop over years and couples
        for t in range(T):
            for i in range(N):

                # a. Set d=1 if u_d uncer probability of contraception
                d[t,i] = u_d[t,i] < 1-pnc[x[t,i],t] 
                
                # b. Birth probability conditional of choice
                # i. Condional on d = 0
                if d[t,i] == 0:
                    # o. Find states and choices
                    csum_p1 = np.cumsum(self.p1_list[t])  # Cumulated sum of p1 
                    # oo. Birth next period
                    dx1[t,i] = 0
                    for val in csum_p1:
                        dx1[t,i] += u_dx[t,i]>val
                # ii. Condional on d = 1 
                else: 
                    # o. Find states and choices
                    csum_p2 = np.cumsum(self.p2_list[t])  # Cumulated sum of p2 
                    dx1[t,i] = 0
                    # oo. Birth next period
                    for val in csum_p2:
                        dx1[t,i] += u_dx[t,i]>val
                
                # c. Find states and choices
                x1[t,i] = np.minimum(x[t,i]+dx1[t,i], self.n-1) # State transition, minimum to avoid exceeding the maximum number of children

                # d. Set x1 to state in next period
                if t < T-1:
                    x[t+1,i] = x1[t,i]
  
        # g. Reshape 
        idx = np.reshape(idx,T*N,order='F')
        time = np.reshape(time,T*N,order='F')
        d   = np.reshape(d,T*N,order='F')
        x   = np.reshape(x,T*N,order='F')   
        x1  = np.reshape(x1,T*N,order='F')  
        dx1 = np.reshape(dx1,T*N,order='F')

        # h. Make columns and name them
        data = {'id': idx,'t': time,'d': d, 'x': x, 'dx1': dx1, 'x1': x1}
        
        # i. Set type to dataframe
        df = pd.DataFrame(data) 

        return(df)

    def read_data(self): 

        # a. Read data 
        with open('carro-mira_new.txt', 'r') as file:
        # i. Read the content of the file
            content = file.read()

            # o. Remove spaces and replace with commas
            content_without_spaces = ','.join(content.split())

            # oo. Create DataFrame with separated columns
            num_columns = 25  # Specify the desired number of columns
            data = np.array(content_without_spaces.split(','))
            reshaped_data = np.reshape(data, (-1, num_columns))
            data = pd.DataFrame(reshaped_data)

        # b. Extract columns 
        idx = data.iloc[:,3]   # Couple id
        t = data.iloc[:,5]     # Woman's age
        cc = data.iloc[:,10]   # Contraception choice
        d = data.iloc[:,14]    # Decision
        x = data.iloc[:,9]     # Number of children
        dx1 = data.iloc[:,8]   # Birth indicator
                    
        # c. Change type to integrer
        idx = idx.astype(int)
        x = x.astype(int)
        dx1 = dx1.astype(int)
        d = d.astype(int)
        t = t.astype(int)           
        t = t-18   # Subtract 18 so that age 18 will become the first period 0

        # d. Collect in a dataframe
        data = {'id': idx, 't' : t,'contraception choice':cc, 'd': d, 'x': x, 'dx1': dx1}
        df = pd.DataFrame(data) 

        # e. Remove sterilized couples, couples with wife's age below 18, and couples with more than 4 children 
        df = df.drop(df[df['contraception choice'] == 3].index, axis=0)
        df = df.drop(df[df['t'] < 0].index, axis=0)
        df = df.drop(df[df['x'] > 4].index, axis=0)

        # f. Save data
        dta = df.drop(['contraception choice'],axis=1)
        
        return dta
    

        