# Dynamic programming 2023 
This code provides the result of the paper To contracept or not to contracept by Julie Cathrine Krabek Sørensen and Caroline Bergholdt Hansen.

This paper uses a dynamic discrete choice model to describe the contraception decision for married couples. Due to the fact that fertile years are limited, this model works within a finite time horizon. Given this time framework, backward induction is used to determine the choice conditional on states and maximum log-likelihood estimates of the parameters of the data. 

All the results can be seen by running the [Contracept.ipynb](Contracept.ipynb) file.
The rest of the files in the repository is:
Model.py - Which have a class of parameters and equation. It is used to calculate utility, state transitions, and read and simulate data.

Solve.py - Which solves the model using backward induction.

Estimate.py - Which calculate the log-likelihood.
