# Macro-in-Python
### Solution of Macroeconomic Models in Python

These files contain my (amateur) approach to solve macroeconomic models using Python. The code is not written for being elegant, neither for speed, therefore, optimization is needed and comments are welcome.

A large part of the code is based on the following resources:

- Introduction to computational economics using fortran by Fehr and Kindermann, 2018
- Recursive Macroeconomic Theory by Sargent and Ljungqvist, 2012
- Dynamic General Equilibrium Modeling by Herr and Maussner, 2009
- João B. Duarte's Ph.D. course on Macro in Python (https://github.com/jbduarte/Advanced_Macro)
- McKay's short course on heterogeneous agent macroeconomics (https://alisdairmckay.com/Notes/HetAgents/index.html)

At the moment the codes include:

- Simple stochastic growth model (Value function iteration, Howard improvement algorithm, Endogeneous Grid method, and Perturbation)
- Simple RBC model with labor choice (Value function iteration, Howard improvement algorithm, and Perturbation)
- A variety of versions of the RBC model with different frictions (Perturbation only)
- Aiyagari model with aggregation using Monte Carlo simulation and an invariante distribution (Value function iteration, and Howard improvement algorithm)
