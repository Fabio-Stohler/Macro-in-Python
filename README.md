# Macro-in-Python
### Solution of Macroeconomic Models in Python

These spyder-files contain my (amateur) approach to solve macroeconomic models using Python. The code is not written for being elegant, neither for speed, therefore, optimization is needed and comments are welcome. I only code to learn the concepts and make first experience with solving these models.

A large part of the code is based on the following resources:

- Introduction to computational economics using fortran by Fehr and Kindermann, 2018
- Recursive Macroeconomic Theory by Sargent and Ljungqvist, 2012
- Dynamic General Equilibrium Modeling by Herr and Maussner, 2009
- João B. Duarte's Ph.D. course on Macro in Python (https://github.com/jbduarte/Advanced_Macro)
- McKay's short course on heterogeneous agent macroeconomics (https://alisdairmckay.com/Notes/HetAgents/index.html)

I am currently working on a solution to Krusell and Smith (1998) and endogeneous grid methods. At the moment the codes include:

- Simple stochastic growth model (Value function iteration, Howard improvement algorithm, and Perturbation)
- Simple RBC model with labor choice (Value function iteration, Howard improvement algorithm, and Perturbation)
- Aiyagari model with aggregation using Monte Carlo and an invariante distribution (Value function iteration, and Howard improvement algorithm)