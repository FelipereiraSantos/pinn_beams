# Introduction
This repository contains the code used in the article "Machine Learning in Structural Engineering: Physics-Informed Neural Networks for Beam Problems". It implements Physics-Informed Neural Networks (PINNs) for solving beam-related problems. The code compares the trained solutions with analytical and other reference solutions.

# Features

-Implementation of PINNs for solving 1D beam problems.

-Governing equations based on the statics of the Timoshenko beam, the stability of the Euler-Bernoulli beam, and a large deflection case study.

-Deflections and rotations are obtained during the learning process in a data-driven solution approach.

-Critical load for buckling problems is obtained via a data-driven discovery framework. 

-Beams with prismatic and varying cross-section area are investigated.

-SciANN-based implementation for ease and flexibility.

# Dependencies

The last version of the code has been tested with the following package versions:

-Python: 3.10

-SciANN: 0.7.0.1

-TensorFlow: 2.12


# Running the Code

-Run the main file.

-A file selection window will appear—choose the input file to be processed.

-The model will be trained automatically.

-Once training is complete, one or more CSV files containing error results will be generated.

-Errors are computed for all implemented problems, using analytical and other reference solutions for comparison.

# Disclaimer

This code is not optimized, and some parts of its structure could be better organized. Contributions and improvements are welcome.



# Paper abstract

The objective of this paper is to provide an accessible introduction to the fundamental principles of physics-informed neural networks (PINNs), to facilitate aspiring researchers to start the journey in this field with minimal coding efforts. A series of structural analysis examples of one-dimensional beams are used to illustrate how the code can be structured. Deflections and cross-section rotations are obtained in a data-driven solution approach, while a data-driven discovery framework is developed to find the beam buckling loads. Regarding the former, the statics of the Timoshenko beam and a geometrically nonlinear case study are investigated, while the discovery framework is used to obtain the buckling load of Euler–Bernoulli beams. The results of PINN simulations are consistent with the reference solutions, with an error of the order of 
$10^{-3}$ for each one of the different error measures.

# Citation

If you find this repository helpful, please consider citing our paper:

Santos, F.P. and Gori, L. "Machine Learning in Structural Engineering: Physics-Informed Neural Networks for Beam Problems". International Journal of Computational Methods.

DOI: https://doi.org/10.1142/S0219876224500907
