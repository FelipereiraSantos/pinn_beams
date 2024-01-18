# @author Felipe Pereira dos Santos
# @since 27 december, 2023
# @version 27 december, 2023

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import tensorflow as tf
from sciann_datagenerator import *
import time
import sys


class Nonlinear_TimoEx:
    """
         Class that represents provide features for the Timoshenko bending beam analysis.

         Based on the problem''s initial and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on
    """

    def __init__(self, network, P, L, E, I, num_training_samples, num_test_samples):
        """
            Constructor of the Euler-Benoulli beam class for stability

            Attributes:
                network (keras network): usually represents a neural network used to approximate the target
                problem solution
                P: Point load at the  beam
                L: beam span
                E: Young modulus
                I: inertia moment
                nu: Poisson coefficient
                A: cross-section area
                num_training_samples: number of samples for training the model
                num_test_samples: number of samples for testing the model (predictions)

        """
        self.problem = "Nonlinear_TimoEx"
        self.P = P

        self.L = L
        self.L_aux = 1
        self.E = E
        self.I = I
        self.alpha = (self.P * self.L ** 2)/(self.E * self.I)

        if self.alpha >= 1:
            self.alpha = round(self.alpha)
        else:
            self.alpha = round(self.alpha, 2)

        print("alpha: ", self.alpha)


        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        # Neural Network Setup.
        dtype = 'float32'

        self.xi = sn.Variable("xi", dtype=dtype)
        # self.u = sn.Functional('u', self.x, network[0], network[1], kernel_initializer=network[2])
        self.rot = sn.Functional('rot', self.xi, network[0], network[1], kernel_initializer=network[2])
        # self.P = sn.Parameter(0.05, inputs=self.x, name='Pcr')
        # self.P = sn.Parameter(1.0, inputs=self.x, name='Pcr')
        # self.alpha = sn.Parameter(1.0, inputs=self.x, name='Alpha')

        self.drot_dx = sn.diff(self.rot, self.xi)
        self.d2rot_dx2 = sn.diff(self.rot, self.xi, order=2)

        self.eqDiff1 = self.d2rot_dx2 + self.alpha * sn.cos(self.rot)

        self.variables = [self.xi]

    def model_info(self):
        """
        Method to write the physical model information in the text file output that contains the
        elvaluation of the MSE errors

        """
        model_parameters = 'Number of training samples: ' + str(self.num_training_samples) + \
                           '\nP: ' + str(self.P) + ' N | ' + 'L: ' + str(self.L) + ' m | ' + 'E: ' +\
                           str(self.E) + ' N/m² | ' + 'I: ' + str(self.I) + ' m^4 | ' + 'a: ' + str(self.a) + ' m\n'
        return model_parameters


    def fixed_free(self, problem):
        """
             Method to setting the features for a cantilever beam with an axial load

        """

        # Reference solution for the predictions ======================================
        xi = np.linspace(0, self.L_aux, int(self.num_test_samples))
        self.x_test = xi
        rot_ref = self.reference_solution(xi, problem)

        self.ref_solu = rot_ref
        # Reference solution for the predictions ======================================

        # Boundary and initial conditions
        BC_left_1 = (self.xi == 0.) * (self.rot)

        BC_right_1 = (self.xi == self.L_aux) * (self.drot_dx)

        # Loss function
        self.targets = [self.eqDiff1,
                   BC_left_1, BC_right_1]

        dg = DataGeneratorX(X=[0., self.L_aux],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 1 * ['bc-left'] + 1 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()


    def reference_solution(self, xi, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
         In this case, the reference solution was extracted from [1] page 126, for n = 4.

         [1] Timoshenko, S. P., & Gere, J. M. (1982). Mecânica dos Sólidos. Volume 1.

         For each alpha value, there is a correspondent m that generates the solution in terms of th rotation theta.

        """
        theta = (np.pi / 2) * np.array([0.079, 0.156, 0.228, 0.294, 0.498, 0.628, 0.714, 0.774, 0.817, 0.849, 0.874, 0.894, 0.911])
        alpha = np.array([0.25, 0.50, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        dic_theta = dict(zip(alpha, theta))

        theta_ref = dic_theta[self.alpha]

        return theta_ref
