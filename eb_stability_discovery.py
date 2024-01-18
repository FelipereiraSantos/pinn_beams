# @author Felipe Pereira dos Santos
# @since 25 October, 2023
# @version 25 October, 2023

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import tensorflow as tf
from sciann_datagenerator import *
import time
import sys


class EB_Stability_Discovery:
    """
         Class that represents provide features for the Euler-Bernoulli beam stability
         analysis of parameter discovery

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
                num_training_samples: number of samples for training the model
                num_test_samples: number of samples for testing the model (predictions)

        """
        # Neural Network Setup.
        dtype = 'float32'
        self.problem = "EB_stability_discovery"
        self.L = L
        self.E = E
        self.I = I
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        self.x = sn.Variable("x", dtype=dtype)
        self.u = sn.Functional('u', self.x, network[0], network[1], kernel_initializer=network[2])
        # self.rot = sn.Functional('rot', self.x, network[0], network[1], kernel_initializer=network[2])
        self.P = sn.Parameter(P, inputs=self.x, name='Pcr')
        # self.alpha = sn.Parameter(1.0, inputs=self.x, name='Alpha')

        self.du_dx = sn.diff(self.u, self.x)
        self.d2u_dx2 = sn.diff(self.u, self.x, order=2)
        self.d3u_dx3 = sn.diff(self.u, self.x, order=3)
        self.d4u_dx4 = sn.diff(self.u, self.x, order=4)

        self.eqDiff1 = (self.E * self.I) * self.d4u_dx4 + self.P * self.d2u_dx2
        # self.eqDiff1 = (self.E * self.I) * self.d2u_dx2 + self.P * self.u

        self.variables = [self.x]

    def model_info(self):
        """
        Method to write the physical model information in the text file output that contains the
        elvaluation of the MSE errors

        """
        model_parameters = 'Number of training samples: ' + str(self.num_training_samples) + \
                           '\nP: ' + str(self.P) + ' N | ' + 'L: ' + str(self.L) + ' m | ' + 'E: ' +\
                           str(self.E) + ' N/m² | ' + 'I: ' + str(self.I) + '\n'
        return model_parameters

    def pinned_pinned(self, problem):
        """
             Method to setting the features for a simply-supported beam with an axial load

        """
        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        P_ref = self.reference_solution(x, problem)

        self.ref_solu = P_ref
        # Reference solution for the predictions ======================================

        # Boundary and initial conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.d2u_dx2)
        BC_left_3 = (self.x == 0.) * (self.du_dx - 0.5)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.d2u_dx2)
        BC_right_3 = (self.x == self.L) * (self.du_dx + 0.5)

        # Loss function
        self.targets = [self.eqDiff1,
                        BC_left_1, BC_left_2,BC_left_3,
                        BC_right_1, BC_right_2,BC_right_3]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 3 * ['bc-left'] + 3 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_pinned(self, problem):
        """
             Method to setting the features for a fixed-pinned beam with an axial load

        """
        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        P_ref = self.reference_solution(x, problem)

        self.ref_solu = P_ref
        # Reference solution for the predictions ======================================

        # Boundary and initial conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.du_dx)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.d2u_dx2)
        BC_right_3 = (self.x == self.L) * (self.du_dx - 0.5)


        # Loss function
        self.targets = [self.eqDiff1,
                        BC_left_1, BC_left_2,
                        BC_right_1, BC_right_2,BC_right_3]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 2 * ['bc-left'] + 3 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_free(self, problem):
        """
             Method to setting the features for a cantilever beam with an axial load

        """
        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        P_ref = self.reference_solution(x, problem)

        self.ref_solu = P_ref
        # Reference solution for the predictions ======================================

        # Boundary and initial conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.du_dx)

        BC_right_1 = (self.x == self.L) * (self.d2u_dx2)
        BC_right_2 = (self.x == self.L) * (self.du_dx - 0.5)
        BC_right_3 = (self.x == self.L) * (self.d3u_dx3 + self.du_dx * self.P / (self.E * self.I))

        # Loss function
        self.targets = [self.eqDiff1,
                        BC_left_1, BC_left_2,
                         BC_right_1, BC_right_2, BC_right_3]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 2 * ['bc-left'] + 3 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_fixed(self, problem):
        """
             Method to setting the features for a fixed-fixedbeam with an axial load

        """
        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        P_ref = self.reference_solution(x, problem)

        self.ref_solu = P_ref
        # Reference solution for the predictions ======================================

        # Boundary and initial conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.du_dx)
        BC_left_mid = (self.x == self.L/4) * (self.du_dx - 0.5)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.du_dx)
        BC_right_mid = (self.x == 3 * self.L / 4) * (self.du_dx + 0.5)

        # Loss function
        self.targets = [self.eqDiff1,
                        BC_left_1, BC_left_2,
                        BC_left_mid,
                        BC_right_mid,
                        BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[0., self.L/4, 3*self.L/4, self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 2 * ['supports'] + 1 * ['supports'] + 1 * ['supports'] + 2 * ['supports'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def reference_solution(self, x, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
        """
        # Aditional data based on the reference solution for the predictions ======================================
        if problem[1] == "pinned" and problem[2] == "pinned":
            P = 1.0 * (np.pi ** 2 * self.E * self.I) / self.L ** 2
        elif problem[1] == "fixed" and problem[2] == "pinned":
            P = 2.046 * (np.pi ** 2 * self.E * self.I) / self.L ** 2
        elif problem[1] == "fixed" and problem[2] == "fixed":
            P = 4.0 * (np.pi ** 2 * self.E * self.I) / self.L ** 2
        elif problem[1] == "fixed" and problem[2] == "free":
            P = 0.25 * (np.pi ** 2 * self.E * self.I) / self.L ** 2

        return P