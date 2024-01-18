# @author Felipe Pereira dos Santos
# @since 13 june, 2023
# @version 13 june, 2023

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import tensorflow as tf
from sciann_datagenerator import *
import time
import sys


class EB_Stability:
    """
         Class that represents provide features for the Timoshenko bending beam analysis.

         Based on the problem''s initial and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on
    """

    def __init__(self, network, P, L, E, I, a, num_training_samples, num_test_samples):
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
        self.problem = "EB_stability"
        self.P = P

        self.L = L
        self.E = E
        self.I = I
        self.a = a
        # P_cr = ((np.pi ** 2 * self.E * self.I) / (4 * self.L ** 2))
        # self.P = 4 * P_cr
        # self.k = np.sqrt(np.abs(self.P) / (self.E * self.I))
        # self.G = E / (2 * (1 + nu))
        # self.nu = nu
        # self.k = 5. / 6.
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        # Neural Network Setup.
        dtype = 'float32'

        self.x = sn.Variable("x", dtype=dtype)
        self.u = sn.Functional('u', self.x, network[0], network[1], kernel_initializer=network[2])
        # self.rot = sn.Functional('rot', self.x, network[0], network[1], kernel_initializer=network[2])
        # self.P = sn.Parameter(0.05, inputs=self.x, name='Pcr')
        # self.P = sn.Parameter(1.0, inputs=self.x, name='Pcr')
        # self.alpha = sn.Parameter(1.0, inputs=self.x, name='Alpha')

        self.du_dx = sn.diff(self.u, self.x)
        self.d2u_dx2 = sn.diff(self.u, self.x, order=2)
        self.d3u_dx3 = sn.diff(self.u, self.x, order=3)
        self.d4u_dx4 = sn.diff(self.u, self.x, order=4)

        self.eqDiff1 =(self.E * self.I) * self.d4u_dx4 + self.P * self.d2u_dx2
        # self.eqDiff1 = (self.E * self.I) * self.d2u_dx2 + self.P * self.u

        self.variables = [self.x]

    def model_info(self):
        """
        Method to write the physical model information in the text file output that contains the
        elvaluation of the MSE errors

        """
        model_parameters = 'Number of training samples: ' + str(self.num_training_samples) + \
                           '\nP: ' + str(self.P) + ' N | ' + 'L: ' + str(self.L) + ' m | ' + 'E: ' +\
                           str(self.E) + ' N/m² | ' + 'I: ' + str(self.I) + ' m^4 | ' + 'a: ' + str(self.a) + ' m\n'
        return model_parameters

    def pinned_pinned(self, problem):
        """
             Method to setting the features for a simply-supported beam with an axial load

        """

        self.w = 1
        self.eqDiff1 = (self.E * self.I) * self.d4u_dx4 + self.P * self.d2u_dx2 + self.w  # The sign in the load is due to the reference axis

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        x, u_ref, rot_ref = self.reference_solution(x, problem)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Boundary and initial conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.d2u_dx2)
        # BC_left_3 = (self.x == 0.) * (self.du_dx - 0.5)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.d2u_dx2)
        # BC_right_3 = (self.x == self.L) * (self.du_dx + 0.5)

        # Loss function
        self.targets = [self.eqDiff1,
                        BC_left_1, BC_left_2,
                        BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    # def pinned_pinned(self, problem):
    #     """
    #          Method to setting the features for a simply-supported beam with an axial load
    #
    #     """
    #
    #     # Reference solution for the predictions ======================================
    #     x = np.linspace(0, self.L, int(self.num_test_samples))
    #     self.x_test = x
    #     x, u_ref, rot_ref = self.reference_solution(x, problem)
    #     self.ref_solu = [u_ref, rot_ref]
    #     # Reference solution for the predictions ======================================
    #
    #     # Boundary and initial conditions
    #     BC_left_1 = (self.x == 0.) * (self.u)
    #     BC_left_2 = (self.x == 0.) * (self.d2u_dx2)
    #     BC_left_3 = (self.x == 0.) * (self.du_dx - 0.5)
    #
    #     BC_right_1 = (self.x == self.L) * (self.u)
    #     BC_right_2 = (self.x == self.L) * (self.d2u_dx2)
    #     BC_right_3 = (self.x == self.L) * (self.du_dx + 0.5)
    #
    #     # Loss function
    #     self.targets = [self.eqDiff1,
    #                     BC_left_1, BC_left_2,BC_left_3,
    #                     BC_right_1, BC_right_2,BC_right_3]
    #
    #     dg = DataGeneratorX(X=[0., self.L],
    #                         num_sample=self.num_training_samples,
    #                         targets=1 * ['domain'] + 3 * ['bc-left'] + 3 * ['bc-right'])
    #
    #     # Creating the training input points
    #     self.input_data, self.target_data = dg.get_data()

    # def fixed_pinned(self):
    #     """
    #          Method to setting the features for a fixed-pinned beam with an axial load
    #
    #     """
    #     zero = tf.constant(0.0)
    #     one = tf.constant(1.0)
    #     # Reference solution for the predictions ======================================
    #     x = np.linspace(0, self.L, int(self.num_test_samples))
    #     self.x_test = x
    #     u_ref = (self.a / np.cos(self.k * self.L)) * (np.cos(self.k * x) - 1)
    #
    #     self.ref_solu = u_ref
    #     # Reference solution for the predictions ======================================
    #
    #     # Boundary and initial conditions
    #     BC_left_1 = (self.x == 0.) * (self.u)
    #     BC_left_2 = (self.x == 0.) * (self.du_dx)
    #
    #     BC_right_1 = (self.x == self.L) * (self.u)
    #     BC_right_2 = (self.x == self.L) * (self.d2u_dx2)
    #     BC_right_3 = (self.x == self.L) * (self.rot - one)
    #
    #     # Loss function
    #     self.targets = [self.eqDiff1, self.eqDiff2,
    #                     BC_left_1, BC_left_2,
    #                     BC_right_1, BC_right_2,BC_right_3]
    #
    #     dg = DataGeneratorX(X=[0., self.L],
    #                         num_sample=self.num_training_samples,
    #                         targets=2 * ['domain'] + 2 * ['bc-left'] + 3 * ['bc-right'])
    #
    #     # Creating the training input points
    #     self.input_data, self.target_data = dg.get_data()
    #
    # def fixed_fixed(self):
    #     """
    #          Method to setting the features for a fixed-fixed beam with an axial load
    #
    #     """
    #     zero = tf.constant(0.0)
    #     one = tf.constant(1.0)
    #     # Reference solution for the predictions ======================================
    #     x = np.linspace(0, self.L, int(self.num_test_samples))
    #     self.x_test = x
    #     u_ref = (self.a / np.cos(self.k * self.L)) * (np.cos(self.k * x) - 1)
    #
    #     self.ref_solu = u_ref
    #     # Reference solution for the predictions ======================================
    #
    #     # Boundary and initial conditions
    #     BC_left_1 = (self.x == 0.) * (self.u)
    #     BC_left_2 = (self.x == 0.) * (self.du_dx)
    #     BC_left_3 = (self.x == 0.) * (self.rot)
    #     BC_left = (self.x == self.L/4) * (self.du_dx - one)
    #     middle1 = (self.x == self.L/2) * (self.u - one)
    #     middle2 = (self.x == self.L / 2) * (self.du_dx)
    #
    #
    #     BC_right_1 = (self.x == self.L) * (self.u)
    #     BC_right_2 = (self.x == self.L) * (self.du_dx)
    #     BC_right_3 = (self.x == self.L) * (self.rot)
    #     BC_right = (self.x == 3 *self.L / 4) * (self.du_dx + one)
    #
    #     # Loss function
    #     self.targets = [self.eqDiff1,
    #                     BC_left_1, BC_left_2,
    #                     BC_left,
    #                     BC_right,
    #                     BC_right_1, BC_right_2
    #                     ]
    #
    #     dg = DataGeneratorX(X=[0., self.L/4, 3*self.L/4, self.L],
    #                         num_sample=self.num_training_samples,
    #                         targets=1 * ['domain'] + 2 * ['supports'] + 1 * ['supports'] + 1 * ['supports'] + 2 * ['supports'])
    #
    #     # Creating the training input points
    #     self.input_data, self.target_data = dg.get_data()


    # def fixed_free(self, problem):
    #     """
    #          Method to setting the features for a cantilever beam with an axial load
    #
    #     """
    #     zero = tf.constant(0.0)
    #     one = tf.constant(1.0)
    #     # Reference solution for the predictions ======================================
    #     x = np.linspace(0, self.L, int(self.num_test_samples))
    #     self.x_test = x
    #     u_ref = (self.a / np.cos(self.k * self.L)) * (np.cos(self.k * x) - 1)
    #
    #     self.ref_solu = u_ref
    #     # Reference solution for the predictions ======================================
    #
    #     # Boundary and initial conditions
    #     BC_left_1 = (self.x == 0.) * (self.u)
    #     BC_left_2 = (self.x == 0.) * (self.du_dx)
    #     # BC_left_3 = (self.x == 0.) * (self.d3u_dx3)
    #
    #     BC_right_1 = (self.x == self.L) * (self.d2u_dx2)
    #     # BC_right_2 = (self.x == self.L) * (self.d3u_dx3 + self.du_dx * self.k ** 2)
    #     BC_right_2 = (self.x == self.L) * (self.du_dx - 0.5)
    #
    #     # Loss function
    #     self.targets = [self.eqDiff1,
    #                BC_left_1, BC_left_2,
    #                BC_right_1, BC_right_2]
    #
    #     dg = DataGeneratorX(X=[0., self.L],
    #                         num_sample=self.num_training_samples,
    #                         targets=1 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])
    #
    #     # Creating the training input points
    #     self.input_data, self.target_data = dg.get_data()

    def fixed_free_2specie(self, problem):
        """
             Method to setting the features for a cantilever beam of a two-species problem

        """
        self.pr = problem[3]
        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        x, u_ref, rot_ref = self.reference_solution(x, problem)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Boundary and initial conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.du_dx)
        BC_left_3 = (self.x == 0.) * (self.d3u_dx3)

        BC_right_1 = (self.x == self.L) * (self.d2u_dx2 + (self.k ** 2 * self.a))
        BC_right_2 = (self.x == self.L) * (self.d3u_dx3 + self.du_dx * self.k ** 2)

        # Loss function
        self.targets = [self.eqDiff1,
                   BC_left_1, BC_left_2,BC_left_3,
                   BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 3 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def reference_solution(self, x, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
        """
        # Aditional data based on the reference solution for the predictions ======================================
        if problem[1] == "pinned" and problem[2] == "pinned":
            P_cr = np.pi ** 2 * self.E * self.I / self.L ** 2
            u_cte = (np.pi / 2) * np.sqrt(self.P / P_cr)
            u = -(((self.w * self.L ** 4) / (16 * self.E * self.I * u_cte ** 4)) * (
                        np.cos(u_cte - 2 * u_cte * x / self.L) / np.cos(u_cte) - 1) - (
                                  (self.w * self.L ** 2) / (8 * self.E * self.I * u_cte ** 2)) * x * (self.L - x))

            rot = -((2 * u_cte / self.L) * ((self.w * self.L ** 4) / (16 * self.E * self.I * u_cte ** 4)) * (
                        np.sin(u_cte - 2 * u_cte * x / self.L) / np.cos(u_cte)) - (
                                    (self.w * self.L ** 2) / (8 * self.E * self.I * u_cte ** 2)) * (self.L - 2 * x))

        elif problem[1] == "fixed" and problem[2] == "free" and self.pr == "2specie":
           u = (self.a / np.cos(self.k * self.L)) * (np.cos(self.k * x) - 1)
           rot = -(self.a / np.cos(self.k * self.L)) * self.k * np.sin(self.k * x)

        return [x, u, rot]

