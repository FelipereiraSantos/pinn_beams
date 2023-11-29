# @author Felipe Pereira dos Santos
# @since 27 august, 2023
# @version 27 august, 2023

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import tensorflow as tf
from sciann_datagenerator import *
import time
import sys
# import sciann.math.sign as sign


class EB_Dynamics:
    """
         Class that represents provide features for the Euler-Bernoulli beam dynamics analysis.

         Based on the problem''s initial and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on
    """

    def __init__(self, network, P, L, E, I, rho, A, num_training_samples, num_test_samples):
        """
            Constructor of the Euler-Benoulli single beam stability class.

            Attributes:
                network (keras network): usually represents a neural network used to approximate the target
                problem solution
                P: force load at the beam
                L: beam span
                E: Young modulus
                I: inertia moment
                rho: material density
                A: cross-section area
                num_training_samples: number of samples for training the model
                num_test_samples: number of samples for testing the model (predictions)

            Args:
                gradient: (GradientLayer_net_u): used to compute the derivatives needed for the target problem

        """

        self.P = P
        self.L = L
        self.E = E
        self.I = I
        self.rho = rho
        self.A = A
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        self.timeL = 5 # Analsis duration time [s]

        # Neural Network Setup.
        dtype = 'float32'

        self.x = sn.Variable("x", dtype=dtype)  # Space variable
        self.t = sn.Variable("t", dtype=dtype)  # Time variable

        zero = tf.constant(0.0)
        one = tf.constant(1.0)

        def cosine_act(x):
            # act = tf.math.sin(x)
            # act = tf.math.cos(x)
            f = 0.1
            # act = x - (1 - tf.math.cos(2 * f * x))/(2 * f)
            # act = x + tf.math.sin(x) ** 2

            a = 15
            act = x + tf.math.sin(a * x) ** 2  / a
            return act

        self.u = sn.Functional('u', [self.x, self.t], network[0], cosine_act, kernel_initializer=network[2])
        self.rot = sn.Functional('rot', [self.x, self.t], network[0], network[1], kernel_initializer=network[2])

        self.du_dx = sn.diff(self.u, self.x)
        self.d2u_dx2 = sn.diff(self.u, self.x, order=2)
        self.d3u_dx3 = sn.diff(self.u, self.x, order=3)
        self.d4u_dx4 = sn.diff(self.u, self.x, order=4)

        self.du_dt = sn.diff(self.u, self.t)
        self.d2u_dt2 = sn.diff(self.u, self.t, order=2)

        self.eqDiff1 = self.d4u_dx4 + (self.rho*self.A/(self.E * self.I)) * self.d2u_dt2  #Governing differential equation
        self.eqDiff2 = self.du_dx - self.rot
        self.eqDiff3 = self.u * (0.5 - 0.5 * np.sign(self.timeL - 0.01))
        self.eqDiff4 = self.du_dt * (0.5 - 0.5 * np.sign(self.timeL - 0.01))

        # self.eqDiff3 = (1 - sn.sign(self.t)) * self.u
        # self.eqDiff4 = (1 - sn.sign(self.t)) * self.du_dt
        self.eqDiff5 =  self.d2u_dt2 * (0.5 - 0.5 * np.sign(self.timeL - 0.01))

        self.variables = [self.x, self.t]

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
             Method for to setting the features for the simply supported beam

             Inputs:
                 x_1: relative to the first governing differential equation
                 x_2: relative to the second governing differential equation
                 x_3: relative to the deflection (elastic curve)
                 x_4: relative to the derivative of the rotation (bending moment)

             Outputs:
                 u_1: expected result based on x_1
                 u_2: expected result based on x_2
                 u_3: expected result based on x_3
                 u_4: expected result based on x_4

             x_train: array of training parameters (inputs of the neural network)
             y_train: array of the expected results (correspondent 'correct' outputs for the x_train)
             xL: correspondent array of layers for training the physics-informed neural network
             uL: correspondent array of for expected results of the physics-informed neural network

        """

        zero = tf.constant(0.0)
        one = tf.constant(1.0)
        Tforce = tf.constant(1.0)
        # Reference solution for the predictions ======================================
        # x = np.linspace(0, self.L, int(self.num_test_samples))
        # t = np.linspace(0, self.timeL, int(self.num_test_samples))
        x, t = np.meshgrid(
            np.linspace(0, self.L, int(self.num_test_samples)),
            np.linspace(0, self.timeL, int(self.num_test_samples))
            # np.linspace(0, self.L, 4),
            # np.linspace(0, self.timeL, 2)
        )
        self.x_test = x
        self.t_test = t
        # u_ref = (self.a / np.cos(self.k * self.L)) * (np.cos(self.k * x) - 1)
        u_ref = 0.5

        self.ref_solu = u_ref
        # Reference solution for the predictions ======================================

        # self.eqDiff2 = -self.E*self.I*self.d3u_dx3 - self.P # Representation of the force load at the beam's tip

        # Boundary conditions
        BC_left_1 = (self.x == 0.0) * (self.u)
        BC_left_2 = (self.x == 0.0) * (self.du_dx)

        BC_right_1 = (self.x == self.L) * (self.d2u_dx2)
        BC_right_2 = (self.x == self.L) * (self.E * self.I * self.d3u_dx3 - self.P * (0.5 - 0.5*np.sign(self.t - Tforce)))
        # BC_right_2 = (self.x == self.L) * (self.d3u_dx3)

        # self.eqDiff5 = self.E * self.I * self.d3u_dx3 - self.P * (0.5 - 0.5*np.sign(self.t - Tforce))
        # Loss function
        self.targets = [self.eqDiff1,self.eqDiff3, self.eqDiff4,self.eqDiff5,
                   BC_left_1,BC_left_2,
                   BC_right_1,  BC_right_2]

        dg = DataGeneratorXT(X=[0., self.L],
                             T=[0., self.timeL],
                            num_sample=self.num_training_samples,
                            targets=4 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def plotting(self, t_test, u_test):
        # err_u = np.sqrt(np.linalg.norm(u_test - u_ref)) / np.linalg.norm(u_ref)
        # err_rot = np.sqrt(np.linalg.norm(rot_test - rot_ref)) / np.linalg.norm(rot_ref)

        # err_u = "{:.3e}".format(err_u)
        # err_rot = "{:.3e}".format(err_rot)

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        # fig.subplots_adjust(bottom=0.15, left=0.2)
        # str(round(err_u, 3))
        # ax[0].plot(x_test, u_test, 'r', x_test, u_ref, 'b')
        ax[0].plot(t_test, u_test, 'r', )
        ax[0].set_xlabel('t [s]')
        ax[0].set_ylabel('displacements [m]')
        # ax[0].text(0.01, 0.01, "error disp: " + str(err_u),
        #            verticalalignment='bottom', horizontalalignment='left',
        #            transform=ax[0].transAxes,
        #            color='black', fontsize=8)
        # ax[0].text(0.15, 3, "error disp: " + str(err_u), fontsize=15)
        ax[0].grid()
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.legend(loc='best')

        # ax[1].plot(x_test, rot_test, 'r')
        # ax[1].set_xlabel('x [m]')
        # ax[1].set_ylabel('rad []')
        # ax[1].text(0.01, 0.01, "error rot: " + str(err_rot),
        #            verticalalignment='bottom', horizontalalignment='left',
        #            transform=ax[1].transAxes,
        #            color='black', fontsize=8)
        # ax[1].grid()
        # plt.grid(color='black', linestyle='--', linewidth=0.5)
        # plt.legend(loc='best')
        # plt.savefig('eb_dynamic_ffr_P_0.001_32_300.pdf')

        plt.show()

