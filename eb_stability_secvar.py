# @author Felipe Pereira dos Santos
# @since 21 july, 2023
# @version 21 july, 2023

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import tensorflow as tf
from sciann_datagenerator import *
import time
import sys


class EB_Stability_secvar:
    """
         Class that represents provide features for the Timoshenko bending beam analysis.

         Based on the problem''s initial and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on
    """

    def __init__(self, network, P, L, E, I, a, num_training_samples, num_test_samples):
        """
            Constructor of the Euler-Benoulli single beam stability class.

            Attributes:
                network: list of settings of a neural network used to approximate the target
                problem solution [size, activation function, initialiser]
                P: Point load at the beam
                L: beam span
                E: Young modulus
                I: inertia moment
                a: distance of the applied P load from the beam axis

                num_training_samples: number of samples for training the model
                num_test_samples: number of samples for testing the model (predictions)

            Args:
                gradient: (GradientLayer_net_u): used to compute the derivatives needed for the target problem

        """

        #self.P = P
        self.L = L
        self.E = E
        self.I = I
        self.a = a
        # self.G = E / (2 * (1 + nu))
        # self.nu = nu
        # self.k = 5. / 6.
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        # Neural Network Setup.
        dtype = 'float32'

        self.x = sn.Variable("x", dtype=dtype)

        zero = tf.constant(0.0)
        one = tf.constant(1.0)

        self.u = sn.Functional('u', self.x, network[0], network[1], kernel_initializer=network[2])
        self.rot = sn.Functional('rot', self.x, network[0], network[1], kernel_initializer=network[2])
        self.P = sn.Parameter(0.02, inputs=self.x, name='Pcr')
        # self.alpha = sn.Parameter(15*np.pi/180, inputs=self.x, name='alpha')
        # self.alpha = sn.Parameter(0.5, inputs=self.x, name='alpha')
        # self.alpha = 0.5

        self.du_dx = sn.diff(self.u, self.x)
        self.d2u_dx2 = sn.diff(self.u, self.x, order=2)
        self.d3u_dx3 = sn.diff(self.u, self.x, order=3)
        self.d4u_dx4 = sn.diff(self.u, self.x, order=4)

        # self.eqDiff1 = self.d4u_dx4 + (self.k ** 2) * self.d2u_dx2

        self.variables = [self.x]

    def model_info(self):
        """
        Method to write the physical model information in the text file output that contains the
        evaluation of the MSE errors

        DISCLAIMER: this method might be unused


        """
        model_parameters = 'Number of training samples: ' + str(self.num_training_samples) + \
                           '\nP: ' + str(self.P) + ' N | ' + 'L: ' + str(self.L) + ' m | ' + 'E: ' +\
                           str(self.E) + ' N/m² | ' + 'I: ' + str(self.I) + ' m^4 | ' + 'a: ' + str(self.a) + ' m\n'
        return model_parameters


    def free_fixed(self, problem):
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

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        P_ref = self.reference_solution(x, problem)

        self.ref_solu = P_ref
        # Reference solution for the predictions ======================================

        I = self.I*((self.a + self.x)/self.a)**2
        # I = self.I
        # self.eqDiff1 = self.E * I * self.d2u_dx2 + self.P * self.u
        # EI = 1.03084 * 10 ** 10 * (0.05 + 0.005 * self.x ** 2) ** 4
        # E = (2.1 - 2.2*self.x + 1.1*self.x**2)*10**11
        # EI = 1.03084 * 10 ** 2 * ( 0.005*self.x**2 - 0.03*self.x + 0.095) ** 4
        # d = 0.02
        # I = np.pi*d**4/64
        self.eqDiff1 = self.E*I * self.d2u_dx2 + self.P * self.u
        self.eqDiff2 = self.du_dx - self.rot

        # Boundary conditions
        BC_left_1 = (self.x == 0.0) * (self.u)
        BC_left_2 = (self.x == 0.0) * (self.du_dx - 0.5)
        # BC_left_3 = (self.x == self.L) * (self.d3u_dx3)

        BC_right_1 = (self.x == self.L) * (self.du_dx)
        # BC_right_2 = (self.x == self.L) * (self.d3u_dx3 + self.du_dx * self.k ** 2)

        # Loss function
        self.targets = [self.eqDiff1,
                   BC_left_1,BC_left_2,
                   BC_right_1]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 2 * ['bc-left'] + 1 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def reference_solution(self, x, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
         In this case, the reference solution was extracted from [1] page 126.

         [1] Timoshenko, S. P., & Gere, J. M. (1963). Theory of elastic stability. International student edition,
         second edition, McGraw-Hill.

         For each inertia ratio I_1/I_2, there is a correspondent m that generates the solution in terms of P_cr.

        """
        inertia_ratio = np.linspace(0, 1, 11)
        m = np.array([0.250, 1.350, 1.593, 1.763, 1.904, 2.023, 2.128, 2.223, 2.311, 2.392, np.pi ** 2 /4])
        P_cr = m * self.E * self. I / self.L ** 2
        dic_P = dict(zip(inertia_ratio, P_cr))

        P = dic_P[problem]

        return P

