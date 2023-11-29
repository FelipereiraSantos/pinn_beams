# @author Felipe Pereira dos Santos
# @since 23 june, 2023
# @version 23 june, 2023

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import tensorflow as tf
from sciann_datagenerator import *
from data_generator import*
# from data_generator import *
import time
import sys


class EulerBernoulli:
    """
         Class that represents provide features for the Buler Bernoulli bending beam analysis.

         Based on the problem''s initial and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on
    """

    def __init__(self, network, w, L, E, I, nu, A, num_training_samples, num_test_samples):
        """
            Constructor of the Euler-Bernoulli beam class.

            Attributes:
                network (keras network): usually represents a neural network used to approximate the target
                problem solution
                w: distributed load over the beam
                L: beam span
                E: Young modulus
                I: inertia moment
                nu: Poisson coefficient
                A: cross-section area
                num_training_samples: number of samples for training the model
                num_test_samples: number of samples for testing the model (predictions)


        """

        self.w = w
        self.L = L
        self.E = E
        self.I = I
        self.A = A
        self.G = E / (2 * (1 + nu))  # Shear modulus
        self.nu = nu
        self.k = 5. / 6.   # Form factor specific for a rectangular cross-section
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        # Neural Network Setup.
        dtype = 'float32'

        self.x = sn.Variable("x", dtype=dtype)

        zero = tf.constant(0.0)
        one = tf.constant(1.0)

        self.u = sn.Functional('u', self.x, network[0], network[1], kernel_initializer=network[2])
        self.rot = sn.Functional('rot', self.x, network[0], network[1], kernel_initializer=network[2])
        # self.M = sn.Functional('M', self.x, network[0], network[1], kernel_initializer=network[2])

        self.du_dx = sn.diff(self.u, self.x)
        self.drot_dx = sn.diff(self.rot, self.x)

        self.d2u_dx2 = sn.diff(self.u, self.x, order=2)
        self.d2rot_dx2 = sn.diff(self.rot, self.x, order=2)

        self.d3u_dx3 = sn.diff(self.u, self.x, order=3)
        self.d4u_dx4 = sn.diff(self.u, self.x, order=4)

        # Construcion of the Euler-Bernoulli beam differential equations
        self.eqDiff1 = self.d4u_dx4 - self.w/(self.E * I)
        self.eqDiff2 = self.du_dx - self.rot

        self.variables = [self.x]


    def model_info(self):
        """
        Method to write the physical model information in the text file output that contains the
        elvaluation of the MSE errors

        """
        model_parameters = 'Number of training samples: ' + str(self.num_training_samples) + \
                           '\nw: ' + str(self.w) + ' N/m | ' + 'L: ' + str(self.L) + ' m | ' + 'k: ' + str(self.k) +\
                           '\nA: ' + str(self.A) + ' m² | ' + 'G: ' + str(self.G) + ' N/m² | ' + 'E: ' + \
                           str(self.E) + ' N/m² | ' + 'I: ' + str(self.I) + ' m^4 | ' + 'nu: ' + str(self.nu) + ' []\n'
        return model_parameters


    def varying_sec(self, problem):

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        x, u_ref, rot_ref = self.reference_solution(x, problem)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        if problem[4] == "parabolic":
            self.eqDiff1 = self.d2u_dx2 + (self.w*self.L**2/(12*self.E*self.I))*(8*self.L**3*self.x + 4*self.L*self.x**3 - self.x**4)/(self.L + self.x)**4
        else:
            self.eqDiff1 = self.d2u_dx2 + (self.w * self.L ** 4 * self.x * (2 * self.L - self.x)) / (2 * self.E * self.I * (self.L + self.x) ** 4)
        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.d2u_dx2)

        BC_right_1 = (self.x == self.L) * (self.du_dx)
        BC_right_2 = (self.x == self.L) * (self.rot)

        # BC_right_1 = (self.x == self.L) * (self.u)
        # BC_right_2 = (self.x == self.L) * (self.d2u_dx2)
        # BC_right_3 = (self.x == self.L) * (self.rot)

        # Loss function
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1, BC_left_2,
                        BC_right_1, BC_right_2]

        # Additional training data===============================================
        if isinstance(problem[3], int):
            print("Additional data was added to training")
            x_input = np.linspace(0, self.L, int(problem[3]))
            data = self.additional_data(x_input, problem)
            # self.input_data, self.target_data = self.get_inputs_with_data()

            self.targets.append(sn.Data(self.u))
            self.targets.append(sn.Data(self.rot))

            dg = DataGenerator1D(X=[0., self.L],
                                 num_sample=self.num_training_samples,
                                 targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'] + 2 * ['data'],
                                 data=data)

        else:

            dg = DataGeneratorX(X=[0., self.L],
                                num_sample=self.num_training_samples,
                                targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])
            #
            # dg = DataGeneratorX(X=[0., self.L],
            #                     num_sample=self.num_training_samples,
            #                     targets=2 * ['domain'] + 4 * ['supports'])

            # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def pinned_pinned(self, problem):
        """
             Method for setting the features for the simply supported Timoshenko beam

             x_test: array of collocation points to evaluate the trained model and the reference solution
             u_ref: reference solution for further comparisons (obtained analytical or numerically)
             input_data: points distributed all over the problem domain for training
             target_data: corresponded labels of the input_data points ("true" values"
             targets: target constraints involving the differential equations and boundary conditions
                    eqDiff: refers to one or more differential equations of the problem
                    BC_left: refers to the target points at the left boundary of the beam
                    BC_right: refers to the target points at the right boundary of the beam

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        x, u_ref, rot_ref = self.reference_solution(x, problem)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        if problem[4] == "parabolic":
            self.eqDiff1 = self.d4u_dx4 - (4 * self.w) / (self.E * self.I * self.L ** 2) * (
                        self.L * self.x - self.x ** 2)

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.d2u_dx2)
        BC_left_3 = (self.x == 0.) * (self.drot_dx)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.d2u_dx2)
        BC_right_3 = (self.x == self.L) * (self.drot_dx)

        # Loss function
        self.targets = [self.eqDiff1, self.eqDiff2,
                   BC_left_1, BC_left_2,
                   BC_right_1, BC_right_2]

        # Additional training data===============================================
        if isinstance(problem[3], int):
            print("Additional data was added to training")
            x_input = np.linspace(0, self.L, int(problem[3]))
            data = self.additional_data(x_input, problem)
            # self.input_data, self.target_data = self.get_inputs_with_data()

            self.targets.append(sn.Data(self.u))
            self.targets.append(sn.Data(self.rot))

            dg = DataGenerator1D(X=[0., self.L],
                                num_sample=self.num_training_samples,
                                targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'] + 2 * ['data'],
                                 data=data)

        else:

            dg = DataGeneratorX(X=[0., self.L],
                                num_sample=self.num_training_samples,
                                targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

            # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_free(self, problem):
        """
             Method for setting the features for the cantilever (fixed-free) Timoshenko beam

             x_test: array of collocation points to evaluate the trained model and the reference solution
             u_ref: reference solution for further comparisons (obtained analytical or numerically)
             input_data: points distributed all over the problem domain for training
             target_data: corresponded labels of the input_data points ("true" values"
             targets: target constraints involving the differential equations and boundary conditions
                    eqDiff: refers to one or more differential equations of the problem
                    BC_left: refers to the target points at the left boundary of the beam

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        x, u_ref, rot_ref = self.reference_solution(x, problem)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================
        if problem[5] == "Iparabolic":
            one = tf.constant(1.0)
            a =  tf.constant(-1 / 18)
            b =  tf.constant(3 / 20)
            c =  tf.constant(3 / 20)
            pi = tf.constant(np.pi)
            # self.eqDiff1 = self.d2u_dx2 - (64 * self.w / (np.pi * self.E)) * ((self.L - self.x)/(a * self.x ** 2 + b * self.x + c) ** 4)
            #self.eqDiff1 = self.d4u_dx4 - 4*(64 * self.w / (np.pi * self.E)) * ((4 * a * self.x + 2 * b - (self.L - self.x) * (2 * a - (5 * (2 * a * self.x + b) ** 2)/(a * self.x ** 2 + b * self.x + c))) / (a * self.x ** 2 + b * self.x + c) ** 5)
            self.eqDiff1 = self.d2u_dx2 - (64 * self.w / (pi * self.E)) * ((self.L - self.x) / (a * self.x ** 2 + b * self.x + c) ** 4)
            # self.eqDiff3 = self.M - (64 * self.w / (pi * self.E)) * ((self.L - self.x) / (a * self.x ** 2 + b * self.x + c) ** 4)
            # self.eqDiff4 = self.M - self.E*self.I*self.d2u_dx2
        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.du_dx)
        BC_left_3 = (self.x == 0.) * (self.rot)
        # BC_left_4 = (self.x == 0.) * (self.d2u_dx2 - self.w * self.L)

        BC_right_1 = (self.x == self.L) * (self.d2u_dx2)
        BC_right_2 = (self.x == self.L) * (self.d3u_dx3)
        # BC_right_3 = (self.x == self.L) * (self.M)
        # BC_right_2 = (self.x == self.L) * (self.d2rot_dx2)

        # Loss function
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1, BC_left_2, BC_left_3, BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=2 * ['domain'] + 3 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_fixed(self, problem):
        """
             Method for setting the features for the double-fixed (fixed-fixed) Timoshenko beam

             x_test: array of collocation points to evaluate the trained model and the reference solution
             u_ref: reference solution for further comparisons (obtained analytical or numerically)
             input_data: points distributed all over the problem domain for training
             target_data: corresponded labels of the input_data points ("true" values"
             targets: target constraints involving the differential equations and boundary conditions
                    eqDiff: refers to one or more differential equations of the problem
                    BC_left: refers to the target points at the left boundary of the beam
                    BC_right: refers to the target points at the right boundary of the beam

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        x, u_ref, rot_ref = self.reference_solution(x, problem)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.du_dx)
        BC_left_3 = (self.x == 0.) * (self.rot)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.du_dx)
        BC_right_3 = (self.x == self.L) * (self.rot)

        # Loss function
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1, BC_left_2, BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_pinned(self, problem):
        """
             Method for setting the features for the fixed-pinned Timoshenko beam

             x_test: array of collocation points to evaluate the trained model and the reference solution
             u_ref: reference solution for further comparisons (obtained analytical or numerically)
             input_data: points distributed all over the problem domain for training
             target_data: corresponded labels of the input_data points ("true" values"
             targets: target constraints involving the differential equations and boundary conditions
                    eqDiff: refers to one or more differential equations of the problem
                    BC_left: refers to the target points at the left boundary of the beam
                    BC_right: refers to the target points at the right boundary of the beam

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        x, u_ref, rot_ref = self.reference_solution(x, problem)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.du_dx)
        BC_left_3 = (self.x == 0.) * (self.rot)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.d2u_dx2)

        # Loss function
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1, BC_left_2, BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def reference_solution(self, x, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
        """
        # Aditional data based on the reference solution for the predictions ======================================
        # x = np.linspace(0, self.L, int(problem[3]))
        if problem[1] == "pinned" and problem[2] == "pinned":
            if problem[4] == "parabolic":
                u = (self.w / (90 * self.E * self.I * self.L ** 2)) * (-x ** 6 + 3 * self.L * x ** 5 - 5 * self.L ** 3 * x ** 3 + 3 * self.L ** 5 * x)
                rot = (self.w / (90 * self.E * self.I * self.L ** 2)) * (-6 * x ** 5 + 15 * self.L * x ** 4 - 15 * self.L ** 3 * x ** 2 + 3 * self.L ** 5)
            else:
                u = -(self.w / (24 * self.E * self.I)) * (-x ** 4 + 2 * self.L * (x ** 3) - (self.L ** 3) * x)
                rot = -(self.w / (24 * self.E * self.I)) * (-4 * x ** 3 + 6 * self.L * x ** 2 - self.L ** 3)
        elif problem[1] == "fixed" and problem[2] == "free":
            if problem[5] == "Iparabolic":
                u = -(self.w / (24 * self.E * self.I)) * (-x ** 4 + 4 * self.L * x ** 3 - 6 * self.L ** 2 * x ** 2)
                rot = -(self.w / (24 * self.E * self.I)) * (-4 * x ** 3 + 12 * self.L * x ** 2 - 12 * self.L ** 2 * x)

                # x_var = smp.symbols('x_var', real=True)
                # a, b, c = smp.symbols('a b c', real=True)
                # L = smp.symbols('L', real=True, positive=True)
                # q = smp.symbols('q', real=True)
                # cte = smp.symbols('cte', real=True)
                # E = smp.symbols('E', real=True, positive=True)
                # pi = smp.symbols('pi', real=True, positive=True)
                # # L = self.L
                # # a = -1 / 18
                # # b = 3 / 20
                # # c = 3 / 20
                # # q = self.w
                # # E = self.E
                # # pi = np.pi
                # cte = 64*q/(E*pi)
                # f = cte*((L - x_var) / (a*x_var**2 + b*x_var + c) ** 4)  # d2u_dx2
                # g = smp.integrate(f, x_var)
                # C1 = -smp.integrate(f, x_var).subs(x_var, 0)
                # rot_aux1 = g + C1
                # rot_aux2 = rot_aux1.subs(L, self.L).subs(a, -1/18).subs(b, 3/20).subs(c, 3/20).subs(q, self.w).subs(E, self.E).subs(pi, np.pi)
                # rot_numeric = smp.lambdify(x_var, rot_aux2)
                # h = smp.integrate(rot_aux1, x_var)
                # C2 = -smp.integrate(rot_aux1, x_var).subs(x_var, 0)
                # u_aux1 = h + C2
                # u_aux2 = u_aux1.subs(L, self.L).subs(a, -1/18).subs(b, 3/20).subs(c, 3/20).subs(q, self.w).subs(E, self.E).subs(pi, np.pi)
                # u_numeric = smp.lambdify(x_var, u_aux2)
                # u = u_numeric(x)
                # rot = rot_numeric(x)
            else:
                u = -(self.w / (24 * self.E * self.I)) * (-x ** 4 + 4 * self.L * x ** 3 - 6 * self.L ** 2 * x ** 2)
                rot = -(self.w / (24 * self.E * self.I)) * (-4 * x ** 3 + 12 * self.L * x ** 2 - 12 * self.L ** 2 * x)
        elif problem[1] == "fixed" and problem[2] == "fixed":
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 2 * x ** 2)
            rot = (self.w / (24 * self.E * self.I)) * (4 * x ** 3 - 6 * self.L * x ** 2 + 2 * self.L ** 2 * x)
        elif problem[1] == "fixed" and problem[2] == "pinned":
            u = -(self.w / (48* self.E * self.I)) * (-2 * x ** 4 + 5 * self. L * x ** 3 - 3 * self.L ** 2 * x ** 2)
            rot = -(self.w / (48 * self.E * self.I)) * (-8 * x ** 3 + 15 * self.L * x ** 2 - 6 * self.L ** 2 * x)
        elif problem[1] == "varying_sec":
            if problem[4] == "parabolic":
                u = -(self.w*self.L**2/(12*self.E*self.I))*(-13*self.L**4/(6*(self.L + x)**2) + 12*self.L**3/(self.L + x) - 325*self.L * x/24 -\
                    8*self.L*x*np.log(2*self.L) + 2*self.L*(13*self.L+4*x)*np.log(self.L + x) - x**2/2 - 26*self.L**2*np.log(self.L) - 59*self.L**2/6)
                rot = -(self.w*self.L**2/(12*self.E*self.I))*(8*self.L*np.log(self.L + x) - x + (self.L**2*(31*self.L**2 + 72*self.L*x + 54*x**2))/(3*(self.L +x)**3)- \
                    (133*self.L/24 + 8*self.L*np.log(2*self.L)))

            else:
                u = -((self.w * L ** 4) / (2 * self.E * self.I)) * ((9 * L ** 2 * x + 14 * L * x ** 2 + x ** 3) / (8 * L * (L + x)**2) - np.log(1 + x / L))
                # rot1 = -((self.w * L ** 4) / (2 * self.E * self.I)) * (L ** 3 + 3 * L ** 2 * x - 5 * L * x ** 2 + x ** 3)/(8 * L * (L + x) ** 3)
                rot = -((self.w * L ** 3) / (16 * self.E * self.I)) * (1 - 8 * L * x ** 2/(L + x) ** 3)

        return [x, u, rot]

    def additional_data(self, x, problem):
        """
         Method to add data from the reference solution to the model training data.
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
        """
        # Aditional data based on the reference solution for the predictions ======================================
        # x = np.linspace(0, self.L, int(problem[3]))
        if problem[1] == "pinned" and problem[2] == "pinned":
            u = (self.w / (24 * self.E * self.I)) * (-x ** 4 + 2 * self.L * (x ** 3) - (self.L ** 3) * x)
            rot = (self.w / (24 * self.E * self.I)) * (-4 * x ** 3 + 6 * self.L * x ** 2 + self.L ** 3)
        elif problem[1] == "fixed" and problem[2] == "free":
            u = (self.w / (24 * self.E * self.I)) * (-x ** 4 + 4 * self.L * (x ** 3) - 6 * (self.L ** 2) * (x ** 2))
            rot = (self.w / (6 * self.E * self.I)) * (-x ** 3 + 3 * self.L * x ** 2 + 3 * self.L ** 2 * x)
        elif problem[1] == "fixed" and problem[2] == "fixed":
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 2 * x ** 2)
            rot = (self.w / (24 * self.E * self.I)) * (4 * x ** 3 - 6 * self.L * x ** 2 + 2 * self.L ** 2 * x)
        elif problem[1] == "fixed" and problem[2] == "pinned":
            u = (self.w / (48* self.E * self.I)) * (-2 * x ** 4 + 5 * L * x ** 3 - 3 * self.L ** 2 * x ** 2)
            rot = (1 / (6 * self.E * self.I)) * (
                        self.w * x ** 3 + 3 * C1 * x ** 2 - 3 * self.w * self.L ** 2 * x - 6 * C1 * self.L * x)

        # u_add = np.tile(u, int(self.num_training_samples / problem[3]))
        # rot_add = np.tile(rot, int(self.num_training_samples / problem[3]))
        # x_add = np.tile(x, int(self.num_training_samples / problem[3]))
        self.u_add = u
        self.rot_add = rot
        self.x_add = x

        return [x, u, rot]
        # return x_add, u_add, rot_add

    def plotting(self, x_test, u_ref, u_test, rot_ref, rot_test):
        err_u = np.sqrt(np.linalg.norm(u_test - u_ref)) / np.linalg.norm(u_ref)
        err_rot = np.sqrt(np.linalg.norm(rot_test - rot_ref)) / np.linalg.norm(rot_ref)

        err_u = "{:.3e}".format(err_u)
        err_rot = "{:.3e}".format(err_rot)

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        # fig.subplots_adjust(bottom=0.15, left=0.2)
        # str(round(err_u, 3))
        ax[0].plot(x_test, u_test, 'r', x_test, u_ref, 'b')
        # ax[0].plot(x_test, u_test, 'r',)
        ax[0].set_xlabel('x [m]')
        ax[0].set_ylabel('displacements [m]')
        ax[0].text(0.01, 0.01, "error disp: " + str(err_u),
                   verticalalignment='bottom', horizontalalignment='left',
                   transform=ax[0].transAxes,
                   color='black', fontsize=8)
        # ax[0].text(0.15, 3, "error disp: " + str(err_u), fontsize=15)
        ax[0].grid()
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.legend(loc='best')

        ax[1].plot(x_test, rot_test, 'r', x_test, rot_ref, 'b')
        # ax[1].plot(x_test, rot_test, 'r')
        ax[1].set_xlabel('x [m]')
        ax[1].set_ylabel('rad []')
        ax[1].text(0.01, 0.01, "error rot: " + str(err_rot),
                   verticalalignment='bottom', horizontalalignment='left',
                   transform=ax[1].transAxes,
                   color='black', fontsize=8)
        ax[1].grid()
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.legend(loc='best')
        # plt.savefig('eb_varsec_pp_P_0.001_32_300.pdf')

        plt.show()
