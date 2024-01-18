# @author Felipe Pereira dos Santos
# @since 12 june, 2023
# @version 15 june, 2023

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import tensorflow as tf
from sciann_datagenerator import *
from data_generator import*
# from data_generator import *
import time
import sys


class Timoshenko:
    """
         Class that represents provide features for the Timoshenko bending beam analysis.

         Based on the problem's initial and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on
    """

    def __init__(self, network, w, L, E, I, nu, A, num_training_samples, num_test_samples):
        """
            Constructor of the Timoshenko beam class.

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

        self.problem = "Tk_bending"
        self.w = w
        self.L = L
        self.E = E
        self.I = I
        self.A = A
        self.G = E / (2 * (1 + nu))  # Shear modulus
        self.nu = nu
        self.k = 5. / 6.   # Form factor specific for a rectangular cross-section
        # self.k = 9. / 10   # Form factor specific for a circular cross-section
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        # Neural Network Setup.
        dtype = 'float32'

        self.x = sn.Variable("x", dtype=dtype)


        self.u_aux = sn.Functional('u', self.x, network[0], network[1], kernel_initializer=network[2])
        self.rot_aux = sn.Functional('rot', self.x, network[0], network[1], kernel_initializer=network[2])
        # self.M = sn.Functional('M', self.x, network[0], network[1], kernel_initializer=network[2])

        self.u = self.u_aux
        self.rot = self.rot_aux

        self.du_dx = sn.diff(self.u, self.x)
        self.drot_dx = sn.diff(self.rot, self.x)

        self.d2u_dx2 = sn.diff(self.u, self.x, order=2)
        self.d2rot_dx2 = sn.diff(self.rot, self.x, order=2)

        # d4u_dx4 = sn.diff(u, x, order=4)

        # Construcion of the Timoshenko beam differential equations
        self.eqDiff1 = ((self.E * self.I) / (self.k * self.A * self.G)) * self.d2rot_dx2 + (self.du_dx - self.rot)
        self.eqDiff2 = (self.d2u_dx2 - self.drot_dx) + (self.w / (self.k * self.A * self.G))

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

        if problem[3] == "LinearShape":
            self.linear_shape()

        elif problem[3] == "ParabolicShape":
            self.parabolic_shape()

        elif problem[3] == "ParabolicLoad":
            self.eqDiff2 = (self.d2u_dx2 - self.drot_dx) + (-self.x ** 2 + self.L * self.x) * (4 * self.w) / (self.k * self.A * self.G * self.L ** 2)


        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.drot_dx)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.drot_dx)

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

            if problem[3] == "LinearShape":
                # Loss function

                BC_right_1 = (self.x == self.L) * (self.du_dx)
                BC_right_2 = (self.x == self.L) * (self.rot)
                self.targets = [self.eqDiff1, self.eqDiff2,
                                BC_left_1, BC_left_2,
                                BC_right_1, BC_right_2]

                dg = DataGeneratorX(X=[0., self.L],
                                    num_sample=self.num_training_samples,
                                    targets=2 * ['domain'] + 2 * ['bc-left']+ 2 * ['bc-right'])

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
        # x = np.linspace(0, self.L, int(self.num_test_samples))
        # self.x_test = x
        # u_ref = (self.w/(24 * self.E * self.I))*(x ** 4 - 4 * self.L * x ** 3 + 6 * self.L ** 2 * x ** 2) + (self.w/(2 * self.k * self.G * self.A))*(-x ** 2 + 2 * self.L * x)
        # rot_ref = (self.w/(6 * self.E * self.I))*(x ** 3 - 3 * self.L* x ** 2 + 3 * self.L ** 2 * x)
        # self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================
        if problem[3] == "ParabolicShape":
            self.parabolic_shape()
            # diff_I = (pi/16)*(a * x ** 2 + b * x + c) ** 3 * (2*a*self.x + b)

            # For a point load at the end of the beam
            # cte = 64 * self.w / (self.E * pi)
            # self.eqDiff1 = self.drot_dx - cte * ((self.L-self.x) / (a * self.x ** 2 + b * self.x + c) ** 4)
            # self.eqDiff2 = self.du_dx - self.rot + (self.E/self.G)*((a * self.x ** 2 + b * self.x + c)*(2*a*self.x +b)/4 * self.drot_dx + (a * self.x ** 2 + b * self.x + c)**2/16 * self.d2rot_dx2)

            # self.eqDiff1 = self.d2u_dx2 - (64 * self.w / (np.pi * self.E)) * ((self.L - self.x)/(a * self.x ** 2 + b * self.x + c) ** 4)
            #self.eqDiff1 = self.d4u_dx4 - 4*(64 * self.w / (np.pi * self.E)) * ((4 * a * self.x + 2 * b - (self.L - self.x) * (2 * a - (5 * (2 * a * self.x + b) ** 2)/(a * self.x ** 2 + b * self.x + c))) / (a * self.x ** 2 + b * self.x + c) ** 5)
            # self.eqDiff1 = self.d2u_dx2 - (64 * self.w / (pi * self.E)) * ((self.L - self.x) / (a * self.x ** 2 + b * self.x + c) ** 4)
            # self.eqDiff3 = self.M - (64 * self.w / (pi * self.E)) * ((self.L - self.x) / (a * self.x ** 2 + b * self.x + c) ** 4)
            # self.eqDiff4 = self.M -  (self.E * I) * self.drot_dx

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.rot)

        BC_right_1 = (self.x == self.L) * (self.drot_dx)
        BC_right_2 = (self.x == self.L) * (self.du_dx - self.rot)


        # Loss function
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1,BC_left_2,
                        BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=3 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

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
        C1 = -(self.w * self.L * (self.k * self.G * self.A * self.L ** 2 + 12 * self.E * self.I)) / (24 * self.E * self.I + 2 * self.k * self.G * self.A * self.L ** 2);
        u_ref = (self.w/(24 * self.E * self.I))*(x ** 4 - 2 * self.L * x ** 3 + self.L ** 2 * x ** 2) + (self.w/(2 * self.k * self.G * self.A))*(-x ** 2 + self.L * x)
        rot_ref = (1/(6*self.E * self.I))*(self.w * x ** 3 + 3 * C1 * x ** 2 - self.w * self.L ** 2 * x -3 * C1 * self.L * x)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.rot)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.rot)

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
        C1 = -(5 * self.w * self.k * self.G * self.A * self.L ** 3 + 12 * self.w * self.L * self.E * self.I)/(8 * (self.k * self.G * self.A * self.L ** 2 + 3 * self.E * self.I))
        u_ref = (1/(self.E * self.I))*(self.w * x ** 4/24 + C1 * x ** 3/6 - (self.w * self.L ** 2/4) * x ** 2 - (C1 * self.L/2) * x ** 2) - (self.w/(2 * self.k * self.G * self.A)) * x ** 2 - (C1 * x)/(self.k * self.G * self.A)
        rot_ref = (1/(6 * self.E * self.I))*(self.w * x ** 3 + 3 * C1 * x ** 2 - 3 * self.w * self.L ** 2 * x -6 * C1 * self.L * x)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.rot)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.drot_dx)

        # Loss function
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1, BC_left_2, BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def linear_shape(self):
        # Construcion of the Timochenko beam differential equations
        C = ((self.E * self.I) / (self.k * self.A * self.G * self.L ** 2))

        self.eqDiff1 = C * (4 * (self.L + self.x) * self.drot_dx + self.d2rot_dx2 * (self.L + self.x) ** 2) + (
                    self.du_dx - self.rot)
        self.eqDiff2 = 2 * (self.L + self.x) * (self.du_dx - self.rot) + (self.L + self.x) ** 2 * (
                    self.d2u_dx2 - self.drot_dx) + ((self.w * self.L ** 2) / (self.k * self.A * self.G))

    def parabolic_shape(self):
        qi = 0.25
        qm = 0.50
        qf = 0.20
        a = (-4 * qm + 2 * qf + 2 * qi) / self.L ** 2
        b = (4 * qm - qf - 3 * qi) / self.L
        c = qi
        pi = tf.constant(np.pi)

        # b = 0.2
        # h = (a * self.x ** 2 + b * self.x + c)
        #
        # I_var = b * h ** 3 / 12
        # diff_I_var = b * 3 * h ** 2 * (2 * a * self.x + b) / 12
        # A = b * h
        # diff_A = b * (2 * a * self.x + b)
        # I_A = h ** 2 / 12 # I over A
        #
        # diff_I_A = 3 * h * (2*a*self.x + b) / 12
        #
        # cte = self.w / (self.E)
        # self.eqDiff1 = (self.du_dx - self.rot) + (self.E / (self.G * self.k)) * (
        #             diff_I_A * self.drot_dx + I_A * self.d2rot_dx2)



        I_var = (pi / 64) * (a * self.x ** 2 + b * self.x + c) ** 4
        diff_I_var = (pi/16) * (a * self.x ** 2 + b * self.x + c) ** 3 * (2*a*self.x + b)
        diff_A = (pi/2)*(a * self.x ** 2 + b * self.x + c)*(2*a*self.x + b)
        A = (pi/4)*(a * self.x ** 2 + b * self.x + c)**2
        diff_I_A = (1/4)*(a * self.x ** 2 + b * self.x + c)*(2*a*self.x + b)
        I_A = (1/16)*(a * self.x ** 2 + b * self.x + c)**2 # I over A
        cte = 32 * self.w / (self.E * pi)
        # self.eqDiff1 = sn.diff(self.E * I_var * self.drot_dx, self.x) + A * self.G * self.k * (self.du_dx - self.rot)
        # self.eqDiff1 = (self.du_dx - self.rot) + (self.E/(self.G * self.k))*(diff_I_A*self.drot_dx + I_A*self.d2rot_dx2)

        self.eqDiff1 = self.drot_dx - cte * (
                    (-2 * self.L * self.x + self.x ** 2 + self.L ** 2) / (a * self.x ** 2 + b * self.x + c) ** 4) # From the bending moment

        # self.eqDiff3 = self.E * (diff_I_var * self.drot_dx  + I_var * self.d2rot_dx2) + self.G*self.k * A * (self.du_dx - self.rot)
        self.eqDiff2 = (self.w/(self.G*self.k)) + (diff_A *(self.du_dx - self.rot) + A *(self.d2u_dx2 - self.drot_dx))
        # self.eqDiff2 = (self.w/(self.G*self.k))*(self.L - self.x) - A * (self.du_dx - self.rot)

        # Define the file path
        file_path = 'C:/Users/felip/git/beam_pinns/NumericalResults/StaticResults/CaseStudies/ParabolicShape/FixedFree/Tk_ffr_q_ParabolicShape_ref_circ_200.csv'

        # Using numpy's genfromtxt function to read the CSV file
        # Skipping the first row if it contains headers
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

        # Separating the columns into numpy arrays
        x = data[:, 0]
        u_ref = data[:, 1]
        rot_ref = data[:, 2]
        self.ref_solu = [u_ref, rot_ref]

        mesh = [5, 11, 17]
        str_mesh = ['Tk_ffr_q_ParabolicShape_ref_circ_5.csv', 'Tk_ffr_q_ParabolicShape_ref_circ_11.csv',
                    'Tk_ffr_q_ParabolicShape_ref_circ_17.csv']
        self.mesh_ref = []
        for i, num in enumerate(mesh):
            # Define the file path
            file_path = 'C:/Users/felip/git/beam_pinns/NumericalResults/StaticResults/CaseStudies/ParabolicShape/FixedFree/' + \
                        str_mesh[i]

            # Using numpy's genfromtxt function to read the CSV file
            # Skipping the first row if it contains headers
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

            # Separating the columns into numpy arrays
            x_m = data[:, 0]
            u_ref_m = data[:, 1]
            rot_ref_m = data[:, 2]
            aux_mesh = [x_m, u_ref_m, rot_ref_m]
            self.mesh_ref.append(aux_mesh)

    def reference_solution(self, x, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
        """
        # Aditional data based on the reference solution for the predictions ======================================
        # x = np.linspace(0, self.L, int(problem[3]))
        if problem[1] == "pinned" and problem[2] == "pinned":
            if problem[3] == "LinearShape":
                if x.size % 2 > 0:
                    x_half = x[:(int(x.size /2) + 1)]
                else:
                    x_half = x[:int(x.size /2)]
                u_half = -((self.w * self.L ** 4) / (2 * self.E * self.I)) * (
                            self.L * (3 * self.L + 4 * x_half) / (2 * (self.L + x_half) ** 2) + np.log(self.L + x_half) - 1.5 - np.log(
                        self.L) - \
                            x_half / (8 * self.L)) - ((self.w * self.L ** 2) / (self.G * self.A * self.k)) * (
                                2 * self.L / (self.L + x_half) + np.log(self.L + x_half) - 2 - np.log(self.L))
                rot_half = -((self.w * self.L ** 4) / (2 * self.E * self.I)) * (x_half ** 2 / (self.L + x_half) ** 3) + (
                            (self.w * self.L ** 3) / (16 * self.E * self.I))
                if x.size % 2 > 0:
                    u = np.concatenate((u_half, u_half[:-1][::-1]))
                    rot = np.concatenate((rot_half, rot_half[:-1][::-1]))
                else:
                    u = np.concatenate((u_half, u_half[::-1]))
                    rot = np.concatenate((rot_half, rot_half[::-1]))
                u_half = -((self.w * self.L ** 4) / (2 * self.E * self.I)) * (
                        self.L * (3 * self.L + 4 * x) / (2 * (self.L + x) ** 2) + np.log(self.L + x) - 1.5 - np.log(
                    self.L) - \
                        x / (8 * self.L)) - ((self.w * self.L ** 2) / (self.G * self.A * self.k)) * (
                                 2 * self.L / (self.L + x) + np.log(self.L + x) - 2 - np.log(self.L))
                rot_half = -((self.w * self.L ** 4) / (2 * self.E * self.I)) * (x ** 2 / (self.L + x) ** 3) + (
                        (self.w * self.L ** 3) / (16 * self.E * self.I))
                u = u_half
                rot = rot_half
                print(rot_half)
                sys.exit()

            elif problem[3] == "ParabolicLoad":
                u = (4 * self.w / (self.E * self.I * self.L ** 2)) * (-x ** 6 / 360 + self.L * x ** 5 / 120 - self.L ** 3 * x ** 3 / 72 + self.L ** 5 * x / 120)\
                    + (4 * self.w / (self.k * self.G * self.A * self.L ** 2)) * (x ** 4 / 12 - self.L * x ** 3 / 6 + self.L ** 3 * x / 12)
                rot = (4 * self.w / (self.E * self.I * self.L ** 2)) * (-x ** 5 / 60 + self.L * x ** 4 / 24 - self.L ** 3 * x ** 2 / 24 + self.L ** 5 / 120)
            else:
                u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 3 * x) + \
                    (self.w) / (2 * self.k * self.G * self.A) * (-x ** 2 + self.L * x)
                rot = (self.w / (24 * self.E * self.I)) * (4 * x ** 3 - 6 * self.L * x ** 2 + self.L ** 3)
        elif problem[1] == "fixed" and problem[2] == "free":
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 4 * self.L * x ** 3 + 6 * self.L ** 2 * x ** 2) + (
                        self.w / (2 * self.k * self.G * self.A)) * (-x ** 2 + 2 * self.L * x)
            rot = (self.w / (6 * self.E * self.I)) * (x ** 3 - 3 * self.L * x ** 2 + 3 * self.L ** 2 * x)
        elif problem[1] == "fixed" and problem[2] == "fixed":
            C1 = -(self.w * self.L * (self.k * self.G * self.A * self.L ** 2 + 12 * self.E * self.I)) / (
                        24 * self.E * self.I + 2 * self.k * self.G * self.A * self.L ** 2);
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 2 * x ** 2) + (
                        self.w / (2 * self.k * self.G * self.A)) * (-x ** 2 + self.L * x)
            rot = (1 / (6 * self.E * self.I)) * (
                        self.w * x ** 3 + 3 * C1 * x ** 2 - self.w * self.L ** 2 * x - 3 * C1 * self.L * x)
        elif problem[1] == "fixed" and problem[2] == "pinned":
            C1 = -(5 * self.w * self.k * self.G * self.A * self.L ** 3 + 12 * self.w * self.L * self.E * self.I) / (
                        8 * (self.k * self.G * self.A * self.L ** 2 + 3 * self.E * self.I))
            u = (1 / (self.E * self.I)) * (
                        self.w * x ** 4 / 24 + C1 * x ** 3 / 6 - (self.w * self.L ** 2 / 4) * x ** 2 - (
                            C1 * self.L / 2) * x ** 2) - (self.w / (2 * self.k * self.G * self.A)) * x ** 2 - (
                                C1 * x) / (self.k * self.G * self.A)
            rot = (1 / (6 * self.E * self.I)) * (
                        self.w * x ** 3 + 3 * C1 * x ** 2 - 3 * self.w * self.L ** 2 * x - 6 * C1 * self.L * x)
        # elif problem[1] == "linear_shape":
        #     u = -((self.w *self.L ** 4) / (2 * self.E * self.I)) * (self.L * (3 * self.L + 4 * x)/(2 * (self.L + x)**2) + np.log(self.L + x) - 1.5 - np.log(self.L) - \
        #     x / (8 * self.L)) - ((self.w *self.L ** 2)/(self.G * self.A * self.k )) * (2*self.L /(self.L + x) + np.log(self.L + x) - 2 - np.log(self.L))
        #     rot = -((self.w *self.L ** 4) / (2 * self.E * self.I)) * (x ** 2/(self.L + x) ** 3) + ((self.w *self.L ** 3) / (16 * self.E * self.I))

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
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 3 * x) + \
                (self.w) / (2 * self.k * self.G * self.A) * (-x ** 2 + self.L * x)
            rot = (self.w / (24 * self.E * self.I)) * (4 * x ** 3 - 6 * self.L * x ** 2 + self.L ** 3)
        elif problem[1] == "fixed" and problem[2] == "free":
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 4 * self.L * x ** 3 + 6 * self.L ** 2 * x ** 2) + (
                        self.w / (2 * self.k * self.G * self.A)) * (-x ** 2 + 2 * self.L * x)
            rot = (self.w / (6 * self.E * self.I)) * (x ** 3 - 3 * self.L * x ** 2 + 3 * self.L ** 2 * x)
        elif problem[1] == "fixed" and problem[2] == "fixed":
            C1 = -(self.w * self.L * (self.k * self.G * self.A * self.L ** 2 + 12 * self.E * self.I)) / (
                        24 * self.E * self.I + 2 * self.k * self.G * self.A * self.L ** 2);
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 2 * x ** 2) + (
                        self.w / (2 * self.k * self.G * self.A)) * (-x ** 2 + self.L * x)
            rot = (1 / (6 * self.E * self.I)) * (
                        self.w * x ** 3 + 3 * C1 * x ** 2 - self.w * self.L ** 2 * x - 3 * C1 * self.L * x)
        elif problem[1] == "fixed" and problem[2] == "pinned":
            C1 = -(5 * self.w * self.k * self.G * self.A * self.L ** 3 + 12 * self.w * self.L * self.E * self.I) / (
                        8 * (self.k * self.G * self.A * self.L ** 2 + 3 * self.E * self.I))
            u = (1 / (self.E * self.I)) * (
                        self.w * x ** 4 / 24 + C1 * x ** 3 / 6 - (self.w * self.L ** 2 / 4) * x ** 2 - (
                            C1 * self.L / 2) * x ** 2) - (self.w / (2 * self.k * self.G * self.A)) * x ** 2 - (
                                C1 * x) / (self.k * self.G * self.A)
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
        ax[1].set_xlabel('x [m]')
        ax[1].set_ylabel('rad []')
        ax[1].text(0.01, 0.01, "error rot: " + str(err_rot),
                   verticalalignment='bottom', horizontalalignment='left',
                   transform=ax[1].transAxes,
                   color='black', fontsize=8)
        ax[1].grid()
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.legend(loc='best')
        # plt.savefig('tk_varsec_0.001_32_300.pdf')

        plt.show()

# def timoshenko_beam():
#     # Beam initial data --------------------------
#     # Distributed load [N/m]
#     # w = -35000
#     w = -35
#
#     b = 0.2
#     h = 0.5
#
#     # Beam length [m]
#     L = 2
#
#     # Young Modulus [N/m²]
#     # E = 2 * 10 ** 11
#     E = 2
#
#     # Inertia Moment [m^4]
#     # I = 0.000038929334
#     I = b * h ** 3 / 12
#
#     # Poisson coefficient
#     nu = 0.3
#
#     # Area [m^2]
#     A = b * h
#
#     G = E / (2 * (1 + nu))
#
#     k = 5.0 / 6.0
#
#     # Number of training points
#     num_training_samples = 10000
#
#     # Number of test points for the predictions
#     num_test_samples = int(0.10 * num_training_samples)
#
#     zero = tf.constant(0.0)
#     one = tf.constant(1.0)
#
#     # x_test = np.linspace(0, L, int(num_test_samples))
#     # u_ref = (w / (24 * E * I)) * (x_test ** 4 - 2 * L * x_test ** 3 + L ** 3 * x_test) + \
#     #         (w) / (2 * k * G * A) * (-x_test ** 2 + L * x_test)
#
#     # Neural Network Setup.
#     dtype = 'float32'
#
#     x = sn.Variable("x", dtype=dtype)
#
#     u = sn.Functional('u', x, [40, 40, 40], 'tanh')
#     rot = sn.Functional('rot', x, [40, 40, 40], 'tanh')
#
#
#     du_dx = sn.diff(u, x)
#     drot_dx = sn.diff(rot, x)
#
#     d2u_dx2 = sn.diff(u, x, order=2)
#     d2rot_dx2 = sn.diff(rot, x, order=2)
#
#     # d4u_dx4 = sn.diff(u, x, order=4)
#
#     eqDiff1 = ((E * I) / (k * A * G)) * d2rot_dx2 + (du_dx - rot)
#     eqDiff2 = (d2u_dx2 - drot_dx) + (w / (k * A * G)) * one
#     # eqDiff1 = d4u_dx4 + w / (E * I)
#     # eqDiff2 = (E * I) * d3rot_dx3 - w
#
#     BC_left_1 = (x == 0.) * (u)
#     BC_left_2 = (x == 0.) * (drot_dx)
#
#     BC_right_1 = (x == L) * (u)
#     BC_right_2 = (x == L) * (drot_dx)
#
#
#     targets = [eqDiff1, eqDiff2,
#                BC_left_1, BC_left_2,
#                BC_right_1, BC_right_2]
#
#     dg = DataGeneratorX(X=[0.,L],
#                          num_sample=num_training_samples,
#                          targets=2*['domain'] + 2*['bc-left'] + 2*['bc-right'])
#
#     input_data, target_data = dg.get_data()
#
#
#     model = sn.SciModel([x], targets, optimizer='adam', loss_func="mse")
#
#     print("The data fitting process (training) has started")
#     start = time.time()
#     model.train(input_data, target_data , batch_size=32, epochs=200, verbose=0, learning_rate=1e-4)
#     end = time.time()
#     print("The data fitting process (training) has ended")
#     print("\nThe execution time was: ", (end - start), "seconds")
#
#     u_test = u.eval([x_test])
#     rot_test = rot.eval([x_test])
#
#     plt.plot(x_test, u_test, 'r', x_test, u_ref, 'b')  # Predicted solution
#     # plt.plot(x_test, u_test, 'r')  # Predicted solution
#     plt.xlabel('x [m]')
#     plt.ylabel('displacements [m]')
#     plt.grid()
#     plt.grid(color='black', linestyle='--', linewidth=0.5)
#     plt.legend(loc='best')
#     # plt.savefig('1.tk_pinned_pinned.pdf')
#     plt.show()
