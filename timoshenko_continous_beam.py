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


class Timoshenko_Continuous:
    """
         Class that represents provide features for the Timoshenko bending continuous beam analysis.

         Based on the problem''s initial and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on
    """

    def __init__(self, beam_seg):
        """
            Constructor of the Timoshenko continuous beam class.

            Attributes:
                beam_seg: each segment of the beam

        """

        self.beam_seg = beam_seg

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


    def beam_conditions(self, problem):
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

        # Reference solution for the predictions ======================================
        x1 = self.beam_seg[0].x
        u1 = self.beam_seg[0].u
        rot1 = self.beam_seg[0].rot
        du1_dx = self.beam_seg[0].du_dx
        drot1_dx = self.beam_seg[0].drot_dx
        d2rot1_dx2 = self.beam_seg[0].d2rot_dx2

        x2 = self.beam_seg[0].x
        u2 = self.beam_seg[1].u
        rot2 = self.beam_seg[1].rot
        du2_dx = self.beam_seg[1].du_dx
        drot2_dx = self.beam_seg[1].drot_dx
        d2rot2_dx2 = self.beam_seg[1].d2rot_dx2

        x3 = self.beam_seg[0].x
        u3 = self.beam_seg[2].u
        rot3 = self.beam_seg[2].rot
        du3_dx = self.beam_seg[2].du_dx
        drot3_dx = self.beam_seg[2].drot_dx
        d2rot3_dx2 = self.beam_seg[2].d2rot_dx2


        # Boundary conditions
        support1_1 = (x1 == 0.) * (u1)
        support1_2 = (x1 == 0.) * (drot1_dx)

        support2_1 = (x1 == 1.0) * (u1)
        support2_2 = (x2 == 0.0) * (u2)

        support3_1 = (x2 == 1.0) * (u2)
        support3_2 = (x3 == 0.0) * (u3)

        support4_1 = (x3 == 1.0) * (u3)
        support4_2 = (x3 == 1.0) * (drot3_dx)

        u12_1 = (x1 == 1.0) * (u1)
        u12_2 = (x2 == 0.0) * (u2)
        u12 = u12_1 - u12_2
        support2_3 = u12

        u23_2 = (x2 == 1.0) * (u2)
        u23_3 = (x3 == 0.0) * (u3)
        u23 = u23_2 - u23_3

        rot12_1 = (x1 == 1.0) * (rot1)
        rot12_2 = (x2 == 0.0) * (rot2)
        rot12 = rot12_1 - rot12_2

        rot23_2 = (x2 == 1.0) * (rot2)
        rot23_3 = (x3 == 0.0) * (rot3)
        rot23 = rot23_2 - rot23_3

        drot_dx12_1 = (x1 == 1.0) * (drot1_dx)
        drot_dx12_2 = (x2 == 0.0) * (drot2_dx)
        drot_dx12 = drot_dx12_1 - drot_dx12_2

        drot_dx23_2 = (x1 == 1.0) * (drot2_dx)
        drot_dx23_3 = (x2 == 0.0) * (drot3_dx)
        drot_dx23 = drot_dx23_2 - drot_dx23_3

        # Loss function
        self.targets = [self.beam_seg[0].eqDiff1, self.beam_seg[0].eqDiff2, self.beam_seg[1].eqDiff1, self.beam_seg[1].eqDiff2,
                        self.beam_seg[2].eqDiff1, self.beam_seg[2].eqDiff2,
                        u12, u23, rot12, rot23, drot_dx12, drot_dx23,
                   support1_1, support1_2,
                   support2_1, support2_2,
                   support3_1, support3_2,
                   support4_1, support4_2]

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
        u_ref = (self.w/(24 * self.E * self.I))*(x ** 4 - 4 * self.L * x ** 3 + 6 * self.L ** 2 * x ** 2) + (self.w/(2 * self.k * self.G * self.A))*(-x ** 2 + 2 * self.L * x)
        rot_ref = (self.w/(6 * self.E * self.I))*(x ** 3 - 3 * self.L* x ** 2 + 3 * self.L ** 2 * x)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.rot)

        BC_right_1 = (self.x == self.L) * (self.drot_dx)
        BC_right_2 = (self.x == self.L) * (self.du_dx - self.rot)
        # BC_right_2 = (self.x == self.L) * (self.d2rot_dx2)

        # Loss function
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1, BC_left_2, BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

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

    def reference_solution(self, x, problem):
        """
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
        plt.savefig('tk_ffr_0.001_32_300.pdf')

        plt.show()