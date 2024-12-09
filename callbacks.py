# @author Felipe Pereira dos Santos
# @since 01 October, 2023
# @version 01 October, 2023

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import time
import sciann as sn

# Define a custom callback to evaluate the model at specific epochs.
class EvaluateAtEpoch(Callback):

    """
         Class responsible to evaluate and provide partial error results during the training.

    """
    def __init__(self, evaluate_epochs, pmodel, start_time, file_name):
        """
            Constructor of the callback class

            Attributes:
                evaluate_epochs: list of epochs that this function is called to provide the partial results
                pmodel: refers to the physics-informed model that has been trained
                file_name: used as reference to write a csv file using a name based on the input file's name
                start_time: define the time counting before training
                count_time: list of time counting with respect to each epoch of the list 'evaluate_epochs'
                error: list of the computed errors with respect to each epoch of the list 'evaluate_epochs'

        """
        super().__init__()
        self.evaluate_epochs = evaluate_epochs
        self.pmodel = pmodel
        self.file_name = file_name
        self.start_time = start_time
        self.count_time = []
        self.error = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.evaluate_epochs:
            end_time = time.time()  # Record the end time at the end of the interval
            elapsed_time = end_time - self.start_time  # Calculate the elapsed time

            # Storing the time for every target epoch
            self.count_time.append(elapsed_time)

            # Refers to the buckling load discovery of the Euler-Bernoulli beams and the study case of a beams with a varying cross-section from the Timoshenko book:
            # Timoshenko, S. P., & Gere, J. M. (1963). Theory of Elastic Stability (2nd ed.). McGraw-Hill Book Company.
            if (self.pmodel.problem == 'EB_stability_discovery') or (self.pmodel.problem == 'EB_stability_discovery_timobook'):
                P_exact = self.pmodel.ref_solu
                P_pred = self.pmodel.P.value
                error = np.abs(P_exact - P_pred) / P_exact
                self.error.append(error)

            # Refers to the bending of the Euler-Bernoulli beam-columns (forward problem, stability)
            elif (self.pmodel.problem == 'EB_stability'):
                # Evaluate your model here
                # Making predictions based on the training
                x_test = self.pmodel.x_test
                u_pred = self.pmodel.u.eval([x_test])
                du_dx = sn.diff(self.pmodel.u, self.pmodel.x)
                rot_pred = du_dx.eval([x_test])

                # Recovering the reference solution for further comparisons
                u_ref = self.pmodel.ref_solu[0]
                rot_ref = self.pmodel.ref_solu[1]

                error = self.error_norm(u_ref, u_pred, rot_ref, rot_pred)
                self.error.append(error)

            # Refers to the bending problem of the non-linear beam from the Timoshenko book (forward problem)
            elif(self.pmodel.problem == 'Nonlinear_TimoEx'):
                x_test = self.pmodel.x_test
                rot_exact = self.pmodel.ref_solu
                rot_pred = self.pmodel.rot.eval([x_test])
                error = np.abs(rot_exact - rot_pred[-1]) / rot_exact
                print("rot_pred: ", rot_pred[-1])
                print("rot_exact: ", rot_exact)
                print("error: ", error)
                self.error.append(error)


            else:

                # Evaluate your model here
                # Making predictions based on the training
                x_test = self.pmodel.x_test
                u_pred = self.pmodel.u.eval([x_test])
                rot_pred = self.pmodel.rot.eval([x_test])

                # Recovering the reference solution for further comparisons
                u_ref = self.pmodel.ref_solu[0]
                rot_ref = self.pmodel.ref_solu[1]

                error = self.error_norm(u_ref, u_pred, rot_ref, rot_pred)
                self.error.append(error)


    def error_norm(self, u_ref, u_pred, rot_ref, rot_pred):
        # take square of differences and sum them
        num_u = np.sum(np.power((u_pred - u_ref), 2))
        num_rot = np.sum(np.power((rot_pred - rot_ref), 2))

        # Sum the square values
        dem_u = np.sum(np.power(u_ref, 2))
        dem_rot = np.sum(np.power(rot_ref, 2))
        error = np.sqrt((num_u + num_rot)/(dem_u + dem_rot))
        return error

    # def plotting(self, x_test, u_ref, u_pred, rot_ref, rot_pred, num_epochs):
    #     err_u = np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref)
    #     err_rot = np.linalg.norm(rot_pred - rot_ref) / np.linalg.norm(rot_ref)
    #
    #     err_u = "{:.3e}".format(err_u)
    #     err_rot = "{:.3e}".format(err_rot)
    #
    #     fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    #     # fig.subplots_adjust(bottom=0.15, left=0.2)
    #     # str(round(err_u, 3))
    #     ax[0].plot(x_test, u_pred, 'r', x_test, u_ref, 'b')
    #     ax[0].set_xlabel('x [m]')
    #     ax[0].set_ylabel('displacements [m]')
    #     ax[0].text(0.01, 0.01, "error disp: " + str(err_u),
    #                verticalalignment='bottom', horizontalalignment='left',
    #                transform=ax[0].transAxes,
    #                color='black', fontsize=8)
    #     # ax[0].text(0.15, 3, "error disp: " + str(err_u), fontsize=15)
    #     ax[0].grid()
    #     plt.grid(color='black', linestyle='--', linewidth=0.5)
    #     plt.legend(loc='best')
    #
    #     ax[1].plot(x_test, rot_pred, 'r', x_test, rot_ref, 'b')
    #     ax[1].set_xlabel('x [m]')
    #     ax[1].set_ylabel('rad []')
    #     ax[1].text(0.01, 0.01, "error rot: " + str(err_rot),
    #                verticalalignment='bottom', horizontalalignment='left',
    #                transform=ax[1].transAxes,
    #                color='black', fontsize=8)
    #     ax[1].grid()
    #     plt.grid(color='black', linestyle='--', linewidth=0.5)
    #     plt.legend(loc='best')
    #     plt.savefig(self.file_name + '0.001_32' + '_' + str(num_epochs) + '.pdf')
    #     plt.clf() # Clears the current axis
    #
    #     # plt.show()
