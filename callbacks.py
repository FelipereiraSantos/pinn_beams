# @author Felipe Pereira dos Santos
# @since 01 October, 2023
# @version 01 October, 2023

# import sciann as sn
# import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import time
import sciann as sn

# Define a custom callback to evaluate the model at specific epochs.
class EvaluateAtEpoch(Callback):
    def __init__(self, evaluate_epochs, pmodel, start_time, file_name):
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
            if (self.pmodel.problem == 'EB_stability_discovery') or (self.pmodel.problem == 'EB_stability_discovery_timobook'):
                P_exact = self.pmodel.ref_solu
                P_pred = self.pmodel.P.value
                error = np.abs(P_exact - P_pred) / P_exact
                # print("P_pred: ", P_pred)
                # print("P_exact: ", P_exact)
                # print("error: ", error)
                self.error.append(error)

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
                # print("error: ", error)

                # self.plotting(x_test, u_ref, u_pred, rot_ref, rot_pred, epoch)

            elif(self.pmodel.problem == 'Nonlinear_TimoEx'):
                x_test = self.pmodel.x_test
                rot_exact = self.pmodel.ref_solu
                rot_pred = self.pmodel.rot.eval([x_test])
                error = np.abs(rot_exact - rot_pred[-1]) / rot_exact
                print("rot_pred: ", rot_pred[-1])
                print("rot_exact: ", rot_exact)
                print("error: ", error)
                self.error.append(error)

                # plt.plot(x_test, rot_pred, 'r')
                # plt.grid(color='black', linestyle='--', linewidth=0.5)
                # plt.legend(loc='best')
                # plt.savefig(self.file_name + '_' + str(epoch) + '.pdf')
                # plt.clf()  # Clears the current axis

            else:

                # Evaluate your model here
                # Making predictions based on the training
                x_test = self.pmodel.x_test
                u_pred = self.pmodel.u.eval([x_test])
                rot_pred = self.pmodel.rot.eval([x_test])

                # Recovering the reference solution for further comparisons
                u_ref = self.pmodel.ref_solu[0]
                rot_ref = self.pmodel.ref_solu[1]

                # if x_test.size % 2 > 0:
                #     u_pred = np.concatenate((u_pred, u_pred[:-1][::-1]))
                #     rot_pred = np.concatenate((rot_pred, rot_pred[:-1][::-1]))
                # else:
                #     u_pred = np.concatenate((u_pred, u_pred[::-1]))
                #     rot_pred = np.concatenate((rot_pred, rot_pred[::-1]))

                error = self.error_norm(u_ref, u_pred, rot_ref, rot_pred)
                self.error.append(error)

                # self.plotting(x_test, u_ref, u_pred, rot_ref, rot_pred, epoch)

    def error_norm(self, u_ref, u_pred, rot_ref, rot_pred):
        # take square of differences and sum them
        num_u = np.sum(np.power((u_pred - u_ref), 2))
        num_rot = np.sum(np.power((rot_pred - rot_ref), 2))

        # Sum the square values
        dem_u = np.sum(np.power(u_ref, 2))
        dem_rot = np.sum(np.power(rot_ref, 2))
        error = np.sqrt((num_u + num_rot)/(dem_u + dem_rot))
        return error

    def plotting(self, x_test, u_ref, u_pred, rot_ref, rot_pred, num_epochs):
        err_u = np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref)
        err_rot = np.linalg.norm(rot_pred - rot_ref) / np.linalg.norm(rot_ref)

        err_u = "{:.3e}".format(err_u)
        err_rot = "{:.3e}".format(err_rot)

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        # fig.subplots_adjust(bottom=0.15, left=0.2)
        # str(round(err_u, 3))
        ax[0].plot(x_test, u_pred, 'r', x_test, u_ref, 'b')
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

        ax[1].plot(x_test, rot_pred, 'r', x_test, rot_ref, 'b')
        ax[1].set_xlabel('x [m]')
        ax[1].set_ylabel('rad []')
        ax[1].text(0.01, 0.01, "error rot: " + str(err_rot),
                   verticalalignment='bottom', horizontalalignment='left',
                   transform=ax[1].transAxes,
                   color='black', fontsize=8)
        ax[1].grid()
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.legend(loc='best')
        plt.savefig(self.file_name + '0.001_32' + '_' + str(num_epochs) + '.pdf')
        plt.clf() # Clears the current axis

        # plt.show()
