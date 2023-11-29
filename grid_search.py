# @author Felipe Pereira dos Santos
# @since 12 june, 2023
# @version 13 june, 2023

import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
import sciann as sn

class GridSearch:
    """
            Class for multiparameter testing

            Based on the problem''s initial and boundary conditions, the tasks of this class are:

                1. Simulate the combination of the neural network hyperparameters
                2. Generate the results in terms of plots and errors
       """

    def __init__(self, neural_net, optimizer, epochs_num, bc_size, pmodel, file_name):
        """
            Constructor of the Euler-Bernoulli class.

            Attributes:
                neural_net: list of neural networks to be tested. Each one is in the form:
                    layers: list(int)
                    activation: str
                    k_init: str
                optimizer: list of optimizers to be tested. Each one has its particular properties.
                    For instance, Adam's algorithms are defined as follows:
                    [Adam, learning_rate1, learning_rate2, learning_rate3, ...] = [str, float, float, float, ...]
                epochs_num: list with the number of epochs to be tested. Each one is an integer
                bc_size: list with the bath_sizes to be tested. Each one is an integer
                model_info: information regarding the model t be solved
                file_name: str (name of the file with all the training information)
            Args:
                gradient: (GradientLayer_net_u): used to compute the derivatives needed for the target problem

        """

        self.neural_net = neural_net
        self.optimizer = optimizer
        self.epochs_num = epochs_num
        self.bc_size = bc_size
        self.pmodel = pmodel
        self.file_name = file_name
    @classmethod
    def grid_search(clc):
        check_line = ''
        neural_nets = []
        optimizers = []
        with open(fd.askopenfilename(initialdir='GridSearch'), 'r') as f:
            for line in f:
                if '%BEGIN_NEURAL_NETS' in line:
                    print('\nReading the neural networks-----------------BEGIN')
                    f.readline()  # Reading the header of this section
                    check_line = f.readline()  # Reading the first line that matters
                    while '%END_NEURAL_NETS' not in check_line:
                        # neural_nets.append(eval(check_line.split(' ')))
                        neural_nets.append(eval(check_line))
                        check_line = f.readline()
                    print('Reading the neural networks-----------------END')
                if '%BEGIN_OPTIMIZERS' in line:
                    print('\nReading the optimizers-----------------BEGIN')
                    check_line = f.readline()  # Reading the first line that matters
                    while '%END_OPTIMIZERS' not in check_line:
                        optimizers.append(eval(check_line))
                        check_line = f.readline()
                    print('Reading the optimizers-----------------END')
                if '%BEGIN_BATCHES' in line:
                    print('\nReading the batch sizes-----------------BEGIN')
                    check_line = f.readline()  # Reading the first line that matters
                    while '%END_BATCHES' not in check_line:
                        bc_sizes = eval(check_line)
                        check_line = f.readline()
                    print('Reading the batch sizes-----------------END')
                if '%BEGIN_EPOCHS' in line:
                    print('\nReading the epochs-----------------BEGIN')
                    check_line = f.readline()  # Reading the first line that matters
                    while '%END_EPOCHS' not in check_line:
                        epochs = eval(check_line)
                        check_line = f.readline()
                    print('Reading the epochs-----------------END')
                check_line = f.readline()
        return neural_nets, optimizers, epochs, bc_sizes

    def training(self):
        count = 1
        mse_values = []
        mse_rot_values = []
        with open("ResultsSciann/" + str(self.file_name) + ".txt", "w") as f:
            f.write('\nlayers: ' + str(self.neural_net[0]) +
                    '\nactivation: ' + str(self.neural_net[1]) + '\nk_init: ' + str(self.neural_net[2]))
            f.write('\n\n' + self.pmodel.model_info() + '\n')
            for optimizer in self.optimizer:
                if str(optimizer[0]) == 'Adam':
                    learning_rate = optimizer[1:]
                    for lr in learning_rate:
                        model = sn.SciModel(self.pmodel.variables, self.pmodel.targets, optimizer='adam', loss_func="mse")
                        for bs in self.bc_size:
                            for epoch in self.epochs_num:
                                print("The data fitting process (training) has started for the model: ", count)
                                # self.PINN_model.fit(self.pmodel.x_train, self.pmodel.y_train, epochs=epoch, batch_size=bs, verbose=0)
                                model.train(self.pmodel.input_data, self.pmodel.target_data, batch_size=bs, epochs=epoch, verbose=0,
                                            learning_rate=lr)
                                # y_pred = self.pmodel.network.predict(self.pmodel.x_test)
                                y_pred = self.pmodel.u.eval([self.pmodel.x_test])
                                rot_pred = self.pmodel.rot.eval([self.pmodel.x_test])

                                mse = mean_squared_error(self.pmodel.ref_solu[0], y_pred)
                                mse_rot = mean_squared_error(self.pmodel.ref_solu[1], rot_pred)

                                mean_abs = np.mean(np.abs(self.pmodel.ref_solu[0] - y_pred))
                                mean_abs_rot = np.mean(np.abs(self.pmodel.ref_solu[1] - rot_pred))

                                print("The data fitting process (training) has ended for the model: ", count)
                                print('L_rate: %s, batch_size: %s, Epochs: %s, MSE: %s' % (lr, bs, epoch, mse))
                                res = 'Adam: L_rate: ' + str(lr) + ' | batch_size: ' + str(bs) + ' | Epochs: ' + str(epoch) + ' | MSE_u: ' + \
                                      str(mse) + ' | Mean(|yreal-ypred|_u): ' + str(mean_abs) + ' | MSE_rot: ' + \
                                      str(mse_rot) + ' | Mean(|yreal-ypred|_rot): ' + str(mean_abs_rot) + '\n'
                                f.write(res)
                                mse_values.append(mse)
                                mse_rot_values.append(mse_rot)
                                count = count + 1

                                x_test = self.pmodel.x_test
                                u_ref = self.pmodel.ref_solu[0]
                                rot_ref = self.pmodel.ref_solu[1]
                                u_test = y_pred
                                rot_test = rot_pred

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

                                plt.savefig('GridSearch/' + self.file_name + '_' + str(lr) + '_' + str(bs) + '_' + str(epoch) + '.pdf')
                                plt.close()

                                # plt.plot(self.pmodel.x_test, y_pred, 'r', self.pmodel.x_test, self.pmodel.ref_solu[1], 'b')  # Predicted solution
                                # plt.xlabel('x [m]')
                                # plt.ylabel('displacements [m]')
                                # plt.grid()
                                # plt.grid(color='black', linestyle='--', linewidth=0.5)
                                # plt.legend(loc='best')
                                # plt.savefig('GridSearch/' + self.file_name + '_' + str(lr) + '_' + str(bs) + '_' + str(epoch) + '.pdf')
                                # plt.close()
                # elif optimizer[0] == 'L_bgf':
                #     self.PINN_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                #                             loss=tf.keras.losses.mse,
                #                             metrics=tf.keras.metrics.mse,
                #                             )
                #     for bs in self.bc_size:
                #         for epoch in self.epochs_num:
                #             print("The data fitting process (training) has started for the model: ", count)
                #             self.PINN_model.fit(self.pmodel.x_train, self.pmodel.y_train, epochs=epoch, batch_size=bs,
                #                                 verbose=0)
                #             y_pred = self.pmodel.network.predict(self.pmodel.x_test)
                #             mse = mean_squared_error(self.pmodel.ref_solu[1], y_pred)
                #             mean_abs = np.mean(np.abs(self.pmodel.ref_solu[1] - y_pred))
                #             print("The data fitting process (training) has ended for the model: ", count)
                #             print('L_rate: %s, batch_size: %s, Epochs: %s, MSE: %s' % (lr, bs, epoch, mse))
                #             res = 'LBGST optimizer' + ' | batch_size: ' + str(bs) + ' | Epochs: ' + str(
                #                 epoch) + ' | MSE: ' + str(mse) + ' | Mean(|yreal-ypred|): ' + str(mean_abs) + '\n'
                #             f.write(res)
                #             mse_values.append(mse)
                #             j = j + 1
                #             count = count + 1

            min_mse = np.amin(np.array(mse_values))
            min_mse_rot = np.amin(np.array(mse_rot_values))
            f.write('\nMinimum mse: ' + str(min_mse) + ' | Index: ' + str(mse_values.index(min_mse) + 1) +
                    '\nMinimum mse_rot: ' + str(min_mse_rot) + ' | Index: ' + str(mse_rot_values.index(min_mse_rot) + 1))
