# @author Felipe Pereira dos Santos
# @since 29 September, 2023
# @version 08 October, 2023

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog as fd
from pre_processor import PreProcessor
from processor import Processor
import os
import gc

class NumericalResults:
    """
            Class to generate the numerical results based on a grid search-like approach

    """

    def __init__(self, initial_features, m_parameters, neural_net, optimizers, epochs, bc_sizes, file_name, problem, mesh):
        """
            Constructor to generate numerical results.

            Attributes:
                initial_features: defines whether the callback function will be used or not during the training,
                and the defines the mesh (list) to make predictions
                m_parameters: parameters of the model, such as: beam length, Young modulus, cross-section area, etc
                neural_net: list of neural networks to be tested. Each one is in the form:
                    layers: list(int)
                    activation function: str
                    k_init: str (kernel initialization)
                optimizers: list of optimizers to be tested. Each one has its particular properties.
                    For instance, Adam's algorithms are defined as follows:
                    [Adam, learning_rate1, learning_rate2, learning_rate3, ...] = [str, float, float, float, ...]
                epochs: list with the number of epochs to be tested. Each one is an integer number related to the number
                of iterations during the training process
                bc_sizes: list with the bath_sizes to be tested. Each one is an integer
                file_name: str (name of the file with all the training information)
                problem: list of strings to inform the problem that will be solved
                mesh: list of integers to compose the number of points to be tested after training


        """

        self.initial_features = initial_features
        self.m_parameters = m_parameters
        self.neural_net = neural_net
        self.optimizers = optimizers
        self.epochs = epochs
        self.bc_sizes = bc_sizes
        self.file_name = file_name
        self.problem = problem
        self.mesh = mesh
    '''
    '''
    @classmethod
    def file_reading(clc):
        """
        Class method to read the input file and return its information to the menu

        Returns: the following lists:
            problems: problems to be solved
            model_parameters: parameters for the model analysis, such as Young modulus, beam length, etc
            neural_nets: neural networks and their configurations
            optimizers: optimizers for the training process
            epochs: epochs to train the model
            bc_sizes: batch sizes to divide the training points
            file_name (not a list): name of the read input file for later use in the output file's name

        """

        check_line = ''
        neural_nets = []
        optimizers = []
        problems = []
        model_parameters = []

        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        # Open the file dialog to select a file and capture the file path
        file_path = fd.askopenfilename(initialdir='NumericalResults')
        with open(file_path, 'r') as f:
            # Getting the file name to use it later on the output files
            file_name, _ = os.path.splitext(os.path.basename(file_path))
            for line in f:
                if '%BEGIN_INITIAL_FEATURES' in line:
                    print('\nReading the initial features-----------------BEGIN')
                    f.readline()  # Reading the header of this section
                    check_line = f.readline()  # Reading the first line that matters
                    while '%END_INITIAL_FEATURES' not in check_line:
                        initial_features = eval(check_line)
                        check_line = f.readline()
                    print('Reading the initial features-----------------END')
                if '%BEGIN_PROBLEMS' in line:
                    print('\nReading the problems-----------------BEGIN')
                    check_line = f.readline()  # Reading the first line that matters
                    while '%END_PROBLEMS' not in check_line:
                        problems.append(eval(check_line))
                        check_line = f.readline()
                    print('Reading the problemns-----------------END')
                if '%BEGIN_MODEL_PARAMETERS' in line:
                    print('\nReading the model parameters-----------------BEGIN')
                    f.readline()  # Reading the header of this section
                    check_line = f.readline()  # Reading the first line that matters
                    while '%END_MODEL_PARAMETERS' not in check_line:
                        model_parameters .append(eval(check_line))
                        check_line = f.readline()
                    print('Reading the model parameters-----------------END')
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
        return initial_features, problems, model_parameters, neural_nets, optimizers, epochs, bc_sizes, file_name

    def training(self):
        """
        Method to train the model in a grid search approach based on the input file,
        For a particular problem and neural network, the number of epochs, optimizers,
        and batch sizes are varied via for loops.

        After training, the model is tested against a reference solution (analytic, numeric, etc)
        for the number of points defined in the mesh (list). A L2 error is computed for each mesh,
        and a CSV file is generated to store this information. Moreover, the model results
        (displacements, rotations, etc) are stored in CSV files correspondents to each mesh.

        Finally, a CSV file is also created to store the information of the loss function
        values obtained during the training and the correspondent epochs.

        """

        # The problem is defined, as well as the weights and biases, activation function and neural network
        pmodel = PreProcessor.initialize_model(self.problem, self.neural_net, self.m_parameters)

        # Loops for variation of the parameters defined in the input file
        for optimizer in self.optimizers:
            if str(optimizer[0]) == 'Adam':
                learning_rate = optimizer[1:]
                for lr in learning_rate:
                    for bs in self.bc_sizes:
                        for epoch in self.epochs:
                            # String to characterize the output file names
                            net_par = str(lr) + '_' + str(bs) + '_' + str(epoch)

                            processor = Processor(self.initial_features)
                            model, history = processor.training_model(pmodel, self.problem, lr, bs, epoch, self.file_name)

    def save_csv_error(self, pmodel, problem, mesh, net_par):
        L = pmodel.L
        headers = ['x', 'Dy', 'Rz']
        e_u = []
        e_rot = []
        for num in mesh:
            x_test = np.linspace(0, L, num)

            [x, u_ref, rot_ref] = pmodel.reference_solution(x_test, problem)

            u_pred = pmodel.u.eval(x_test)
            rot_pred = pmodel.rot.eval(x_test)

            err_u = np.sqrt(np.linalg.norm(u_pred - u_ref)) / np.linalg.norm(u_ref)
            err_rot = np.sqrt(np.linalg.norm(rot_pred - rot_ref)) / np.linalg.norm(rot_ref)

            results = [x_test, u_pred, rot_pred]

            # Combine the arrays into a structured array
            combined_array = np.column_stack(results)
            np.savetxt("NumericalResults/" + self.file_name + '_' + net_par + '_' + str(num) + '.csv', combined_array, delimiter=',', header=','.join(headers),
                       comments='')

            e_u.append(err_u)
            e_rot.append(err_rot)

        print("Fora do for")
        error = [mesh, e_u, e_rot]

        headers = ['mesh', 'error_u', 'error_rot']

        fmt = ('%d', '%g', '%g')

        combined_array = np.column_stack(error)
        np.savetxt("NumericalResults/" + self.file_name + '_' + net_par + '_error.csv', combined_array, delimiter=',', header=','.join(headers), comments='',
                       fmt=fmt)

    def save_csv_loss(self, h, num_epochs, net_par):
        epochs = range(1, (num_epochs + 1))
        loss = h.history['loss']
        headers = ['epochs', 'loss']
        fmt = ('%d', '%g')

        combined_array = np.column_stack([epochs, loss])
        np.savetxt("NumericalResults/" + self.file_name + '_' + net_par + '_loss.csv', combined_array, delimiter=',', header=','.join(headers), comments='',
                       fmt=fmt)

    def plotting_loss_function(self, h, num_epochs):
        epochs = range(1, (num_epochs + 1))
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, h.history['loss'], label='loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('NumericalResults/' + self.file_name + '_' + str(num_epochs) + '_loss' + '.pdf')
        plt.close()

    def plotting_time_cost(self, time_c, epochs):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, time_c, label='time')
        plt.xlabel('epochs')
        plt.ylabel('time [s]')
        plt.legend()
        plt.savefig('NumericalResults/' + self.file_name + '_timeXepochs' + '.pdf')
        plt.close()

    def save_csv_error(self, error, epochs):
        headers = ['epochs', 'error']
        fmt = ('%d', '%g')

        combined_array = np.column_stack([epochs, error])
        np.savetxt("NumericalResults/" + self.file_name + '_error.csv', combined_array, delimiter=',', header=','.join(headers), comments='',
                       fmt=fmt)

    def plotting_error(self, error, epochs):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, error, label='error')
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.legend()
        plt.savefig('NumericalResults/' + self.file_name + '_error' + '.pdf')
        plt.close()


