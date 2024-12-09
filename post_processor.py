# @author Felipe Pereira dos Santos
# @since 15 june, 2023
# @version 06 October, 2023

import matplotlib.pyplot as plt
import numpy as np

class PostProcessor:
    """
        Class that represents a post-processor for this program. The main tasks of this class are
        plotting and printing results and comparisons.

        DISCLAIMER: some of the following method may not being in use, or may not being working.
    """

    @classmethod
    def save_csv_error_mesh(cls, pmodel, problem, mesh, net_par, file_name):
        '''
        Method to make predictions based on the provided mesh, compute the error, and
        generate CSV files with the results of displacements/rotations and the
        error computed for both in a L2-norm form
        '''

        L = pmodel.L
        headers = ['x', 'Dy', 'Rz']
        e_u = []
        e_rot = []
        error = []

        for i, num in enumerate(mesh):
            x_test = np.linspace(0, L, num)

            if problem[3] == 'ParabolicShape':
                x, u_ref, rot_ref = pmodel.mesh_ref[i]

            else:
                [x, u_ref, rot_ref] = pmodel.reference_solution(x_test, problem)

            u_pred = pmodel.u.eval(x_test)
            rot_pred = pmodel.rot.eval(x_test)

            def error_norm(u_ref, u_pred, rot_ref, rot_pred):
                # take square of differences and sum them
                num_u = np.sum(np.power((u_pred - u_ref), 2))
                num_rot = np.sum(np.power((rot_pred - rot_ref), 2))

                # Sum the square values
                dem_u = np.sum(np.power(u_ref, 2))
                dem_rot = np.sum(np.power(rot_ref, 2))
                error = np.sqrt((num_u + num_rot) / (dem_u + dem_rot))
                return error

            err = error_norm(u_ref, u_pred, rot_ref, rot_pred)

            results = [x_test, u_pred, rot_pred]

            # Combine the arrays into a structured array
            combined_array = np.column_stack(results)
            # np.savetxt("NumericalResults/" + file_name + '_' + net_par + '_' + str(num) + '.csv', combined_array, delimiter=',', header=','.join(headers),
            #            comments='')
            np.savetxt("NumericalResults/" + file_name + '.csv', combined_array, delimiter=',', header=','.join(headers),
                       comments='')

            error.append(err)

        err_res = [mesh, error]

        headers = ['mesh', 'error']

        fmt = ('%d', '%g')

        combined_array = np.column_stack(err_res)
        np.savetxt("NumericalResults/" + file_name + '_' + net_par + '_error_mesh.csv', combined_array, delimiter=',', header=','.join(headers), comments='',
                       fmt=fmt)


    @classmethod
    def save_csv_loss(cls, h, num_epochs, net_par, file_name):
        epochs = range(1, (num_epochs + 1))
        loss = h.history['loss']
        headers = ['epochs', 'loss']
        fmt = ('%d', '%g')

        combined_array = np.column_stack([epochs, loss])
        np.savetxt("NumericalResults/" + file_name + '_loss.csv', combined_array, delimiter=',', header=','.join(headers), comments='',
                       fmt=fmt)

    @classmethod
    def plotting_loss_function_pdf(cls, h, num_epochs, net_par, file_name):
        epochs = range(1, (num_epochs + 1))
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, h.history['loss'], label='loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('NumericalResults/' + file_name + '_' + net_par + '_' + str(num_epochs) + '_loss' + '.pdf')
        plt.close()

    @classmethod
    def plotting_time_cost_pdf(cls, time_c, epochs, net_par, file_name):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, time_c, label='time')
        plt.xlabel('epochs')
        plt.ylabel('time [s]')
        plt.legend()
        plt.savefig('NumericalResults/' + file_name + '_' + net_par + '_timeXepochs' + '.pdf')
        plt.close()

    @classmethod
    def save_csv_error(cls, error, epochs, net_par, file_name):
        headers = ['epochs', 'error']
        fmt = ('%d', '%g')

        combined_array = np.column_stack([epochs, error])
        np.savetxt("NumericalResults/" + file_name + '.csv', combined_array, delimiter=',', header=','.join(headers), comments='',
                       fmt=fmt)

    @classmethod
    def plotting_error_pdf(cls, error, epochs, net_par, file_name):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, error, label='error')
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.legend()
        plt.savefig('NumericalResults/' + file_name + '_' + net_par + '_error' + '.pdf')
        plt.close()


    @classmethod
    def plotting(clc, pmodel, results):
        """
            Method to plot the results of the numerical analysis, such as: the comparison of the preditions with
            a reference solution, when its avalaible; loss results or mean squared error or error, and so on.
        """

        pmodel.plotting(*results)

    def toString(self):
        """
            Method to print some information, such as: problem settings, network framework, global error
            between the predictions and reference solution, etc
        """
        rep = 30  # Number of symbols [-]  to be printed
        sModel = ''.join(['\n\n\n' + '-' * rep + 'BEGIN OF THE STRUCTURAL MODEL INFORMATION' + '-' * rep + '\n\n\n',
                          'Model: ', self.__model_type, '\nNumber of degrees of freedom per node: ',
                          str(self.get_number_dof()),
                          '\nNumber of nodes: ', str(len(self.__model_nodes)), '\nNumber of cross sections: ',
                          str(len(self.__model_cross_sections)),
                          '\nNumber of elements: ', str(len(self.__model_elements)), '\nNumber of materials: ',
                          str(len(self.__model_materials)),
                          '\n ', '-' * rep, 'MODEL NODES', '-' * rep, '\n',
                          '\n'.join(nodes.toString() for nodes in self.__model_nodes), '\n',
                          '\n ', '-' * rep, 'MODEL MATERIALS', '-' * rep, '\n',
                          '\n'.join(mat.toString() for mat in self.__model_materials), '\n',
                          '\n ', '-' * rep, 'MODEL CROSS-SECTIONS', '-' * rep, '\n',
                          '\n'.join(cros_sec.toString() for cros_sec in self.__model_cross_sections), '\n',
                          '\n ' + '-' * rep, 'MODEL DEGENERATIONS', '-' * rep + '\n',
                          '\n'.join(deg.toString() for deg in self.__model_degenerations), '\n',
                          '\n ', '-' * rep, 'MODEL ELEMENTS', '-' * rep, '\n',
                          '\n'.join(elem.toString() for elem in self.__model_elements), '\n'])
