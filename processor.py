# @author Felipe Pereira dos Santos
# @since 06 October, 2023
# @version 06 October, 2023
import sys

from input_information import InputInformation
import sciann as sn
from callbacks import EvaluateAtEpoch
import time
from post_processor import PostProcessor
import os

class Processor:
    """
         Class that represents a processor for this program.

         The tasks of this class are:

             1. Train the model
    """

    def __init__(self, cback_config):

        self.cback_config = cback_config

    def training_model(self, pmodel, problem, lr, bs, epoch, count, file_name):

        if self.cback_config[0] == "on":
            start = (self.cback_config[1])[0]  # Start epoch
            end = (self.cback_config[1])[1]   # Last epoch
            step = (self.cback_config[1])[2]  # Interval between epochs
            evaluate_epochs = list(range(start, (end + step), step))  # Epochs for analysis in the callback feature

            # Training parameters to compose the name of the output files
            net_par = str(lr) + '_' + str(bs) + '_' + str(epoch)

            # Defining the model
            model = sn.SciModel(pmodel.variables, pmodel.targets, optimizer='adam',
                                loss_func="mse")
            # model = sn.SciModel(pmodel.variables, pmodel.targets, optimizer='adam',
            #                     loss_func="mse", load_weights_from='weights_test.hdf5')
            # model.save_weights("weights_test.hdf5")
            # sys.exit()
            print("The model number {} was defined. ".format(count))

            start_time = time.time()  # Record the start time at the beginning of the interval

            # Create an instance of the custom callback.
            evaluate_callback = EvaluateAtEpoch(evaluate_epochs, pmodel, start_time, file_name)
            print("\nStarting the training!")
            print("The data fitting process (training) has started for the model: ", count)
            history = model.train(pmodel.input_data, pmodel.target_data, batch_size=bs, epochs=epoch, verbose=0,
                                  learning_rate=lr, callbacks=[evaluate_callback], stop_loss_value=1e-15)
            end_time = time.time()
            print("Time of training in seconds: ", (end_time - start_time))
            # model.save_weights("Tk_statics_pp_q_weights.hdf5")

            # For discover of parameters
            # if (pmodel.problem != "EB_stability_discovery") and (pmodel.problem != "EB_stability_discovery_timobook"):
            #     # Mesh to make predictions and save the results as CSV files
            #     mesh = self.cback_config[2]
            #     PostProcessor.save_csv_error_mesh(pmodel, problem, mesh, net_par, file_name)  # Error for each mesh

            # PostProcessor.save_csv_loss(history, epoch, net_par, file_name)  # Loss function for each epoch
            PostProcessor.save_csv_error(evaluate_callback.error, evaluate_epochs, net_par, file_name) # Error for each epoch
            # PostProcessor.plotting_loss_function_pdf(history, epoch, net_par, file_name)
            # PostProcessor.plotting_time_cost_pdf(evaluate_callback.count_time, evaluate_epochs, net_par, file_name)
            # PostProcessor.plotting_error_pdf(evaluate_callback.error, evaluate_epochs, net_par, file_name)
            # x_test = pmodel.x_test
            # u_pred = pmodel.u.eval(x_test)
            # rot_pred = pmodel.rot.eval(x_test)
            # u_ref = pmodel.ref_solu[0]
            # rot_ref = pmodel.ref_solu[1]
            # results = [x_test, u_ref, u_pred, rot_ref, rot_pred]
            # PostProcessor.plotting(pmodel, results)

            return model, history

        else:
            model = sn.SciModel(pmodel.variables, pmodel.targets, optimizer='adam',
                                loss_func="mse")
            print("The model number {} was defined. ".format(count))

            print("The data fitting process (training) has started for the model: ", count)
            history = model.train(pmodel.input_data, pmodel.target_data, batch_size=bs, epochs=epoch, verbose=0,
                                  learning_rate=lr, stop_loss_value=1e-12)

            return model, history
