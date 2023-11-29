# @author Felipe Pereira dos Santos
# @since 12 june, 2023
# @version 12 june, 2023

from input_information import InputInformation

class PreProcessor:
    """
         Class that represents a preprocessor for this program. All information of the
         target problem is filled here, as well as the initial settings regarding the machine
         learning approach

         The tasks of this class are:

             1. Initialize the input data class where the problem's initial settings are defined,
             such as the material data, training parameters, and neural network framework
             2. Initialize both neural networks: the usual one and also the physics-informed one
    """

    @classmethod
    def initialize_model(clc, problem, net, m_parameters):
        """
            This method is responsible to trigger the class tasks

            Attributes:
                problem (list): name of the problem to be solved. It also can contain its
                boundary conditions

            Returns:
                network: usual neural network
                PINN_model: physics-informed neural network constructed for the target problem
                x_train: input training parameters
                y_train: output training parameters (target, labels)
                ref_solu: reference solution to compare with the predictions
                x_nn: test parameters to perform the predictions

        """

        pmodel = InputInformation(problem, net, m_parameters).input_data()

        return pmodel