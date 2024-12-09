# @author Felipe Pereira dos Santos
# @since 12 june, 2023
# @version 12 june, 2023

from input_information import InputInformation

class PreProcessor:
    """
         Class that represents a preprocessor for this program. The physics-informed neural network is build,
         and this class return it.
         All information of the target problem is filled in the input data method, as well as the initial settings regarding the machine
         learning approach.
    """

    @classmethod
    def initialize_model(clc, problem, net, m_parameters):
        """
            This method is responsible to trigger the class tasks

            Attributes:
                problem (list): name of the problem to be solved. It also can contain its
                boundary conditions

            Returns:
                pmodel: the physics-informed neural network for the target problem

        """

        pmodel = InputInformation(problem, net, m_parameters).input_data()

        return pmodel