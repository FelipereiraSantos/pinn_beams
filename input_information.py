# @author Felipe Pereira dos Santos
# @since 12 june, 2023
# @version 12 june, 2023
import sys

import numpy as np
from timoshenko_beam import Timoshenko
from eb_stability import EB_Stability
from euler_bernoulli_beam import EulerBernoulli
from eb_stability_discovery import EB_Stability_Discovery
from eb_stability_discovery_timobook import EB_Stability_Discovery_TimoBook
from nonlinear_timoex import Nonlinear_TimoEx

class InputInformation:
    """
         Class that represents the input information of the model

         The tasks of this class is to build the physics-informed neural network (pmodel) based on the
         information data provided by the input file.

    """

    def __init__(self, problem, network_info, model_parameters):
        """
            Constructor of the InputInformation class.

            Args:
                problem (list): it defines the target problem, such as bending Euler-Bernoulli or Timoshenko beam,
                non-linear beam, etc. It is a list containing strings with the problem name, and
                other strings informing the boundary conditions: pinned-pinned, fixed-free and so on.
                network_info: list of settings of a neural network used to approximate the target
                problem solution [size, activation function, initialiser]
                model_parameters: load, geometry and material for the beam problem
        """
        self.problem = problem
        self.network = network_info
        self.model_parameters = model_parameters

    def input_data(self):
        """
            Method to initialize the data input based on the problem type

            Returns:
                neural_net_info (list): neural network settings
                num_test_samples (int): size of the test inputs for the predictions
                x_train (numpy array): input training parameters
                y_train (numpy array): output training parameters (target, labels)
                problem_variables (list): material (young modulus, poisson coefficient, etc)
                and geometric properties (area, inertia moment, etc) depending on the problem
                ref_solu (numpy array): reference solution to compare with the predictions
                x_nn (numpy array): set of test parameters to perform the predictions

        """

        if self.problem[0] == "EB_bending":
            return self.EB_bending_data(*self.model_parameters)
        elif self.problem[0] == "Tk_bending":
            return self.Tk_bending_data(*self.model_parameters)
        elif self.problem[0] == "EB_stability":
            return self.EB_stability_data(*self.model_parameters)
        # elif self.problem[0] == "EB_Stability_secvar":
        #     return self.EB_stability_secvarying_data(*self.model_parameters)
        elif self.problem[0] == "EB_stability_discovery":
            return self.EB_stability_discovery_data(*self.model_parameters)
        elif self.problem[0] == "EB_stability_discovery_timobook":
            return self.EB_stability_discovery_data_timobook(*self.model_parameters)
        elif self.problem[0] == "Nonlinear_TimoEx":
            return self.Nonlinear_TimoEx_data(*self.model_parameters)


    def EB_bending_data(self, w, b, h, L, E, nu,  num_training_samples):
        """
             Method that represents the Euler-Bernoulli beam problem and its settings

             w: Distributed load [N/m]
             b: horizontal dimension of a rectangular cross-section [m]
             h: vertical dimension of a rectangular cross-section [m]
             L: beam length [m]
             E: Young Modulus [N/m2]
             nu: Poisson coefficient []
        """

        # Inertia Moment [m^4]
        I = b * h ** 3 / 12

        # Area [m^2]
        A = b * h

        # Defining the list of the problem parameters (material and geometry)
        problem_parameters = [w, L, E, I, nu, A]

        # Number of test points for the predictions
        num_test_samples = int(0.10 * num_training_samples)

        eb = EulerBernoulli(self.network, *problem_parameters, num_training_samples, num_test_samples)

        if self.problem[1] == "pinned" and self.problem[2] == "pinned":
            eb.pinned_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "free":
            eb.fixed_free(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "pinned":
            eb.fixed_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "fixed":
            eb.fixed_fixed(self.problem)
        elif self.problem[1] == "varying_sec":
            eb.varying_sec(self.problem)

        return eb

    def Tk_bending_data(self, w, b, h, L, E, nu,  num_training_samples):
        """
             Method that represents the Timoshenko beam problem and its settings

             w: Distributed load [N/m]
             b: horizontal dimension (width) of a rectangular cross-section [m]
             h: vertical dimension (height) of a rectangular cross-section [m]
             L: beam length [m]
             E: Young Modulus [N/m2]
             nu: Poisson coefficient []
        """

        # Inertia Moment [m^4]
        I = b * h ** 3 / 12

        # Area [m^2]
        A = b * h

        # Defining the list of the problem parameters (material and geometry)
        problem_parameters = [w, L, E, I, nu, A]

        # Number of test points for the predictions
        num_test_samples = int(0.10 * num_training_samples)

        tk = Timoshenko(self.network, *problem_parameters, num_training_samples, num_test_samples)

        if self.problem[1] == "pinned" and self.problem[2] == "pinned":
            tk.pinned_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "free":
            tk.fixed_free(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "pinned":
            tk.fixed_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "fixed":
            tk.fixed_fixed(self.problem)
        # elif self.problem[1] == "varying_sec":
        #     tk.varying_sec(self.problem)

        return tk

    def Nonlinear_TimoEx_data(self, P, b, h, L, E, num_training_samples):
        """
             Method that represents the non-linear case study from the Timoshenko book of Mechanics of Materials.

             [1] Timoshenko, S. P., & Gere, J. M. (1982). Mecânica dos Sólidos. Volume 1.
             This is a translated version (Portuguese) of the Mechanics of Materials book.
        """
        # Beam initial data --------------------------
        # Inertia Moment [m^4]
        I = b * h ** 3 / 12

        # Defining the list of the problem parameters (material and geometry)
        problem_parameters = [P, L, E, I]

        # Number of test points for the predictions
        num_test_samples = int(0.10 * num_training_samples)

        nonlinear_disc = Nonlinear_TimoEx(self.network, *problem_parameters, num_training_samples, num_test_samples)
        print("\nInput data section is finished.")

        if self.problem[1] == "fixed" and self.problem[2] == "free":
            nonlinear_disc.fixed_free(self.problem)

        return nonlinear_disc

    def EB_stability_discovery_data(self, P, b, h, L, E, num_training_samples):
        """
             Method that represents the Euler-Bernoulli beam for stability problems and its settings for
             discovery of parameters. In this case, after training the buckling load will be learnt.
        """
        # Beam initial data --------------------------
        # Inertia Moment [m^4]
        I = b * h ** 3 / 12

        # Defining the list of the problem parameters (material and geometry)
        problem_parameters = [P, L, E, I]

        # Number of test points for the predictions
        num_test_samples = int(0.10 * num_training_samples)

        eb_s_disc = EB_Stability_Discovery(self.network, *problem_parameters, num_training_samples, num_test_samples)
        print("\nInput data section is finished.")

        if self.problem[1] == "pinned" and self.problem[2] == "pinned":
            eb_s_disc.pinned_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "free":
            eb_s_disc.fixed_free(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "pinned":
            eb_s_disc.fixed_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "fixed":
            eb_s_disc.fixed_fixed(self.problem)

        return eb_s_disc

    def EB_stability_data(self, P, b, h, L, E, a, num_training_samples):
        """
             Method that represents the Euler-Bernoulli beam for stability problems and its settings
             (This is a forward problem, that is to say, after training the deformed configurations of
             the problem will be hopefully adequate)
        """

        # Beam initial data --------------------------
        # Inertia Moment [m^4]
        I = b * h ** 3 / 12

        # Defining the list of the problem parameters (material and geometry)
        problem_parameters = [P, L, E, I, a]

        # Number of test points for the predictions
        num_test_samples = int(0.10 * num_training_samples)

        eb_s = EB_Stability(self.network, *problem_parameters, num_training_samples, num_test_samples)

        if self.problem[1] == "pinned" and self.problem[2] == "pinned":
            eb_s.pinned_pinned(self.problem)
        # elif self.problem[1] == "fixed" and self.problem[2] == "free" and a == 0:
        #     eb_s.fixed_free(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "free" and a != 0:
            eb_s.fixed_free_2specie(self.problem)
        # elif self.problem[1] == "fixed" and self.problem[2] == "pinned":
        #     eb_s.fixed_pinned(self.problem)
        # elif self.problem[1] == "fixed" and self.problem[2] == "fixed":
        #     eb_s.fixed_fixed(self.problem)

        return eb_s

    def EB_stability_discovery_data_timobook(self, P, b, h, L, E, inertia_ratio, n, num_training_samples):
        """
            Method that represents the Euler-Bernoulli beam for stability problems and its settings.
            This is a method for the specific problem extracted in [1] page 126. . See also the paper [2].

            This is a discovery problem, the discovery of the buckling load for this study case.

            [1] Timoshenko, S. P., & Gere, J. M. (1963). Theory of elastic stability. International student edition,
            second edition, McGraw-Hill.

            [2] Chen, Y., Cheung, Y., & Xie, J. (1989). Buckling loads of columns with varying cross sections. Journal of Engineering
            Mechanics, 115(3), 662–667.
        """
        # Beam initial data --------------------------
        # Inertia Moment [m^4]
        I = b * h ** 3 / 12
        # a = L * np.sqrt(inertia_ratio)/(1 - np.sqrt(inertia_ratio))
        aux = inertia_ratio ** (1/n)
        a = L * aux/(1 - aux)

        # Defining the list of the problem parameters (material and geometry)
        problem_parameters = [P, L, E, I, inertia_ratio, a]

        # Number of test points for the predictions
        num_test_samples = int(0.10 * num_training_samples)

        eb_s_disc = EB_Stability_Discovery_TimoBook(self.network, *problem_parameters, num_training_samples, num_test_samples)
        print("\nInput data section is finished.")

        if self.problem[1] == "fixed" and self.problem[2] == "free" and n == 2:
            eb_s_disc.fixed_free_n2(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "free" and n == 4:
            eb_s_disc.fixed_free_n4(self.problem)

        return eb_s_disc


    #
    # def EB_stability_secvarying_data(self, P, b, h, L, E, a, num_training_samples):
    #     """
    #          Method that represents the Euler-Bernoulli beam for stability problems and its settings.
    #          This is a method for the specific problem extracted in [1] page 126. See also the paper [2]
    #
    #      [1] Timoshenko, S. P., & Gere, J. M. (1963). Theory of elastic stability. International student edition,
    #      second edition, McGraw-Hill.
    #
    #      [2] Chen, Y., Cheung, Y., & Xie, J. (1989). Buckling loads of columns with varying cross sections. Journal of Engineering
    #      Mechanics, 115(3), 662–667.
    #     """
    #
    #     # Beam initial data --------------------------
    #     # Axial load [N/m]
    #     P = 1
    #
    #     b = 1
    #     h = 1
    #
    #     # Beam length [m]
    #     L = 3
    #
    #     # Young Modulus [N/m²]
    #     E = 2
    #
    #     # Inertia Moment [m^4]
    #     I = b * h ** 3 / 12
    #
    #     # Poisson coefficient
    #     nu = 0.3
    #
    #     # Area [m^2]
    #     A = b * h
    #
    #     # The length beyond the beam origin (see the paper)
    #     a = L/(np.sqrt(5) - 1) # I1/I2 = 0.2
    #     # a = np.sqrt(2) * L / (np.sqrt(5) - np.sqrt(2)) # I1/I2 = 0.4
    #     # a = np.sqrt(3) * L / (np.sqrt(5) - np.sqrt(3))  # I1/I2 = 0.6
    #     # a = np.sqrt(4) * L / (np.sqrt(5) - np.sqrt(4))  # I1/I2 = 0.8
    #
    #     # Defining the list of the problem parameters (material and geometry)
    #     problem_parameters = [P, L, E, I, a]
    #
    #     # Number of training points
    #     num_training_samples = 5000
    #
    #     # Number of test points for the predictions
    #     num_test_samples = int(0.10 * num_training_samples)
    #
    #     eb_s = EB_Stability_secvar(self.network, *problem_parameters, num_training_samples, num_test_samples)
    #
    #     if self.problem[1] == "free" and self.problem[2] == "fixed":
    #         eb_s.free_fixed()
    #
    #     return eb_s
