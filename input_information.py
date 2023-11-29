# @author Felipe Pereira dos Santos
# @since 12 june, 2023
# @version 12 june, 2023
import sys

import numpy as np
from timoshenko_beam import Timoshenko
from eb_stability import EB_Stability
from eb_stability_secvar import EB_Stability_secvar
from euler_bernoulli_beam import EulerBernoulli
from eb_dynamics import EB_Dynamics
from eb_stability_discovery import EB_Stability_Discovery
from eb_stability_discovery_timobook import EB_Stability_Discovery_TimoBook

class InputInformation:
    """
         Class that represents the input information of the model

         The tasks of this class are:
             1. Define the material and geometry of the problem
             2. Define the number of training samples and test samples
             3. Define the training variables and their labels
             4. Define the neural network structure (number of layers and nodes, activation function, etc)
             5. Define the reference solution to be compared with the prediction of the
             trained model later on. It can be setted as 'None' with the lack of such solution.

         Attributes:
         problem: it defines the target problem, such as bending euler-bernoulli or timoshenko beam,
         axial beam, etc. This variable can be a list containing a string with the problem name, and
         other strings informing the boundary conditions: pinned-pinned, fixed-free and so on.
    """

    def __init__(self, problem, network_info, model_parameters):
        """
            Constructor of the InputInformation class.

            Args:
                problem (list): name of the problem to be solved. It also can contain its
                boundary conditions
        """
        self.problem = problem
        self.network = network_info
        self.model_parameters = model_parameters
        # if self.network[2] == 'on':
        #     print('Loading the weights')
        #     self.network[2] = load_weights_from='weights_test.hdf5'

    def input_data(self):
        """
            Method to initialize the data inputs based on the problem type

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
        elif self.problem[0] == "EB_Stability_secvar":
            return self.EB_stability_secvarying_data(*self.model_parameters)
        elif self.problem[0] == "EB_stability_discovery":
            return self.EB_stability_discovery_data(*self.model_parameters)
        elif self.problem[0] == "EB_stability_discovery_timobook":
            return self.EB_stability_discovery_data_timobook(*self.model_parameters)
        elif self.problem[0] == "EB_dynamics":
            return self.EB_dynamics_data()
        elif self.problem[0] == "Tk_continuous_bending":
            return self.Tk_continuous_bending_data()
        elif self.problem[0] == "axial_beam":
            return self.axial_beam_data()


    def EB_dynamics_data(self):
        """
             Method that represents the Euler-Bernoulli dynamics beam problem and its settings
        """

        # Beam initial data --------------------------
        # Point load [N]
        P = 100

        b = 0.01
        h = 0.01

        # Beam length [m]
        L = 1

        # Young Modulus [N/m²]
        E = 206800*10**6

        # Inertia Moment [m^4]
        # I = 0.000038929334
        I = b * h ** 3 / 12

        # Poisson coefficient
        nu = 0.3

        # Area [m^2]
        A = b * h

        # Density
        rho = 7830 # [kg/m^3]

        # Defining the list of the problem parameters (material and geometry)
        problem_parameters = [P, L, E, I, rho, A]

        # Number of training points
        num_training_samples = 8000

        # Number of test points for the predictions
        num_test_samples = int(0.10 * num_training_samples)

        eb_dy = EB_Dynamics(self.network, *problem_parameters, num_training_samples, num_test_samples)

        if self.problem[1] == "pinned" and self.problem[2] == "pinned":
            eb_dy.pinned_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "free":
            eb_dy.fixed_free(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "pinned":
            eb_dy.fixed_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "fixed":
            eb_dy.fixed_fixed(self.problem)
        elif self.problem[1] == "varying_sec":
            eb_dy.varying_sec(self.problem)

        return eb_dy

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

        tk = Timoshenko(self.network, *problem_parameters, num_training_samples, num_test_samples)

        if self.problem[1] == "pinned" and self.problem[2] == "pinned":
            tk.pinned_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "free":
            tk.fixed_free(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "pinned":
            tk.fixed_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "fixed":
            tk.fixed_fixed(self.problem)
        elif self.problem[1] == "varying_sec":
            tk.varying_sec(self.problem)

        return tk

    def EB_stability_discovery_data(self, P, b, h, L, E, num_training_samples):
        """
             Method that represents the Euler-Bernoulli beam for stability problems and its settings for
             discovery of parameters
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
        elif self.problem[1] == "fixed" and self.problem[2] == "free" and a == 0:
            eb_s.fixed_free(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "free" and a != 0:
            eb_s.fixed_free_2specie(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "pinned":
            eb_s.fixed_pinned(self.problem)
        elif self.problem[1] == "fixed" and self.problem[2] == "fixed":
            eb_s.fixed_fixed(self.problem)

        return eb_s

    def EB_stability_discovery_data_timobook(self, P, b, h, L, E, inertia_ratio, n, num_training_samples):
        """
            Method that represents the Euler-Bernoulli beam for stability problems and its settings.
            This is a method for the specific problem extracted in [1] page 126.

            [1] Timoshenko, S. P., & Gere, J. M. (1963). Theory of elastic stability. International student edition,
            second edition, McGraw-Hill.
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



    def EB_stability_secvarying_data(self, P, b, h, L, E, a, num_training_samples):
        """
             Method that represents the Euler-Bernoulli beam for stability problems and its settings.
             This is a method for the specific problem extracted in [1] page 126.

         [1] Timoshenko, S. P., & Gere, J. M. (1963). Theory of elastic stability. International student edition,
         second edition, McGraw-Hill.
        """

        # Beam initial data --------------------------
        # Axial load [N/m]
        P = 1

        b = 1
        h = 1

        # Beam length [m]
        L = 3

        # Young Modulus [N/m²]
        E = 2

        # Inertia Moment [m^4]
        I = b * h ** 3 / 12

        # Poisson coefficient
        nu = 0.3

        # Area [m^2]
        A = b * h

        # The length beyong the beam origin (see paper)
        a = L/(np.sqrt(5) - 1) # I1/I2 = 0.2
        # a = np.sqrt(2) * L / (np.sqrt(5) - np.sqrt(2)) # I1/I2 = 0.4
        # a = np.sqrt(3) * L / (np.sqrt(5) - np.sqrt(3))  # I1/I2 = 0.6
        # a = np.sqrt(4) * L / (np.sqrt(5) - np.sqrt(4))  # I1/I2 = 0.8

        # Defining the list of the problem parameters (material and geometry)
        problem_parameters = [P, L, E, I, a]

        # Number of training points
        num_training_samples = 5000

        # Number of test points for the predictions
        num_test_samples = int(0.10 * num_training_samples)

        eb_s = EB_Stability_secvar(self.network, *problem_parameters, num_training_samples, num_test_samples)

        if self.problem[1] == "pinned" and self.problem[2] == "pinned":
            eb_s.pinned_pinned()
        elif self.problem[1] == "free" and self.problem[2] == "fixed":
            eb_s.free_fixed()
        elif self.problem[1] == "fixed" and self.problem[2] == "pinned":
            eb_s.fixed_pinned()
        elif self.problem[1] == "fixed" and self.problem[2] == "fixed":
            eb_s.fixed_fixed()

        return eb_s

    def Tk_continuous_bending_data(self):
        """
             Method that represents the Timoshenko beam problem and its settings
        """

        # Beam initial data --------------------------
        # Beam segments
        num_s = 3

        # Distributed load [N/m]
        w = -1

        b = 0.2
        h = 0.5

        # Beam length [m]
        L_1 = 3
        L_2 = 2
        L_3 = 1

        # Young Modulus [N/m²]
        E = 100

        # Inertia Moment [m^4]
        # I = 0.000038929334
        I = b * h ** 3 / 12

        # Poisson coefficient
        nu = 0.3

        # Area [m^2]
        A = b * h

        # Defining the list of the problem parameters (material and geometry)
        problem_parameters = [[w, L_1, E, I, nu, A], [w, L_2, E, I, nu, A], [w, L_3, E, I, nu, A]]

        # Number of training points
        num_training_samples = 5000

        # Number of test points for the predictions
        # num_test_samples = 23
        num_test_samples = int(0.10 * num_training_samples)

        tk = []
        for i in range(num_s):
            tk[i] = Timoshenko(self.network, *problem_parameters[i], num_training_samples, num_test_samples)

        return tk

