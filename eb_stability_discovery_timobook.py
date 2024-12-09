# @author Felipe Pereira dos Santos
# @since 04 November, 2023
# @version 04 November, 2023


import sciann as sn
from sciann_datagenerator import *


class EB_Stability_Discovery_TimoBook:
    """
         Class that represents provide features for the Euler-Bernoulli beam stability
         analysis of parameter discovery for a varying cross-section

         Based on the problem''s initial and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on


            This is a discovery problem, the discovery of the buckling load for this study case.

            [1] Timoshenko, S. P., & Gere, J. M. (1963). Theory of elastic stability. International student edition,
            second edition, McGraw-Hill.

            [2] Chen, Y., Cheung, Y., & Xie, J. (1989). Buckling loads of columns with varying cross sections. Journal of Engineering
            Mechanics, 115(3), 662–667.
    """

    def __init__(self, network, P, L, E, I, inertia_ratio, a, num_training_samples, num_test_samples):
        """
            Constructor of the Euler-Benoulli single beam stability class.

            Attributes:
                network: list of settings of a neural network used to approximate the target
                problem solution [size, activation function, initialiser]
                w: distributed load over the beam
                L: beam span
                E: Young modulus
                I: inertia moment
                nu: Poisson coefficient
                A: cross-section area
                num_training_samples: number of samples for training the model
                num_test_samples: number of samples for testing the model (predictions)


        """

        self.problem = "EB_stability_discovery_timobook"
        self.L = L
        self.E = E
        self.I = I
        self.a = a
        self.inertia_ratio = inertia_ratio
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        # Neural Network Setup.
        dtype = 'float32'

        self.x = sn.Variable("x", dtype=dtype)

        self.u = sn.Functional('u', self.x, network[0], network[1], kernel_initializer=network[2])
        self.rot = sn.Functional('rot', self.x, network[0], network[1], kernel_initializer=network[2])
        self.P = sn.Parameter(P, inputs=self.x, name='Pcr')

        self.du_dx = sn.diff(self.u, self.x)
        self.d2u_dx2 = sn.diff(self.u, self.x, order=2)
        self.d3u_dx3 = sn.diff(self.u, self.x, order=3)
        self.d4u_dx4 = sn.diff(self.u, self.x, order=4)

        # I_fx = self.I * (self.x / self.a) ** 2
        I_fx = self.I * ((self.x + self.a) / self.a) ** 2
        self.eqDiff1 = self.E * I_fx * self.d2u_dx2 + self.P * self.u

        self.variables = [self.x]

    def model_info(self):
        """
        Method to write the physical model information in the text file output that contains the
        elvaluation of the MSE errors

        """
        model_parameters = 'Number of training samples: ' + str(self.num_training_samples) + \
                           '\nP: ' + str(self.P) + ' N | ' + 'L: ' + str(self.L) + ' m | ' + 'E: ' +\
                           str(self.E) + ' N/m² | ' + 'I: ' + str(self.I) + ' m^4 | ' + 'a: ' + str(self.a) + ' m\n'
        return model_parameters


    def fixed_free_n2_OK(self, problem):
        """
             Method to setting the features for a cantilever varying section beam with an axial load

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        P_ref = self.reference_solution_n2(x, problem)

        self.ref_solu = P_ref
        # Reference solution for the predictions ======================================

        # I = self.I*((self.a + self.x)/self.a)**2
        # I = self.I
        # self.eqDiff1 = self.E * I * self.d2u_dx2 + self.P * self.u
        # EI = 1.03084 * 10 ** 10 * (0.05 + 0.005 * self.x ** 2) ** 4
        # E = (2.1 - 2.2*self.x + 1.1*self.x**2)*10**11
        # EI = 1.03084 * 10 ** 2 * ( 0.005*self.x**2 - 0.03*self.x + 0.095) ** 4
        # d = 0.02
        # I = np.pi*d**4/64

        # I_fx = self.I * (self.x / self.a) ** 2
        # self.eqDiff1 = self.E * I_fx * self.d2u_dx2 + self.P * self.u

        # Boundary conditions
        BC_left_1 = (self.x == 0.0) * (self.du_dx - 0.5)
        BC_left_2 = (self.x == 0.0) * (self.u)
        # BC_left_3 = (self.x == 0.) * (self.x ** 2 * self.d3u_dx3 + 2 * self.x * self.d2u_dx2 + self.du_dx * (self.P * self.a ** 2) / (self.E * self.I))

        BC_right_1 = (self.x == self.L) * (self.du_dx)

        # Boundary conditions
        # BC_left_1 = (self.x == self.a) * (self.du_dx - 0.5)
        # # BC_left_2 = (self.x == self.a) * (self.d2u_dx2)
        # # BC_left_3 = (self.x == 0.) * (self.x ** 2 * self.d3u_dx3 + 2 * self.x * self.d2u_dx2 + self.du_dx * (self.P * self.a ** 2) / (self.E * self.I))
        #
        # BC_right_1 = (self.x == (self.a + self.L)) * (self.du_dx)
        # BC_right_2 = (self.x == (self.a + self.L)) * (self.u)

        # Loss function
        self.targets = [self.eqDiff1,
                   BC_left_1, BC_left_2,
                   BC_right_1]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 2 * ['bc-left'] + 1 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_free_n2(self, problem):
        """
             Method to setting the features for a cantilever varying section beam with an axial load

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        P_ref = self.reference_solution_n2(x, problem)

        self.ref_solu = P_ref
        # Reference solution for the predictions ======================================


        I_fx = self.I * ((self.x + self.a) / self.a) ** 2
        self.eqDiff1 = self.E * I_fx * self.d2u_dx2 + self.P * self.u

        # Boundary conditions
        BC_left_1 = (self.x == 0.0) * (self.du_dx - 0.5)
        BC_left_2 = (self.x == 0.0) * (self.u)

        BC_right_1 = (self.x == self.L) * (self.du_dx)

        # I_fx = self.I * ((self.L + self.a - self.x) /(self.a)) ** 2
        # self.eqDiff1 = self.E * I_fx * self.d2u_dx2 + self.P * self.u
        #
        #
        # # Boundary conditions
        # BC_left_1 = (self.x == 0.0) * (self.du_dx)
        # # BC_left_2 = (self.x == 0.0) * (self.u - 1)
        # # BC_left_3 = (self.x == 0.) * (self.x ** 2 * self.d3u_dx3 + 2 * self.x * self.d2u_dx2 + self.du_dx * (self.P * self.a ** 2) / (self.E * self.I))
        #
        # BC_right_1 = (self.x == self.L) * (self.d2u_dx2)
        # BC_right_2 = (self.x == self.L) * (self.u)
        # BC_right_3 = (self.x == self.L) * (self.du_dx - 0.5)
        # partial_I = ((self.L + self.a - self.x) /self.a) ** 2
        # diff_aux = sn.diff(partial_I * self.d2u_dx2, self.x)
        # BC_right_3 = (self.x == self.L) * ((self.d3u_dx3 * (self.L + self.a - self.x) /(self.a)) ** 2 - 2*((self.L + self.a - self.x) /(self.a)) * self.d2u_dx2 + self.du_dx * self.P / (self.E * self.I))
        # BC_right_4 = (self.x == self.L) * (self.d3u_dx3 - 2 * self.d2u_dx2 + self.du_dx * self.P / (self.E * self.I))

        # Loss function
        self.targets = [self.eqDiff1,
                   BC_left_1,BC_left_2,
                   BC_right_1]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 2 * ['bc-left'] + 1 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_free_n2_Timoref(self, problem):
        """
             Method to setting the features for a cantilever varying section beam with an axial load

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        P_ref = self.reference_solution_n2(x, problem)

        self.ref_solu = P_ref
        # Reference solution for the predictions ======================================

        I_fx = self.I * (self.x / self.a) ** 2
        self.eqDiff1 = self.E * I_fx * self.d2u_dx2 + self.P * self.u

        # Boundary conditions
        BC_left_1 = (self.x == self.a) * (self.u)
        BC_left_2 = (self.x == self.a) * (self.d2u_dx2)

        partial_I = (self.x / self.a) ** 2
        diff_aux = sn.diff(partial_I * self.d2u_dx2, self.x)
        # BC_left_3 = (self.x == self.a) * (diff_aux + self.du_dx * self.P / (self.E * self.I))
        BC_left_3 = (self.x == self.a) * ((2 * self.a * self.d2u_dx2 + self.a ** 2 * self.d3u_dx3) + self.du_dx * (self.P * self.a ** 2) / (self.E * self.I))

        BC_right_1 = (self.x == (self.L + self.a)) * (self.u - 1)
        BC_right_2 = (self.x == (self.L + self.a)) * (self.du_dx)

        # Loss function
        self.targets = [self.eqDiff1,
                        BC_left_1, BC_left_2, BC_left_3,
                        BC_right_1, BC_right_2]

        dg = DataGeneratorX(X=[self.a, (self.L + self.a)],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 3 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_free_n4(self, problem):
        """
             Method to setting the features for a cantilever varying section beam with an axial load

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        P_ref = self.reference_solution_n4(x, problem)

        self.ref_solu = P_ref
        # Reference solution for the predictions ======================================


        I_fx = self.I * ((self.x + self.a) / self.a) ** 4
        self.eqDiff1 = self.E * I_fx * self.d2u_dx2 + self.P * self.u

        # Boundary conditions
        BC_left_1 = (self.x == 0.0) * (self.du_dx - 0.5)
        BC_left_2 = (self.x == 0.0) * (self.u)

        BC_right_1 = (self.x == self.L) * (self.du_dx)

        # Loss function
        self.targets = [self.eqDiff1,
                   BC_left_1, BC_left_2,
                   BC_right_1]

        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 2 * ['bc-left'] + 1 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def reference_solution_n2(self, x, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
         In this case, the reference solution was extracted from [1] page 126, for n = 2.

         [1] Timoshenko, S. P., & Gere, J. M. (1963). Theory of elastic stability. International student edition,
         second edition, McGraw-Hill.

         For each inertia ratio I_1/I_2, there is a correspondent m that generates the solution in terms of P_cr.

        """
        # I_2 = self.I * (self.L / self.a) ** 2
        I_2 = self.I * ((self.L + self.a) / self.a) ** 2
        inertia_ratio_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        m = np.array([1.350, 1.593, 1.763, 1.904, 2.023, 2.128, 2.223, 2.311, 2.392, np.pi ** 2 /4])
        P_cr = m * self.E * I_2 / self.L ** 2
        dic_P = dict(zip(inertia_ratio_range, P_cr))

        P = dic_P[self.inertia_ratio]

        return P

    def reference_solution_n4(self, x, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
         In this case, the reference solution was extracted from [1] page 126, for n = 4.

         [1] Timoshenko, S. P., & Gere, J. M. (1963). Theory of elastic stability. International student edition,
         second edition, McGraw-Hill.

         For each inertia ratio I_1/I_2, there is a correspondent m that generates the solution in terms of P_cr.

        """
        # I_2 = self.I * (self.L / self.a) ** 4
        I_2 = self.I * ((self.L + self.a) / self.a) ** 4
        inertia_ratio_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        m = np.array([1.202, 1.505, 1.710, 1.870, 2.002, 2.116, 2.217, 2.308, 2.391, np.pi ** 2 /4])
        P_cr = m * self.E * I_2 / self.L ** 2
        dic_P = dict(zip(inertia_ratio_range, P_cr))

        P = dic_P[self.inertia_ratio]

        return P

