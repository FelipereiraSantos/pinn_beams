# @author Felipe Pereira dos Santos
# @since 27 December, 2023
# @version 27 December, 2023

import sciann as sn
from sciann_datagenerator import *


class Nonlinear_TimoEx:
    """
         Class responsible to provide features for the non-linear example beam-column analysis.

         Based on the problem's governing equations and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on


         This problem and its reference results were extracted from the following book:

         [1] Timoshenko, S. P., & Gere, J. M. (1982). Mecânica dos Sólidos. Volume 1.
         This is a translated version (Portuguese) of the Mechanics of Materials book
    """

    def __init__(self, network, P, L, E, I, num_training_samples, num_test_samples):
        """
            Constructor of the non-linear beam study case with a vertical point load at the beam's tip

            Attributes:
                network (keras network): list of settings of a neural network used to approximate the target
                problem solution [size, activation function, initialiser]
                P: Point load at the beam's tip
                L: beam span
                E: Young modulus
                I: inertia moment
                nu: Poisson coefficient
                A: cross-section area
                num_training_samples: number of samples for training the model
                num_test_samples: number of samples for testing the model (predictions)

        """
        self.problem = "Nonlinear_TimoEx"
        self.P = P

        self.L = L
        self.L_aux = 1
        self.E = E
        self.I = I
        self.alpha = (self.P * self.L ** 2)/(self.E * self.I)

        if self.alpha >= 1:
            self.alpha = round(self.alpha)
        else:
            self.alpha = round(self.alpha, 2)

        print("alpha: ", self.alpha)


        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        # Neural Network Setup.
        dtype = 'float32'

        self.xi = sn.Variable("xi", dtype=dtype)
        self.rot = sn.Functional('rot', self.xi, network[0], network[1], kernel_initializer=network[2])

        self.drot_dx = sn.diff(self.rot, self.xi)
        self.d2rot_dx2 = sn.diff(self.rot, self.xi, order=2)

        self.eqDiff1 = self.d2rot_dx2 + self.alpha * sn.cos(self.rot)

        self.variables = [self.xi]

    def model_info(self):
        """
        Method to write the physical model information in the text file output that contains the
        evaluation of the MSE errors

        DISCLAIMER: this method might be unused

        """
        model_parameters = 'Number of training samples: ' + str(self.num_training_samples) + \
                           '\nP: ' + str(self.P) + ' N | ' + 'L: ' + str(self.L) + ' m | ' + 'E: ' +\
                           str(self.E) + ' N/m² | ' + 'I: ' + str(self.I) + ' m^4 | ' + 'alpha: ' + str(self.alpha) + ' m\n'
        return model_parameters


    def fixed_free(self, problem):
        """
             Method for defining the loss function of the non-linear beam a vertical load P at the beam's tip.
             In this method, the input data and target data (in the PINN context) are also generated.

        """

        # Reference solution for the predictions ======================================
        xi = np.linspace(0, self.L_aux, int(self.num_test_samples))
        self.x_test = xi
        rot_ref = self.reference_solution(xi, problem)

        self.ref_solu = rot_ref
        # Reference solution for the predictions ======================================

        # Boundary and initial conditions
        BC_left_1 = (self.xi == 0.) * (self.rot)
        BC_right_1 = (self.xi == self.L_aux) * (self.drot_dx)

        # Loss function
        self.targets = [self.eqDiff1,
                   BC_left_1, BC_right_1]

        dg = DataGeneratorX(X=[0., self.L_aux],
                            num_sample=self.num_training_samples,
                            targets=1 * ['domain'] + 1 * ['bc-left'] + 1 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()


    def reference_solution(self, xi, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
         In this case, the reference solution was extracted from [1] page 126, for n = 4.

         [1] Timoshenko, S. P., & Gere, J. M. (1982). Mecânica dos Sólidos. Volume 1.

         This is a translated version (Portuguese) of the Mechanics of Materials book

         For each alpha value, there is a correspondent 'm' that generates the solution in terms of the rotation 'theta'.

        """

        # The following rotation values (theta) were extracted from the book used as reference
        theta = (np.pi / 2) * np.array([0.079, 0.156, 0.228, 0.294, 0.498, 0.628, 0.714, 0.774, 0.817, 0.849, 0.874, 0.894, 0.911])
        alpha = np.array([0.25, 0.50, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        dic_theta = dict(zip(alpha, theta))

        theta_ref = dic_theta[self.alpha]

        return theta_ref
