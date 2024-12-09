# @author Felipe Pereira dos Santos
# @since 12 june, 2023
# @version 15 june, 2023


import sciann as sn
import tensorflow as tf
from sciann_datagenerator import *
from data_generator import*
import sys


class Timoshenko:
    """
         Class responsible to provide features for the Timoshenko bending beam analysis.

         Based on the problem's governing equations and boundary conditions, the tasks of this class are:

             1. Create the inputs and outputs for the physics-informed neural network
             2. Build the reference solution to compare with the predictions later on
    """

    def __init__(self, network, w, L, E, I, nu, A, num_training_samples, num_test_samples):
        """
            Constructor of the Timoshenko beam class.

            Attributes:
                network: list of settings of a neural network used to approximate the target
                problem solution [size, activation function, initialiser]
                w: distributed load over the beam span
                L: beam span
                E: Young modulus
                I: inertia moment
                nu: Poisson coefficient
                A: cross-section area
                num_training_samples: number of samples for training the model
                num_test_samples: number of samples for testing the model (predictions)

            Comments:
            It is possible to supress some boundary conditions and force them in the neural network approximation.
            If hard constraints are used, the loss function must be built accordingly.


        """

        self.problem = "Tk_bending"
        self.w = w
        self.L = L
        self.E = E
        self.I = I
        self.A = A
        self.G = E / (2 * (1 + nu))  # Shear modulus
        self.nu = nu
        self.k = 5. / 6.   # Form factor specific for a rectangular cross-section
        # self.k = 9. / 10   # Form factor specific for a circular cross-section
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        # Neural Network Setup.
        dtype = 'float32'

        self.x = sn.Variable("x", dtype=dtype)

        self.u_aux = sn.Functional('u', self.x, network[0], network[1], kernel_initializer=network[2])
        self.rot_aux = sn.Functional('rot', self.x, network[0], network[1], kernel_initializer=network[2])
        # self.M = sn.Functional('M', self.x, network[0], network[1], kernel_initializer=network[2])

        # Deflection (u) and cross-section rotations (rot)
        self.u = self.x * self.u_aux
        self.rot = self.x * self.rot_aux

        # # Hard constraint for a cantilever beam
        # If hard constraints are used, the loss function must be built accordingly
        # self.u = self.x * self.u_aux
        # self.rot = self.x * self.rot_aux

        self.du_dx = sn.diff(self.u, self.x)
        self.drot_dx = sn.diff(self.rot, self.x)

        self.d2u_dx2 = sn.diff(self.u, self.x, order=2)
        self.d2rot_dx2 = sn.diff(self.rot, self.x, order=2)

        # Building the Timoshenko beam differential equations
        self.eqDiff1 = ((self.E * self.I) / (self.k * self.A * self.G)) * self.d2rot_dx2 + (self.du_dx - self.rot)
        self.eqDiff2 = (self.d2u_dx2 - self.drot_dx) + (self.w / (self.k * self.A * self.G))

        self.variables = [self.x]


    def model_info(self):
        """
        Method to write the physical model information in the text file output that contains the
        evaluation of the MSE errors

        DISCLAIMER: this method might be unused

        """
        model_parameters = 'Number of training samples: ' + str(self.num_training_samples) + \
                           '\nw: ' + str(self.w) + ' N/m | ' + 'L: ' + str(self.L) + ' m | ' + 'k: ' + str(self.k) +\
                           '\nA: ' + str(self.A) + ' m² | ' + 'G: ' + str(self.G) + ' N/m² | ' + 'E: ' + \
                           str(self.E) + ' N/m² | ' + 'I: ' + str(self.I) + ' m^4 | ' + 'nu: ' + str(self.nu) + ' []\n'
        return model_parameters

    def pinned_pinned(self, problem):
        """
             Method for setting the features for the simply supported Timoshenko beam

             x_test: array of collocation points to evaluate the trained model and the reference solution
             u_ref: reference solution for further comparisons (obtained analytically or numerically)
             input_data: points distributed all over the problem domain for training
             target_data: corresponded labels of the input_data points ("true" values)
             targets: target constraints involving the differential equations and boundary conditions (loss function)
                    eqDiff: refers to one or more differential equations of the problem
                    BC_left: refers to the target points at the left boundary of the beam
                    BC_right: refers to the target points at the right boundary of the beam

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        x, u_ref, rot_ref = self.reference_solution(x, problem)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Different problem setting for the pinned-pinned case
        # Linear shape of the cross-section, parabolic shape of the cross-section and parabolic load for a
        # prismatic cross-section
        if problem[3] == "LinearShape":
            self.linear_shape()

        elif problem[3] == "ParabolicShape":
            self.parabolic_shape()

        elif problem[3] == "ParabolicLoad":
            self.eqDiff2 = (self.d2u_dx2 - self.drot_dx) + (-self.x ** 2 + self.L * self.x) * (4 * self.w) / (self.k * self.A * self.G * self.L ** 2)


        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.drot_dx)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.drot_dx)

        # Loss function (self.targets)
        self.targets = [self.eqDiff1, self.eqDiff2,
                   BC_left_1, BC_left_2,
                   BC_right_1, BC_right_2]

        # Additional training data===============================================
        # This is not working yet. It refers to the possibility to add data points ('true' values from reference
        # solution) on the training
        if isinstance(problem[3], int):
            print("Additional data was added to training")
            x_input = np.linspace(0, self.L, int(problem[3]))
            data = self.additional_data(x_input, problem)
            # self.input_data, self.target_data = self.get_inputs_with_data()

            # Loss function (self.targets)
            self.targets.append(sn.Data(self.u))
            self.targets.append(sn.Data(self.rot))

            # Data generation based on the loss function terms
            dg = DataGenerator1D(X=[0., self.L],
                                num_sample=self.num_training_samples,
                                targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'] + 2 * ['data'],
                                 data=data)

        else:
            # Data generation based on the loss function terms
            dg = DataGeneratorX(X=[0., self.L],
                                num_sample=self.num_training_samples,
                                targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

            if problem[3] == "LinearShape":
                BC_right_1 = (self.x == self.L) * (self.du_dx)
                BC_right_2 = (self.x == self.L) * (self.rot)

                # Loss function (self.targets)
                self.targets = [self.eqDiff1, self.eqDiff2,
                                BC_left_1, BC_left_2,
                                BC_right_1, BC_right_2]

                dg = DataGeneratorX(X=[0., self.L],
                                    num_sample=self.num_training_samples,
                                    targets=2 * ['domain'] + 2 * ['bc-left']+ 2 * ['bc-right'])

            # Creating the training input points
        self.input_data, self.target_data = dg.get_data()


    def fixed_free(self, problem):
        """
             Method for setting the features for the cantilever (fixed-free) Timoshenko beam

             x_test: array of collocation points to evaluate the trained model and the reference solution
             u_ref: reference solution for further comparisons (obtained analytically or numerically)
             input_data: points distributed all over the problem domain for training
             target_data: corresponded labels of the input_data points ("true" values)
             targets: target constraints involving the differential equations and boundary conditions
                    eqDiff: refers to one or more differential equations of the problem
                    BC_left: refers to the target points at the left boundary of the beam

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        x, u_ref, rot_ref = self.reference_solution(x, problem)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        if problem[3] == "ParabolicShape":
            self.parabolic_shape()

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.rot)

        BC_right_1 = (self.x == self.L) * (self.drot_dx)
        BC_right_2 = (self.x == self.L) * (self.du_dx - self.rot)

        # Loss function (self.targets)
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1, BC_left_2, BC_right_1, BC_right_2]

        # # If hard constraints are used, the loss function must be built accordingly
        # self.targets = [self.eqDiff1, self.eqDiff2,
        #                 BC_right_1, BC_right_2]

        # Data generation based on the loss function terms
        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=2 * ['domain'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()


    def fixed_fixed(self, problem):
        """
             Method for setting the features for the double-fixed (fixed-fixed) Timoshenko beam

             x_test: array of collocation points to evaluate the trained model and the reference solution
             u_ref: reference solution for further comparisons (obtained analytical or numerically)
             input_data: points distributed all over the problem domain for training
             target_data: corresponded labels of the input_data points ("true" values)
             targets: target constraints involving the differential equations and boundary conditions
                    eqDiff: refers to one or more differential equations of the problem
                    BC_left: refers to the target points at the left boundary of the beam
                    BC_right: refers to the target points at the right boundary of the beam

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        C1 = -(self.w * self.L * (self.k * self.G * self.A * self.L ** 2 + 12 * self.E * self.I)) / (24 * self.E * self.I + 2 * self.k * self.G * self.A * self.L ** 2);
        u_ref = (self.w/(24 * self.E * self.I))*(x ** 4 - 2 * self.L * x ** 3 + self.L ** 2 * x ** 2) + (self.w/(2 * self.k * self.G * self.A))*(-x ** 2 + self.L * x)
        rot_ref = (1/(6*self.E * self.I))*(self.w * x ** 3 + 3 * C1 * x ** 2 - self.w * self.L ** 2 * x -3 * C1 * self.L * x)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.rot)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.rot)

        # Loss function (self.targets)
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1, BC_left_2, BC_right_1, BC_right_2]

        # Data generation based on the loss function terms
        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def fixed_pinned(self, problem):
        """
             Method for setting the features for the fixed-pinned Timoshenko beam

             x_test: array of collocation points to evaluate the trained model and the reference solution
             u_ref: reference solution for further comparisons (obtained analytical or numerically)
             input_data: points distributed all over the problem domain for training
             target_data: corresponded labels of the input_data points ("true" values"
             targets: target constraints involving the differential equations and boundary conditions
                    eqDiff: refers to one or more differential equations of the problem
                    BC_left: refers to the target points at the left boundary of the beam
                    BC_right: refers to the target points at the right boundary of the beam

        """

        # Reference solution for the predictions ======================================
        x = np.linspace(0, self.L, int(self.num_test_samples))
        self.x_test = x
        C1 = -(5 * self.w * self.k * self.G * self.A * self.L ** 3 + 12 * self.w * self.L * self.E * self.I)/(8 * (self.k * self.G * self.A * self.L ** 2 + 3 * self.E * self.I))
        u_ref = (1/(self.E * self.I))*(self.w * x ** 4/24 + C1 * x ** 3/6 - (self.w * self.L ** 2/4) * x ** 2 - (C1 * self.L/2) * x ** 2) - (self.w/(2 * self.k * self.G * self.A)) * x ** 2 - (C1 * x)/(self.k * self.G * self.A)
        rot_ref = (1/(6 * self.E * self.I))*(self.w * x ** 3 + 3 * C1 * x ** 2 - 3 * self.w * self.L ** 2 * x -6 * C1 * self.L * x)
        self.ref_solu = [u_ref, rot_ref]
        # Reference solution for the predictions ======================================

        # Boundary conditions
        BC_left_1 = (self.x == 0.) * (self.u)
        BC_left_2 = (self.x == 0.) * (self.rot)

        BC_right_1 = (self.x == self.L) * (self.u)
        BC_right_2 = (self.x == self.L) * (self.drot_dx)

        # Loss function (self.targets)
        self.targets = [self.eqDiff1, self.eqDiff2,
                        BC_left_1, BC_left_2, BC_right_1, BC_right_2]

        # Data generation based on the loss function terms
        dg = DataGeneratorX(X=[0., self.L],
                            num_sample=self.num_training_samples,
                            targets=2 * ['domain'] + 2 * ['bc-left'] + 2 * ['bc-right'])

        # Creating the training input points
        self.input_data, self.target_data = dg.get_data()

    def linear_shape(self):
        # Construcion of the Timochenko beam differential equations
        C = ((self.E * self.I) / (self.k * self.A * self.G * self.L ** 2))

        self.eqDiff1 = C * (4 * (self.L + self.x) * self.drot_dx + self.d2rot_dx2 * (self.L + self.x) ** 2) + (
                    self.du_dx - self.rot)
        self.eqDiff2 = 2 * (self.L + self.x) * (self.du_dx - self.rot) + (self.L + self.x) ** 2 * (
                    self.d2u_dx2 - self.drot_dx) + ((self.w * self.L ** 2) / (self.k * self.A * self.G))

    def parabolic_shape(self):
        qi = 0.25
        qm = 0.50
        qf = 0.20
        a = (-4 * qm + 2 * qf + 2 * qi) / self.L ** 2
        b = (4 * qm - qf - 3 * qi) / self.L
        c = qi
        pi = tf.constant(np.pi)

        I_var = (pi / 64) * (a * self.x ** 2 + b * self.x + c) ** 4
        diff_I_var = (pi/16) * (a * self.x ** 2 + b * self.x + c) ** 3 * (2*a*self.x + b)
        diff_A = (pi/2)*(a * self.x ** 2 + b * self.x + c)*(2*a*self.x + b)
        A = (pi/4)*(a * self.x ** 2 + b * self.x + c)**2
        diff_I_A = (1/4)*(a * self.x ** 2 + b * self.x + c)*(2*a*self.x + b)
        I_A = (1/16)*(a * self.x ** 2 + b * self.x + c)**2 # I over A
        cte = 32 * self.w / (self.E * pi)
        # self.eqDiff1 = sn.diff(self.E * I_var * self.drot_dx, self.x) + A * self.G * self.k * (self.du_dx - self.rot)
        # self.eqDiff1 = (self.du_dx - self.rot) + (self.E/(self.G * self.k))*(diff_I_A*self.drot_dx + I_A*self.d2rot_dx2)

        self.eqDiff1 = self.drot_dx - cte * (
                    (-2 * self.L * self.x + self.x ** 2 + self.L ** 2) / (a * self.x ** 2 + b * self.x + c) ** 4) # From the bending moment

        # self.eqDiff3 = self.E * (diff_I_var * self.drot_dx  + I_var * self.d2rot_dx2) + self.G*self.k * A * (self.du_dx - self.rot)
        self.eqDiff2 = (self.w/(self.G*self.k)) + (diff_A *(self.du_dx - self.rot) + A *(self.d2u_dx2 - self.drot_dx))
        # self.eqDiff2 = (self.w/(self.G*self.k))*(self.L - self.x) - A * (self.du_dx - self.rot)

        # Define the file path to get the reference solution
        file_path = 'C:/Users/felip/git/beam_pinns/NumericalResults/StaticResults/CaseStudies/ParabolicShape/FixedFree/Tk_ffr_q_ParabolicShape_ref_circ_200.csv'

        # Using numpy's genfromtxt function to read the CSV file
        # Skipping the first row if it contains headers
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

        # Separating the columns into numpy arrays
        x = data[:, 0]
        u_ref = data[:, 1]
        rot_ref = data[:, 2]
        self.ref_solu = [u_ref, rot_ref]

        mesh = [5, 11, 17]
        str_mesh = ['Tk_ffr_q_ParabolicShape_ref_circ_5.csv', 'Tk_ffr_q_ParabolicShape_ref_circ_11.csv',
                    'Tk_ffr_q_ParabolicShape_ref_circ_17.csv']
        self.mesh_ref = []
        for i, num in enumerate(mesh):
            # Define the file path
            file_path = 'C:/Users/felip/git/beam_pinns/NumericalResults/StaticResults/CaseStudies/ParabolicShape/FixedFree/' + \
                        str_mesh[i]

            # Using numpy's genfromtxt function to read the CSV file
            # Skipping the first row if it contains headers
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

            # Separating the columns into numpy arrays
            x_m = data[:, 0]
            u_ref_m = data[:, 1]
            rot_ref_m = data[:, 2]
            aux_mesh = [x_m, u_ref_m, rot_ref_m]
            self.mesh_ref.append(aux_mesh)

    def reference_solution(self, x, problem):
        """
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc
        """
        # Aditional data based on the reference solution for the predictions ======================================
        # x = np.linspace(0, self.L, int(problem[3]))
        if problem[1] == "pinned" and problem[2] == "pinned":
            if problem[3] == "LinearShape":
                if x.size % 2 > 0:
                    x_half = x[:(int(x.size /2) + 1)]
                else:
                    x_half = x[:int(x.size /2)]
                u_half = -((self.w * self.L ** 4) / (2 * self.E * self.I)) * (
                            self.L * (3 * self.L + 4 * x_half) / (2 * (self.L + x_half) ** 2) + np.log(self.L + x_half) - 1.5 - np.log(
                        self.L) - \
                            x_half / (8 * self.L)) - ((self.w * self.L ** 2) / (self.G * self.A * self.k)) * (
                                2 * self.L / (self.L + x_half) + np.log(self.L + x_half) - 2 - np.log(self.L))
                rot_half = -((self.w * self.L ** 4) / (2 * self.E * self.I)) * (x_half ** 2 / (self.L + x_half) ** 3) + (
                            (self.w * self.L ** 3) / (16 * self.E * self.I))
                if x.size % 2 > 0:
                    u = np.concatenate((u_half, u_half[:-1][::-1]))
                    rot = np.concatenate((rot_half, rot_half[:-1][::-1]))
                else:
                    u = np.concatenate((u_half, u_half[::-1]))
                    rot = np.concatenate((rot_half, rot_half[::-1]))
                u_half = -((self.w * self.L ** 4) / (2 * self.E * self.I)) * (
                        self.L * (3 * self.L + 4 * x) / (2 * (self.L + x) ** 2) + np.log(self.L + x) - 1.5 - np.log(
                    self.L) - \
                        x / (8 * self.L)) - ((self.w * self.L ** 2) / (self.G * self.A * self.k)) * (
                                 2 * self.L / (self.L + x) + np.log(self.L + x) - 2 - np.log(self.L))
                rot_half = -((self.w * self.L ** 4) / (2 * self.E * self.I)) * (x ** 2 / (self.L + x) ** 3) + (
                        (self.w * self.L ** 3) / (16 * self.E * self.I))
                u = u_half
                rot = rot_half
                print(rot_half)
                sys.exit()

            elif problem[3] == "ParabolicLoad":
                u = (4 * self.w / (self.E * self.I * self.L ** 2)) * (-x ** 6 / 360 + self.L * x ** 5 / 120 - self.L ** 3 * x ** 3 / 72 + self.L ** 5 * x / 120)\
                    + (4 * self.w / (self.k * self.G * self.A * self.L ** 2)) * (x ** 4 / 12 - self.L * x ** 3 / 6 + self.L ** 3 * x / 12)
                rot = (4 * self.w / (self.E * self.I * self.L ** 2)) * (-x ** 5 / 60 + self.L * x ** 4 / 24 - self.L ** 3 * x ** 2 / 24 + self.L ** 5 / 120)
            else:
                u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 3 * x) + \
                    (self.w) / (2 * self.k * self.G * self.A) * (-x ** 2 + self.L * x)
                rot = (self.w / (24 * self.E * self.I)) * (4 * x ** 3 - 6 * self.L * x ** 2 + self.L ** 3)
        elif problem[1] == "fixed" and problem[2] == "free":
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 4 * self.L * x ** 3 + 6 * self.L ** 2 * x ** 2) + (
                        self.w / (2 * self.k * self.G * self.A)) * (-x ** 2 + 2 * self.L * x)
            rot = (self.w / (6 * self.E * self.I)) * (x ** 3 - 3 * self.L * x ** 2 + 3 * self.L ** 2 * x)
        elif problem[1] == "fixed" and problem[2] == "fixed":
            C1 = -(self.w * self.L * (self.k * self.G * self.A * self.L ** 2 + 12 * self.E * self.I)) / (
                        24 * self.E * self.I + 2 * self.k * self.G * self.A * self.L ** 2);
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 2 * x ** 2) + (
                        self.w / (2 * self.k * self.G * self.A)) * (-x ** 2 + self.L * x)
            rot = (1 / (6 * self.E * self.I)) * (
                        self.w * x ** 3 + 3 * C1 * x ** 2 - self.w * self.L ** 2 * x - 3 * C1 * self.L * x)
        elif problem[1] == "fixed" and problem[2] == "pinned":
            C1 = -(5 * self.w * self.k * self.G * self.A * self.L ** 3 + 12 * self.w * self.L * self.E * self.I) / (
                        8 * (self.k * self.G * self.A * self.L ** 2 + 3 * self.E * self.I))
            u = (1 / (self.E * self.I)) * (
                        self.w * x ** 4 / 24 + C1 * x ** 3 / 6 - (self.w * self.L ** 2 / 4) * x ** 2 - (
                            C1 * self.L / 2) * x ** 2) - (self.w / (2 * self.k * self.G * self.A)) * x ** 2 - (
                                C1 * x) / (self.k * self.G * self.A)
            rot = (1 / (6 * self.E * self.I)) * (
                        self.w * x ** 3 + 3 * C1 * x ** 2 - 3 * self.w * self.L ** 2 * x - 6 * C1 * self.L * x)

        return [x, u, rot]

    def additional_data(self, x, problem):
        """
         Method to add data from the reference solution to the model training data.
         The reference solution contains the target values for the predictions
         Ex: analytical solution, other numerical results with great accuracy, experimental data, etc

         DISCLAIMER: this might be unused, since this part of the code is not working yet
        """
        # Aditional data based on the reference solution for the predictions ======================================
        # x = np.linspace(0, self.L, int(problem[3]))
        if problem[1] == "pinned" and problem[2] == "pinned":
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 3 * x) + \
                (self.w) / (2 * self.k * self.G * self.A) * (-x ** 2 + self.L * x)
            rot = (self.w / (24 * self.E * self.I)) * (4 * x ** 3 - 6 * self.L * x ** 2 + self.L ** 3)
        elif problem[1] == "fixed" and problem[2] == "free":
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 4 * self.L * x ** 3 + 6 * self.L ** 2 * x ** 2) + (
                        self.w / (2 * self.k * self.G * self.A)) * (-x ** 2 + 2 * self.L * x)
            rot = (self.w / (6 * self.E * self.I)) * (x ** 3 - 3 * self.L * x ** 2 + 3 * self.L ** 2 * x)
        elif problem[1] == "fixed" and problem[2] == "fixed":
            C1 = -(self.w * self.L * (self.k * self.G * self.A * self.L ** 2 + 12 * self.E * self.I)) / (
                        24 * self.E * self.I + 2 * self.k * self.G * self.A * self.L ** 2);
            u = (self.w / (24 * self.E * self.I)) * (x ** 4 - 2 * self.L * x ** 3 + self.L ** 2 * x ** 2) + (
                        self.w / (2 * self.k * self.G * self.A)) * (-x ** 2 + self.L * x)
            rot = (1 / (6 * self.E * self.I)) * (
                        self.w * x ** 3 + 3 * C1 * x ** 2 - self.w * self.L ** 2 * x - 3 * C1 * self.L * x)
        elif problem[1] == "fixed" and problem[2] == "pinned":
            C1 = -(5 * self.w * self.k * self.G * self.A * self.L ** 3 + 12 * self.w * self.L * self.E * self.I) / (
                        8 * (self.k * self.G * self.A * self.L ** 2 + 3 * self.E * self.I))
            u = (1 / (self.E * self.I)) * (
                        self.w * x ** 4 / 24 + C1 * x ** 3 / 6 - (self.w * self.L ** 2 / 4) * x ** 2 - (
                            C1 * self.L / 2) * x ** 2) - (self.w / (2 * self.k * self.G * self.A)) * x ** 2 - (
                                C1 * x) / (self.k * self.G * self.A)
            rot = (1 / (6 * self.E * self.I)) * (
                        self.w * x ** 3 + 3 * C1 * x ** 2 - 3 * self.w * self.L ** 2 * x - 6 * C1 * self.L * x)

        # u_add = np.tile(u, int(self.num_training_samples / problem[3]))
        # rot_add = np.tile(rot, int(self.num_training_samples / problem[3]))
        # x_add = np.tile(x, int(self.num_training_samples / problem[3]))
        self.u_add = u
        self.rot_add = rot
        self.x_add = x

        return [x, u, rot]
        # return x_add, u_add, rot_add

    def plotting(self, x_test, u_ref, u_test, rot_ref, rot_test):
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
        # plt.savefig('tk_varsec_0.001_32_300.pdf')

        plt.show()