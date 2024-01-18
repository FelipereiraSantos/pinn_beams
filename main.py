import sys
from numerical_results import NumericalResults
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

def implementation():
    num_results = 'on'  # on or off
    mesh = [5, 11, 17, 21]

    if num_results == 'on':
        k = 0
        initial_features, problems, model_parameters, neural_nets, optimizers, epochs, bc_sizes = NumericalResults.file_reading()
        for j, problem in enumerate(problems):
            # bc_s = []  # Batch size list to use an adaptative batch size approach
            # bc_s.append(bc_sizes[j])
            # print(bc_s)
            k = k + 1
            m_parameters = model_parameters[j]
            for i, net in enumerate(neural_nets):
                # print("\nThe training of the following neural network has started: " + str(net[0]))
                # file_name = (problem[1])[0] + problem[0] + str(net[0]) + str(m_parameters[5]) + "_" + str(k)
                # file_name = 'Tk_statics_ffr_q_' + str(m_parameters[6]) + '_' + str(k) + str(net[i])
                file_name = 'Tk_ffr_q_ParabolicShape_error_' + str(k)
                gs = NumericalResults(initial_features, m_parameters, net, optimizers, epochs, bc_sizes, file_name, problem[1], mesh)
                gs.training()
                print("\nThe training of the following neural network has ended: " + str(net[2]))
            print("The following model is trained: ", k)
        print("\nThe numerical results are available!")
    sys.exit()

if __name__ == '__main__':
    # print(device_lib.list_local_devices())
    # np.random.seed(1)

    # Disabling the gpu for tensorflow
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Check if GPU is available
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # Verifique se a GPU está disponível
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Configurando o dispositivo de processamento
    if tf.test.is_gpu_available():
        with tf.device('/device:GPU:0'):
          implementation()

    else:
        print("GPU is not available. Using CPU instead.")
        implementation()


    # problem = ["Tk_bending", "fixed", "pinned", "nada", "nadaparabolic", "nIparabolic"]  # EB_dynamics, Tk_bending, EB_stability, Tk_continuous_bending, EB_Stability_secvar
    # size = 6
    # L = 2
    # x = np.linspace(-L, L, size)
    # print("x: ",x)
    # size2 = int(size / 2)
    # func = x ** 2
    # print("func: ", func)
    # if x.size % 2 > 0:
    #     x2 = x[:(size2 + 1)]
    #     func2 = x2 ** 2
    # else:
    #     x2 = x[:size2]
    #     func2 = x2 ** 2
    # print("x2: ", x2)
    # print("func2: ", func2)
    #
    # if x.size % 2 > 0:
    #     func3 = np.concatenate((func2, func2[:-1][::-1]))
    # else:
    #     func3 = np.concatenate((func2, func2[::-1]))
    #
    # print("func3: ", func3)
    #
    # sys.exit()


    # num_results = 'on'  # on or off
    # mesh = [5, 11, 17, 21]
    # if num_results == 'on':
    #     initial_features, problems, model_parameters, neural_nets, optimizers, epochs, bc_sizes = NumericalResults.file_reading()
    #     for j, problem in enumerate(problems):
    #         m_parameters = model_parameters[j]
    #         for i, net in enumerate(neural_nets):
    #             # print("\nThe training of the following neural network has started: " + str(net[0]))
    #             file_name = (problem[1])[0] + problem[0] + str(net[0])
    #             gs = NumericalResults(initial_features, m_parameters, net, optimizers, epochs, bc_sizes, file_name, problem[1], mesh)
    #             gs.training()
    #             i = i + 1
    #             print("\nThe training of the following neural network has ended: " + str(net[0]))
    #         print("\nThe numerical results are available!")
    #         sys.exit()
    #


    # gr_search = 'off'  # on or off
    # if gr_search == 'on':
    #     neural_nets, optimizers, epochs, bc_sizes = GridSearch.grid_search()
    #     i = 1
    #     for net in neural_nets:
    #         file_name = problem[0] + '_fp_q_Grid' + '_net' + str(i)
    #         pmodel = PreProcessor.initialize_model(problem, net)
    #         gs = GridSearch(net, optimizers, epochs, bc_sizes, pmodel, file_name)
    #         gs.training()
    #         i = i + 1

    # print("Pcr: {}".format(pmodel.P.value))
    # print("Alpha: {}".format(pmodel.alpha.value))

   # dynamics analysis ==============================
    # x_test = pmodel.x_test
    # t_test = pmodel.t_test
    # u_test = pmodel.u.eval([x_test, t_test])
    # rot_test = pmodel.rot.eval([x_test, t_test])

    # pmodel.plotting(t_test[:, -1], u_test[:, -1])

    # x_test = pmodel.x_test
    # u_test = pmodel.u.eval([x_test])
    # rot_test = pmodel.rot.eval([x_test])
    # # pmodel.plotting(x_test, pmodel.ref_solu[0], u_test, pmodel.ref_solu[1], rot_test)
    #
    # # Plotting the results
    # results = [x_test, pmodel.ref_solu[0], u_test, pmodel.ref_solu[1], rot_test]
    # if problem[1] == "varying_sec":
    #     x_test = np.linspace(0, 2 * pmodel.L, 2 * pmodel.num_test_samples)
    #     u_ref = np.concatenate([pmodel.ref_solu[0], pmodel.ref_solu[0][::-1]])
    #     rot_ref = np.concatenate([pmodel.ref_solu[1], pmodel.ref_solu[1][::-1]])
    #     u_test = np.concatenate([u_test, u_test[::-1]])
    #     rot_test = np.concatenate([rot_test, rot_test[::-1]])
    #     results = [x_test, u_ref, u_test, rot_ref, rot_test]
    # PostProcessor.plotting(pmodel, results)



