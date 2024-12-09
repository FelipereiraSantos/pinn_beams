import sys
from numerical_results import NumericalResults
import tensorflow as tf
import os

def implementation():
    num_results = 'on'  # on or off
    mesh = [5, 11, 17, 21]
    if num_results == 'on':
        k = 0
        initial_features, problems, model_parameters, neural_nets, optimizers, epochs, bc_sizes, file_name = NumericalResults.file_reading()
        for j, problem in enumerate(problems):
            m_parameters = model_parameters[j]
            for i, net in enumerate(neural_nets):
                # File name to use as the file
                file_name = file_name + str(k + 1)
                gs = NumericalResults(initial_features, m_parameters, net, optimizers, epochs, bc_sizes, file_name, problem[1], mesh)
                gs.training()
                print("\nThe training of the following neural network has ended: " + str(net[2]))
            print("The following model is trained: ", (k + 1))
        print("\nThe numerical results are available!")
    # sys.exit()

if __name__ == '__main__':

    # Disabling the gpu for tensorflow, or forcing the use of cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Check if GPU is available
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Configurando o dispositivo de processamento
    if tf.test.is_gpu_available():
        with tf.device('/device:GPU:0'):
          implementation()

    else:
        print("GPU is not available. Using CPU instead.")
        implementation()
