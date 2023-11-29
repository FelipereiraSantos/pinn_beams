# ==============================================================================
# Copyright 2021 SciANN -- Ehsan Haghighat.
# All Rights Reserved.
#
# Licensed under the MIT License.
#
# A guide for generating collocation points for PINN solvers.
#
# Includes:
#    - DataGeneratorX:
#           Generate 1D collocation grid.
#    - DataGeneratorXY:
#           Generate 2D collocation grid for a rectangular domain.
#    - DataGeneratorXT:
#           Generate 1D time-dependent collocation grid.
#    - DataGeneratorXYT:
#           Generate 2D time-dependent collocation grid  for a rectangular domain.
# ==============================================================================
import sys

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

cycol = cycle('bgrcmk')


class DataGenerator1D:
    """ Generates 1D collocation grid for training PINNs
    # Arguments:
      X: [X0, X1]
      targets: list and type of targets you wish to impose on PINNs.
          ('domain', 'bc-left', 'bc-right', 'all')
      num_sample: total number of collocation points.
    # Examples:
      >> dg = DataGeneratorX([0., 1.], ["domain", "bc-left", "bc-right"], 10000)
      >> input_data, target_data = dg.get_data()
    """

    def __init__(self,
                 X,
                 targets,
                 num_sample,
                 data):
        'Initialization'
        self.Xdomain = X
        self.targets = targets
        self.num_sample = num_sample
        self.input_data = None
        self.target_data = None
        self.set_data(data)

    def __len__(self):
        return self.input_data[0].shape[0]

    def set_data(self, data):
        self.input_data, self.target_data = self.get_inputs_with_data(data)
        # self.input_data, self.target_data = self.generate_data()

    def get_data(self):
        return self.input_data, self.target_data

    def generate_data_general(self):
        num_support = 3
        # distribute half inside domain half on the boundary
        num_sample = int(self.num_sample / 2)

        counter = 0
        # domain points
        x_dom = np.linspace(self.Xdomain[0], self.Xdomain[(num_support - 1)], num_sample, endpoint=False)
        # x_dom = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
        ids_dom = np.arange(x_dom.shape[0])
        counter += ids_dom.size

        # Supports
        supports = []
        ids_sup = []
        points = int(num_sample / num_support)
        for i in range(0, num_support):
            # left bc points
            x_i = np.full(points, self.Xdomain[i])
            ids_i = np.arange(x_i.shape[0]) + counter
            counter += ids_i.size

            supports.append(x_i)
            ids_sup.append()

        # # right bc points
        # x_bc_right = np.full(num_sample - int(num_sample / 2), self.Xdomain[1])
        # ids_bc_right = np.arange(x_bc_right.shape[0]) + counter
        # counter += ids_bc_right.size

        #
        # ids_bc = np.concatenate([ids_bc_left, ids_bc_right])
        # ids_all = np.concatenate([ids_dom, ids_bc])

        ids_bc = np.concatenate([ids_sup])
        ids_all = np.concatenate([ids_dom, ids])

        ids = {
            'domain': ids_dom,
            'bc-left': ids_bc_left,
            'bc-right': ids_bc_right,
            'bc': ids_bc,
            'supports': ids_sup,
            'all': ids_all
        }

        assert all([t in ids.keys() for t in self.targets]), \
            'accepted target types: {}'.format(ids.keys())

        input_data = [
            np.concatenate([x_dom, supports]).reshape(-1, 1),
        ]
        total_sample = input_data[0].shape[0]

        target_data = []
        for i, tp in enumerate(self.targets):
            target_data.append(
                (ids[tp], 'zeros')
            )

        return input_data, target_data

    def generate_data(self):
        # distribute half inside domain half on the boundary
        num_sample = int(self.num_sample / 2)

        counter = 0
        # domain points
        x_dom = np.linspace(self.Xdomain[0], self.Xdomain[1], num_sample, endpoint=False)
        # x_dom = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
        ids_dom = np.arange(x_dom.shape[0])
        counter += ids_dom.size

        # left bc points
        x_bc_left = np.full(int(num_sample / 2), self.Xdomain[0])
        ids_bc_left = np.arange(x_bc_left.shape[0]) + counter
        counter += ids_bc_left.size

        # right bc points
        x_bc_right = np.full(num_sample - int(num_sample / 2), self.Xdomain[1])
        ids_bc_right = np.arange(x_bc_right.shape[0]) + counter
        counter += ids_bc_right.size

        # Aditional data from a reference solution (numerical, experimental, etc)


        ids_bc = np.concatenate([ids_bc_left, ids_bc_right])
        ids_all = np.concatenate([ids_dom, ids_bc])

        ids = {
            'domain': ids_dom,
            'bc-left': ids_bc_left,
            'bc-right': ids_bc_right,
            'bc': ids_bc,
            'all': ids_all
        }

        assert all([t in ids.keys() for t in self.targets]), \
            'accepted target types: {}'.format(ids.keys())

        input_data = [
            np.concatenate([x_dom, x_bc_left, x_bc_right]).reshape(-1, 1),
        ]
        total_sample = input_data[0].shape[0]

        target_data = []
        for i, tp in enumerate(self.targets):
            target_data.append(
                (ids[tp], 'zeros')
            )

        return input_data, target_data

    def get_inputs_with_data(self, data):
        # distribute half inside domain half on the boundary
        num_sample = int(self.num_sample / 2)
        dom_sample = int(1 * num_sample)

        counter = 0
        # domain points
        x_dom = np.linspace(self.Xdomain[0], self.Xdomain[1], dom_sample, endpoint=False)
        # x_dom = np.linspace(self.Xdomain[0], self.Xdomain[1], num_sample, endpoint=False)
        # x_dom = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
        ids_dom = np.arange(x_dom.shape[0])
        counter += ids_dom.size

        # left bc points
        x_bc_left = np.full(int(num_sample / 2), self.Xdomain[0])
        ids_bc_left = np.arange(x_bc_left.shape[0]) + counter
        counter += ids_bc_left.size

        # right bc points
        x_bc_right = np.full(num_sample - int(num_sample / 2), self.Xdomain[1])
        ids_bc_right = np.arange(x_bc_right.shape[0]) + counter
        counter += ids_bc_right.size

        # Aditional data from a reference solution (numerical, experimental, etc)
        data_sample = 3*num_sample

        x = data[0]
        u = data[1]
        rot = data[2]

        x_add = np.tile(x, int(data_sample/u.shape[0]))
        u_add = np.tile(u, int(data_sample/u.shape[0]))
        rot_add = np.tile(rot, int(data_sample/u.shape[0]))

        data_add = [x_add, u_add, rot_add]

        ids_data = np.arange(x_add.shape[0]) + counter

        ids_bc = np.concatenate([ids_bc_left, ids_bc_right])
        ids_all = np.concatenate([ids_dom, ids_bc, ids_data])
        # ids_all = np.concatenate([ids_dom, ids_bc])

        ids = {
            'domain': ids_dom,
            'bc-left': ids_bc_left,
            'bc-right': ids_bc_right,
            'bc': ids_bc,
            'all': ids_all,
            'data': ids_data
        }

        assert all([t in ids.keys() for t in self.targets]), \
            'accepted target types: {}'.format(ids.keys())

        input_data = [
            # np.concatenate([x_dom, x_add, x_bc_left, x_bc_right]).reshape(-1, 1)
            np.concatenate([x_dom, x_bc_left, x_bc_right, x_add]).reshape(-1, 1),
        ]
        total_sample = input_data[0].shape[0]

        target_data = []
        j = 1
        for i, tp in enumerate(self.targets):
            out = 'zeros'
            if i > (len(self.targets) - 3):
                out = data_add[j].reshape(-1, 1)
                j =+ 1
            target_data.append(
                (ids[tp], out)
            )
        # target_data.append(u_add)

        # print("Input: ", input_data)
        # print("Target: ", target_data)
        # sys.exit()
        # for i, tp in enumerate(self.targets):
        #     out = 'zeros'
        #     if tp == 'data':
        #         out = u_add
        #     target_data.append(
        #         out
        #     )

        return input_data, target_data

    def get_test_grid(self, Nx=1000):
        xs = np.linspace(self.Xdomain[0], self.Xdomain[1], Nx)
        return xs

    def plot_sample_batch(self, batch_size=500):
        ids = np.random.choice(len(self), batch_size, replace=False)
        x_data = self.input_data[0][ids, :]
        y_data = np.random.uniform(-.1, .1, x_data.shape)
        plt.scatter(x_data, y_data)
        plt.xlabel('x')
        plt.ylabel('Random vals')
        plt.ylim(-1, 1)
        plt.title('Sample batch = {}'.format(batch_size))
        plt.show()

    def plot_data(self):
        fig = plt.figure()
        for t, (t_idx, t_val) in zip(self.targets, self.target_data):
            x_data = self.input_data[0][t_idx, :]
            y_data = np.random.uniform(-.1, .1, x_data.shape)
            plt.scatter(x_data, y_data, label=t, c=next(cycol))
        plt.ylim(-1, 1)
        plt.xlabel('x')
        plt.ylabel('Random vals')
        plt.title('Training Data')
        plt.legend(title="Training Data", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        plt.show()


