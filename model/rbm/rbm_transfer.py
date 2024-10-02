import torch
import copy
import sys
import numpy as np
import os
import pickle
import itertools


class RBMTransfer(object):
    """
    handle the transfer for multilayer perceptron model
    """

    def __init__(self, model_target, graph_target, base_model_path, base_model_number=None, divide=1.):
        """
        Initialize an RBM transfer object.
        Args:
            model_target: the target model that we want to transfer the params
            graph_target: the graph of the target task
            base_model_path: the path of the base model that will be transferred
            base_model_number: the number of the base model that wants to be transferred (number of experiments, default: None, we get the latest model)
            divide: to divide the parameters by some values to avoid nan
        """
        self.model_target = model_target
        self.graph_target = graph_target
        self.base_model_path = base_model_path
        self.base_model_number = base_model_number
        self.divide = divide
        self.initialize()

    def initialize(self):
        """
        Initialize the transfer
        """
        # Get the base model from the path
        self.learner_base = self.get_base_model()
        self.model_base = self.learner_base.model
        self.graph_base = self.learner_base.hamiltonian.graph

        # Initialize the transferred weight and biases from the target model
        self.W_transfer = self.model_target.W_array
        self.bv_transfer = self.model_target.bv_array
        self.bh_transfer = self.model_target.bh_array

        self.params_base = self.model_base.get_parameters()

        # Divide parameter of the base network
        self.W_base = self.params_base[0] / self.divide
        self.bv_base = self.params_base[1] / self.divide
        self.bh_base = self.params_base[2] / self.divide

    def get_base_model(self):
        """
        Load the base learner model.
        Return:
            base learner model

        """
        # If base model number is not specified, get the latest trained model
        if self.base_model_number is None:
            dir_names = [int(f) for f in os.listdir(
                self.base_model_path) if os.path.isdir(self.base_model_path + f)]
            self.base_model_number = max(dir_names)

        # Load the base model
        self.base_model_path = '%s/%d/model.p' % (
            self.base_model_path, self.base_model_number)
        base_model = pickle.load(open(self.base_model_path, 'rb'))
        return base_model

    def cutpaste(self):
        """
        If the base model and target model is the same, we just cut and paste the parameters for the transfer.
        """
        assert self.model_target.num_visible >= self.model_base.num_visible, "Number of visible node in the model must be larger than or equal to the number of visible node in the base model!"
        self.model_target.set_parameters(self.params_base)

    def expand_hidden(self, k_val):
        """
        Transfer for expanding hidden layer.
        """

        half_col = self.W_transfer.shape[1] // 2
        multiplier = self.W_transfer.shape[1] // self.W_base.shape[1]

        self.bv_transfer = self.bv_base

        if k_val == 0:
            self.W_transfer[:, :half_col] = self.W_base
            self.bh_transfer[:, :half_col] = self.bh_base
        elif k_val == 1:
            self.W_transfer = np.repeat(self.W_base, multiplier, 1)
            self.bh_transfer = np.repeat(self.bh_base, multiplier, 1)
        else:
            self.W_transfer = np.tile(self.W_base, (1, multiplier))
            self.bh_transfer = np.tile(self.bh_base, (1, multiplier))

        self.model_target.set_parameters(
            [self.W_transfer, self.bv_transfer, self.bh_transfer])

    def tiling(self, k_val):
        assert self.model_target.num_visible >= self.model_base.num_visible and self.model_target.num_visible % self.model_base.num_visible == 0, "Number of visible node in the model must be larger than or equal to and divisible by the number of visible node in the base model!"
        assert self.graph_base.length % k_val == 0, "k must be divisible by the number of visible node in base model!"

        p_val = int(self.graph_target.length / self.graph_base.length)

        base_coor = []
        for point in range(self.graph_base.num_points):
            old_coor = np.array(self.graph_base._point_to_coordinate(point))

            new_coor = (old_coor // k_val) * \
                (k_val * p_val) + (old_coor % k_val)

            to_iter = []
            for dd in range(self.graph_target.dimension):
                temp = []
                for pp in range(p_val):
                    temp.append(new_coor[dd] + pp * k_val)
                to_iter.append(temp)

            new_coordinates = []
            if self.graph_target.dimension == 1:
                new_coordinates = [[a] for a in to_iter[0]]
            else:
                for kk in to_iter:
                    if len(new_coordinates) == 0:
                        new_coordinates = kk
                    else:
                        new_coordinates = [list(cc[0] + [cc[1]]) if isinstance(cc[0], list) else list(
                            cc) for cc in list(itertools.product(new_coordinates, kk))]

            for coord in new_coordinates:
                quadrant = [int(c / self.graph_base.length) for c in coord]
                hid_pos = 0
                for ddd in range(self.graph_base.dimension):
                    hid_pos += quadrant[ddd] * (p_val ** ddd)

                target_point = self.graph_target._coordinate_to_point(coord)

                self.W_transfer[int(target_point), int(hid_pos * self.W_base.shape[1]):int(
                    (hid_pos + 1) * self.W_base.shape[1])] = self.W_base[point, :]

        if k_val == 1:
            ind = np.array([[a, a] for a in range(
                0, self.bv_base.shape[1], 1)]).flatten()
            self.bv_transfer = self.bv_base[:, ind]
            ind = np.array([[a, a] for a in range(
                0, self.bh_base.shape[1], 1)]).flatten()
            self.bh_transfer = self.bh_base[:, ind]
        elif k_val == 2:
            ind = np.array([[a, a+1, a, a+1]
                           for a in range(0, self.bv_base.shape[1], 2)]).flatten()
            self.bv_transfer = self.bv_base[:, ind]
            ind = np.array([[a, a+1, a, a+1]
                           for a in range(0, self.bh_base.shape[1], 2)]).flatten()
            self.bh_transfer = self.bh_base[:, ind]
        elif k_val == 4:
            ind = np.array([[a, a+1, a+2, a+3, a, a+1, a+2, a+3]
                           for a in range(0, self.bv_base.shape[1], 4)]).flatten()
            self.bv_transfer = self.bv_base[:, ind]
            ind = np.array([[a, a+1, a+2, a+3, a, a+1, a+2, a+3]
                           for a in range(0, self.bh_base.shape[1], 4)]).flatten()
            self.bh_transfer = self.bh_base[:, ind]
        else:
            self.bv_transfer = np.concatenate((self.bv_base, self.bv_base), 1)
            self.bh_transfer = np.concatenate((self.bh_base, self.bh_base), 1)

        self.model_target.set_parameters(
            [self.W_transfer, self.bv_transfer, self.bh_transfer])

    def tiling_sample(self, k_val):
        """ 
            Tiling sample for sample transfer
        """
        if hasattr(self.learner_base, 'samples_observ'):
            samples = self.learner_base.samples_observ.numpy()
        else:
            samples = self.learner_base.samples.numpy()
        multiplier = self.model_target.num_visible // self.model_base.num_visible

        if k_val == 0:
            tiled_samples = samples
        elif k_val == 1:
            ind = np.array([[a, a]
                           for a in range(0, samples.shape[1], 1)]).flatten()
            tiled_samples = samples[:, ind]
        elif k_val == 2:
            ind = np.array([[a, a+1, a, a+1]
                           for a in range(0, samples.shape[1], 2)]).flatten()
            tiled_samples = samples[:, ind]
        elif k_val == 4:
            ind = np.array([[a, a+1, a+2, a+3, a, a+1, a+2, a+3]
                           for a in range(0, samples.shape[1], 4)]).flatten()
            tiled_samples = samples[:, ind]
        else:
            tiled_samples = np.concatenate((samples, samples), 1)

        return tiled_samples

    def decimate(self, k_val):
        """
            Decimate parameters to transfer from large to small system.
        """
        if k_val == 0:
            ind = np.arange(self.model_target.num_visible)
            ind_hidden = np.arange(self.model_target.num_hidden)
        elif k_val == 1:
            ind = np.array(
                [a for a in range(0, self.model_base.num_visible, 2)]).flatten()
            ind_hidden = np.array(
                [a for a in range(0, self.model_base.num_hidden, 2)]).flatten()
        else:
            ind = np.arange(0, self.model_base.num_visible, 4)
            ind_hidden = np.arange(0, self.model_base.num_hidden, 4)
        self.W_transfer = self.W_base[ind, :][:, ind_hidden]
        self.bv_transfer = self.bv_base[:, ind]
        self.bh_transfer = self.bh_base[:, ind_hidden]
        self.model_target.set_parameters(
            [self.W_transfer, self.bv_transfer, self.bh_transfer])
