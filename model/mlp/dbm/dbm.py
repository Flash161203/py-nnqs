import matplotlib.pyplot as plt
from model.mlp import MLP
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')


class DBM(MLP):
    """
    This class is used to define Deep Boltzmann Machine with real and
    positive wavefunction.
    """

    def __init__(self, num_visible, num_hidden=[256], num_expe=None, use_bias=True, freeze_layer=[]):
        """
        Construct a multilayer perceptron model for real positive wavefunction.

        Args:
            num_visible: number of input nodes in the input layer.
            num_hidden: number of hidden nodes in the hidden layer.
            num_expe: number of experiments to determine the seed.
        """

        super().__init__(num_visible, num_hidden)
        self.use_bias = use_bias
        self.freeze_layer = freeze_layer
        self.num_expe = num_expe

        if num_expe is not None:
            np.random.seed(num_expe)
            torch.manual_seed(num_expe)

        self.build_model()

    def build_model(self):
        """
        Create the model with PyTorch
        """
        layers = []
        in_features = self.num_visible

        for ii in range(self.num_layer):
            layers.append(
                nn.Linear(in_features, self.num_hidden[ii], bias=self.use_bias))
            layers.append(nn.Lambda(lambda x: torch.log(torch.cosh(x))))
            in_features = self.num_hidden[ii]

        self.model = nn.Sequential(*layers)

    def log_val(self, x):
        """
            Calculate log(\Psi(x))
            Args:
                x: the input x
        """
        return torch.sum(self.model(x), axis=1, keepdim=True) / self.num_visible

    def log_val_diff(self, xprime, x):
        """
            Calculate log(\Psi(x')) - log(\Psi(x))
            Args:
                xprime: x'
                x: x
        """
        log_val_1 = self.log_val(xprime)
        log_val_2 = self.log_val(x)
        return log_val_1 - log_val_2

    def derlog(self, x):
        """
            Calculate $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW)$ where W can be the weights or the biases.
        """
        output = torch.exp(self.log_val(x))
        output.backward(torch.ones_like(output))
        gradients = [param.grad for param in self.model.parameters()]

        gradients_new = []
        for ii, grad in enumerate(gradients):
            if ii in self.freeze_layer:
                grad = grad * 0.

            # reshape so it can be divided by output
            grad = grad.reshape(grad.shape[0], -1) / output
            gradients_new.append(grad)

        return gradients_new

    def get_parameters(self):
        """
        Get the parameters for this model
        """
        if self.model is None:
            return self.params
        else:
            return [param.data for param in self.model.parameters()]

    def set_parameters(self, params):
        """
        Set the parameters for this model for transfer learning or loading model purposes
        Args:
            params: the parameters to be set.
        """
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), params):
                param.copy_(torch.tensor(new_param))

    def param_difference(self, first_param, last_param):
        """
        Calculate the difference between two parameters.
        This is equal to the sum of the mean squared difference of all parameters (weights and biases).
        """
        sum_diff = 0.
        for (par1, par2) in zip(first_param[1], last_param[1]):
            sum_diff += np.mean((par1 - par2) ** 2)

        return sum_diff

    def visualize_param(self, params, path):
        """
        Visualize every parameter.
        Args:
            params: the parameters to visualize
            path: the path to save the visualization
        """
        epoch = params[0]
        for ii, param in enumerate(params[1]):
            # Reshape for bias
            if len(param.shape) == 1:
                param = np.reshape(param, (param.shape[0], 1))

            plt.figure()
            if ii % 2 == 0:
                plt.title("Weight layer %d at epoch %d" % (ii + 1, epoch))
            else:
                plt.title("Bias layer %d at epoch %d" % (ii + 1, epoch))
            plt.imshow(param, cmap='hot', interpolation='nearest')
            plt.xticks(np.arange(0, param.shape[1], 1.0))
            plt.yticks(np.arange(0, param.shape[0], 1.0))
            plt.colorbar()
            plt.tight_layout()
            if ii % 2 == 0:
                plt.savefig(path + '/weight-layer-%d-%d.png' % (ii+1, epoch))
            else:
                plt.savefig(path + '/bias-layer-%d-%d.png' % (ii+1, epoch))
            plt.close()

    def get_name(self):
        """
        Get the name of the model
        """
        hidden_layer_str = '-'.join([str(hid) for hid in self.num_hidden])
        return 'rbm-%s' % (hidden_layer_str)

    def make_pickle_object(self):
        """
        PyTorch object cannot be pickled so needs to be handled
        save the last param first and make it none
        """
        self.params = self.get_parameters()
        self.model = None
        self.activation = None

    def __str__(self):
        hidden_layer_str = '-'.join([str(hid) for hid in self.num_hidden])
        return 'RBM %s' % (hidden_layer_str)

    def to_xml(self):
        stri = ""
        stri += "<model>\n"
        stri += "\t<type>DBM</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<num_visible>%d</num_visible>\n" % self.num_visible
        stri += "\t\t<num_hidden>%s</num_hidden>\n" % self.num_hidden
        stri += "\t\t<num_expe>%s</num_expe>\n" % self.num_expe
        stri += "\t</params>\n"
        stri += "</model>\n"
        return stri
