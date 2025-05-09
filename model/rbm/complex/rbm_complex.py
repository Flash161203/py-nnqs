import matplotlib.pyplot as plt
from model.rbm import RBM
import torch
import copy
from functools import partial
import numpy as np
import matplotlib
matplotlib.use('Agg')


class RBMComplex(RBM):
    """
    This class is used to define a restricted Boltzmann machine with complex weights for complex wavefunction.

    $\Psi(x) = e^{ax} \times \Pi_{j=1}^{H} \cosh(W_jx + b_j)$

    where a = bv is the visible bias, b = bh is the hidden bias, and H is the number of hidden nodes.
    """

    def __init__(self, num_visible, density=2, initializer=np.random.randn, use_bias=True, num_expe=None, freeze_pos=None, freeze_pos2=None):
        """
        Construct an RBM model for real positive wavefunction
        Args:
            num_visible: number of visible
            density: the number hidden layer define as density * num_visible
            initializer: the initialization of the weights
            use_bias: use bias or not
            num_expe: number of experiment to determine the seed
        """
        super().__init__(num_visible, density)
        self.initializer = initializer
        self.use_bias = use_bias
        self.num_expe = num_expe

        self.freeze_pos = freeze_pos
        self.freeze_pos2 = freeze_pos2

        if num_expe is not None:
            np.random.seed(num_expe)
            torch.manual_seed(num_expe)

        self.build_model()

    def build_model(self):
        """
        Build the RBM model
        """
        # Randomly initialize the weights and bias
        self.random_initialize()
        # Create the model
        self.create_variable()

    def random_initialize(self):
        """
        Randomly initialize an array based on the initializer. Biases array are initialized zero.
        """
        # Weights (W)
        self.W_array = self.initializer(size=(self.num_visible, self.num_hidden)) + \
            self.initializer(size=(self.num_visible, self.num_hidden)) * 1j
        # Visible bias (a)
        self.bv_array = np.zeros((1, self.num_visible))
        # Hidden bias (b)
        self.bh_array = np.zeros((1, self.num_hidden))

    def create_variable(self):
        """
        Create model by creating a parameters variable, which is the weight, visible bias, hidden bias 
        """
        self.W = torch.nn.Parameter(torch.tensor(
            self.W_array.astype(np.complex64)), requires_grad=True)
        self.bv = torch.nn.Parameter(torch.tensor(
            self.bv_array.astype(np.complex64)), requires_grad=True)
        self.bh = torch.nn.Parameter(torch.tensor(
            self.bh_array.astype(np.complex64)), requires_grad=True)
        self.model = self
        self.trainable_weights = [self.W, self.bv, self.bh]

    def parameters(self):
        """
        Get the parameters of the model
        """
        return self.trainable_weights

    def log_val(self, x):
        """
            Calculate log(\Psi(x)) = ax + \sum_{j=1}^H log(cosh(Wx + b)) 
            Args:
                x: the x
        """
        x = x.to(torch.complex64)

        # Calculate theta = Wx + b
        theta = torch.matmul(x, self.W) + self.bh

        # Calculate \sum_{j=1}^H log(cosh(Wx + b))
        sum_ln_thetas = torch.sum(
            torch.log(torch.cosh(theta)), dim=1, keepdim=True)

        # calculate ax
        ln_bias = torch.matmul(x, self.bv.T)

        return sum_ln_thetas + ln_bias

    def log_val_diff(self, xprime, x):
        """
            Calculate log(\Psi(x')) - log(\Psi(x))
            Args:
                xprime: x'
                x: x
        """
        xprime = xprime.to(torch.complex64)
        x = x.to(torch.complex64)
        log_val_1 = self.log_val(xprime)
        log_val_2 = self.log_val(x)
        return log_val_1 - log_val_2

    def derlog(self, x):
        """
        Calculate $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW)$ where W can be the weights or the biases.
        """
        sample_size = x.shape[0]
        x = x.to(torch.complex64)

        psix = torch.exp(self.log_val(x))

        # Calculate theta = Wx + b
        theta = torch.matmul(x, self.W) + self.bh

        if self.use_bias:
            D_bv = x
            D_bh = torch.tanh(theta)
        else:
            D_bv = x * 0.0
            D_bh = torch.tanh(theta) * 0.0

        # D_W(x) = x * tanh(Wx+b)
        D_w = torch.reshape(torch.tanh(theta), (sample_size, 1, self.num_hidden)
                            ) * torch.reshape(x, (sample_size, self.num_visible, 1))

        D_bv = torch.reshape(
            D_bv, (sample_size, 1, self.num_visible)) / psix.unsqueeze(2)
        D_bh = torch.reshape(
            D_bh, (sample_size, 1, self.num_hidden)) / psix.unsqueeze(2)

        if self.freeze_pos is not None:
            D_w = D_w.detach().numpy()
            D_w[self.freeze_pos] = 0.
            if self.freeze_pos2 is not None:
                D_w[self.freeze_pos2] = 0.

            D_w = torch.tensor(D_w)

        derlogs = [D_w, D_bv, D_bh]
        return derlogs

    def param_difference(self, first_param, last_param):
        """
        Calculate the difference between two parameters.
        This is equals to the sum of the mean squared difference of all parameters (weights and biases)
        """
        sum_diff = 0.
        for (par1, par2) in zip(first_param[1], last_param[1]):
            sum_diff += np.mean(np.abs(par1 - par2) ** 2)

        return sum_diff

    def visualize_param(self, params, path):
        """
        Visualize every parameters
        Args:
            params: the parameters that visualize
            path: the path to save the visualization
        """
        epoch = params[0]
        for ii, param in enumerate(params[1]):
            param = param.real
            plt.figure()
            if ii == 0:
                plt.title("Weight at epoch %d" % (epoch))
            elif ii == 1:
                plt.title("Visible Bias at epoch %d" % (epoch))
            elif ii == 2:
                plt.title("Hidden Bias at epoch %d" % (epoch))

            plt.imshow(param, cmap='hot', interpolation='nearest')
            plt.xticks(np.arange(0, param.shape[1], 1.0))
            plt.yticks(np.arange(0, param.shape[0], 1.0))
            plt.colorbar()
            plt.tight_layout()
            if ii == 0:
                plt.savefig(path + '/weight-layer-%d.png' % (epoch))
            elif ii == 1:
                plt.savefig(path + '/visbias-layer-%d.png' % (epoch))
            elif ii == 2:
                plt.savefig(path + '/hidbias-layer-%d.png' % (epoch))
            plt.close()

    def get_parameters(self):
        """
        Get the parameter of this model
        """
        return [self.W.detach().numpy(), self.bv.detach().numpy(), self.bh.detach().numpy()]

    def set_parameters(self, params):
        """
        Set the parameters for this model for transfer learning or loading model purposes
        Args:
            params: the parameters to be set.
        """
        self.W.data = torch.tensor(params[0])
        self.bv.data = torch.tensor(params[1])
        self.bh.data = torch.tensor(params[2])

    def get_name(self):
        """
        Get the name of the model
        """
        return 'rbmcomplex-%d' % (self.num_hidden)

    def make_pickle_object(self):
        """
        Make pickle object for RBM
        Nothing to do for RBM
        """
        pass

    def is_complex(self):
        return True

    def __str__(self):
        return 'RBMComplex %d' % (self.num_hidden)

    def to_xml(self):
        stri = ""
        stri += "<model>\n"
        stri += "\t<type>rbm_complex</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<num_visible>%d</num_visible>\n" % self.num_visible
        stri += "\t\t<num_hidden>%d</num_hidden>\n" % self.num_hidden
        stri += "\t\t<density>%d</density>\n" % self.density
        stri += "\t\t<initializer>%s</initializer>\n" % str(self.initializer)
        stri += "\t\t<use_bias>%s</use_bias>\n" % str(self.use_bias)
        stri += "\t\t<num_expe>%s</num_expe>\n" % str(self.num_expe)
        stri += "\t</params>\n"
        stri += "</model>\n"
        return stri
