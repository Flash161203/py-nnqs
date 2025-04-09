from model.mlp import MLP
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')


class MLPRealPos(MLP):
    """
    This class is used to define a multilayer perceptron with real and 
    positive wavefunction. 

    \Psi(x) = \sigma_N(...\sigma_2(\sigma_1(xW_1 + b_1)W_2 + b_2)....)

    where \sigma_i is the activation function of layer i, W_i is the weights, and
    b is the biases.

    For Cai and Liu paper (PhysRevB.97.035116):
        activation_output = sigmoid
        activation_hidden = tanh
    """

    def __init__(self, num_visible, num_hidden=[256], activation_hidden='ReLU', activation_output=None, num_expe=None, use_bias=True, freeze_layer=[], freeze_pos=None):
        """
        Construct a multilayer perceptron model for real positive wavefunction.

        Args:
            num_visible: number of input nodes in the input layer.
            num_hidden: number of hidden nodes in the hidden layer represented in an array.
            activation_hidden: the activation in the hidden layer.
            activation_output: the activation in the output layer.
            num_expe: number of experiments to determine the seed.
            use_bias: whether to use bias or not.
            freeze_layer: a list to freeze the weights or the biases.
                          where the index 0 and 1 refers to the weights and biases
                          from input layer to the first hidden layer, respectively,
                          and so on.
        """

        super().__init__(num_visible, num_hidden)
        self.activation_hidden = getattr(
            nn, activation_hidden)() if activation_hidden else None
        self.activation_output = getattr(
            nn, activation_output)() if activation_output else None
        self.use_bias = use_bias
        self.freeze_layer = freeze_layer
        self.num_expe = num_expe

        self.freeze_pos = freeze_pos

        # Set the same seed
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
            out_features = self.num_hidden[ii]
            layers.append(
                nn.Linear(in_features, out_features, bias=self.use_bias))

            if self.activation_hidden:
                layers.append(self.activation_hidden)

            in_features = out_features

        layers.append(nn.Linear(in_features, 1, bias=self.use_bias))
        if self.activation_output:
            layers.append(self.activation_output)

        self.model = nn.Sequential(*layers)

    def log_val(self, x):
        """
            Calculate log(\Psi(x))
            Args:
                x: the input x
        """
        return torch.log(self.model(x))

    def eval_log_val(self, x):
        """
            Evaluate log(\Psi(x)) without gradient tracking.
            Args:
                x: the input x
        """
        with torch.no_grad():
            return torch.log(self.model(x))

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

    '''
    def derlog(self, x):
        """
            Calculate $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW) = d \log \Psi(x) / dW$ where W can be the weights or the biases.
        """
        output = torch.exp(self.log_val(x))
        output.backward(torch.ones_like(output))
        gradients = [param.grad for param in self.model.parameters()]

        gradients_new = []
        for ii, grad in enumerate(gradients):
            if ii in self.freeze_layer:
                grad = grad * 0.

            # Normalize the gradients by output
            grad = grad / output

            if ii == 0 and self.freeze_pos is not None:
                grad = grad.detach().numpy()
                grad[self.freeze_pos] = 0.
                grad = torch.tensor(grad)

            gradients_new.append(grad)

        return gradients_new
    '''

    def derlog(self, x):
        # x is assumed to have shape (sample_size, num_visible)
        sample_size = x.shape[0]
        # Initialize a list of tensors to hold per-sample gradients for each parameter.
        per_sample_grads = [
            torch.zeros((sample_size,) + param.shape,
                        dtype=torch.float32, device=x.device)
            for param in self.model.parameters()
        ]
        # Loop over each sample to compute its gradient
        for i in range(sample_size):
            self.model.zero_grad()
            xi = x[i: i + 1]  # Keep the batch dimension (1, num_visible)
            # Compute the output; note that log_val should track gradients.
            output = torch.exp(self.log_val(xi))
            # Backpropagate the gradient for this single sample.
            output.backward(torch.ones_like(output))
            # For each parameter, store its per-sample gradient normalized by the output.
            for idx, param in enumerate(self.model.parameters()):
                grad_val = param.grad / output.item()  # Normalize by the scalar output
                # Apply freezing conditions if needed.
                if idx in self.freeze_layer:
                    grad_val = grad_val * 0.0
                if idx == 0 and self.freeze_pos is not None:
                    grad_np = grad_val.detach().cpu().numpy()
                    grad_np[self.freeze_pos] = 0.0
                    grad_val = torch.tensor(grad_np, device=x.device)
                per_sample_grads[idx][i] = grad_val.clone()
        # Optionally, reshape each per-sample gradient tensor to (sample_size, -1)
        # per_sample_grads = [g.view(sample_size, -1) for g in per_sample_grads]
        return per_sample_grads

    def get_parameters(self):
        """
        Get the parameters for this model
        """
        return [param.detach().numpy() for param in self.model.parameters()]

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
        This is equal to the sum of the mean squared difference of all parameters (weights and biases)
        """
        sum_diff = 0.
        for (par1, par2) in zip(first_param[1], last_param[1]):
            sum_diff += np.mean((par1 - par2) ** 2)

        return sum_diff

    def visualize_param(self, params, path):
        """
        Visualize every parameter
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
        return 'mlprealpos-%s' % (hidden_layer_str)

    def make_pickle_object(self):
        """
        PyTorch object cannot be pickled easily, so needs to be handled
        save the last param first and make it none
        """
        self.params = self.get_parameters()
        self.model = None
        self.activation_hidden = None
        self.activation_output = None

    def __str__(self):
        return 'MLPRealPositive %s' % (self.num_hidden)

    def to_xml(self):
        stri = ""
        stri += "<model>\n"
        stri += "\t<type>MLPRealPositive</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<num_visible>%d</num_visible>\n" % self.num_visible
        stri += "\t\t<num_hidden>%s</num_hidden>\n" % self.num_hidden
        stri += "\t\t<activation_output>%s</activation_output>\n" % str(
            self.activation_output)
        stri += "\t\t<activation_hidden>%s</activation_hidden>\n" % str(
            self.activation_hidden)
        stri += "\t\t<use_bias>%s</use_bias>\n" % str(self.use_bias)
        stri += "\t\t<num_expe>%s</num_expe>\n" % str(self.num_expe)
        stri += "\t\t<freeze_layer>%s</freeze_layer>\n" % str(
            self.freeze_layer)
        stri += "\t</params>\n"
        stri += "</model>\n"
        return stri
