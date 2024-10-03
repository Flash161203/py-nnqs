from __future__ import print_function
import torch
import time
import numpy as np
import copy
import scipy.stats


class Learner:
    """
    This class is used to specified all the learning process and saving data for logging purposes. 

    TODO: Minibatch training
    """

    def __init__(self, hamiltonian, model, sampler, optimizer, num_epochs=1000,
                 minibatch_size=0, window_period=50, reference_energy=None, stopping_threshold=0.05,
                 store_model_freq=0, observables=[], observable_freq=0, use_sr=False, transfer_sample=None):
        """
        Construct a learner objects
        Args:
            hamiltonian: Hamiltonian of the model
            model: the machine learning model used
            sampler: the sampler used to train
            optimizer: the optimizer for training
            num_epochs: the number of epochs for training (Default: 1000)
            minibatch_size: the number of minibatch training (Default: 0)
            window_period: the number of windows for logging purposes (Default: 50) 
            reference_energy: reference energy value if there is any (Default: None)
            stopping_threshold: stopping threshold for the training defined as mean(elocs)/std(elocs) (Default: 0.05)    
            store_model_freq: store the model only at epochs that is the multiplier of this value. Zero means nothing is stored. By default the model at the first and last epochs are saved. (Default: 0)
            observables: observables value to compute (Default: [])
            observable_freq: compute the observables only at epochs that is the multiplier of this value. Zero means nothing is stored. By default observables are calculated at the last epoch. (Default: 0)

        """
        self.hamiltonian = hamiltonian
        self.model = model
        self.sampler = sampler
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.window_period = window_period
        self.reference_energy = reference_energy
        self.stopping_threshold = stopping_threshold
        self.store_model_freq = store_model_freq
        self.observables = observables
        self.observable_freq = observable_freq
        self.use_sr = use_sr
        self.transfer_sample = transfer_sample

        self.ground_energy = []
        self.ground_energy_std = []
        self.energy_windows = []
        self.energy_windows_std = []
        self.rel_errors = []
        self.times = []
        self.observables_value = []
        self.model_params = []

        self.samples = []

        if self.minibatch_size == 0 or self.minibatch_size > self.sampler.num_samples:
            self.minibatch_size = self.sampler.num_samples

        self.div = 1.0

    def learn(self):
        # Reset array
        self.reset_memory_array()

        # Get initial sample
        if self.transfer_sample is not None:
            self.samples = torch.tensor(
                self.transfer_sample, dtype=torch.float32)
        else:
            self.samples = torch.tensor(self.sampler.get_initial_random_samples(
                self.model.num_visible), dtype=torch.float32)
        print('===== Training start')

        for epoch in range(self.num_epochs):
            start = time.time()
            #####################################
            ####### TRAINING PROCESS ############
            #####################################

            # 1. Calculate local energy
            elocs = self.get_local_energy(self.samples)
            energy, energy_std, energy_window, energy_window_std, rel_error = self.process_energy_and_error(
                elocs)

            # Some processing
            # Print status
            print('Epoch: %d, energy: %.4f, std: %.4f, std / mean: %.4f, relerror: %.5f' % (
                epoch, energy, energy_std, energy_std / np.abs(energy), rel_error), end='')

            # stop if it is NaN (fail)
            if np.isnan(energy):
                params = [a.clone() for a in self.model.get_parameters()]
                for div in np.arange(1.1, 3.0, 0.1):
                    print("Retrying dividing weights by %.1f" % div)
                    self.model.set_parameters([p / div for p in params])

                    elocs = self.get_local_energy(self.samples)
                    energy, energy_std, energy_window, energy_window_std, rel_error = self.process_energy_and_error(
                        elocs)

                    # Some processing
                    # Print status
                    print('Epoch: %d, energy: %.4f, std: %.4f, std / mean: %.4f, relerror: %.5f' % (
                        epoch, energy, energy_std, energy_std / np.abs(energy), rel_error))
                    if not np.isnan(energy):
                        self.div = div
                        break

            if np.isnan(energy):
                print('Fail NaN')
                break

            # check stopping criterion
            if energy_std / np.abs(energy) < self.stopping_threshold:
                print('Stopping criterion reached!')
                break

            # save weight
            self.store_model(epoch)

            # calculate observable
            if self.observable_freq != 0 and epoch % self.observable_freq == 0:
                self.calculate_observables(epoch)

            # 2. Calculate gradient
            if self.use_sr:
                grads = self.get_gradient_sr(
                    self.samples, self.minibatch_size, elocs, epoch)
            else:
                grads = self.get_gradient(
                    self.samples, self.minibatch_size, elocs)

            # 3. Apply gradients
            self.optimizer.zero_grad()
            for g, p in zip(grads, self.model.model.parameters()):
                p.grad = g
            self.optimizer.step()

            # 4. Get new sample
            self.samples = self.sampler.sample(
                self.model, self.samples, self.minibatch_size)

            #####################################
            #####################################
            #####################################

            # Calculating additional stuffs
            end = time.time()
            time_interval = end - start
            self.times.append(time_interval)

            print(', time: %.5f' % time_interval)

        print('===== Training finish')
        # save the last data
        self.store_model(epoch, last=True)
        self.calculate_observables(epoch)

    def get_local_energy(self, samples):
        """
            Calculate local energy from a given samples
            $E_{loc}(x) = \sum_{x'} H_{x,x'} \Psi(x') / \Psi(x)$
            In this part, we instead do $log(\Psi(x')) - log(\Psi(x))$
            Args:
                samples: samples that we want to calculate the local energy
            Return:
                The local energy of each given samples
        """
        # Calculate $H_{x,x'}$
        hamiltonian = self.hamiltonian.calculate_hamiltonian_matrix(
            samples, len(samples))
        # Calculate $log(\Psi(x')) - log(\Psi(x))$
        lvd = self.hamiltonian.calculate_ratio(
            samples, self.model, len(samples))

        # Sum over x'
        if self.model.is_complex():
            eloc_array = torch.sum(
                (torch.exp(lvd) * hamiltonian.to(torch.complex64)), axis=1, keepdim=True)
        elif self.model.is_real():
            eloc_array = torch.sum((lvd * hamiltonian), axis=1, keepdim=True)
        else:
            eloc_array = torch.sum(
                (torch.exp(lvd) * hamiltonian), axis=1, keepdim=True)

        return eloc_array

    def get_gradient(self, samples, sample_size, eloc):
        """
            Calculate the gradient of E[\Psi] defined as 
            $2Re[  <E_{loc}D_{W}> - <E_{loc}><D_{W}> ]$
            where D_W is the gradient of the neural network w.r.t to its output defined as
            $D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW)$ where W can be the weights or the biases.

            Args:
                samples: the samples to calculate gradient
                sample_size:  the sample size
                eloc: the local energy E_{loc}
        """
        # Get D_{W} from the model
        derlogs = self.model.derlog(samples)

        # Calculate <E_{loc}>
        eloc_mean = torch.mean(eloc, dim=0, keepdim=True)

        grads = []
        for ii, derlog in enumerate(derlogs):
            old_shape = derlog.shape
            derlog = derlog.view(sample_size, -1)

            # Calculate <D_{W}>
            derlog_mean = torch.mean(derlog, dim=0, keepdim=True)

            # Calculate <E_loc D_{W}>
            ed = torch.mean(torch.conj(derlog) * eloc, dim=0, keepdim=True)

            # Calculate  $2Re[  <E_{loc}D_{W}> - <E_{loc}><D_{W}> ]$
            grad = (ed - derlog_mean * eloc_mean)

            grads.append(grad.view(*old_shape[1:]))

        return grads

    def get_gradient_sr(self, samples, sample_size, eloc, epoch):
        """
            Calculate the gradient of E[\Psi] defined as 
            $2Re[  <E_{loc}D_{W}> - <E_{loc}><D_{W}> ]$
            where D_W is the gradient of the neural network w.r.t to its output defined as
            $D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW)$ where W can be the weights or the biases.

            Args:
                samples: the samples to calculate gradient
                sample_size:  the sample size
                eloc: the local energy E_{loc}
        """
        # Get D_{W} from the model
        derlogs = self.model.derlog(samples)
        old_shapes = [derlog.shape for derlog in derlogs]

        # Calculate <E_{oc}>
        eloc_mean = torch.mean(eloc, dim=0, keepdim=True)

        # Calculate O_k
        all_derlogs = torch.cat([derlog.view(sample_size, -1)
                                for derlog in derlogs], dim=1)

        # Calculate <O_k>
        all_derlogs_mean = torch.mean(all_derlogs, dim=0, keepdim=True)

        # Calculate <O^*_k O_k>
        all_derlogs_derlogs_mean = torch.einsum(
            'ij, ik->jk', torch.conj(all_derlogs), all_derlogs) / len(samples)

        # Calculate S_kk = <O^*_k O_k> - <O_k><O^*_k>
        S_kk = all_derlogs_derlogs_mean - \
            torch.conj(all_derlogs_mean) * all_derlogs_mean.t()

        # Regularize S_kk to make sure it is invertible
        regularizer = max(100 * (0.9 ** (epoch+1)), 1e-4)
        S_kk_diag_reg = torch.diag(regularizer * torch.diag(S_kk))
        S_kk_reg = S_kk + S_kk_diag_reg

        # Calculate <D_{W}>
        derlog_mean = torch.mean(all_derlogs, dim=0, keepdim=True)

        # Calculate <E_loc D_{W}>
        ed = torch.mean(torch.conj(all_derlogs) * eloc, dim=0, keepdim=True)

        # Calculate  $2Re[  <E_{loc}D_{W}> - <E_{loc}><D_{W}> ]$
        grad = ed - derlog_mean * eloc_mean

        # inv(S_kk) * grads
        S_inv = torch.inverse(S_kk_reg)

        final_grads = torch.matmul(S_inv, grad.t())

        grads = []
        prev = 0
        for old_shape in old_shapes:
            final_grad = final_grads[prev:prev +
                                     torch.prod(torch.tensor(old_shape[1:]))]
            prev += torch.prod(torch.tensor(old_shape[1:]))
            grads.append(final_grad.view(*old_shape[1:]))
        return grads

    def process_energy_and_error(self, elocs):
        """
            Process the energy and error by calculating the energy, energy over windows and relative error.
            Args:
                elocs: the local energies array of one epoch
        """
        # Calculate ground state energy mean and std
        ground_energy = np.real(np.mean(elocs))
        ground_energy_std = np.real(np.std(elocs))

        self.ground_energy.append(ground_energy)
        self.ground_energy_std.append(ground_energy_std)

        # Calculate energy over windows
        energy_window = np.mean(self.ground_energy[-self.window_period:])
        energy_window_std = np.std(self.ground_energy[-self.window_period:])

        self.energy_windows.append(energy_window)
        self.energy_windows_std.append(energy_window_std)

        # Calculate relative error
        if self.reference_energy is None:
            rel_error = 0.0
        else:
            rel_error = np.abs(
                (ground_energy - self.reference_energy) / self.reference_energy)
        self.rel_errors.append(rel_error)

        return ground_energy, ground_energy_std, energy_window, energy_window_std, rel_error

    def calculate_observables(self, epoch):
        """
            Calculate observables if any.
            Args:
                epoch: epoch for log purposes
        """
        # Get the probability $|\Psi(x)|^2$ from samples
        confs, count_ = np.unique(
            self.samples.cpu().numpy(), axis=0, return_counts=True)
        prob_out = count_ / len(self.samples)

        # Calculate each observables
        value_map = {}
        for obs in self.observables:
            obs_value = (obs.get_value_ferro(prob_out, confs),
                         obs.get_value_antiferro(prob_out, confs))
            value_map[obs.get_name()] = obs_value

        self.observables_value.append((epoch, value_map))

    def reset_memory_array(self):
        """
        Reset memory array to all empty
        """
        self.ground_energy = []
        self.ground_energy_std = []
        self.energy_windows = []
        self.energy_windows_std = []
        self.rel_errors = []
        self.times = []
        self.samples = []
        self.model_params = []
        self.observables_value = []

    def store_model(self, epoch, last=False):
        """
        Store the model parameters in model_params at each epoch based on store_model_freq if needed.
        First and last epoch always stored.
        Args:
            epoch: the epoch 
            last: to mark if it is the last epoch or not
        """
        if last or epoch == 0:
            self.model_params.append((epoch, self.model.get_parameters()))
        else:
            if self.store_model_freq != 0 and epoch % self.store_model_freq == 0:
                self.model_params.append((epoch, self.model.get_parameters()))

    def make_pickle_object(self):
        """
        Create pickle object to save.
        """
        temp_learner = copy.copy(self)
        # pickle cannot save a PyTorch object so it needs to be removed
        temp_learner.model.make_pickle_object()
        return temp_learner

    def to_xml(self):
        str = ""
        str += "<learner>\n"
        str += "\t<params>\n"
        str += "\t\t<optimizer>%s</optimizer>\n" % self.optimizer
        str += "\t\t<lr>%.5f</lr>\n" % self.optimizer.param_groups[0]['lr']
        str += "\t\t<epochs>%d</epochs>\n" % self.num_epochs
        str += "\t\t<minibatch>%d</minibatch>\n" % self.minibatch_size
        str += "\t\t<window_period>%d</window_period>\n" % self.window_period
        str += "\t\t<stopping_threshold>%.5f</stopping_threshold>\n" % self.stopping_threshold
        str += "\t\t<div>%.5f</div>\n" % self.div
        str += "\t</params>\n"
        str += "</learner>\n"
        return str
