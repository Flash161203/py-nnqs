from __future__ import print_function
import torch
import time
import numpy as np
import copy
import scipy.stats


class Learner:
    """
    This class is used to specify all the learning process and saving data for logging purposes. 
    """

    def __init__(self, hamiltonian, model, sampler, optimizer, num_epochs=1000,
                 minibatch_size=0, window_period=50, reference_energy=None, stopping_threshold=0.05,
                 store_model_freq=0, observables=[], observable_freq=0, use_sr=False, transfer_sample=None,
                 lambda_mul=0):
        """
        Construct a learner object.
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
            store_model_freq: store the model only at epochs that are a multiple of this value. Zero means nothing is stored.
            observables: observables value to compute (Default: [])
            observable_freq: compute the observables only at epochs that are a multiple of this value. Zero means nothing is stored.
            use_sr: whether to use stochastic reconfiguration (Default: False)
            transfer_sample: initial sample to transfer (Default: None)
            lambda_mul: coefficient for the excited state orthogonal penalty (Default: 0)
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
        self.lambda_mul = lambda_mul

        # Use unified energy lists for both ground and excited state training.
        self.energy = []
        self.energy_std = []
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
        # Reset arrays
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

            # Print status
            print('Epoch: %d, energy: %.4f, std: %.4f, std / mean: %.4f, relerror: %.5f' % (
                epoch, energy, energy_std, energy_std / np.abs(energy), rel_error), end='')

            # Stop if energy is NaN (fail)
            if np.isnan(energy):
                params = [p.clone() for p in self.model.get_parameters()]
                for div in np.arange(1.1, 3.0, 0.1):
                    print("Retrying dividing weights by %.1f" % div)
                    self.model.set_parameters([p / div for p in params])

                    elocs = self.get_local_energy(self.samples)
                    energy, energy_std, energy_window, energy_window_std, rel_error = self.process_energy_and_error(
                        elocs)
                    print('Epoch: %d, energy: %.4f, std: %.4f, std / mean: %.4f, relerror: %.5f' % (
                        epoch, energy, energy_std, energy_std / np.abs(energy), rel_error))
                    if not np.isnan(energy):
                        self.div = div
                        break

            if np.isnan(energy):
                print('Fail NaN')
                break

            # Check stopping criterion
            if energy_std / np.abs(energy) < self.stopping_threshold:
                print('Stopping criterion reached!')
                break

            # Save model parameters
            self.store_model(epoch)

            # Calculate observables if required
            if self.observable_freq != 0 and epoch % self.observable_freq == 0:
                self.calculate_observables(epoch)

            # 2. Calculate gradient
            if self.use_sr:
                grads = self.get_gradient_sr(
                    self.samples, self.minibatch_size, elocs, epoch)
            else:
                grads = self.get_gradient(
                    self.samples, self.minibatch_size, elocs)

            # 3. Apply gradients using PyTorch optimizer
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
            end = time.time()
            time_interval = end - start
            self.times.append(time_interval)
            print(', time: %.5f' % time_interval)

        print('===== Training finish')
        self.store_model(epoch, last=True)
        self.calculate_observables(epoch)

    def learn_excited_states(self, prior_models, prior_samples, state_index=1):
        """
        Carries out the training for an excited state.
        Args:
            prior_models: list of prior states' converged models (e.g., ground state model, first excited state, etc.)
            prior_samples: list of samples from prior converged models
            state_index: index of the excited state to find (non-zero values correspond to excited states)
        """
        if state_index == 0:
            print('Passed state_index: {}. Please use learn() for the ground state'.format(
                state_index))
            return

        self.reset_memory_array()

        if self.transfer_sample is not None:
            self.samples = torch.tensor(
                self.transfer_sample, dtype=torch.float32)
        else:
            self.samples = torch.tensor(self.sampler.get_initial_random_samples(
                self.model.num_visible), dtype=torch.float32)
        print('===== Training start (Excited State)')

        for epoch in range(self.num_epochs):
            start = time.time()
            # 1. Calculate local energy and overlap sum
            elocs = self.get_local_energy(self.samples)
            oloc_sum = self.get_oloc_sum(prior_models, prior_samples)
            energy, energy_std, energy_window, energy_window_std, rel_error = self.process_energy_and_error(
                elocs, oloc_sum)

            print('Epoch: %d, oloc_sum: %.6f, energy: %.4f, std: %.4f, std/mean: %.4f, relerror: %.5f' % (
                epoch, np.mean(oloc_sum.detach().cpu().numpy()), energy, energy_std, energy_std / np.abs(energy), rel_error), end='')

            if np.isnan(energy):
                params = [p.clone() for p in self.model.get_parameters()]
                for div in np.arange(1.1, 5.0, 0.1):
                    print("Retrying dividing weights by %.1f" % div)
                    self.model.set_parameters([p / div for p in params])
                    elocs = self.get_local_energy(self.samples)
                    oloc_sum = self.get_oloc_sum(prior_models, prior_samples)
                    energy, energy_std, energy_window, energy_window_std, rel_error = self.process_energy_and_error(
                        elocs, oloc_sum)
                    print('Epoch: %d, energy: %.4f, std: %.4f, std/mean: %.4f, relerror: %.5f' % (
                        epoch, energy, energy_std, energy_std / np.abs(energy), rel_error))
                    if not np.isnan(energy):
                        self.div = div
                        break

            # Check stopping criterion (including overlap sum near zero)
            if (energy_std / np.abs(energy) < self.stopping_threshold and
                np.abs(np.std(elocs.detach().cpu().numpy()) / np.mean(elocs.detach().cpu().numpy())) < self.stopping_threshold and
                    np.abs(np.mean(oloc_sum.detach().cpu().numpy())) <= 0.010):
                print('Stopping criterion reached!')
                break

            self.store_model(epoch)
            if self.observable_freq != 0 and epoch % self.observable_freq == 0:
                self.calculate_observables(epoch)

            # 2. Calculate excited state gradient
            grads = self.get_gradient_excited(
                prior_models, prior_samples, self.minibatch_size, eloc=elocs)
            self.optimizer.zero_grad()
            for g, p in zip(grads, self.model.model.parameters()):
                p.grad = g
            self.optimizer.step()

            # 3. Get new sample
            self.samples = self.sampler.sample(
                self.model, self.samples, self.minibatch_size)

            end = time.time()
            time_interval = end - start
            self.times.append(time_interval)
            print(', time: %.5f' % time_interval)

        print('===== Training finish (Excited State)')
        self.store_model(epoch, last=True)
        self.calculate_observables(epoch)

    def get_local_energy(self, samples):
        """
        Calculate local energy for the given samples:
          E_loc(x) = sum_{x'} H_{x,x'} Psi(x')/Psi(x)
        In our implementation, we calculate it via the log ratios.
        """
        # Calculate H_{x,x'}
        hamiltonian = self.hamiltonian.calculate_hamiltonian_matrix(
            samples, len(samples))
        # Calculate log(Psi(x')) - log(Psi(x))
        lvd = self.hamiltonian.calculate_ratio(
            samples, self.model, len(samples))

        # Sum over x'
        if self.model.is_complex():
            eloc_array = torch.sum(
                torch.exp(lvd) * hamiltonian.to(torch.complex64), dim=1, keepdim=True)
        elif self.model.is_real():
            eloc_array = torch.sum(lvd * hamiltonian, dim=1, keepdim=True)
        else:
            eloc_array = torch.sum(
                torch.exp(lvd) * hamiltonian, dim=1, keepdim=True)
        return eloc_array

    def get_gradient(self, samples, sample_size, eloc):
        """
        Calculate the gradient:
         2 Re[ <E_loc D_W> - <E_loc><D_W> ]
        where D_W = (1/Psi) * dPsi/dW.
        """
        derlogs = self.model.derlog(samples)
        eloc_mean = torch.mean(eloc, dim=0, keepdim=True)
        grads = []
        for ii, derlog in enumerate(derlogs):
            old_shape = derlog.shape
            derlog = derlog.view(sample_size, -1)
            derlog_mean = torch.mean(derlog, dim=0, keepdim=True)
            ed = torch.mean(torch.conj(derlog) * eloc, dim=0, keepdim=True)
            grad = ed - derlog_mean * eloc_mean
            grads.append(grad.view(*old_shape[1:]))
        return grads

    def get_gradient_sr(self, samples, sample_size, eloc, epoch):
        """
        Calculate the gradient using stochastic reconfiguration.
        """
        derlogs = self.model.derlog(samples)
        old_shapes = [d.shape for d in derlogs]
        eloc_mean = torch.mean(eloc, dim=0, keepdim=True)
        all_derlogs = torch.cat([d.view(sample_size, -1)
                                for d in derlogs], dim=1)
        all_derlogs_mean = torch.mean(all_derlogs, dim=0, keepdim=True)
        all_derlogs_derlogs_mean = torch.einsum(
            'ij,ik->jk', torch.conj(all_derlogs), all_derlogs) / len(samples)
        S_kk = all_derlogs_derlogs_mean - \
            torch.conj(all_derlogs_mean) * all_derlogs_mean.t()
        regularizer = max(100 * (0.9 ** (epoch + 1)), 1e-4)
        S_kk_diag_reg = torch.diag(regularizer * torch.diag(S_kk))
        S_kk_reg = S_kk + S_kk_diag_reg
        derlog_mean = torch.mean(all_derlogs, dim=0, keepdim=True)
        ed = torch.mean(torch.conj(all_derlogs) * eloc, dim=0, keepdim=True)
        grad = ed - derlog_mean * eloc_mean
        S_inv = torch.inverse(S_kk_reg)
        final_grads = torch.matmul(S_inv, grad.t())
        grads = []
        prev = 0
        for old_shape in old_shapes:
            numel = int(torch.prod(torch.tensor(old_shape[1:])))
            final_grad = final_grads[prev:prev+numel]
            prev += numel
            grads.append(final_grad.view(*old_shape[1:]))
        return grads

    def get_oloc_sum(self, prev_state_models, prev_state_samples):
        """
        Calculate the sum of overlaps with prior states.
        """
        net_overlap = 0
        for prior_model, prior_samples in zip(prev_state_models, prev_state_samples):
            oloc_recvd = self.get_oloc(prior_model, prior_samples)
            net_overlap = net_overlap + oloc_recvd
        return net_overlap

    def get_oloc(self, prior_model, prior_samples):
        """
        Calculate the overlap term:
           oloc = mean[exp(log(Psi_exc) - log(Psi_prior))] * exp(log(Psi_prior) - log(Psi_exc))
        where the log(Psi) values come from the models.
        """
        if isinstance(prior_model, dict):
            E_psiExc_psiPrior = torch.exp(self.model.log_val(prior_samples) -
                                          self.get_log_val_from_mapping(mapping=prior_model, samples=prior_samples))
            E_psiPrior_psiExc = torch.exp(self.get_log_val_from_mapping(mapping=prior_model, samples=self.samples) -
                                          self.model.log_val(self.samples))
        else:
            E_psiExc_psiPrior = torch.exp(self.model.log_val(prior_samples) -
                                          prior_model.log_val(prior_samples).to(torch.complex64))
            E_psiPrior_psiExc = torch.exp(prior_model.log_val(self.samples).to(torch.complex64) -
                                          self.model.log_val(self.samples))
        oloc = torch.mean(E_psiExc_psiPrior, dim=0,
                          keepdim=True) * E_psiPrior_psiExc
        return oloc

    def get_gradient_excited(self, prior_models, prior_samples, sample_size, eloc):
        """
        Calculate the gradient for excited states.
        """
        # Ground state gradient term
        derlogs = self.model.derlog(self.samples)
        eloc_mean = torch.mean(eloc, dim=0, keepdim=True)
        grads_0 = []
        for derlog in derlogs:
            old_shape = derlog.shape
            d = derlog.view(sample_size, -1)
            derlog_mean = torch.mean(d, dim=0, keepdim=True)
            ed = torch.mean(torch.conj(d) * eloc, dim=0, keepdim=True)
            grad = ed - derlog_mean * eloc_mean
            grads_0.append(grad.view(*old_shape[1:]))

        # Orthogonal penalty gradient terms
        orth_grads_curr_iter = []
        for prior_model, prior_samples in zip(prior_models, prior_samples):
            if isinstance(prior_model, dict):
                E_psi_i_psi = torch.exp(self.get_log_val_from_mapping(mapping=prior_model, samples=self.samples) -
                                        self.model.log_val(self.samples))
                E_psi_psi_i = torch.exp(self.model.log_val(prior_samples) -
                                        self.get_log_val_from_mapping(mapping=prior_model, samples=prior_samples))
            else:
                E_psi_i_psi = torch.exp(prior_model.log_val(self.samples).to(torch.complex64) -
                                        self.model.log_val(self.samples))
                E_psi_psi_i = torch.exp(self.model.log_val(prior_samples) -
                                        prior_model.log_val(prior_samples).to(torch.complex64))
            grads_1 = []
            grads_2 = []
            grads_3 = []
            for derlog in derlogs:
                old_shape = derlog.shape
                d = derlog.view(sample_size, -1)
                exp_1 = torch.mean(torch.conj(
                    d) * E_psi_i_psi, dim=0, keepdim=True)
                exp_2 = torch.mean(E_psi_psi_i, dim=0, keepdim=True)
                term1 = self.lambda_mul * exp_1 * exp_2
                grads_1.append(term1.view(*old_shape[1:]))

                exp_1_val = torch.mean(E_psi_i_psi)
                exp_1_1 = torch.mean(d, dim=0, keepdim=True) + \
                    torch.mean(torch.conj(d), dim=0, keepdim=True)
                exp_2_val = torch.mean(E_psi_psi_i, dim=0, keepdim=True)
                term2 = - self.lambda_mul * exp_1_1 * exp_1_val * exp_2_val
                grads_2.append(term2.view(*old_shape[1:]))

                exp_1b = torch.mean(E_psi_i_psi, dim=0, keepdim=True)
                exp_2b = torch.mean(d * E_psi_psi_i, dim=0, keepdim=True)
                term3 = self.lambda_mul * exp_1b * exp_2b
                grads_3.append(term3.view(*old_shape[1:]))

            if len(orth_grads_curr_iter) == 0:
                orth_grads_curr_iter = [g1 + g2 + g3 for g1,
                                        g2, g3 in zip(grads_1, grads_2, grads_3)]
            else:
                orth_grads_curr_iter = [prev + g1 + g2 + g3 for prev, g1, g2,
                                        g3 in zip(orth_grads_curr_iter, grads_1, grads_2, grads_3)]

        gradient = [g0 + orth for g0,
                    orth in zip(grads_0, orth_grads_curr_iter)]
        return gradient

    def process_energy_and_error(self, elocs, oloc_sum=None):
        """
        Process energy and error.
        If oloc_sum is provided, the excited state energy is computed as:
          E = mean(elocs + lambda_mul * oloc_sum)
        Otherwise, ground state energy is computed.
        """
        if oloc_sum is not None:
            energy_val = np.real(
                np.mean(elocs.detach().cpu().numpy() + self.lambda_mul * oloc_sum.detach().cpu().numpy()))
            energy_std_val = np.real(
                np.std(elocs.detach().cpu().numpy() + self.lambda_mul * oloc_sum.detach().cpu().numpy()))
        else:
            energy_val = np.real(np.mean(elocs.detach().cpu().numpy()))
            energy_std_val = np.real(np.std(elocs.detach().cpu().numpy()))
        self.energy.append(energy_val)
        self.energy_std.append(energy_std_val)
        energy_window = np.mean(self.energy[-self.window_period:])
        energy_window_std = np.std(self.energy[-self.window_period:])
        self.energy_windows.append(energy_window)
        self.energy_windows_std.append(energy_window_std)
        if self.reference_energy is None:
            rel_error = 0.0
        else:
            rel_error = np.abs(
                (energy_val - self.reference_energy) / self.reference_energy)
        self.rel_errors.append(rel_error)
        return energy_val, energy_std_val, energy_window, energy_window_std, rel_error

    def calculate_observables(self, epoch):
        """
        Calculate observables if any.
        """
        samples_np = self.samples.detach().cpu().numpy()
        confs, count_ = np.unique(samples_np, axis=0, return_counts=True)
        prob_out = count_ / len(samples_np)
        value_map = {}
        for obs in self.observables:
            obs_value = (obs.get_value_ferro(prob_out, confs),
                         obs.get_value_antiferro(prob_out, confs))
            value_map[obs.get_name()] = obs_value
        self.observables_value.append((epoch, value_map))

    def reset_memory_array(self):
        """
        Reset all memory arrays.
        """
        self.energy = []
        self.energy_std = []
        self.energy_windows = []
        self.energy_windows_std = []
        self.rel_errors = []
        self.times = []
        self.samples = []
        self.model_params = []
        self.observables_value = []

    def store_model(self, epoch, last=False):
        """
        Store model parameters at the given epoch.
        """
        if last or epoch == 0:
            self.model_params.append((epoch, self.model.get_parameters()))
        else:
            if self.store_model_freq != 0 and epoch % self.store_model_freq == 0:
                self.model_params.append((epoch, self.model.get_parameters()))

    def print_freq_map_from_sample(self):
        """
        Print frequency map of configurations (interpreting each sample as a binary string).
        """
        samples_np = self.samples.detach().cpu().numpy()
        bin_confs = []
        for sample in samples_np:
            conf = [1 if c == 1 else 0 for c in sample]
            bin_confs.append(conf)
        freq_map = {}
        for each_sample in bin_confs:
            num = 0
            for bit in each_sample:
                num = 2 * num + bit
            freq_map[num] = freq_map.get(num, 0) + 1
        print('Samples (as binary strings):')
        print('Percentage of configs from all:', 100 *
              len(freq_map.keys()) / (2**self.model.num_visible))
        print(freq_map)

    def get_log_val_from_mapping(self, mapping, samples):
        """
        Returns an array of log|Psi| values corresponding to the given samples using a provided mapping.
        """
        samples_np = samples.detach().cpu().numpy()
        log_psi_values = []
        for sample in samples_np:
            conf = [1 if c == 1 else 0 for c in sample]
            bitstring = ''.join(str(b) for b in conf)
            log_psi_values.append(mapping[bitstring])
        return np.array(log_psi_values).reshape(-1, 1)

    def make_pickle_object(self):
        """
        Create a pickleable object.
        """
        temp_learner = copy.copy(self)
        # PyTorch objects cannot be pickled so remove the model
        temp_learner.model.make_pickle_object()
        return temp_learner

    def to_xml(self):
        s = ""
        s += "<learner>\n"
        s += "\t<params>\n"
        s += "\t\t<optimizer>%s</optimizer>\n" % self.optimizer
        s += "\t\t<lr>%.5f</lr>\n" % self.optimizer.param_groups[0]['lr']
        s += "\t\t<epochs>%d</epochs>\n" % self.num_epochs
        s += "\t\t<minibatch>%d</minibatch>\n" % self.minibatch_size
        s += "\t\t<window_period>%d</window_period>\n" % self.window_period
        s += "\t\t<stopping_threshold>%.5f</stopping_threshold>\n" % self.stopping_threshold
        if self.reference_energy:
            s += "\t\t<reference_energy>%.5f</reference_energy>\n" % self.reference_energy
        s += "\t\t<div>%.5f</div>\n" % self.div
        s += "\t</params>\n"
        s += "</learner>\n"
        return s
