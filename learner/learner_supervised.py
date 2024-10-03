from __future__ import print_function
import torch
import time
import numpy as np
import copy
import scipy.stats


class LearnerSupervised:
    """
    This class is used to specified all the learning process and saving data for logging purposes. 

    TODO: Minibatch training
    """

    def __init__(self, hamiltonian, model, reference_energy=None,
                 observables=[], num_samples=1000):
        """
        Construct a learner objects
        Args:
            hamiltonian: Hamiltonian of the model
            model: the machine learning model used
            num_samples: number of samples used for training (Default: 1000)
            observables: observables value to compute (Default: [])
        """
        self.hamiltonian = hamiltonian
        self.model = model
        self.observables = observables
        self.num_samples = num_samples
        self.reference_energy = reference_energy

        self.model_params = []
        self.observables_value = []

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_fn = loss
        self.metrics = metrics

    def fit(self, train_data, val_data=None, epochs=1000, callbacks=[], batch_size=0):
        self.num_epochs = epochs
        self.batch_size = batch_size

        # Save initial model parameters
        self.store_model(0)

        # Prepare data
        x_train, y_train = train_data
        if val_data is not None:
            x_val, y_val = val_data

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(x_train)
            loss = self.loss_fn(y_pred, y_train)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Print metrics
            with torch.no_grad():
                if len(self.metrics) > 0:
                    metric_values = [metric(y_pred, y_train).item()
                                     for metric in self.metrics]
                    print(
                        f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()} - Metrics: {metric_values}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

            # Validation
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(x_val)
                    val_loss = self.loss_fn(y_val_pred, y_val)
                    if len(self.metrics) > 0:
                        val_metric_values = [
                            metric(y_val_pred, y_val).item() for metric in self.metrics]
                        print(
                            f"Validation - Loss: {val_loss.item()} - Metrics: {val_metric_values}")
                    else:
                        print(f"Validation - Loss: {val_loss.item()}")

            # Callback updates
            for callback in callbacks:
                callback.on_epoch_end(epoch)

        # Final model storage
        self.store_model(epochs, last=True)

        # Handle EnergyCallback (if used)
        for callback in callbacks:
            if 'EnergyCallback' in callback.__class__.__name__:
                self.samples = callback.samples

        # Final observables calculation
        self.calculate_observables(epochs)

    def get_overlap(self, true_wavefunction):
        pred_wavefunction = self.get_wave_function().detach().numpy()
        return np.abs(np.dot(pred_wavefunction.T, true_wavefunction)) ** 2 / np.sum(np.abs(pred_wavefunction) ** 2)

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
            samples, samples.shape[0])
        # Calculate $log(\Psi(x')) - log(\Psi(x))$
        lvd = self.hamiltonian.calculate_ratio(
            samples, self.model, samples.shape[0])

        # Sum over x'
        eloc_array = torch.sum(
            (torch.exp(lvd) * hamiltonian), axis=1, keepdim=True)

        return eloc_array

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

    def store_model(self, epoch, last=False):
        """
        Store the model parameters in model_params at each epoch based on store_model_freq if needed.
        First and last epoch always stored.
        Args:
            epoch: the epoch 
            last: to mark if it is the last epoch or not
        """
        if last or epoch == 0:
            self.model_params.append(
                (epoch, [p.clone() for p in self.model.parameters()]))

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
        str += "<learner_supervised>\n"
        str += "\t<params>\n"
        str += "\t\t<optimizer>%s</optimizer>\n" % self.optimizer
        str += "\t\t<lr>%.5f</lr>\n" % self.optimizer.param_groups[0]['lr']
        str += "\t\t<epochs>%d</epochs>\n" % self.num_epochs
        str += "\t\t<minibatch>%d</minibatch>\n" % self.batch_size
        str += "\t</params>\n"
        str += "</learner_supervised>\n"
        return str
