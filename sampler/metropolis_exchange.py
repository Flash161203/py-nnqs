from sampler import Sampler
import torch
import numpy as np


class MetropolisExchange(Sampler):
    """
    This class is used to do a metropolis exchange sampling. 
    In metropolis exchange, the next sample is determined by flipping two random spins.
    It is similar to metropolis local, the difference is only on the get_new_config.
    This sampling is used to maintain total sz to zero particularly in Heisenberg model.
    """

    def __init__(self, num_samples, num_steps, total_sz=0):
        Sampler.__init__(self, num_samples)
        self.num_steps = num_steps
        self.total_sz = total_sz

    # set total Sz to be 0
    def get_initial_random_samples(self, sample_size, num_samples=None):
        """
            Get initial random samples with size [num_samples, sample_size]
            from a random uniform. However it must conforms to the total sz specification
            Args:
                sample_size: the number of particles
                num_samples: number of samples
            Return:
                initial random samples
        """
        if num_samples is None:
            num_samples = self.num_samples

        assert self.total_sz <= sample_size
        assert (self.total_sz + sample_size) % 2 == 0

        # mix +1 and -1 equally
        plus = np.ones(int((sample_size + self.total_sz) / 2)) * 1.0
        minus = np.ones(int((sample_size - self.total_sz) / 2)) * -1.0
        model = np.concatenate((plus, minus))

        # shuffle them
        init_data = []
        for i in range(num_samples):
            data = np.copy(model)
            np.random.shuffle(data)
            init_data.append(data)

        init_data = np.array(init_data, np.float32)
        init_data = np.reshape(init_data, (num_samples, sample_size))

        return init_data

    def sample(self, model, initial_sample, num_samples):
        """
            Do a metropolis local sample from a given initial sample
            and model to get \Psi(x).
            Args:
                model: model to calculate \Psi(x)
                initial_sample: the initial sample
                num_samples: number of samples returned

            Return:
                new samples
        """
        sample = initial_sample
        for i in range(self.num_steps):
            sample = self.sample_once(model, sample, num_samples)

        return sample

    def sample_once(self, model, starting_sample, num_samples):
        """
            Do a one metropolis step from a given starting sample, model is used to calculate probability.

            Args:
                model: the model to calculate probability |\Psi(x)|^2
                starting_samples: the initial samples
                num_samples: number of samples returned
            Return:
                new samples from one metropolis exchange
        """
        # Get new configuration by flipping the spin of two random spins
        new_config = self.get_new_config(starting_sample, num_samples)

        # Calculate the ratio of the new configuration and old configuration probability by computing |log(psi(x')) - log(psi(x))|^2
        if model.is_real():
            ratio = torch.abs(model.log_val_diff(
                new_config, starting_sample)) ** 2
        else:
            ratio = torch.abs(torch.exp(model.log_val_diff(
                new_config, starting_sample))) ** 2

        # Sampling
        random = torch.rand((num_samples, 1))

        # Calculate acceptance
        accept = torch.squeeze(ratio > random)
        accept = accept.unsqueeze(1).expand_as(starting_sample)

        # Reject and accept samples
        sample = torch.where(accept, new_config, starting_sample)
        return sample

    def get_new_config(self, sample, num_samples):
        """
            Get a new configuration by flipping two random spins
            Args: 
                sample: the samples that want to be flipped randomly
                num_samples: the number of samples
            Return:
                new samples with two randomly flipped spins
        """
        num_points = int(sample.shape[1])
        position1 = torch.randint(0, num_points, (num_samples,))
        position2 = torch.randint(0, num_points, (num_samples,))
        row_indices = torch.arange(num_samples).unsqueeze(1)
        col_indices1 = position1.unsqueeze(1)
        col_indices2 = position2.unsqueeze(1)
        indices1 = torch.cat((row_indices, col_indices1), dim=1)
        indices2 = torch.cat((row_indices, col_indices2), dim=1)
        elements1 = sample.gather(1, col_indices1)
        elements2 = sample.gather(1, col_indices2)
        old1 = torch.zeros_like(sample).scatter_(1, col_indices1, elements1)
        old2 = torch.zeros_like(sample).scatter_(1, col_indices2, elements2)
        new1 = torch.zeros_like(sample).scatter_(1, col_indices1, elements2)
        new2 = torch.zeros_like(sample).scatter_(1, col_indices2, elements1)
        return sample - old1 - old2 + new1 + new2

    def get_all_samples(self, model, initial_sample, num_samples):
        all_samples = []
        sample = initial_sample
        for i in range(self.num_steps):
            sample = self.sample_once(model, sample, num_samples)
            all_samples.append(sample)

        return all_samples

    def to_xml(self):
        str = ""
        str += "<sampler>\n"
        str += "\t<type>metropolis_exchange</type>\n"
        str += "\t<params>\n"
        str += "\t\t<num_samples>%d</num_samples>\n" % self.num_samples
        str += "\t\t<num_steps>%d</num_steps>\n" % self.num_steps
        str += "\t\t<total_sz>%d</total_sz>\n" % self.total_sz
        str += "\t</params>\n"
        str += "</sampler>\n"
        return str
