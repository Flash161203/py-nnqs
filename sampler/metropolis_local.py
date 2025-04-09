from sampler import Sampler
import torch


class MetropolisLocal(Sampler):
    """
    This class is used to do a Metropolis local sampling. 
    In Metropolis local, the next sample is determined by flipping one random spin.
    """

    def __init__(self, num_samples, num_steps):
        """
        Construct a Metropolis Local sampler

        Args:
            num_samples: number of samples
            num_steps: number of steps in Metropolis
        """
        super().__init__(num_samples)
        self.num_steps = num_steps

    def get_initial_random_samples(self, sample_size, num_samples=None):
        """
            Get initial random samples with size [num_samples, sample_size]
            from a random uniform distribution.
            Args:
                sample_size: the number of particles
                num_samples: number of samples
            Return:
                initial random samples
        """
        if num_samples is None:
            num_samples = self.num_samples

        init_data = torch.randint(
            0, 2, (num_samples, sample_size), dtype=torch.int64)
        init_data = torch.where(init_data == 0, -1, 1)

        return init_data.float()

    def sample(self, model, initial_sample, num_samples):
        """
            Do a Metropolis local sample from a given initial sample
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
            Do one Metropolis step from a given starting sample, model is used to calculate probability.

            Args:
                model: the model to calculate probability |\Psi(x)|^2
                starting_sample: the initial sample
                num_samples: number of samples returned
            Return:
                new samples from one Metropolis local step
        """
        # Get new configuration by flipping a random spin
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
            Get a new configuration by flipping a random spin
            Args: 
                sample: the samples that want to be flipped randomly
                num_samples: the number of samples
            Return:
                new samples with a randomly flipped spin
        """
        num_points = int(sample.shape[1])

        row_indices = torch.arange(num_samples).unsqueeze(1)
        col_indices = torch.randint(
            0, num_points, (num_samples, 1), dtype=torch.int64)
        indices = torch.cat([row_indices, col_indices], 1)

        elements = sample.gather(1, col_indices)
        old = torch.zeros_like(sample).scatter_(1, col_indices, elements)
        new = torch.zeros_like(sample).scatter_(1, col_indices, -elements)
        return sample - old + new

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
        str += "\t<type>metropolis_local</type>\n"
        str += "\t<params>\n"
        str += "\t\t<num_samples>%d</num_samples>\n" % self.num_samples
        str += "\t\t<num_steps>%d</num_steps>\n" % self.num_steps
        str += "\t</params>\n"
        str += "</sampler>\n"
        return str
