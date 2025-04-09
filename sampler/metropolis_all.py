from sampler import Sampler
import torch


class MetropolisAll(Sampler):
    """
    This class is used to do a metropolis all sampling. 
    In metropolis all, we consider all possible sample.
    """

    def __init__(self, num_samples, num_steps):
        """
        Construct a Metropolis All sampler

        Args:
            num_samples: number of samples
            num_steps: number of steps in metropolis
        """
        Sampler.__init__(self, num_samples)
        self.num_steps = num_steps

    def get_initial_random_samples(self, sample_size, num_samples=None):
        """
            Get initial random samples with size [num_samples, sample_size]
            from a random uniform.
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
        init_data = torch.where(init_data == 0, torch.tensor(-1,
                                dtype=torch.int64), torch.tensor(1, dtype=torch.int64))

        return init_data.float()

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
            # Get new configuration by flipping the spin a random spin
            new_config = self.get_new_config(sample, num_samples)

            # Calculate the ratio of the new configuration and old configuration probability by computing |log(psi(x')) - log(psi(x))|^2
            if model.is_real():
                ratio = torch.abs(model.log_val_diff(new_config, sample)) ** 2
            else:
                ratio = torch.abs(
                    torch.exp(model.log_val_diff(new_config, sample))) ** 2

            # Sampling
            random = torch.rand(num_samples, 1)

            # Calculate acceptance
            accept = (ratio > random).view(-1, 1).expand_as(sample)

            # Reject and accept samples
            sample = torch.where(accept, new_config, sample)
        return sample

    def get_new_config(self, sample, num_samples):
        """
            Get a new configuration by create a new random uniform sample
            Args: 
                sample: the samples that want to be flipped randomly
                num_samples: the number of samples
            Return:
                new samples with a randomly flipped spin
        """
        new_sample = torch.randint(
            0, 2, sample.shape, dtype=torch.int64, device=sample.device)

        return torch.where(new_sample == 0, torch.tensor(-1, dtype=torch.int64, device=sample.device), new_sample).float()

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
        str += "\t<type>metropolis_all</type>\n"
        str += "\t<params>\n"
        str += "\t\t<num_samples>%d</num_samples>\n" % self.num_samples
        str += "\t\t<num_steps>%d</num_steps>\n" % self.num_steps
        str += "\t</params>\n"
        str += "</sampler>\n"
        return str
