from sampler import Sampler
import torch


class Gibbs(Sampler):
    """
    This class is used to do Gibbs sampling. 
    Gibbs sampling is a special case of the Metropolis algorithm where the acceptance ratio is 1.
    Only works for RBM machine with real positive wave function
    """

    def __init__(self, num_samples, num_steps=1):
        """
        Construct a Gibbs sampler

        Args:
            num_samples: number of samples
            num_steps: number of steps (1 step = sample h from p(h| v) and sample v from p (v|h)
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
            Perform Gibbs sampling from a given initial sample
            by sampling from p(v | h) in the RBM.
            Args:
                model: model to calculate \Psi(x)
                initial_sample: the initial sample
                num_samples: number of samples returned

            Return:
                new samples
        """
        sample = initial_sample
        for i in range(self.num_steps):
            sample = model.get_new_visible(sample)

        return sample

    def get_all_samples(self, model, initial_sample, num_samples):
        """ 
            Get samples from Gibbs sampling
        """
        all_samples = []
        sample = initial_sample
        for i in range(self.num_steps):
            sample = model.get_new_visible(sample)
            all_samples.append(sample)

        return all_samples

    def to_xml(self):
        str = ""
        str += "<sampler>\n"
        str += "\t<type>gibbs</type>\n"
        str += "\t<params>\n"
        str += "\t\t<num_samples>%d</num_samples>\n" % self.num_samples
        str += "\t\t<num_steps>%d</num_steps>\n" % self.num_steps
        str += "\t</params>\n"
        str += "</sampler>\n"
        return str
