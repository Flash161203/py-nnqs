import torch
from hamiltonian import Hamiltonian
import itertools
import numpy as np
import scipy
import scipy.sparse.linalg


class FromArray(Hamiltonian):
    """
    """

    def __init__(self, graph, array, seed=None):
        """
        """

        super.__init__(graph)

        self.seed = seed
        # Set the same seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Q * D * inv(Q)
        self.hamiltonian = array
        self.num_particles = int(np.log2(self.hamiltonian.shape[0]))

        # [diagonal non-diagonal]
        self.shiftmat = 1 << np.arange(self.num_particles)[::-1]

        self.confs = []
        for i in range(2 ** self.num_particles):
            conf_bin = format(i, '#0%db' % (self.num_particles + 2))
            # configuration in binary -1 1
            conf = np.array(
                [1. if c == '1' else -1. for c in conf_bin[2:]], dtype='float32')
            self.confs.append(conf)

        self.confs = torch.tensor(self.confs)

    def bin2int(self, b):
        return b.dot(self.shiftmat)

    def calculate_hamiltonian_matrix(self, samples, num_samples):
        """
        Calculate the Hamiltonian matrix $H_{x,x'}$ from a given sample x.
        Only non-zero elements are returned.

        Args:
            samples: The samples 
            num_samples: number of samples

        Return:
            The Hamiltonian where the first column contains the diagonal, which is $-J \sum_{i,j} x_i x_j$.
            The rest of the column contains the off-diagonal, which is -h for every spin flip. 
            Therefore, the number of columns equals the number of particles + 1 and the number of rows = num_samples
        """
        samples_numpy = samples.numpy()
        samples_numpy[samples_numpy == -1] = 0
        index = np.array([self.bin2int(tes)
                         for tes in samples_numpy[:]], dtype='int32')

        return torch.tensor(self.hamiltonian[index, :], dtype=torch.float32)

    def calculate_ratio(self, samples, model, num_samples):
        """
        Calculate the ratio of \Psi(x') and \Psi(x) from a given x 
        as log(\Psi(x')) - log(\Psi(x)).
        \Psi is defined in the model. 
        However, the Hamiltonian determines which x' gives non-zero.

        Args:
            samples: the samples x
            model: the model used to define \Psi
            num_samples: the number of samples
        Return:
            The ratio where the first column contains \Psi(x) / \Psi(x).
            The rest of the column contains the non-zero \Psi(x') / \Psi(x).
            In the Ising model, this corresponds to x' where exactly one of the spins x is flipped. 
            Therefore, the number of columns equals the number of particles + 1 and the number of rows = num_samples
        """
        lvd = []
        for ii, sample in enumerate(samples):
            lvd.append(model.log_val_diff(self.confs, samples[ii:ii+1, :]))
        return torch.stack(lvd)[:, :, 0]

    def diagonalize(self):
        """
        Diagonalize Hamiltonian with exact diagonalization.
        Only works for small systems (<= 10)!
        """
        # Calculate the eigenvalue
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.hamiltonian)

    def diagonalize_sparse(self):
        """
        Diagonalize Hamiltonian with exact diagonalization using sparse matrix methods.
        Only works for small (<= 20) systems!
        """
        self.eigen_values, self.eigen_vectors = scipy.sparse.linalg.eigs(
            self.hamiltonian, k=1, which='SR')

    def get_name(self):
        """ 
        Get the name of the Hamiltonian
        """
        return 'fromarray_%d' % self.num_particles

    def __str__(self):
        return "FromArray %d" % self.num_particles

    def to_xml(self):
        stri = ""
        stri += "<hamiltonian>\n"
        stri += "\t<type>from_array</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<seed>%s</seed>\n" % self.seed
        stri += "\t\t<num_particles>%s</particles>\n" % self.num_particles
        stri += "\t</params>\n"
        stri += "</hamiltonian>\n"
        return stri
