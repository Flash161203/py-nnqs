import numpy as np


class Hamiltonian(object):
    """
    Base class for Hamiltonian.

    This class defines the hamiltonian of the quantum many-body system.
    You must define how to get the Hamiltonian matrix.
    """

    def __init__(self, graph):
        self.graph = graph
        self.hamiltonian = None
        self.eigen_values = None
        self.eigen_vectors = None

        # Pauli matrices
        self.SIGMA_X = np.array([[0, 1], [1, 0]])
        self.SIGMA_Y = np.array([[0, -1j], [1j, 0]])
        self.SIGMA_Z = np.array([[1, 0], [0, -1]])

    # Calculates the Hamiltonian matrix from list of samples. Returns a tensor.

    def calculate_hamiltonian_matrix(self, samples, num_samples):
        # implemented in subclass
        raise NotImplementedError

    def calculate_ratio(self, samples, machine, num_samples):
        # implemented in subclass
        raise NotImplementedError

    def diagonalize(self):
        # implemented in subclass
        raise NotImplementedError

    def diagonalize_sparse(self, num_states=1):
        # implemented in subclass
        raise NotImplementedError

    def get_gs_energy(self):
        """
        Get ground state energy $E_0$
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize() or diagonalize_sparse()!")
        else:
            return np.real(np.min(self.eigen_values))

    def get_gs(self):
        """
        Get ground state $\Psi_{GS}$
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize() or diagonalize_sparse()!")
        else:
            return self.eigen_vectors[:, np.argmin(self.eigen_values)]

    def get_gs_probability(self):
        """
        Get ground state probability $|\Psi_{GS}|^2$
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize() or diagonalize_sparse()!")
        else:
            return np.abs(self.get_gs()) ** 2

    def get_energy(self, n=0):
        """
        Get n-th state energy $E_n$
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize() or diagonalize_sparse()!")
        else:
            return np.real(sorted(self.eigen_values)[n])

    def get_state(self, n=0):
        """
        Get n-th state $\Psi_n$
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize() or diagonalize_sparse()!")
        else:
            # argsort the eigenvalues, get the index at the n-th place in the sorted list
            # return the vector at this index in the eigenvector matrix
            idx = np.argsort(self.eigen_values)[n]
            return self.eigen_vectors[:, idx]

    def get_state_probability(self, n=0):
        """
        Get probability of n-th excited state $|\Psi_n|^2$
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize() or diagonalize_sparse()!")
        else:
            return np.abs(self.get_state(n)) ** 2

    def get_full_hamiltonian(self):
        """
        Get the full Hamiltonian matrix H
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize() or diagonalize_sparse()!")
        else:
            return self.hamiltonian

    def get_gs_local_energy(self):
        """
        Get the ground state local energy
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize() or diagonalize_sparse()!")
        else:
            gs = self.get_gs()
            eloc = np.matmul(self.hamiltonian, gs) / gs
            return eloc

    def is_sparse(self):
        return False
