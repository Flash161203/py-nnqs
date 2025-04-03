import torch
from hamiltonian import Hamiltonian
import itertools
import numpy as np
import scipy
import scipy.sparse.linalg


class IsingJ1J2(Hamiltonian):
    """
    This class is used to define Transverse Field Ising Hamiltonian.
    Nearest neighbor interaction along z-axis with magnitude J_1,
    next nearest neighbor interaction along z-axis with magnitude J_2,
    external magnetic field along x-axis with magnitude h

    $H_{IJ} = -J_1 \sum_{<i,j>} \sigma^z_i \sigma^z_j -J_2 \sum_{<<i,j>>} \sigma^z_i \sigma^z_j - h \sum_{i} \sigma^x_i $ 
    """

    def __init__(self, graph, j1=1.0, h=1.0, j2=1.0):
        """
        Construct an Ising J1-J2 model.

        Args:
            j1: magnitude of the nearest neighbor interaction along z-axis
            h: magnitude of external magnetic field along x-axis
            j2: magnitude of the next nearest neighbor interaction along z-axis
        """

        super().__init__(graph)
        self.j1 = j1
        self.h = h
        self.j2 = j2

    def calculate_hamiltonian_matrix(self, samples, num_samples):
        """
        Calculate the Hamiltonian matrix $H_{x,x'}$ from given samples x.
        Only non-zero elements are returned.

        Args:
            samples: The samples 
            num_samples: number of samples

        Return:
            The Hamiltonian where the first column contains the diagonal, which is $-J_1 \sum_{<i,j>} x_i x_j - J_2 \sum_{<<i,j>>} x_i, x_j$.
            The rest of the column contains the off-diagonal, which is -h for every spin flip. 
            Therefore, the number of columns equals the number of particles + 1 and the number of rows = num_samples
        """

        # Diagonal element of the Hamiltonian
        # $-J \sum_{i,j} x_i x_j$
        diagonal = torch.zeros((num_samples,))
        for (s, s_2) in self.graph.bonds:
            diagonal += -self.j1 * samples[:, s] * samples[:, s_2]

        for (s, s_2) in self.graph.bonds_next:
            diagonal += -self.j2 * samples[:, s] * samples[:, s_2]

        diagonal = diagonal.view(num_samples, 1)

        # Off-diagonal element of the Hamiltonian
        # $-h$ for every spin flip
        off_diagonal = torch.full(
            (num_samples, self.graph.num_points), -self.h)
        hamiltonian = torch.cat([diagonal, off_diagonal], dim=1)

        return hamiltonian

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
            In the Ising model, this corresponds to x' where exactly one spin x is flipped. 
            Therefore, the number of columns equals the number of particles + 1 and the number of rows = num_samples
        """

        # Calculate log(\Psi(x)) - log(\Psi(x))
        lvd = model.log_val_diff(samples, samples)

        # Calculate log(\Psi(x')) - log(\Psi(x)) where x' is non-zero when x is flipped at one position.
        for pos in range(self.graph.num_points):
            # Flip spin at position pos
            new_config = samples.clone()
            flipped = new_config[:, pos].neg().view(num_samples, 1)
            if pos == 0:
                new_config = torch.cat((flipped, samples[:, pos + 1:]), dim=1)
            elif pos == self.graph.num_points - 1:
                new_config = torch.cat((samples[:, :pos], flipped), dim=1)
            else:
                new_config = torch.cat(
                    (samples[:, :pos], flipped, samples[:, pos + 1:]), dim=1)

            lvd = torch.cat(
                (lvd, model.log_val_diff(new_config, samples)), dim=1)
        return lvd

    def diagonalize(self):
        """
        Diagonalize the Hamiltonian with exact diagonalization.
        Only works for small systems (<= 10)!
        """
        num_particles = self.graph.num_points
        # Initialize zero Hamiltonian
        H = np.zeros((2 ** num_particles, 2 ** num_particles))

        # Calculate self energy
        for i in range(num_particles):
            togg_vect = np.zeros(num_particles)
            togg_vect[i] = 1
            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_X)
                else:
                    temp = np.kron(temp, np.identity(2))
            H -= self.h * temp

        # Calculate interaction energy
        for i, a in self.graph.bonds:
            togg_vect = np.zeros(num_particles)
            togg_vect[i] = 1
            togg_vect[a] = 1

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_Z)
                else:
                    temp = np.kron(temp, np.identity(2))

            H -= self.j1 * temp

        # Calculate interaction next nearest energy
        for i, a in self.graph.bonds_next:
            togg_vect = np.zeros(num_particles)
            togg_vect[i] = 1
            togg_vect[a] = 1

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_Z)
                else:
                    temp = np.kron(temp, np.identity(2))

            H -= self.j2 * temp

        # Calculate the eigenvalues
        self.eigen_values, self.eigen_vectors = np.linalg.eig(H)
        self.hamiltonian = H

    def diagonalize_sparse(self):
        """
        Diagonalize the Hamiltonian with exact diagonalization using a sparse matrix.
        Only works for small (<= 20) systems!
        """

        num_particles = self.graph.num_points
        num_confs = 2 ** num_particles

        # Constructing the COO sparse matrix
        row_ind = []
        col_ind = []
        data = []
        for row in range(num_confs):
            # print(row, num_confs)
            # configuration in binary 0 1
            conf_bin = format(row, '#0%db' % (num_particles + 2))
            # configuration in binary -1 1
            conf = [1 if c == '1' else -1 for c in conf_bin[2:]]

            # Diagonal = -J1 \sum SiSj -J2 \sum SiSj
            row_ind.append(row)
            col_ind.append(row)
            total_j1 = 0
            for (i, j) in self.graph.bonds:
                total_j1 += conf[i] * conf[j]

            total_j1 *= -self.j1

            total_j2 = 0

            for (i, j) in self.graph.bonds_next:
                total_j2 += conf[i] * conf[j]

            total_j2 *= -self.j2

            data.append(total_j1 + total_j2)

            # Flip one by one
            xor = 1
            for ii in range(num_particles):
                # flipped the configuration
                conf_flipped_bin = format(row ^ xor, '#0%db' % num_particles)

                row_ind.append(row)
                col_ind.append(row ^ xor)
                data.append(-self.h)

                # shift left to flip other bit locations
                xor = xor << 1

        row_ind = np.array(row_ind)
        col_ind = np.array(col_ind)

        data = np.array(data, dtype=float)

        mat_coo = scipy.sparse.coo_matrix((data, (row_ind, col_ind)))

        self.eigen_values, self.eigen_vectors = scipy.sparse.linalg.eigs(
            mat_coo, k=1, which='SR')
        self.hamiltonian = mat_coo

    def get_name(self):
        """ 
        Get the name of the Hamiltonian
        """
        if self.graph.pbc:
            bc = 'pbc'
        else:
            bc = 'obc'
        return 'isingj1j2_%dd_%d_%.3f_%.3f_%.3f_%s' % (
            self.graph.dimension, self.graph.length, self.h,
            self.j1, self.j2, bc)

    def __str__(self):
        return "Ising J1-J2 %dD, h=%.2f, j1=%.2f, j2=%.2f" % (self.graph.dimension, self.h, self.j1, self.j2)

    def to_xml(self):
        str = ""
        str += "<hamiltonian>\n"
        str += "\t<type>ising j1-j2</type>\n"
        str += "\t<params>\n"
        str += "\t\t<j1>%.2f</j1>\n" % self.j1
        str += "\t\t<j2>%.2f</j2>\n" % self.j2
        str += "\t\t<h>%.2f</h>\n" % self.h
        str += "\t</params>\n"
        str += "</hamiltonian>\n"
        return str
