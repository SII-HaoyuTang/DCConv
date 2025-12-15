import torch
import torch.nn as nn

from .Polynomials import HydrogenWaveFunction


class DCConv3dKernelUnitPolynomials(nn.Module):
    """
    A neural network layer that uses different hydrogen atom wave functions as basis functions.
    The network parameters are the coefficients in front of each basis function.

    :param N: Maximum principal quantum number (n)
    :param L: Maximum azimuthal quantum number (l)
    :param M: Maximum magnetic quantum number (m)
    """

    def __init__(self, N: int, L: int, M: int):
        """
        Initializes the DCConv3d layer.

        :param N: Maximum principal quantum number (n)
        :param L: Maximum azimuthal quantum number (l)
        :param M: Maximum magnetic quantum number (m)
        """
        super(DCConv3dKernelUnitPolynomials, self).__init__()
        self.N = N
        self.L = L
        self.M = M
        self.polynomials = []
        self._Polynomial_initial()

        # Initialize the coefficients for each polynomial as trainable parameters
        self.coefficients = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(len(self.polynomials))]
        )

    def _Polynomial_initial(self):
        """
        Initializes the list of HydrogenWaveFunction objects based on the given N, L, and M.

        :param N: Maximum principal quantum number (n)
        :param L: Maximum azimuthal quantum number (l)
        :param M: Maximum magnetic quantum number (m)
        """

        for n in range(1, self.N + 1):  # n should start from 1
            for l in range(
                0, min(n, self.L + 1)
            ):  # l should be less than or equal to n and L
                for m in range(-l, l + 1):  # m should be between -l and l
                    if abs(m) <= self.M:
                        self.polynomials.append(HydrogenWaveFunction(n, l, m))

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the DCConv3d layer.

        :param position: Position tensor containing r, theta, phi in shape (OutN, n, 3), where
            OutN is the number of output grids, n is the number of convolution grids, and 3 represents
            the coordinates (r, theta, phi).
        :return: The output tensor of shape (OutN, n)
        """
        OutN, n, _ = position.shape
        output = torch.zeros((OutN, n), dtype=position.dtype)

        # First loop over the output grids
        for i in range(OutN):
            # Second loop over the convolution grids
            for j in range(n):
                r, theta, phi = position[i, j, 0], position[i, j, 1], position[i, j, 2]
                # @SII-HaoyuTang
                r = float(r)
                theta = float(theta)
                phi = float(phi)
                wave_function_values = torch.stack(
                    [
                        polynomial.forward(r, theta, phi)
                        for polynomial in self.polynomials
                    ]
                )
                # Claude Code suggests to do this, and you should check @SII-HaoyuTang
                coeffs = torch.stack([c for c in self.coefficients])
                weighted_sum = (wave_function_values * coeffs).sum()
                output[i, j] = weighted_sum

        return output


class DCConv3dKernelPolynomials(nn.Module):
    """
    A neural network layer that combines multiple DCConv3dKernelUnitPolynomials layers to form a 3D convolutional kernel.

    :param OutC: Number of output channels
    :param InC: Number of input channels
    :param N: Maximum principal quantum number (n)
    :param L: Maximum azimuthal quantum number (l)
    :param M: Maximum magnetic quantum number (m)
    """

    def __init__(self, OutC: int, InC: int, N: int, L: int, M: int):
        """
        Initializes the DCConv3dKernelPolynomials layer.

        :param OutC: Number of output channels
        :param InC: Number of input channels
        :param N: Maximum principal quantum number (n)
        :param L: Maximum azimuthal quantum number (l)
        :param M: Maximum magnetic quantum number (m)
        """
        super(DCConv3dKernelPolynomials, self).__init__()
        self.OutC = OutC
        self.InC = InC
        self.N = N
        self.L = L
        self.M = M

        # Create a list of DCConv3dKernelUnitPolynomials layers
        self.kernel_unit_list = nn.ModuleList(
            [DCConv3dKernelUnitPolynomials(N, L, M) for _ in range(OutC * InC)]
        )

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the DCConv3dKernelPolynomials layer.

        :param position: Position tensor containing r, theta, phi in shape (OutN, n, 3), where
            OutN is the number of output grids, n is the number of convolution grids, and 3 represents
            the coordinates (r, theta, phi).
        :return: The output tensor of shape (OutC, InC, OutN, n)
        """
        OutN, n, _ = position.shape
        output = torch.zeros((self.OutC, self.InC, OutN, n), dtype=position.dtype)

        # Loop over the output channels
        for i in range(self.OutC):
            # Loop over the input channels
            for j in range(self.InC):
                # Compute the output for each channel combination
                output[i, j] = self.kernel_unit_list[i * self.InC + j].forward(position)

        return output
