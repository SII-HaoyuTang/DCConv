import torch
import torch.nn as nn

from .polynomials_torch import HWFR

from typing import List


class DCConv3dKernelUnitPolynomials(nn.Module):
    """
    A neural network layer that uses different hydrogen atom wave functions as basis functions.
    The network parameters are the coefficients in front of each basis function.

    :param N: Maximum principal quantum number (n)
    :param K: Maximum azimuthal quantum number (k)
    :param M: Maximum magnetic quantum number (m)
    """

    def __init__(self, N: int, K: int, M: int):
        """
        Initializes the DCConv3d layer.

        :param N: Maximum principal quantum number (n)
        :param K: Maximum azimuthal quantum number (k)
        :param M: Maximum magnetic quantum number (m)
        """
        super(DCConv3dKernelUnitPolynomials, self).__init__()
        self.N: int = N
        self.K: int = K
        self.M: int = M
        self.polynomials: List[torch.autograd.Function] = []
        self._Polynomial_initial()

        # Initialize the coefficients for each polynomial as trainable parameters
        self.coefficients: torch.nn.ParameterList = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(len(self.polynomials))]
        )

    def _Polynomial_initial(self):
        """
        Initializes the polynomial list with HWFR instances based on the specified ranges for n, k, and m.
        """

        for n in range(1, self.N + 1):  # n should start from 1
            for k in range(
                0, min(n, self.K + 1)
            ):  # l should be less than or equal to n and L
                for m in range(-k, k + 1):  # m should be between -l and l
                    if abs(m) <= self.M:
                        self.polynomials.append(HWFR(n, k, m))

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the DCConv3d layer.

        :param position: Position tensor containing r, theta, phi in shape (OutN, conv_nums, 3), where
            OutN is the number of output grids, conv_nums is the number of convolution grids, and 3 represents
            the coordinates (r, theta, phi).
        :return: The output tensor of shape (OutN, conv_nums)
        """
        output: torch.Tensor = torch.stack(
            [
                polynomial(position) * coeffs
                for polynomial, coeffs in (self.polynomials, self.coefficients)
            ],
            dim=0,
        ).sum(dim=0)

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
        self.OutC: int = OutC
        self.InC: int = InC
        self.N: int = N
        self.L: int = L
        self.M: int = M

        # Create a list of DCConv3dKernelUnitPolynomials layers
        self.kernel_unit_list: nn.ModuleList = nn.ModuleList(
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
                output[i, j] = self.kernel_unit_list[i * self.InC + j](position)

        return output
