from typing import List

import torch
import torch.nn as nn

from .polynomials_torch import HWFR


class DCConv3dKernelPolynomials(nn.Module):
    """
    Evaluates a set of 3D polynomials at given positions and applies coefficients to the result, designed for use in
    deep learning models.

    This class is used to compute the value of a series of 3D polynomials at specified positions.
     The polynomials are defined by their order (N), degree (K), and range (M). The evaluation takes into account the
     constraints on the polynomial parameters and then multiplies the evaluated polynomial values with predefined
     coefficients using Einstein summation. The output is a tensor that can be used in further computations within a
     neural network.
    """

    def __init__(self, OutC: int, InC: int, N: int, K: int, M: int):
        """ """
        if not (isinstance(N, int) and N > 0):
            raise ValueError("n must be a positive integer")
        if not (isinstance(K, int) and 0 <= K < N):
            raise ValueError("l must be an integer between 0 and n-1")
        if not (isinstance(M, int) and -K <= M <= K):
            raise ValueError("m must be an integer between -l and l")

        super(DCConv3dKernelPolynomials, self).__init__()
        self.OutC: int = OutC
        self.InC: int = InC
        self.N: int = N
        self.K: int = K
        self.M: int = M
        self.total_polynomial_nums = N * (K + 1) * min(2 * K + 1, M)

        # Create the coefficients with tensor of (OutC, InC, polynomial_nums)
        self.coefficients: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn((OutC, InC, self.total_polynomial_nums))
        )

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of a set of polynomials at given positions and applies coefficients to the result.

        Summary:
        This method evaluates a series of polynomials at specified positions, taking into account
        the constraints on the polynomial parameters (n, k, m). It then multiplies the evaluated
        polynomial values with predefined coefficients using Einstein summation to produce the final output.
        The method is designed to work with tensors, allowing for batch processing of the input positions.

        Parameters:
            position: torch.Tensor
                The tensor containing the positions at which to evaluate the polynomials.

        Returns:
            torch.Tensor
                The tensor containing the computed values after applying the polynomial evaluation and
                coefficient multiplication.

        Raises:
            ValueError
                If the input `position` does not have the expected shape or type.
        """
        # Compute the value of the polynomials with given positions.
        n_values = torch.arange(1, self.N + 1)
        k_values = torch.arange(self.K)
        m_values = torch.arange(-self.M, self.M + 1)

        # Generate all valid (n, k, m) combinations
        valid_combinations: List = [
            (n, k, m)
            for n in n_values
            for k in k_values
            for m in m_values
            if abs(m) <= k and k < n
        ]

        # Convert to tensors
        n_tensor = torch.tensor([n for n, _, _ in valid_combinations], dtype=torch.long)
        k_tensor = torch.tensor([k for _, k, _ in valid_combinations], dtype=torch.long)
        m_tensor = torch.tensor([m for _, _, m in valid_combinations], dtype=torch.long)

        # Apply HWFR in a batch
        poly_values: torch.Tensor = torch.stack(
            [
                HWFR(n.item(), k.item(), m.item()).apply(position)
                for n, k, m in zip(n_tensor, k_tensor, m_tensor)
            ],
            dim=2,
        )

        # poly_values matrix element-times coefficients with einstein sum from (OutN, conv_nums, total_polynomial_nums)
        # and (OutC, InC, total_polynomial_nums) to (OutC, InC, OutN, conv_nums)
        output = torch.einsum("ijk, mnk -> mnij", poly_values, self.coefficients)

        return output
