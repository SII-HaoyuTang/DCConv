import math

import numpy as np
import torch
from scipy.special import genlaguerre, sph_harm_y


class HydrogenWaveFunction:
    """
    A class to compute the hydrogen atom wave function using spherical harmonics and associated Laguerre polynomials.
    """

    def __init__(self, n: int, l: int, m: int):
        if not (isinstance(n, int) and n > 0):
            raise ValueError("n must be a positive integer")
        if not (isinstance(l, int) and 0 <= l < n):
            raise ValueError("l must be an integer between 0 and n-1")
        if not (isinstance(m, int) and -l <= m <= l):
            raise ValueError("m must be an integer between -l and l")
        self.n = n
        self.l = l
        self.m = m

    def forward(self, r: float, theta: float, phi: float) -> torch.Tensor:
        """
        Computes the hydrogen atom wave function for given coordinates r, theta, phi.

        :param r: Radial distance
        :param theta: Polar angle in radians
        :param phi: Azimuthal angle in radians
        :return: The value of the hydrogen atom wave function
        """
        if not (isinstance(r, (int, float)) and r >= 0):
            raise ValueError("r must be a non-negative numeric value")
        if not (isinstance(theta, (int, float)) and isinstance(phi, (int, float))):
            raise ValueError("theta and phi must be numeric values")

        # Compute the radial part using associated Laguerre polynomial
        radial_part = self._compute_radial_part(self.n, self.l, r)

        # Compute the angular part using spherical harmonics
        angular_part = self._compute_angular_part(self.l, self.m, theta, phi)

        # Combine the parts to get the wave function
        return torch.tensor(radial_part * angular_part, dtype=torch.float32)

    def _compute_radial_part(self, n: int, l: int, r: float) -> float:
        """
        Computes the radial part of the hydrogen atom wave function using associated Laguerre polynomials.

        :param n: Principal quantum number (positive integer)
        :param l: Azimuthal quantum number (integer between 0 and n-1)
        :param r: Radial distance
        :return: The value of the radial part
        """
        normalization_factor = np.sqrt(
            (2.0 / n) ** 3
            * math.factorial(n - l - 1)
            / (2.0 * n * math.factorial(n + l))
        )
        laguerre_polynomial = genlaguerre(n - l - 1, 2 * l + 1)(2.0 * r / n)
        exponential_term = np.exp(-r / n)
        return (
            normalization_factor
            * (2.0 * r / n) ** l
            * laguerre_polynomial
            * exponential_term
        )

    def _compute_angular_part(self, l: int, m: int, theta: float, phi: float) -> float:
        """
        Computes the angular part of the hydrogen atom wave function using spherical harmonics.

        :param l: Azimuthal quantum number (non-negative integer)
        :param m: Magnetic quantum number (integer between -l and l)
        :param theta: Polar angle in radians
        :param phi: Azimuthal angle in radians
        :return: The value of the angular part
        """
        y_lm = sph_harm_y(m, l, phi, theta)
        if m > 0:
            return np.sqrt(2) * (-1) ** m * np.real(y_lm)
        elif m < 0:
            return np.sqrt(2) * (-1) ** m * np.imag(y_lm)
        else:
            return np.real(y_lm)

    def backward(self, r: float, theta: float, phi: float, gradient: str) -> float:
        """
        Computes the derivative of the hydrogen atom wave function with respect to the specified gradient.

        :param r: Radial distance
        :param theta: Polar angle in radians
        :param phi: Azimuthal angle in radians
        :param gradient: The gradient with respect to which the derivative is computed ('r', 'theta', or 'phi')
        :return: The derivative of the hydrogen atom wave function
        """
        if gradient not in ["r", "theta", "phi"]:
            raise ValueError("Gradient must be either 'r', 'theta', or 'phi'")

        # Compute the radial part
        radial_part = self._compute_radial_part(self.n, self.l, r)

        # Compute the angular part
        angular_part = self._compute_angular_part(self.l, self.m, theta, phi)

        if gradient == "r":
            # Derivative with respect to r
            dr_radial_part = self._compute_radial_part_derivative(self.n, self.l, r)
            return dr_radial_part * angular_part
        elif gradient == "theta":
            # Derivative with respect to theta
            dtheta_angular_part = self._compute_angular_part_derivative(
                self.l, self.m, theta, phi, "theta"
            )
            return radial_part * dtheta_angular_part
        else:
            # Derivative with respect to phi
            dphi_angular_part = self._compute_angular_part_derivative(
                self.l, self.m, theta, phi, "phi"
            )
            return radial_part * dphi_angular_part

    def _compute_radial_part_derivative(self, n: int, l: int, r: float) -> float:
        """
        Computes the derivative of the radial part of the hydrogen atom wave function with respect to r.

        :param n: Principal quantum number (positive integer)
        :param l: Azimuthal quantum number (integer between 0 and n-1)
        :param r: Radial distance
        :return: The derivative of the radial part
        """
        normalization_factor = np.sqrt(
            (2.0 / n) ** 3
            * math.factorial(n - l - 1)
            / (2.0 * n * math.factorial(n + l))
        )
        laguerre_polynomial = genlaguerre(n - l - 1, 2 * l + 1)(2.0 * r / n)
        laguerre_derivative = genlaguerre(n - l - 1, 2 * l + 1).deriv()(2.0 * r / n)
        exponential_term = np.exp(-r / n)
        return (
            normalization_factor
            * (2.0 * r / n) ** l
            * (
                laguerre_derivative * (2.0 / n)
                + laguerre_polynomial * (2.0 * l / r - 1.0 / n)
            )
            * exponential_term
        )

    def _compute_angular_part_derivative(
        self, l: int, m: int, theta: float, phi: float, gradient: str
    ) -> float:
        """
        Computes the derivative of the angular part of the hydrogen atom wave function with respect to the specified gradient.

        :param l: Azimuthal quantum number (non-negative integer)
        :param m: Magnetic quantum number (integer between -l and l)
        :param theta: Polar angle in radians
        :param phi: Azimuthal angle in radians
        :param gradient: The gradient with respect to which the derivative is computed ('theta' or 'phi')
        :return: The derivative of the angular part
        """
        y_lm = sph_harm_y(m, l, phi, theta)
        if gradient == "theta":
            if m > 0:
                return (
                    np.sqrt(2)
                    * (-1) ** m
                    * np.real(
                        m * np.cos(theta) * y_lm
                        - (l + m)
                        * np.sin(theta)
                        * sph_harm_y(m, l, phi, theta + np.pi / 2)
                    )
                )
            elif m < 0:
                return (
                    np.sqrt(2)
                    * (-1) ** m
                    * np.imag(
                        m * np.cos(theta) * y_lm
                        - (l + m)
                        * np.sin(theta)
                        * sph_harm_y(m, l, phi, theta + np.pi / 2)
                    )
                )
            else:
                return np.real(
                    m * np.cos(theta) * y_lm
                    - (l + m) * np.sin(theta) * sph_harm_y(m, l, phi, theta + np.pi / 2)
                )
        else:
            if m > 0:
                return np.sqrt(2) * (-1) ** m * np.real(1j * m * y_lm)
            elif m < 0:
                return np.sqrt(2) * (-1) ** m * np.imag(1j * m * y_lm)
            else:
                return 0.0


if __name__ == "__main__":
    # 创建 HydrogenWaveFunction 的一个实例，设置 n=2, l=1, m=0
    wave_function = HydrogenWaveFunction(n=2, l=1, m=0)

    # 给定空间坐标
    r, theta, phi = [1.0], [math.pi / 4], [math.pi / 2]

    # 计算波函数值
    psi_value = wave_function.forward(r, theta, phi)
    print(f"Wave function value at (r={r}, theta={theta}, phi={phi}): {psi_value}")

    # 计算波函数相对于 r, theta, 和 phi 的导数
    dpsi_dr = wave_function.backward(r, theta, phi, "r")
    dpsi_dtheta = wave_function.backward(r, theta, phi, "theta")
    dpsi_dphi = wave_function.backward(r, theta, phi, "phi")

    # 输出导数值
    print(f"Derivative with respect to r: {dpsi_dr}")
    print(f"Derivative with respect to theta: {dpsi_dtheta}")
    print(f"Derivative with respect to phi: {dpsi_dphi}")
