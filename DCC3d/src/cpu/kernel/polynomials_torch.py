import math
from collections.abc import Callable
from typing import Dict, Tuple, Type

import numpy as np
import torch


class AssociatedLaguerrePoly:
    r"""
    适用于氢原子与导数计算的高效连带拉盖尔多项式实现。

    此类提供了计算特定阶数和次数的连带拉盖尔多项式的功能。通过预先计算系数并使用霍纳法进行数值稳定性的提升，该类能够高效地评估给定x值处的多项式值。
    连带拉盖尔多项式的表达式为 L_n^l (x) = \sum_{k=0}^{n-l} (-1)^{k-1} \frac{(n!)^2}{(n-l-k)!(l+k)!k!} x^k
    该公式自徐光宪《量子化学——基本原理与从头计算法（上册）》P.127 式（3.4.19）。

    如要求得该函数的导数，可以直接令次数加一：
    见徐光宪《量子化学——基本原理与从头计算法（上册）》P.127 式（3.4.14）
    ‘第r次Laguerre多项式的第s阶导数称为（r-s）次s阶的连带Laguerre多项式 ：L_r^s(\rho) = \frac{d^s}{d\rho^s} L_r(\rho)’

    值得注意的是，这与连带拉盖尔函数的一般形式并不相同，并且在这里我们没有进行归一化。

    :raises:
        - ValueError: 如果输入参数n或k不是非负整数。
    """

    def __init__(self, n: int, k: int, dtype=torch.float32):
        """
        更高效的连带拉盖尔多项式实现。

        :param n: 多项式的阶数
        :param k: 多项式次数
        :param dtype: 计算数据类型
        """
        if (not isinstance(n, int)) or (not isinstance(k, int)):
            raise ValueError(f"n and l must be integers, got {n} and {k}")
        self.n: int = n
        self.k: int = k
        self.poly_equal_zero = False
        self.dtype: torch.dtype = dtype

        # 预计算系数
        self.coefficients: torch.Tensor = self._compute_coefficients_vectorized()

    def _compute_coefficients_vectorized(self) -> torch.Tensor | None:
        """
        计算并返回向量化的系数。

        该函数通过预计算阶乘值来加速系数的计算过程。它首先生成一个从0到n+2的阶乘字典，以确保后续计算中所需的所有阶乘值都已准备好。
        接着，使用PyTorch库中的张量操作来高效地处理数组运算，包括确定符号、计算不同部分的阶乘以及最终合成每个系数。
        最后，根据给定公式组合这些值，得到最终的系数向量。如果 n - k < 0，则返回 None。

        :returns: 计算得到的系数向量
        :rtype: torch.Tensor
        """

        def _precompute_factorials() -> Dict[int, int]:
            """预计算阶乘"""
            factorial_cache: Dict[int, int] = {}
            for i in range(self.n + 2):  # +2 确保足够
                factorial_cache[i] = math.factorial(i)
            return factorial_cache

        factorial_cache: Dict[int, int] = _precompute_factorials()

        if self.n - self.k + 1 <= 0:
            self.poly_equal_zero = True
            return None
        m: torch.Tensor = torch.arange(self.n - self.k + 1, dtype=self.dtype)

        # 计算 (-1)^m
        signs: torch.Tensor = torch.where(m % 2 == 0, -1.0, 1.0)

        fact1: torch.Tensor = self.n - self.k - m
        fact2: torch.Tensor = self.k + m
        fact3: torch.Tensor = m

        denom1: torch.Tensor = torch.tensor(
            [factorial_cache[int(_.item())] for _ in fact1],
            dtype=self.dtype,
        )

        denom2: torch.Tensor = torch.tensor(
            [factorial_cache[int(_.item())] for _ in fact2],
            dtype=self.dtype,
        )

        denom3: torch.Tensor = torch.tensor(
            [factorial_cache[int(_.item())] for _ in fact3],
            dtype=self.dtype,
        )

        # 合并系数
        coefficients: torch.Tensor = (
            signs * (factorial_cache[self.n]) ** 2 / (denom1 * denom2 * denom3)
        )

        return coefficients

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of a polynomial at given points using Horner's method.

        Summary:
        This function evaluates a polynomial at specified points. The coefficients of
        the polynomial are stored in the instance and are used alongside the input tensor
        to compute the result through an efficient iterative process known as Horner's
        method. This method minimizes the number of multiplications, making it more
        efficient for evaluating polynomials.

        Parameters:
            x (torch.Tensor): Input tensor containing the points at which to evaluate
                              the polynomial, layout: (OutN, conv_nums)

        Returns:
            torch.Tensor: A tensor containing the evaluated polynomial values at each
                          point in `x`, layout: (OutN, conv_nums)

        Raises:
            RuntimeError: If there is a mismatch in data types that cannot be resolved.
        """
        if self.poly_equal_zero:
            return torch.zeros_like(x)

        # 确保数据类型匹配
        x: torch.Tensor = x.to(self.dtype)
        coeffs: torch.Tensor = self.coefficients.to(x.device)

        # 使用霍纳法（Horner's method）计算多项式
        # 对于多项式 a_n*x^n + ... + a_1*x + a_0
        # 从最高次项开始计算
        result: torch.Tensor = torch.zeros_like(x)
        for i in range(self.n - self.k, -1, -1):
            result: torch.Tensor = result * x + coeffs[i]

        return result


class SphericalHarmonicFunc:
    r"""
    在这个类中实现了球谐函数，其可以被表示为：
     Y_l^m(θ, φ) = N_l^m * P_l^m(cosθ) * exp(imφ)
     见徐光宪《量子化学——基本原理与从头计算法（上册）》P.130 式（3.5.10）
    其中归一化系数 N_l^m = (-1)^{\frac{m+|m|}{2}}\sqrt{\frac{(2l+1)(l-|m|)!}{4\pi(l+|m|)!}}
    连带勒让德多项式 P_l^m(x) = (1-x^2)^{\frac{|m|}{2}} \frac{d^{|m|}}{dx^{|m|}}P_l(x)
    其中 P_l(x) 为勒让德多项式，可以被表示为：
    P_l(x) = \sum_{k=0}^{\left[\frac{l}{2} \right]} (-1)^k \frac{(2l-2k)!}{2^lk!(l-k)!(l-2k)!}x^{l-2k}
    """

    def __init__(
        self,
        k: int,
        m: int,
        dtype: torch.dtype = torch.float32,
        normalization: str = "spherical",
    ):
        """
        初始化球谐函数

        Parameters:
        -----------
        l : int
            角量子数，非负整数
        m : int
            磁量子数，整数，满足 |m| ≤ l
        normalization : str
            归一化类型：
            - 'spherical': 标准球谐函数归一化
            - 'racah': Racah 归一化（无 Condon-Shortley 相位）
            - 'schmidt': Schmidt 半归一化
        """
        if k < 0:
            raise ValueError(f"l must be non-negative, got {k}")
        if abs(m) > k:
            raise ValueError(f"|m| must be ≤ l, got m={m}, l={k}")

        self.k: int = k
        self.m: int = m
        self.m_abs: int = abs(m)
        self.dtype: torch.dtype = dtype
        self.normalization: str = normalization

        # 存储归一化系数
        self.normalization_parameters: torch.Tensor = self._compute_normalization()

        # 预计算用于霍纳法的系数（包含归一化常数和可能的符号因子）
        self.horner_coeffs: torch.Tensor = (
            self._compute_assoc_legendre_coeffs() * self.normalization_parameters
        )

        # 预计算多项式部分的导数系数
        self.horner_coeffs_derivative: torch.Tensor = (
            self._compute_polynomial_derivative_coeffs()
        )

    def _compute_normalization(self) -> torch.Tensor:
        """
        Compute the normalization factor for spherical harmonics based on the specified normalization type.

        The method supports three types of normalization: "spherical", "racah", and "schmidt". Each type
        applies a different formula to compute the normalization factor, which is then returned as a torch.Tensor.

        :raises ValueError: If an unknown normalization type is provided.
        :returns: The computed normalization factor as a torch.Tensor.
        """
        k: int = self.k
        m: int = self.m
        m_abs: int = self.m_abs

        if self.normalization == "spherical":
            # 标准球谐函数归一化，包含 Condon-Shortley 相位因子 (-1)^m
            phase: float = (-1) ** m if m >= 0 else 1
            norm: float = phase * math.sqrt(
                (2 * k + 1)
                * math.factorial(k - m_abs)
                / (4 * math.pi * math.factorial(k + m_abs))
            )
            return torch.Tensor([norm])

        elif self.normalization == "racah":
            # Racah 归一化，无 Condon-Shortley 相位因子
            norm: float = math.sqrt(
                (2 * k + 1)
                * math.factorial(k - m_abs)
                / (4 * math.pi * math.factorial(k + m_abs))
            )
            return torch.Tensor([norm])

        elif self.normalization == "schmidt":
            # Schmidt 半归一化
            if m_abs == 0:
                norm: float = math.sqrt((2 * k + 1) / (4 * math.pi))
            else:
                norm: float = math.sqrt(
                    2
                    * (2 * k + 1)
                    * math.factorial(k - m_abs)
                    / (4 * math.pi * math.factorial(k + m_abs))
                )
            return torch.Tensor([norm])

        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

    def _compute_assoc_legendre_coeffs(self) -> torch.Tensor:
        """
        Compute the associated Legendre polynomial coefficients for given degree and order.

        This method calculates the coefficients of the associated Legendre polynomial P_l^m(x) for a specified degree `k` and order `m_abs`. The calculation involves computing the standard Legendre polynomial coefficients and then differentiating them `m_abs` times. The resulting coefficients are used in the representation of spherical harmonics or other related applications where associated Legendre polynomials are required.

        :returns: A tensor containing the coefficients of the associated Legendre polynomial.
        :rtype: torch.Tensor
        """
        k: int = self.k
        m_abs: int = self.m_abs

        if m_abs == 0:
            # 当 m=0 时，就是勒让德多项式
            return self._compute_standard_legendre_coeffs(k)

        # 计算标准勒让德多项式系数
        legendre_coeffs: torch.Tensor = self._compute_standard_legendre_coeffs(k)

        # 对多项式系数求 m_abs 阶导数
        coeffs: torch.Tensor = legendre_coeffs.clone()

        for _ in range(m_abs):
            # 对多项式系数求导：a_n * x^n -> n * a_n * x^{n-1}
            new_coeffs: torch.Tensor = torch.zeros_like(coeffs)
            for n in range(1, len(coeffs)):
                new_coeffs[n - 1] = n * coeffs[n]
            coeffs: torch.Tensor = new_coeffs

        # 注意：连带勒让德函数 P_l^m(x) 包含因子 (1-x^2)^{m/2}
        # 这个因子将在 __call__ 中单独处理

        return coeffs

    def _compute_standard_legendre_coeffs(self, k: int) -> torch.Tensor:
        """
        Computes the standard Legendre polynomial coefficients for a given degree.

        Summary:
        This method calculates the coefficients of the standard Legendre polynomial
        of a specified degree using the Rodrigues formula. The coefficients are
        normalized by dividing by 2^k, where k is the degree of the polynomial.
        The resulting coefficients are returned as a torch.Tensor object.

        Parameters:
            k (int): Degree of the Legendre polynomial to compute.

        Returns:
            torch.Tensor: A tensor containing the computed coefficients of the
            Legendre polynomial, with the dtype set to torch.float64.

        Raises:
            No specific exceptions are documented here; refer to the source code
            for detailed error handling.
        """
        coeffs: torch.Tensor = torch.zeros(k + 1, dtype=torch.float64)

        # 使用 Rodrigues 公式的展开系数
        for i in range(k // 2 + 1):
            coeff: float = (-1) ** i * math.comb(k, i) * math.comb(2 * k - 2 * i, k)
            power: int = k - 2 * i
            coeffs[power] = coeff

        coeffs /= 2**k
        return coeffs

    def _compute_polynomial_derivative_coeffs(self) -> torch.Tensor:
        """

        Compute the derivative coefficients of a polynomial given its Horner coefficients.

        The function calculates the coefficients of the derivative of a polynomial, which is represented
        by its Horner (nested) form. The input is the tensor of coefficients, and the output is the
        tensor representing the coefficients of the derivative polynomial. If the input polynomial is
        constant or empty, it returns an empty tensor.

        Returns:
            torch.Tensor: Coefficients of the derivative polynomial.
        """
        coeffs: torch.Tensor = self.horner_coeffs

        if len(coeffs) <= 1:
            return torch.zeros(0, dtype=coeffs.dtype)

        # 对于多项式 Σ a_n x^n，导数为 Σ n * a_n * x^{n-1}
        deriv_coeffs: torch.Tensor = torch.zeros(len(coeffs) - 1, dtype=coeffs.dtype)
        for n in range(1, len(coeffs)):
            deriv_coeffs[n - 1] = n * coeffs[n]

        return deriv_coeffs

    def __call__(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the function for given theta and phi.

        Summary:
        This method evaluates the function based on the provided theta and phi values. It handles both positive and negative m values by calling the appropriate internal method.

        Parameters:
            theta (torch.Tensor): The theta value for the evaluation, layout: (OutN, conv_nums)
            phi (torch.Tensor): The phi value for the evaluation, layout: (OutN, conv_nums)

        Returns:
            torch.Tensor: The result of the function evaluation, layout: (OutN, conv_nums)
        """
        if self.m >= 0:
            return self._call_positive_m(theta, phi)
        else:
            return self._call_negative_m(theta, phi)

    def _call_positive_m(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        计算给定角度 theta 和 phi 的正 m 值对应的函数值。

        详细说明：
        该函数用于计算球谐函数在给定角度 theta（极角）和 phi（方位角）处的值。具体来说，它处理的是 m 为正值的情况。通过使用霍纳法计算多项式部分，并结合三角函数来完成整个计算过程。

        参数:
            theta (torch.Tensor): 极角 theta 的张量。
            phi (torch.Tensor): 方位角 phi 的张量。

        返回:
            torch.Tensor: 计算得到的函数值。

        异常:
            无

        注意:
            - 该函数假设输入的 theta 和 phi 是合法的角度值。
            - 多项式系数从实例变量 `horner_coeffs` 中获取。
            - 如果 m 的绝对值大于 0，则会计算 (sinθ)^m 并将其乘到多项式的值上。
            - 最后，结果会乘以方位角部分 cos/sin^{imφ}。
        """
        # x = cosθ
        x: torch.Tensor = torch.cos(theta)

        # 使用霍纳法计算多项式部分（包含归一化常数）
        coeffs: torch.Tensor = self.horner_coeffs.to(theta.device)
        poly_val: torch.Tensor = torch.zeros_like(x)
        for i in range(len(coeffs) - 1, -1, -1):
            poly_val = poly_val * x + coeffs[i]

        # 乘以 (1-x^2)^{|m|/2} = (sinθ)^m
        if self.m_abs > 0:
            sin_theta: torch.Tensor = torch.sin(theta)
            # 确保非负（处理数值误差）
            sin_theta = torch.clamp(sin_theta, min=0.0)
            factor: torch.Tensor = torch.pow(sin_theta, self.m_abs)
            poly_val = poly_val * factor

        # 乘以方位角部分 \cos^{mφ}
        azimuth: torch.Tensor = torch.cos(self.m_abs * phi)
        result: torch.Tensor = poly_val * azimuth

        return result

    def _call_negative_m(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """计算 m < 0 时的球谐函数值（复数形式）"""
        # x = cosθ
        x: torch.Tensor = torch.cos(theta)

        # 使用霍纳法计算多项式部分（包含归一化常数）
        coeffs: torch.Tensor = self.horner_coeffs.to(theta.device)
        poly_val: torch.Tensor = torch.zeros_like(x)
        for i in range(len(coeffs) - 1, -1, -1):
            poly_val = poly_val * x + coeffs[i]

        # 乘以 (1-x^2)^{|m|/2} = (sinθ)^{|m|}
        if self.m_abs > 0:
            sin_theta: torch.Tensor = torch.sin(theta)
            # 确保非负（处理数值误差）
            sin_theta = torch.clamp(sin_theta, min=0.0)
            factor: torch.Tensor = torch.pow(sin_theta, self.m_abs)
            poly_val = poly_val * factor

        # 乘以方位角部分 \cos^{mφ}
        azimuth: torch.Tensor = torch.sin(self.m_abs * phi)
        result: torch.Tensor = poly_val * azimuth

        return result

    def backward(
        self, theta: torch.Tensor, phi: torch.Tensor, saved: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        计算球谐函数的梯度（对 θ 和 φ 的偏导数），在m>=0的情况下有：
        \frac{dY}{d\theta} = N\cos(m\phi)\frac{dP_l^m}{d\cos\theta} \frac{d\cos\theta}{d\theta}
                           = -N\cos(m\phi)\sin(\theta) \frac{dP_l^m}{d\cos\theta}
        其中，令x = cos(\theta)，\frac{dP_l^m}{d\cos\theta}可以表示为：
            \frac{dP_l^m}{d\cos\theta} = \frac{\partial}{\partial x} (1-x^2)^{|m|/2} Poly(x)
                                       = -|m|(1-x^2)^{|m|/2-1} x Poly(x) + (1-x^2)^{|m|/2}Poly'(x)
        其中 Poly代表多项式部分的值，由于 x=\cos(\theta)，所以有：
            \frac{dY}{d\theta} = -N\cos(m\phi)\sin(\theta)[-|m|\sin^{|m|-2}(\theta) x Poly + \sin^{|m|}\theta Poly']
                               = \sin^{|m|-1}(\theta) N \cos(m\phi)(|m|Poly - \sin^2\theta Poly')
                               = \frac{1}{\sin\theta} (\sin^{|m|}\theta N cos(m\phi) Poly|m| - N\cos(m\phi)sin^{\m\+2}\theta Poly')
                               = \frac{1}{\sin\theta} (|m| Saved - N \cos(m\phi) sin^{|m|+2}\theta Poly')
        最后一步是由于前向计算中已经保存的数据 Saved = N \cos(m\phi) sin^{|m|} \theta Poly =N \cos(m\phi) P(cos\theta)
        对于\phi，有：
        \frac{\partial Y}{\partial \phi} = - N \sin(m \phi) P(\cos\theta) = - \frac{saved}{\sin(m\phi)} / \cos(m\phi)

        Returns:
        --------
        (dY/dθ, dY/dφ)
        """
        if self.m >= 0:
            return self._gradient_positive_m(theta, phi, saved)
        else:
            return self._gradient_negative_m(theta, phi, saved)

    def _gradient_positive_m(
        self, theta: torch.Tensor, phi: torch.Tensor, saved: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the gradient of a function with respect to theta and phi for positive m values.
        The function computes derivatives using saved values, trigonometric functions, and polynomial coefficients.
        It handles special cases where sin(theta) or cos(m*phi) is close to zero to avoid numerical instability.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the derivative of the function with respect to theta and phi.

        Raises:
            None

        Args:
            theta (torch.Tensor): The theta angles.
            phi (torch.Tensor): The phi angles.
            saved (torch.Tensor): Precomputed values used in the derivative calculation.
        """
        # 计算 x = cosθ 和 sinθ
        x: torch.Tensor = torch.cos(theta)
        sin_theta: torch.Tensor = torch.sin(theta)
        # 处理sin(\theta)=0的情况，在这种情况下正向计算的值本身为零，故导数为0
        sin_theta_judge: torch.Tensor = torch.where(
            sin_theta < 10e-4, torch.inf, sin_theta
        )
        mphi: torch.Tensor = self.m_abs * phi
        sin_mphi: torch.Tensor = torch.sin(mphi)
        cos_mphi: torch.Tensor = torch.cos(mphi)
        # 同上
        cos_mphi_judge: torch.Tensor = torch.where(
            torch.logical_and(-10e-5 < cos_mphi, cos_mphi < 10e-5), torch.inf, cos_mphi
        )

        # 提取多项式部分导数
        deriv_coeffs: torch.Tensor = self.horner_coeffs_derivative.to(theta.device)

        # 计算多项式导数值Poly'
        poly_deriv_val: torch.Tensor = torch.zeros_like(x)
        if len(deriv_coeffs) > 0:
            for i in range(len(deriv_coeffs) - 1, -1, -1):
                poly_deriv_val = poly_deriv_val * x + deriv_coeffs[i]

        # \frac{1}{\sin\theta} (|m| Saved - N \cos(m\phi) sin^{|m|+2}\theta Poly')
        dY_dtheta: torch.Tensor = (
            self.m_abs * saved
            - self.normalization_parameters
            * cos_mphi
            * sin_theta ** (self.m_abs + 2)
            * poly_deriv_val
        ) / sin_theta_judge

        # dY/dφ = - \frac{saved}{\cos(m\phi)}\sin(m\phi)
        dY_dphi: torch.Tensor = -saved * sin_mphi / cos_mphi_judge

        return dY_dtheta, dY_dphi

    def _gradient_negative_m(
        self, theta: torch.Tensor, phi: torch.Tensor, saved: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the gradient of a function with respect to theta and phi for negative m values.

        Summary:
        This function computes the partial derivatives of a given function with respect to theta and phi, specifically tailored for cases where m (a parameter) is negative. It handles special cases where sin(theta) or sin(m*phi) are close to zero to avoid numerical instability.

        Args:
            theta: A tensor representing the theta values.
            phi: A tensor representing the phi values.
            saved: A tensor containing precomputed values used in the derivative calculation.

        Returns:
            A tuple of two tensors: (dY_dtheta, dY_dphi)
            - dY_dtheta: The derivative of the function with respect to theta.
            - dY_dphi: The derivative of the function with respect to phi.

        Raises:
            No specific exceptions are raised within this method. However, it relies on PyTorch operations which may raise exceptions under certain conditions, such as if the input tensors are not on the same device or have incompatible shapes.
        """
        # 计算 x = cosθ 和 sinθ
        x: torch.Tensor = torch.cos(theta)
        sin_theta: torch.Tensor = torch.sin(theta)
        # 处理sin(\theta)=0的情况，在这种情况下正向计算的值本身为零，故导数为0  # noqa: F821
        sin_theta_judge: torch.Tensor = torch.where(
            sin_theta < 10e-4, torch.inf, sin_theta
        )
        mphi: torch.Tensor = self.m_abs * phi
        cos_mphi: torch.Tensor = torch.sin(mphi)
        sin_mphi: torch.Tensor = torch.cos(mphi)
        # 同上
        sin_mphi_judge: torch.Tensor = torch.where(
            torch.logical_and(-10e-5 < sin_mphi, sin_mphi < 10e-5), torch.inf, sin_mphi
        )

        # 提取多项式部分导数
        deriv_coeffs: torch.Tensor = self.horner_coeffs_derivative.to(theta.device)

        # 计算多项式导数值Poly'
        poly_deriv_val: torch.Tensor = torch.zeros_like(x)
        if len(deriv_coeffs) > 0:
            for i in range(len(deriv_coeffs) - 1, -1, -1):
                poly_deriv_val = poly_deriv_val * x + deriv_coeffs[i]

        # \frac{1}{\sin\theta} (|m| Saved - N \sin(m\phi) sin^{|m|+2}\theta Poly')
        dY_dtheta: torch.Tensor = (
            self.m_abs * saved
            - self.normalization_parameters
            * sin_mphi
            * sin_theta ** (self.m_abs + 2)
            * poly_deriv_val
        ) / sin_theta_judge

        # dY/dφ = \frac{saved}{\cos(m\phi)}\sin(m\phi)
        dY_dphi: torch.Tensor = saved * cos_mphi / sin_mphi_judge

        return dY_dtheta, dY_dphi


class AssociatedLaguerreSeries:
    """
    Class for generating and managing a series of associated Laguerre polynomials.

    This class provides functionality to create, update, and access a dictionary
    of callable functions representing associated Laguerre polynomials. The
    polynomials are indexed by their degree (n) and order (k), with the ability to
    extend the series up to a specified maximum n and k. The class ensures that
    all polynomial functions are stored in a dictionary, allowing for efficient
    access and computation.

    :param max_n: Maximum degree of the associated Laguerre polynomials.
    :type max_n: int
    :param max_k: Maximum order of the associated Laguerre polynomials.
    :type max_k: int
    :param dtype: Data type for the polynomial computations.
    :type dtype: torch.dtype

    :raises ValueError: If `max_n` is not a positive integer or if `max_k` is not
                        an integer between 0 and `max_n - 1`.

    :methods:
    - __call__(n, k): Returns the callable function for the associated Laguerre
                      polynomial of degree n and order k.
    - extend(dest_n, dest_l): Extends the series to include polynomials up to
                              the specified degree and order.
    - check(n, k): Checks if the associated Laguerre polynomial of degree n and
                   order k exists in the series.
    """

    def __init__(
        self, max_n: int = 7, max_k: int = 7, dtype: torch.dtype = torch.float32
    ):
        self.max_n: int = max_n
        self.max_k: int = max_k
        self.dtype: torch.dtype = dtype
        self.associated_laguerre_series: Dict[tuple[int, int], Callable] = (
            self._initial_associated_laguerre_funcs()
        )

    def _initial_associated_laguerre_funcs(self) -> Dict[tuple[int, int], Callable]:
        """
        Initializes and returns a dictionary of associated Laguerre functions.

        Summary:
        This method generates a dictionary where the keys are tuples representing (n, k) pairs,
        and the values are callable instances of AssociatedLaguerrePoly. The dictionary is populated
        for all n from 1 to max_n (inclusive) and for all k from 0 to max_k (inclusive).

        Returns:
            Dict[tuple[int, int], Callable]: A dictionary mapping (n, k) pairs to their respective
            associated Laguerre polynomial functions.
        """
        associated_laguerre_funcs: Dict[tuple[int, int], Callable] = {}
        for n in range(self.max_n):
            for k in range(self.max_k + 1):
                key: tuple[int, int] = (n + 1, k)
                func: Callable = AssociatedLaguerrePoly(n + 1, k, self.dtype)
                associated_laguerre_funcs[key] = func
        return associated_laguerre_funcs

    def _update_associated_laguerre_funcs(self) -> None:
        """
        Updates the associated Laguerre functions for the current instance.

        Summary:
        This method iterates over a range of values for n and k, checking if a specific
        condition is met. If the condition is not met, it creates a new AssociatedLaguerrePoly
        function and stores it in the associated_laguerre_series dictionary with a key
        that is a tuple of (n + 1, k).

        Returns:
            None

        Raises:
            Any exceptions raised by the AssociatedLaguerrePoly constructor or during
            dictionary operations.
        """
        for n in range(self.max_n):
            for k in range(self.max_k + 1):
                if self.check(n + 1, k):
                    continue
                else:
                    key: tuple[int, int] = (n + 1, k)
                    func: Callable = AssociatedLaguerrePoly(n + 1, k, self.dtype)
                    self.associated_laguerre_series[key] = func

    def check(self, n: int, k: int) -> bool:
        """
        Checks if the associated Laguerre series for given n and k is available.

        Summary:
        This method verifies the presence of an associated Laguerre series for the specified
        values of n and k in the internal storage. It returns True if the series exists, otherwise False.

        Parameters:
        n (int): The first parameter for the associated Laguerre series.
        k (int): The second parameter for the associated Laguerre series.

        Returns:
        bool: True if the associated Laguerre series for n and k is present, False otherwise.
        """
        if (n, k) in self.associated_laguerre_series:
            return True
        else:
            return False

    def extend(self, dest_n: int, dest_l: int) -> None:
        """
        Extends the current object's maximum values and updates associated Laguerre functions based on the provided destination values.

        Parameters:
        dest_n (int): The destination n value to compare and potentially set as the new maximum.
        dest_l (int): The destination l value to compare and potentially set as the new maximum.

        Raises:
        ValueError: If either dest_n or dest_l is negative.

        Returns:
        None
        """
        self.max_k = max(dest_l, self.max_k)
        self.max_n = max(dest_n, self.max_n)
        self._update_associated_laguerre_funcs()

    def __call__(self, n: int, k: int) -> Callable:
        if n > self.max_n or k > self.max_k:
            self.extend(n, k)
        return self.associated_laguerre_series[(n, k)]


class SphericalHarmonicSeries:
    """
    Represents a series of spherical harmonic functions, allowing for the dynamic extension and retrieval
    of these functions based on specified degrees (k) and orders (m).

    This class is designed to manage a collection of spherical harmonic functions, which are commonly used in
    spherical coordinate systems for representing functions defined on the surface of a sphere. It supports
    initialization with a maximum degree and order, as well as extending the series beyond its initial bounds
    dynamically.

    Attributes:
        max_k (int): The maximum degree (k) of the spherical harmonics.
        max_m (int): The maximum order (m) of the spherical harmonics.
        dtype (torch.dtype): The data type for the spherical harmonic functions.

    Methods:
        extend: Extends the series to include new maximum values for k and m.
        __call__: Retrieves or generates and then returns a spherical harmonic function for given k and m.
    """

    def __init__(
        self, max_k: int = 3, max_m: int = 3, dtype: torch.dtype = torch.float32
    ):
        if not (isinstance(max_k, int) and 0 <= max_k):
            raise ValueError("l must be an integer greater than 0 ")
        if not (isinstance(max_m, int) and -max_k <= max_m <= max_k):
            raise ValueError("m must be an integer between -l and l")

        self.max_k: int = max_k
        self.max_m: int = max_m
        self.dtype: torch.dtype = dtype
        self.spherical_harmonic_series: Dict[tuple[int, int], Callable] = (
            self._initial_spherical_harmonic_funcs()
        )

    def _initial_spherical_harmonic_funcs(self) -> Dict[tuple[int, int], Callable]:
        """
        Generates a dictionary of spherical harmonic functions based on the maximum degree and order.

        Summary:
        This function creates a collection of spherical harmonic functions, each associated with a specific degree and order. The functions are stored in a dictionary where the keys are tuples representing the degree (k) and order (m), and the values are the corresponding callable spherical harmonic functions. The range of degrees and orders is determined by the `max_k` and `max_m` attributes of the class instance. The data type for the computations is specified by the `dtype` attribute.

        Returns:
            Dict[tuple[int, int], Callable]: A dictionary containing spherical harmonic functions indexed by their degree and order.
        """
        spherical_harmonic_series: Dict[tuple[int, int], Callable] = {}
        for k in range(self.max_k + 1):
            for m in range(-self.max_m, self.max_m + 1):
                if abs(m) <= k:
                    func: Callable = SphericalHarmonicFunc(k, m, self.dtype)
                    spherical_harmonic_series[(k, m)] = func
        return spherical_harmonic_series

    def _update_spherical_harmonic_funcs(self) -> None:
        """
        Updates the spherical harmonic functions for the current instance.

        Summary:
        This method iterates over a range of k and m values, checks if a specific
        condition is met using `self.check(k, m)`, and if not, creates a new
        spherical harmonic function. The newly created function is then stored in
        the `spherical_harmonic_series` dictionary with (k, m) as the key.

        Returns:
            None

        Raises:
            Any exceptions raised by SphericalHarmonicFunc or self.check.
        """
        for k in range(self.max_k + 1):
            for m in range(-self.max_m, self.max_m + 1):
                if self.check(k, m):
                    continue
                else:
                    func: Callable = SphericalHarmonicFunc(k, m, self.dtype)
                    self.spherical_harmonic_series[(k, m)] = func

    def check(self, k: int, m: int) -> bool:
        """
        Checks if the spherical harmonic series for given k and m is available.

        Args:
            k (int): The degree of the spherical harmonic.
            m (int): The order of the spherical harmonic.

        Returns:
            bool: True if the spherical harmonic series for the specified k and m exists, False otherwise.
        """
        if (k, m) in self.spherical_harmonic_series:
            return True
        else:
            return False

    def extend(self, dest_k: int, dest_m: int) -> None:
        """
        Extends the current object's maximum values for k and m, and updates
        the spherical harmonic functions accordingly.

        Parameters:
            dest_k (int): The destination k value to compare with the current max_k.
            dest_m (int): The destination m value to compare with the current max_m.

        Raises:
            ValueError: If either dest_k or dest_m is not a positive integer.

        Returns:
            None
        """
        self.max_k = max(dest_k, self.max_k)
        self.max_m = max(dest_m, self.max_m)
        self._update_spherical_harmonic_funcs()

    def __call__(self, k: int, m: int) -> Callable:
        """
        Call method for the class to get a specific spherical harmonic function.

        This method checks if the requested spherical harmonic function for given k and m
        is within the current range. If not, it extends the range to include the requested
        function. It then returns the corresponding spherical harmonic function from the
        precomputed series.

        Args:
            k (int): The degree of the spherical harmonic.
            m (int): The order of the spherical harmonic.

        Returns:
            Callable: The spherical harmonic function corresponding to the given k and m.

        Raises:
            Any potential exceptions raised by the extend method or when accessing
            self.spherical_harmonic_series are not documented here but should be handled
            appropriately in the calling code.
        """
        if k > self.max_k or m > self.max_m:
            self.extend(k, m)
        return self.spherical_harmonic_series[(k, m)]


assoc_laguerre_series = AssociatedLaguerreSeries()
spher_harmonic_series = SphericalHarmonicSeries()


class HydrogenWaveFunc:
    """
    Represents the wave function of a hydrogen atom, allowing for the calculation
    of its value at given positions and the computation of gradients for backpropagation.
    This class is designed to work with PyTorch tensors and supports automatic differentiation.

    The wave function is defined by quantum numbers n, k, and m, which correspond
    to the principal, angular momentum, and magnetic quantum numbers, respectively.
    It calculates both the radial and angular parts of the wave function and combines
    them to produce the final result. The class also provides methods for forward
    and backward passes, essential for training neural networks that incorporate
    this physical model.

    :param n: Principal quantum number (must be a positive integer)
    :param k: Angular momentum quantum number (integer between 0 and n-1)
    :param m: Magnetic quantum number (integer between -k and k)
    :param dtype: Data type for the calculations (default is torch.float32)
    """

    def __init__(self, n: int, k: int, m: int, dtype: torch.dtype = torch.float32):
        if not (isinstance(n, int) and n > 0):
            raise ValueError("n must be a positive integer")
        if not (isinstance(k, int) and 0 <= k < n):
            raise ValueError("l must be an integer between 0 and n-1")
        if not (isinstance(m, int) and -k <= m <= k):
            raise ValueError("m must be an integer between -l and l")
        self.n: int = n
        self.k: int = k
        self.m: int = m
        self.dtype: torch.dtype = dtype

        self.associated_laguerre: Callable = assoc_laguerre_series(n + k, 2 * k + 1)
        self.spherical_harmonic: Callable = spher_harmonic_series(k, m)

        self.associated_laguerre_derivative: Callable = assoc_laguerre_series(
            n + k, 2 * k + 2
        )

        self.radial_normalization_factor: torch.Tensor = torch.Tensor(
            [np.sqrt(math.factorial(n - k - 1) / (2.0 * n * math.factorial(n + k)))]
        )

    def forward(
        self,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the wave function for given positions using a combination of radial and angular parts.
        The radial part is calculated using an associated Laguerre polynomial, and the angular part is computed
        using spherical harmonics. The final wave function is the product of these two components.

        Parameters:
        position (torch.Tensor): A tensor representing the 3D positions (r, theta, phi) for which to compute
        the wave function. The shape should be (batch_size, num_positions, 3).

        Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the following elements:
        - psi: The overall wave function, result of multiplying the radial and angular parts.
        - radial_part: The radial component of the wave function.
        - laguerre_value: The value of the associated Laguerre polynomial used in the radial part.
        - angular_part: The angular component of the wave function, derived from spherical harmonics.

        Raises:
        - Any specific exceptions or errors that this function might raise due to invalid inputs or
          computational issues should be documented here if known.
        """
        r: torch.Tensor = position[:, :, 0]
        theta: torch.Tensor = position[:, :, 1]
        phi: torch.Tensor = position[:, :, 2]
        # 计算径向部分
        laguerre_value: torch.Tensor = self.associated_laguerre(r)

        radial_part: torch.Tensor = (
            torch.exp(-r / 2)
            * (r**self.k)
            * self.radial_normalization_factor
            * laguerre_value
        )

        # 计算角向部分
        angular_part: torch.Tensor = self.spherical_harmonic(phi, theta)

        # 结合两部分得到波函数
        psi: torch.Tensor = radial_part * angular_part

        return psi, radial_part, laguerre_value, angular_part

    def backward(
        self, grad_output: torch.Tensor, saved_data: torch.Tensor
    ) -> torch.Tensor:
        """

        Computes the backward pass for a function involving radial and angular components, typically used in spherical harmonics or similar mathematical constructs. This function calculates the gradient of the output with respect to the input position, using precomputed radial and angular parts.

        Parameters:
        ----------
        grad_output: torch.Tensor
            The gradient of the loss with respect to the output.
        saved_data: torch.Tensor
            A tensor containing saved data from the forward pass, including position, radial part, laguerre value, and angular part.

        Returns:
        -------
        torch.Tensor
            A tensor representing the gradients with respect to the input position, stacked along the first dimension for each component (dr, dtheta, dphi).

        Raises:
        ------
        ValueError
            If `saved_data` does not contain the required number of elements.

        Notes:
        -----
        - The method assumes that `saved_data` is structured as (position, radial_part, laguerre_value, angular_part).
        - It uses the provided `grad_output` to compute the gradients with respect to the input position.
        - The calculation involves both the radial and angular derivatives, combining them to form the final gradient.

        See Also:
        --------
        spherical_harmonic: Method for computing the spherical harmonic and its derivatives.
        associated_laguerre_derivative: Method for calculating the derivative of the associated Laguerre polynomial.
        """
        position, radial_part, laguerre_value, angular_part = saved_data

        r, theta, phi = position[:, :, 0], position[:, :, 1], position[:, :, 2]

        # 径向部分  \frac{\partial \exp^{-r/2}*r^l*L}{\partial r} = l*\exp^{-r/2}*r^{l-1}*L - 0.5 * \exp^{-r/2}*r^l*L + \exp^{-r/2}*r^l*L'
        # 将上式中可以被提出的前向部分提出，就得到了实际使用的表达式
        dradial_dr: torch.Tensor = radial_part * (
            self.k / r - 0.5 + self.associated_laguerre_derivative(r) / laguerre_value
        )

        # 角向部分
        dangular_dtheta, dangular_dphi = self.spherical_harmonic.backward(
            theta, phi, angular_part
        )

        # 计算梯度
        dpsi_dr = grad_output * (dradial_dr * angular_part)
        dpsi_dtheta = grad_output * (radial_part * dangular_dtheta)
        dpsi_dphi = grad_output * (radial_part * dangular_dphi)

        result = torch.stack([dpsi_dr, dpsi_dtheta, dpsi_dphi], dim=-1)

        return result

    def __call__(self, position: torch.Tensor):
        return self.forward(position)


class HydrogenWaveFuncsSeries:
    """
    Represents a series of hydrogen wave functions, allowing for the dynamic generation and retrieval
    of wave functions based on quantum numbers n, l (k), and m. The class supports extending its
    range of precomputed wave functions as needed.

    Attributes:
        max_n (int): The maximum principal quantum number (n) for which to precompute wave functions.
        max_k (int): The maximum azimuthal quantum number (l, denoted as k in the code) for which to
                     precompute wave functions.
        max_m (int): The maximum magnetic quantum number (m) for which to precompute wave functions.
        dtype (torch.dtype): The data type used for computations.
    """

    def __init__(
        self,
        max_n: int = 4,
        max_k: int = 3,
        max_m: int = 3,
        dtype: torch.dtype = torch.float32,
    ):
        if not (isinstance(max_n, int) and max_n > 0):
            raise ValueError("n must be a positive integer")
        if not (isinstance(max_k, int) and 0 <= max_k < max_n):
            raise ValueError("l must be an integer between 0 and n-1")
        if not (isinstance(max_m, int) and -max_k <= max_m <= max_k):
            raise ValueError("m must be an integer between -l and l")

        self.max_n: int = max_n
        self.max_k: int = max_k
        self.max_m: int = max_m
        self.dtype: torch.dtype = dtype
        self.hydrogen_wave_funcs_series: Dict[tuple[int, int, int], Callable] = (
            self._initial_hydrogen_wave_funcs()
        )

    def _initial_hydrogen_wave_funcs(self) -> Dict[tuple[int, int, int], Callable]:
        hydrogen_wave_funcs_series: Dict[tuple[int, int, int], Callable] = {}
        for n in range(self.max_n):
            for k in range(self.max_k + 1):
                for m in range(-self.max_m, self.max_m + 1):
                    if abs(m) <= k and k < n + 1:
                        func: Callable = HydrogenWaveFunc(n + 1, k, m, self.dtype)
                        hydrogen_wave_funcs_series[(n + 1, k, m)] = func
        return hydrogen_wave_funcs_series

    def _update_hydrogen_wave_funcs(self) -> None:
        for n in range(self.max_n):
            for k in range(self.max_k + 1):
                for m in range(-self.max_m, self.max_m + 1):
                    if self.check(n, k, m):
                        continue
                    else:
                        func: Callable = HydrogenWaveFunc(n + 1, k, m, self.dtype)
                        self.hydrogen_wave_funcs_series[(n + 1, k, m)] = func

    def check(self, n: int, k: int, m: int) -> bool:
        if not self.hydrogen_wave_funcs_series[(n, k, m)]:
            return False
        else:
            return True

    def extend(self, dest_n: int, dest_k: int, dest_m: int) -> None:
        self.max_n = max(dest_n, self.max_n)
        self.max_k = max(dest_k, self.max_k)
        self.max_m = max(dest_m, self.max_m)
        self._update_hydrogen_wave_funcs()

    def __call__(self, n: int, k: int, m: int) -> Callable:
        if n > self.max_n or k > self.max_k or m > self.max_m:
            self.extend(n, k, m)
        return self.hydrogen_wave_funcs_series[(n, k, m)]


hydrogen_wave_funcs_series = HydrogenWaveFuncsSeries()


class HydrogenWaveFunctionRegistry:
    """
    注册和提供氢原子波函数的类。

    该类用于动态创建和缓存基于给定量子数 (n, k, m) 的氢原子波函数。
    这些波函数被实现为 `torch.autograd.Function` 的子类，以便在计算图中进行自动微分。
    通过此注册表，可以高效地获取或创建指定量子数的波函数，并且保证相同的量子数组合只创建一次。

    :raises: 如果提供的量子数超出初始化时设定的最大值，则会抛出异常。
    """

    _function_cache: Dict[tuple, Type[torch.autograd.Function]] = {}

    def __init__(self, max_n: int, max_l: int, max_m: int):
        self._initial_hydrogen_wave_function_register(max_n, max_l, max_m)

    def _initial_hydrogen_wave_function_register(self, max_n, max_l, max_m):
        for n in range(max_n):
            for k in range(max_l + 1):
                for m in range(max_m, max_m + 1):
                    _ = self.get_function(n, k, m)

    @classmethod
    def get_function(cls, n: int, k: int, m: int) -> Type[torch.autograd.Function]:
        """
        获取或创建指定量子数的波函数 Function, 该 Function 可以直接被调用。

        Args:
            n, l, m: 量子数

        Returns:
            torch.autograd.Function 的子类
        """
        key: tuple[int, int, int] = (n, k, m)

        if key not in cls._function_cache:
            # 动态创建新的 Function 类
            function_class = cls._create_hydrogen_function(n, k, m)
            cls._function_cache[key] = function_class

        return cls._function_cache[key]

    @classmethod
    def _create_hydrogen_function(
        cls, n: int, k: int, m: int
    ) -> Type[torch.autograd.Function]:
        """
        动态创建氢原子波函数的 autograd Function
        """
        # 定义类名
        # class_name: str = f"HydrogenWaveFunction_{n}_{k}_{m}"

        # 使用 type() 动态创建类
        # HydrogenFunction: Type[torch.autograd.Function] = type[torch.autograd.Function](
        #     class_name,
        #     (torch.autograd.Function,),
        #     {
        #         # 类属性
        #         "n": n,
        #         "k": k,
        #         "m": m,
        #         # 静态方法
        #         "forward": staticmethod(cls._hydrogen_forward),
        #         "backward": staticmethod(cls._hydrogen_backward),
        #         # 元类信息
        #         "__module__": __name__,
        #         "__doc__": f"Hydrogen wave function for n={n}, k={k}, m={m}",
        #     },
        # )

        class HydrogenFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, position: torch.Tensor):
                ctx._n = n
                ctx._k = k
                ctx._m = m
                result = cls._hydrogen_forward(ctx, position)
                return result

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor):
                grad_position = cls._hydrogen_backward(ctx, grad_output)
                return grad_position

        return HydrogenFunction

    @staticmethod
    def _hydrogen_forward(ctx, position: torch.Tensor):
        """
        Computes the forward pass for the hydrogen wave function calculation.

        This method is intended to be used internally as part of a larger computation
        pipeline. It calculates the hydrogen wave function at given positions and saves
        necessary information for backward pass computations.

        :returns: The computed phi value, which represents the wave function.
        :rtype: torch.Tensor

        :raises: This method does not explicitly raise any exceptions but may propagate
                 errors from the `hydrogen_wave_funcs_series` function or other called
                 methods.
        """
        # 获取量子数（通过 ctx 传递）
        ctx.n = ctx._n
        ctx.k = ctx._k
        ctx.m = ctx._m

        phi, radial, laguerre_value, angular = hydrogen_wave_funcs_series(
            ctx.n, ctx.k, ctx.m
        )(position)

        # 保存反向传播需要的信息
        ctx.save_for_backward(position, radial, laguerre_value, angular)

        return phi

    @staticmethod
    def _hydrogen_backward(ctx, grad_output: torch.Tensor):
        """
        Calculates the gradient of the hydrogen wave function with respect to position.

        This method is used internally for the backward pass in autograd, computing the gradients
        of the wave function with respect to the input positions. It utilizes saved context and
        the provided gradient output to compute the necessary derivatives.

        :returns: The gradient of the wave function with respect to the input position.
        :rtype: torch.Tensor

        :raises: This method does not raise any specific exceptions beyond those raised by PyTorch's
                 autograd and tensor operations.
        """
        # 获取量子数（通过 ctx 传递）
        ctx.n = ctx._n
        ctx.k = ctx._k
        ctx.m = ctx._m

        position_grad = hydrogen_wave_funcs_series(ctx.n, ctx.k, ctx.m).backward(
            grad_output, ctx.saved_tensors
        )

        return position_grad, None, None, None

    def __call__(self, n: int, k: int, m: int):
        if not (isinstance(n, int) and n > 0):
            raise ValueError("n must be a positive integer")
        if not (isinstance(k, int) and 0 <= k < n):
            raise ValueError("l must be an integer between 0 and n-1")
        if not (isinstance(m, int) and -k <= m <= k):
            raise ValueError("m must be an integer between -l and l")
        return self.get_function(n, k, m)


HWFR = HydrogenWaveFunctionRegistry(4, 3, 3)
