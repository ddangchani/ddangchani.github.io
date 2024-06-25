---
title: "Eigenfunctions and Kernel Methods"
tags: 
- Machine Learning
- Mercer's Theorem
- Kernel Methods
- Eigenfunctions
use_math: true
---

# Introduction

이번 글에서는 eigenfunctions<sup>고유함수</sup>와 이를 기반으로 하는 커널 방법론들의 해석에 대해 다루어보도록 하겠습니다. 고유함수의 개념은 데이터를 분석하는 과정에서는 요구되지 않을 수 있습니다. 다만, 많은 머신러닝 방법론, 그 중에서도 [Gaussian Process]({% post_url 2022-09-05-Gaussian_Process %}) 기반 방법론들의 이해에 중요한 부분을 차지한다고 생각합니다. 

기본적인 아이디어는 행렬의 고유값과 고유벡터를 생각해보면 이해가 쉽습니다. 행렬 $A$에 대해 $A\mathbf{v} = \lambda \mathbf{v}$를 만족하는 $\mathbf{v}$를 고유벡터<sup>eigenvector</sup>, $\lambda$를 고유값<sup>eigenvalue</sup>이라고 합니다. 이러한 고유벡터와 고유값은 행렬의 성질을 분석하는 데 중요한 역할을 합니다. 마찬가지로, 함수에 대해서도 고유함수와 고유값을 정의할 수 있습니다.

# Eigenfunctions

## Kernel

고유함수를 이해하기 위해서는 먼저 커널<sup>kernel</sup>에 대한 이해가 필요합니다. 커널에 대한 부가적인 내용은 [Kernel Methods]({% post_url 2021-12-05-kernel1 %})에서 다루었으니, 해당 글을 참고하시기 바랍니다. 커널을 간단히 정의하자면, 두 입력 벡터 $\mathbf{x}, \mathbf{x}'\in \mathcal{X}$에 대해 실수값을 반환하는 함수 $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$를 말합니다. 이때, 커널은 대칭성<sup>symmetric</sup>과 양의 정부호성<sup>positive definiteness</sup>을 만족해야 합니다.

- **대칭성**: $k(\mathbf{x},\mathbf{x}') = k(\mathbf{x}',\mathbf{x})$
- **양의 정부호성**: 모든 $L_2$공간의 함수 $f \in L_2(\mathcal{X},\mu)$에 대해 다음이 성립합니다.

    $$
    \int\int k(\mathbf{x},\mathbf{x}')f(\mathbf{x})f(\mathbf{x}')d\mu(\mathbf{x})d\mu(\mathbf{x}') \geq 0
    $$

> Remark. 커널 함수와 공분산 함수<sup>covariance function</sup>는 밀접한 관련이 있습니다. Gaussian process의 경우, 공분산 함수는 다음과 같이 정의됩니다.
>
> 
> 
> $$
> 
> k(\mathbf{x},\mathbf{x}') = \mathrm{Cov}(f(\mathbf{x}), f(\mathbf{x}'))
> 
> 
> $$
> 
>
> 본래 공분산 함수는 확률과정<sup>stoachastic process</sup> 혹은 랜덤필드<sup>random field</sup>의 공분산을 정의합니다. 커널 함수는 반드시 두 입력값에 대한 유사도를 반환하는 것은 아니므로, 두 정의가 완전히 동일하다고 할 수는 없습니다. 다만, [Gaussian process]({% post_url 2022-09-05-Gaussian_Process %})의 경우, 커널 함수가 공분산 함수의 역할을 한다고 생각할 수 있습니다. (명칭의 차이도 존재 : e.g. Gaussian RBF kernel vs Squared Exponential covariance function)

## Definition

커널함수에 대한 고유함수<sup>eigenfunction</sup>은 다음과 같이 정의됩니다.

$$

\int k(\mathbf{x},\mathbf{x}')\phi(\mathbf{x}')d\mu = \lambda \phi(\mathbf{x'})


$$

여기서 $\mu$는 $\mathcal{X}$에서 정의되는 적절한 [측도](https://ddangchani.github.io/mathematics/실해석학3)<sup>measure</sup>이며, $\lambda$는 고유값<sup>eigenvalue</sup>입니다. 이때, $\phi$를 고유함수라고 부릅니다. 이는 행렬의 고유값과 고유벡터의 정의와 유사합니다.

행렬에서는 고유값과 고유벡터가 행렬의 크기만큼 존재하는 것과 달리, 커널 함수의 경우 **무한개**의 고유함수가 존재합니다. (함수공간을 무한차원의 벡터공간으로 볼 수 있기 때문임을 생각하면 좋습니다.)

## Example

확률측도 $\mu$가 밀도함수 $p(\mathbf{x})=\mathcal{N}(0,\sigma^2)$를 가질 때, Gaussian RBF 커널

$$

k(\mathbf{x},\mathbf{x}') = \exp\left(-\frac{\|\mathbf{x}-\mathbf{x}'\|^2}{2\ell^2}\right)


$$

에 대한 고유함수와 고유값은 다음과 같습니다.

$$

\begin{aligned}
\lambda_k &= \sqrt{\frac{2a}{A}}B^k ,\\
\phi_k(x) &= \exp(-(c-a)x^2)H_k(\sqrt{2c}x)
\end{aligned}


$$

여기서 $a=1/4\sigma^2$, $b=(2\ell^2)^{-1}$, $c=\sqrt{a^2+2ab}$, $A=a+b+c$, $B=b/A$ 이고, $H_k$는 아래와 같이 정의되는 에르미트 다항식<sup>Hermite polynomial</sup>입니다.

$$

H_k(x) = (-1)^k e^{x^2}\frac{d^k}{dx^k}e^{-x^2}


$$

## Mercer's Theorem

Mercer의 정리는, 임의의 커널 함수 $k$를 고유값과 고유함수들로 나타낼 수 있다는 정리입니다. 먼저, 커널 함수 $k$에 대해, 다음과 같이 적분 연산자 $T_k$를 정의하도록 합시다.

$$

T_kf(\mathbf{x}) = \int_\mathcal{X} k(\mathbf{x},\mathbf{x}')f(\mathbf{x}')d\mu


$$

이때 Mercer의 정리는 다음과 같습니다.

> **Mercer's Theorem**. 측도 공간 $(\mathcal{X},\mu)$에서 정의된 커널 함수 $k$에 대해 $T_k:L_2(\mathcal{X},\mu)\to L_2(\mathcal{X},\mu)$가 양의 정부호인 경우, $\mu^2$에 대해 거의 모든<sup>almost everywhere</sup> $\mathbf{x},\mathbf{x}'\in\mathcal{X}$에 대해 다음 표현이 성립합니다.
>
> 
> 
> $$
> 
> k(\mathbf{x},\mathbf{x}') = \sum_{i=1}^\infty \lambda_i \phi_i(\mathbf{x})\phi^\ast_i(\mathbf{x}')
> 
> 
> $$
> 

여기서 $\lambda_i$는 양의 실수이고, $\phi_i$는 정규화된<sup>normalized</sup> 고유함수입니다. 정규화된 고유함수는 행렬의 고유벡터와 유사하게, 함수공간의 **정규직교**<sup>orthonormal</sup> 기저를 형성합니다.

# Nyström Method

Mercer의 정리는 커널 함수를 고유함수들로 나타낼 수 있다는 것을 보여줍니다. 확률측도 $\mu$가 밀도함수 $p(\mathbf{x})d\mathbf{x}$를 가질 때, 고유함수 분해는 다음과 같이 근사될 수 있습니다.

$$

\begin{aligned}
\lambda_i\phi_i(\mathbf{x}) &= \int k(\mathbf{x},\mathbf{x}')p(\mathbf{x})\phi_i(\mathbf{x})d\mathbf{x}\\
&\simeq \frac{1}{n}\sum_{j=1}^n k(\mathbf{x}_j,\mathbf{x}')\phi_i(\mathbf{x}_j)
\end{aligned}


$$

이때, $\mathbf{x}_1,\ldots,\mathbf{x}_n$은 샘플링된 데이터 포인트들입니다. 이때, 위 식에 $\mathbf{x}'=\mathbf{x}_j$ 들을 대입하면 ($j=1,\cdots,n$), 이는 다음과 같은 행렬식으로 나타낼 수 있습니다.

$$

\begin{bmatrix}
\lambda_i\phi_i(\mathbf{x}_1) \\
\vdots \\
\lambda_i\phi_i(\mathbf{x}_n)
\end{bmatrix} = \frac{1}{n}\begin{bmatrix}
k(\mathbf{x}_1,\mathbf{x}_1) & \cdots & k(\mathbf{x}_1,\mathbf{x}_n) \\
\vdots & \ddots & \vdots \\
k(\mathbf{x}_n,\mathbf{x}_1) & \cdots & k(\mathbf{x}_n,\mathbf{x}_n)
\end{bmatrix}\begin{bmatrix}
\phi_i(\mathbf{x}_1) \\
\vdots \\
\phi_i(\mathbf{x}_n)
\end{bmatrix}
\tag{1}

$$

여기서 식 (1) 우변의 행렬은 $n\times n$ 크기의 행렬이며, 이를 **Gram matrix** 라고 부릅니다. 이는 커널 함수에 데이터 포인트들을 대입하여 계산한 행렬로 이해할 수 있습니다. 이때, $K$의 고유값분해는 다음과 같이 나타낼 수 있습니다.

$$

\lambda_i^\mathrm{mat} \mathbf{u}_i = K\mathbf{u}_i \tag{2}


$$

$\lambda_i^\mathrm{mat},\mathbf{u}_i$는 각각 Gram matrix $K$의 고유값과 고유벡터입니다. 그러면, 식 (1)과 식 (2)의 관계로부터 $\lambda_i = \lambda_i^\mathrm{mat}/n$임을 알 수 있습니다. 즉, 커널함수의 고유값을 Gram matrix의 고유값으로 **추정**할 수 있습니다.

나아가, 고유함수 $\phi_i$ 역시 다음과 같이 근사할 수 있습니다.

$$

\phi_i(\mathbf{x}) \simeq \frac{\sqrt{n}}{\lambda_i^\mathrm{mat}}\mathbf{k}(\mathbf{x}')^\top \mathbf{u}_i


$$

이때, $\mathbf{k}(\mathbf{x}')$는 $\mathbf{x}'$에 대한 커널 벡터<sup>kernel vector</sup>로, 다음과 같이 정의됩니다.

$$

\mathbf{k}(\mathbf{x}') = \begin{bmatrix}
k(\mathbf{x}_1,\mathbf{x}') \\
\vdots \\
k(\mathbf{x}_n,\mathbf{x}')
\end{bmatrix}


$$

특히, 데이터 샘플 $\mathbf{x}_j$ 에 대해서는 $\phi_i(\mathbf{x}_j) = \sqrt{n}(\mathbf{u}_i)_j$ 임을 알 수 있습니다. 이는 고유벡터의 $j$번째 원소에 $\sqrt{n}$을 곱한 것과 같습니다.

이러한 방법론을 **Nyström method**라고 부릅니다. 요약하자면, Nyström method는 데이터 샘플을 이용하여 Gram matrix의 고유값과 고유벡터를 구하고, 이를 이용하여 고유함수를 추정합니다.


# Reproducing Kernel Hilbert Space

[이전]({% post_url 2022-01-02-kernel2 %})에 커널 트릭을 설명하기 위해 RKHS<sup>Reproducing Kernel Hilbert Space</sup>와 representer theorem에 대해 다룬 바 있습니다. 해당 포스트에서는 *reproducing kernel map construction* 을 통해 RKHS를 정의하였습니다. 여기서는 고유함수를 이용해 RKHS를 정의하는 방법에 대해 다루어보도록 하겠습니다.

## RKHS

실함수 $f$들로 구성된 힐베르트 공간(완비내적공간) $\mathcal{H}$의 내적이 $\langle\cdot,\cdot\rangle_\mathcal{H}$이고, 함수 $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ 이 존재하여 다음 두 조건을 만족하는 경우,

1. 모든 $\mathbf{x}\in\mathcal{X}$에 대해 $k(\mathbf{x},\cdot)\in\mathcal{H}$
2. **Reproducing property** : 모든 $\mathbf{x}\in\mathcal{X}$와 $f\in\mathcal{H}$에 대해 다음이 성립합니다.

    $$
    f(\mathbf{x}) = \langle f, k(\mathbf{x},\cdot)\rangle_\mathcal{H}
    $$

$\mathcal{H}$를 **RKHS**<sup>재생커널힐베르트공간</sup>라고 부릅니다. 또한, $k$를 **재생커널**<sup>reproducing kernel</sup>이라고 부릅니다.

### Moore-Aronszajn Theorem

RKHS $\mathcal{H}$ 이 존재하면, 이에 대응하는 커널 함수 $k$가 유일하게 존재합니다. 또한, 커널 함수 $k$가 존재하면, 이에 대응하는 RKHS $\mathcal{H}$가 유일하게 존재합니다. 이를 **Moore-Aronszajn theorem**이라고 부릅니다.

## RKHS using Eigenfunctions

이제 고유함수를 바탕으로 RKHS를 다시 살펴보도록 하겠습니다. 우선, 임의의 커널 함수 $k$는 다음과 같이 표현될 수 있습니다. (여기서는 **유한개의** 고유함수를 가정합니다)

$$

k(\mathbf{x},\mathbf{x}') = \sum_{i=1}^N \lambda_i \phi_i(\mathbf{x})\phi_i(\mathbf{x}')


$$

이때, 고유함수 $\phi_i$ 들은 함수공간의 정규직교 기저를 형성합니다. 즉, 다음이 성립합니다.

$$

\langle \phi_i, \phi_j\rangle_\mathcal{H} =: \int \phi_i(\mathbf{x})\phi_j(\mathbf{x})d\mu = \delta_{ij}


$$

그렇다면, $\phi_i$ 들을 기저로 하여, 이들의 선형결합으로 힐베르트공간 $\mathcal{H}$을 구성할 수 있습니다.

$$

f(\mathbf{x}) = \sum_{i=1}^N f_i\phi_i(\mathbf{x}), \quad \sum_i f_i^2/\lambda_i < \infty


$$

이때, $\mathcal{H}$의 원소 $f,g$에 대해 내적을 다음과 같이 정의할 수 있습니다. (내적의 성질을 만족합니다)

$$

\langle f, g\rangle_\mathcal{H} = \sum_{i=1}^N \frac{f_i g_i}{\lambda_i}


$$

이렇게 정의한 힐베르트공간 $\mathcal{H}$는, **reproducing property**를 만족합니다.

$$

\langle f, k(\mathbf{x},\cdot)\rangle_\mathcal{H} = \sum_{i=1}^N \frac{f_i \lambda_i\phi_i(\mathbf{x})}{\lambda_i} = f(\mathbf{x})


$$

마찬가지로,

$$

\langle k(\mathbf{x},\cdot), k(\mathbf{x}',\cdot)\rangle_\mathcal{H} = k(\mathbf{x},\mathbf{x}')


$$

역시 성립합니다. 이러한 관계로부터, 고유함수를 기저로 하는 힐베르트공간 $\mathcal{H}$이 RKHS임을 알 수 있습니다.


# References

- Williams, C. K., & Rasmussen, C. E. (2006). Gaussian processes for machine learning (Vol. 2, No. 3, p. 4). Cambridge, MA: MIT press.
