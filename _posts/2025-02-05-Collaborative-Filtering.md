---
title: "Collaborative Filtering"
tags: 
- Recommender system
- Collaborative filtering
category: Recommender system
use_math: true
---

이번 글에서는 추천 시스템 중 협업 필터링<sup>collaborative filtering</sup>에 대한 내용을 개괄적으로 살펴보도록 하겠습니다. 우선, 협업 필터링이란 사용자가 평가하지 않은 항목에 대한 반응을 예측하는 것을 목표로 하여 각 사용자 혹은 항목 간의 유사도를 측정하여 이루어집니다. 협업 필터링은 크게 다음 두 가지로 분류할 수 있습니다.

- 메모리 기반 필터링<sup>Memory-based Filtering</sup> : 유사한 사용자 혹은 유사한 아이템을 기반으로 특정 아이템에 대한 특정 사용자의 값을 예측
- 모델 기반 필터링<sup>Model-based Filtering</sup> : 평가 데이터를 활용하여 decision tree, bayesian model, latent factor model 등 머신러닝 모델을 학습시키고 활용

# 메모리 기반 필터링

메모리 기반 필터링은 또 다시 다음 두 가지 형태로 분류할 수 있습니다.

- 사용자-사용자 간 필터링
- 아이템-아이템 간 필터링

각 세부 항목은, 어떤 것에 대한 유사도를 측정하는지에 기반합니다. 즉, 사용자-사용자 간 필터링은 추천 대상이 되는 사용자와 비슷한 성향을 가진 사용자를 선택하여, 해당 사용자가 선호하는 아이템을 추천해주는 방식입니다. 아이템-아이템 간 필터링은 그 반대로 작동합니다.

이때, 유사도를 측정하는 방식이 필요한데 대표적으로는 코사인 유사도<sup>cosine similarity</sup>를 사용할 수 있습니다. 예를 들어 사용자 $u$와 사용자 $v$의 평가 데이터가 4개의 아이템에 대해 다음과 같이 주어졌다고 가정해봅시다.

$$
\begin{align*}
\text{User}_u & : (r_{u1}, r_{u2}, r_{u3}, r_{u4}) \\
\text{User}_v & : (r_{v1}, r_{v2}, r_{v3}, r_{v4})
\end{align*}
$$

이때, 두 사용자 간의 코사인 유사도는 다음과 같이 정의됩니다.

$$
\text{Cossim}(u, v) = \frac{r_{u1} \cdot r_{v1} + r_{u2} \cdot r_{v2} + r_{u3} \cdot r_{v3} + r_{u4} \cdot r_{v4}}{\sqrt{r_{u1}^2 + r_{u2}^2 + r_{u3}^2 + r_{u4}^2} \cdot \sqrt{r_{v1}^2 + r_{v2}^2 + r_{v3}^2 + r_{v4}^2}}
$$

중요한 것은, 추천 문제의 경우 모든 아이템들에 대한 평가 데이터가 존재하지 않기 때문에, 유사도를 측정할 때는 누락된 데이터를 어떻게 처리할 것인지가 중요합니다. 이때 일반적으로는 **관찰된 데이터**만을 사용하여 유사도를 측정하게 됩니다.

$$
\begin{array}{c|cccc}
& \text{Item}_1 & \text{Item}_2 & \text{Item}_3 & \text{Item}_4 \\
\hline
\text{User}_1 & 5 & 3 & ? & 1 \\
\text{User}_2 & 4 & ? & 4 & 1 \\
\end{array}
$$

위와 같이 사용자-아이템 행렬이 주어진 경우, 아이템 1과 4에 대해서만 다음과 같이 유사도를 측정할 수 있습니다.

$$
\text{Cossim}(\text{User}_1, \text{User}_2) = 
\frac{5 \cdot 4 + 1 \cdot 1}{\sqrt{5^2 + 1^2} \cdot \sqrt{4^2 + 1^2}} = 0.99
$$


아이템-아이템 간 필터링의 경우, 사용자-아이템 행렬을 전치하여 사용자-아이템 행렬을 얻어낸 후, 이를 이용하여 유사도를 측정합니다.

> **Cold Start Problem**
>
> 새로운 사용자, 새로운 아이템이 들어올 경우 추천 시스템이 제대로 작동하지 않는 문제를 의미합니다. 이를 해결하기 위해서는 새로운 사용자 혹은 아이템에 대한 정보를 어떻게 활용할 것인지가 중요합니다.
>
> 사용자 피드백과 별개의 특성(e.g. 사용자의 나이, 성별, 지역 등)이나 아이템의 특성(e.g. 제품군, 제조사, 가격 등)을 활용하여 새로운 사용자 혹은 아이템에 대한 추천을 제공할 수 있습니다.

## Matrix Completion

다음과 같은 사용자-아이템 행렬이 주어졌다고 가정해봅시다. 각 성분은 사용자가 아이템에 대해 평가한 점수를 나타냅니다. 

$$
\begin{array}{c|cccc}
& \text{Item}_1 & \text{Item}_2 & \text{Item}_3 & \text{Item}_4 \\
\hline
\text{User}_1 & 5 & 3 & ? & 1 \\
\text{User}_2 & 4 & ? & 4 & 1 \\
\text{User}_3 & 1 & 1 & 2 & 3 \\
\text{User}_4 & 2 & 2 & 3 & 1 \\
\end{array}
$$

이때, 추천 문제는 위 행렬의 빈칸($?$)을 채우는 것에 대응되며, 이러한 문제를 **Matrix Completion**이라고 합니다. Matrix completion 문제는 다음과 같은 full-matrix solution $M$을 찾는 문제로 정의됩니다.

$$
\underset{M}{\mathrm{minimize}} \sum_{(i, j) \text{ observed}} (M_{ij} - R_{ij})^2
$$

이때 행렬 $M$에 대한 제약조건이 없기 때문에 이는 자명<sup>trivial</sup>한 문제가 됩니다. 이를 해결하기 위해 다음과 같이 Low Rank Approximation 기반의 행렬 분해를 사용할 수 있습니다(2009년 Netflix Prize contest에서 사용된 방법).

$$
(\hat P, \hat Q) = \underset{P \in \mathbb{R}^{N \times K}, Q \in \mathbb{R}^{A \times K}}{\mathrm{argmin}} \sum_{(i, j) \text{ observed}} (R_{ij} - P_i^\top Q_j)^2
$$

여기서 $K$는 새로운 사용자-아이템 행렬의 계수를 의미합니다. 위와 같은 행렬 분해 $\hat R= \hat P \hat Q^\top$의 특징 중 하나는, 분해된 행렬 $\hat P$와 $\hat Q$가 각각 사용자, 아이템에 대한 잠재 요인을 나타낸다는 것입니다(PCA의 개념과 유사). 또한, 위 최적화 문제에서 과적합을 방지하기 위해 다음과 같이 정규화 항을 추가할 수 있습니다.

$$
(\hat P, \hat Q) = \underset{P \in \mathbb{R}^{N \times K}, Q \in \mathbb{R}^{A \times K}}{\mathrm{argmin}} \sum_{(i, j) \text{ observed}} (R_{ij} - P_i^\top Q_j)^2 + \lambda (\|P\|_2^2 + \|Q\|_2^2)
$$

혹은 $P,Q$를 직교행렬로 가정하거나, non-negative로 가정하여 해석가능성을 높이는 방법도 여럿 제안되었습니다.

# Model-based Filtering

모델 기반의 필터링은 사용자의 피드백을 반응변수로, 사용자와 아이템에 대한 context vector를 설명변수로 하는 회귀모형을 학습하는 방식입니다. 앞서 살펴본 사용자-아이템 행렬은 사용자가 아이템에 대해 평가한 점수만을 담고 있기 때문에, 사용자의 피드백(e.g. 좋아요/싫어요, 클릭 여부)을 활용하여 이를 보완할 수 있습니다. 

예시로 다음과 같은 $3\times 3$ 사용자-아이템 행렬과 각 사용자 $i$와 아이템 $j$에 대한 context vector $\mathbf{x}_{ij}$가 주어졌다고 가정해봅시다.

$$
R^0 = 
\stackrel{\text{Item}}{
\begin{bmatrix}
    ? & 3 & ? \\ ? & ? & 4 \\ ? & 2 & 5
\end{bmatrix}}
\Rightarrow R = \begin{bmatrix}
r_{12} \\ r_{23} \\ r_{32} \\ r_{33}
\end{bmatrix} =

\begin{bmatrix}
    3 \\ 4 \\ 2 \\ 5
\end{bmatrix}
$$

우선, 기존에 행렬 형태로 다루었던 $R^0$를 벡터 형태인 $R$로 변환하였습니다.

$$
\begin{array}{c|ccc|ccc|cc}
& \text{U}_1 & \text{U}_2 & \text{U}_3 & \text{I}_1 & \text{I}_2 & \text{I}_3 & \text{V}_1 & \text{V}_2 \\

\mathbf{x}_{12} & 1 & 0 & 0 & 0 & 1 & 0 & -0.5 & 1.9 \\
\mathbf{x}_{23} & 0 & 1 & 0 & 0 & 0 & 1 & 1.5 & 0.2 \\
\mathbf{x}_{32} & 0 & 0 & 1 & 1 & 0 & 0 & 3.0 & 1.3 \\
\mathbf{x}_{33} & 0 & 0 & 1 & 0 & 1 & 0 & -1.1 & 2.5
\end{array}
$$

여기서 $U_i$는 사용자, $I_j$는 아이템, $V_k$는 보조정보를 나타내며 $$\mathbf{x}_{ij}$$의 $$U_{i}$$와 $$I_{j}$$는 $i,j$번째 사용자와 아이템을 나타내는 one-hot vector입니다. $R$의 각 성분 $r_{ij}$을 반응변수로 하면, 다음과 같은 회귀모형을 생각할 수 있습니다.

$$
r_{ij} = f(\mathbf{x}_{ij}) + \epsilon_{ij}
$$

즉, 사용자 $i$와 아이템 $j$에 대한 context vector $$\mathbf{x}_{ij}$$를 입력으로 받아 평가 점수 $$r_{ij}$$를 출력하는 함수 $f$를 찾는 문제입니다. 간단하게는 선형 회귀 모형

$$
r = \beta_0 + \mathbf{x}^\top \beta + \epsilon
$$

을 고려할 수 있고, 나아가 복잡한 모델링을 하기 위해 $f$ 자리에 neural network, decision tree, random forest 등의 머신러닝 모델을 도입할 수 있습니다.

## Factorization Machine

앞서 살펴본 Matrix Completion 문제는 사용자-아이템 행렬을 분해하는 문제였습니다. 이를 일반화하여, 사용자 피드백 및 보조정보를 활용하여 사용자-아이템 행렬을 분해하는 방법을 생각해볼 수 있습니다. **Factorization Machine**은 다음과 같은 모델을 사용하여 사용자-아이템 행렬을 분해하는 방법입니다.

$$
r_{ij} = \beta_0 + \mathbf{x}_{ij}^\top \boldsymbol{\beta} + \sum_{1\le l_1 \le l_2 \le D} (\boldsymbol{\gamma}_{l_1}^\top \boldsymbol{\gamma}_{l_2}) x_{ijl_1} x_{ijl_2} + \epsilon_{ij}
$$

여기서 $D$는 사용자 수 $N$과 아이템 수 $A$, 그리고 보조정보 특성의 수를 합한 것으로 각 $\mathbf{x}_{ij}$ 벡터의 차원을 의미합니다. Factorization machine은 위 모델을 기반으로 다음과 같은 최적화 문제를 풀게 됩니다.

$$
\underset{\beta_0 \in \mathbb{R}, \boldsymbol{\beta} \in \mathbb{R}^{D}, \Gamma\in \mathbb{R}^{D\times K}}{\text{minimize}}\sum_{(i,j) \text{ observed}} \left(r_{ij}-\beta_{0}-\sum_{l=1}^{D}\beta_{l}x_{ij,l} + \sum_{1\le l_{1}\le l_{2}\le D} (\boldsymbol{\gamma}_{l_{1}}^{\top}\boldsymbol{\gamma}_{l_{2}})x_{ij,l_{1}}x_{ij,l_{2}}\right)^{2}
$$

앞서 살펴본 Low-rank matrix factorization은 이러한 Factorization machine의 일종으로 볼 수 있습니다. 만일 각 context vector $$\mathbf{x}_{ij}$$에 보조정보가 포함되어 있지 않다면(즉, $$D=N+A$$가 성립하는 경우), $$\beta_{0}=\cdots=\beta_D=0$$ 으로 두면 위 식은

$$
\boldsymbol{\Gamma} = \begin{bmatrix}\mathbf{P} \\ \mathbf{Q}\end{bmatrix}
$$

일 때 다음과 같습니다:

$$
\underset{\mathbf{P}\in \mathbb{R}^{N\times K},\mathbf{Q}\in \mathbb{R}^{A\times K}}{\text{minimize}}\sum_{(i,j) \text{ observed}}  (r_{ij}-\mathbf{p}_{i}^{\top}\mathbf{q}_{j})^{2}
$$

예를 들어, $N=3,A=3$ 인 경우

$$
\begin{array}{ccccc|cccc}
&& \text{U}_1 & \text{U}_2 & \text{U}_3 & \text{I}_1 & \text{I}_2 & \text{I}_3 \\
\mathbf{x}_{12}= & \big[ &1 & 0 & 0 & 0 & 1 & 0 & \big]
\end{array}
$$

로 표현되기 때문에, 다음이 성립합니다.

$$
\sum_{1 \le l_1 \le l_2 \le D} (\boldsymbol{\gamma}_{l_1}^\top \boldsymbol{\gamma}_{l_2}) x_{12,l_1} x_{12,l_2} = \boldsymbol{\gamma}_{1}^\top \boldsymbol{\gamma}_{5} = \mathbf{p}_{1}^{\top}\mathbf{q}_{2}
$$

마찬가지로, 사용자 $i$와 아이템 $j$에 대해서 다음이 성립합니다.

$$
\sum_{l_1 < l_2} (\boldsymbol{\gamma}_{l_1}^\top \boldsymbol{\gamma}_{l_2}) x_{ij,l_1} x_{ij,l_2} = \mathbf{p}_{i}^{\top}\mathbf{q}_{j}
$$

다음 글에서는 앞선 방법들에 딥러닝을 적용한 다양한 방법론들에 대해 다루어보도록 하겠습니다.


# References
- 온라인 콘텐츠 추천 문제의 통계학적 기초, 최영근, 한국통계학회 2024년 동계학술논문발표회 Tutorial
- Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” _Computer_, vol. 42, no. 8, pp. 30–37, Aug. 2009, doi: [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263).