---
title: "Classic Geostatistical Methods"
tags: 
- Spatial Statistics
- Kriging
category: 
collection: 
use_math: true
---

이번 글에서는 이전에 [Covariance models]({% post_url 2024-02-02-CovModels %})에서 살펴본 연속형 공간자료를 모델링하는 고전적인 공간통계 방법론들에 대해 살펴보도록 하겠다. 이전 글에서 연속형 geostatistical 자료를 표현하는 방법으로 variogram 등을 살펴보았는데, 이러한 방법을 데이터로부터 어떻게 추론해내는지를 이번 글의 주요 관심사로 볼 수 있다.

# Geostatistical Model

## Model

Covariance model에서와 같이, 공간 $\mathcal{D}\subseteq \mathbb{R}^{d}$에서의 확률과정 $Z(s)$를 생각하고 이를 다음과 같은 구조로 나타내자.
$$
Z(s) = \mu(s) + e(s)\tag{1}
$$
이때 $\mu(s) := \mathrm{E}Z(s)$ 이고 이를 *mean function*이라고 하며, $e(s)$는 오차를 나타내는 평균이 0인 확률과정을 나타낸다. 또한, 오차 확률과정에 대해 일반적으로 다음과 같은 약정상성<sup>weakly stationarity</sup>을 가정한다.
$$
\mathrm{Cov}(e(s),e(t)) = C(s-t),\quad \forall s,t\in \mathcal{D}
$$
또한, 약정상과정 $e(\cdot)$의 semivarigram $\gamma$에 대해 다음이 성립한다.
$$
\gamma(h) = C(0)-C(h)
$$

위와 같은 모델 $(1)$은 큰 공간 변동성을 $\mu$를 통해 고려하고, 작은 공간 변동성은 오차 과정 $e$를 통해 고려한다. 그러나 일반적으로 실제 데이터로부터 이들을 구분하는 것은 거의 불가능하다. 실제 데이터는 어떠한 샘플링 과정 한번으로부터의 샘플에 불과하며, 이를 반복적으로 샘플링 할 경우의 문제는 고려하지 않기 때문이다.

> One person's deterministic mean structure may be another person's correlated error structure (Cressie, 1991)

따라서 이를 해결하기 위해 오차 과정 $e$를 다음과 같이 다시 분해하여 고려한다.
$$
e(s) = \eta(s) + \epsilon(s)
$$
여기서 $\eta$는 공간종속성을 갖는 요소인 반면, $\epsilon$은 측정오차를 나타내기 때문에 공간종속성을 갖지 않는다.

# Analyzing

Geostatistical 데이터를 분석하는 단계는 일반적으로 다음과 같이 이루어진다.

1. Mean function $\mathrm{E}Z(s)$ 를 추정한다.
2. Second-order structure (semivariogram 등)을 추정한다.
3. 관측되지 않은 지점에서의 반응변수 값을 예측한다. (Kriging)

이제 각 과정에서의 방법론들을 살펴보도록 하자.

## Mean function estimation

앞선 형태로 공간데이터에 대한 모델링을 진행하기 위해, 우선 기본적인 모수적 모형<sup>parametric model</sup>을 생각하자. 선형회귀모형과 유사하게, 다음과 같은 기본적인 형태를 가정할 수 있다.
$$
\mu(s;\beta) = X(s)^{\top}\beta
$$
여기서 $X(s)$는 지점 $s$에서 관측된 설명변수(공변량)들을 의미한다. 이러한 가정에서는 선형회귀모형에서와 같이 다음과 같은 OLS<sup>Ordinary Least Squares</sup> estimator를 생각할 수 있다.
$$
\hat\beta^{ols}=\arg\min\sum_{i=1}^{n}\left(Y(s_{i})-X(s_{i})^{\top}\beta\right)^{2}.
$$
## 



# References
- 