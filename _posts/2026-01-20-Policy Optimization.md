---
title: Policy Optimization
tags:
  - Reinforcement Learning
  - Deep Learning
  - Policy Optimization 
use_math: true
header: 
    teaser: https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcRsxv4DO2c3w_6MfGlcdHXLtjcdQgod0E2h2_gRnGJU0T7JmJ4ahW2zNp4ASq_4MzN8JN3WSxxl4KT83NgXk8z6hWVOVvwuaio-OUis1rt-cMhZMTc
---

# Setting

통계학적인 관점에서, 강화학습(RL)은 uncertainty 하에서 이루어지는 sequential decision making 혹은 dynamic optimization 이라고 할 수 있다. 일반적인 supervised learning 세팅에서는 관측 데이터 $(X,Y)$ 쌍들로부터 conditional distribution $\Pr(Y|X)$ 를 학습하는 것인 반면, RL에서는 보상(reward)을 최대화하는 policy $\pi$를 찾고자 한다.


## Policy

> Parametric Conditional Distribution

$$
\pi_{\theta}(a|s) = \Pr(A_{t}=a | S_{t} =s;\theta)
$$

Policy는 상태 $s$에서 행동 $a$를 선택할 확률을 나타내는 조건부 확률 분포이다. 

- Input: **state** $s\in \mathcal{S}$, 현재 상태(ex. 현재 게임 화면, 채팅 기록 등)
- Output: **action** $a \in \mathcal{A}$, 취할 수 있는 행동(ex. 캐릭터 이동, 답변 생성에서 Token "The" 선택 등)
- Parameter $\theta$ : weights of neural networks

## Reward

Reward function $r_t(s)$는 시점 $t$에서 상태가 $s$인 agent가 환경으로부터 받는 scalar feedback signal이다. 강화학습의 목표는 **장기적으로 받을 보상의 합**을 최대화하는 것이다.

$$
G_t = \sum_{k=0}^{\infty} \gamma^{k} r_{t+k+1}
$$

- $\gamma$ : **discount factor**, $0 \leq \gamma < 1$ (주로 0.99 사용), 미래 보상에 대한 현재 가치의 감소율

## Value function

다음과 같은 3개의 주요 가치 함수(value function)가 있다.

$$
\begin{aligned}
Q^{\pi}(s_t,a_t) & = \mathbb{E}_{\pi} \left[ G_t | s = s_t, a = a_t \right] &\quad &\text{(State-action value Function)} \\
&=\mathbb{E}_{s_{t}, a_{t},\ldots}\left[\sum_{l=0}^{\infty} \gamma^{l} r_{t+l+1} \right] \\
V^{\pi}(s_t) & = \mathbb{E}_{\pi} \left[ G_t | s = s_t \right] &\quad &\text{(Value Function)} \\
&= \mathbb{E}_{a_t, s_{t+1}, \ldots} \left[\sum_{l=0}^{\infty} \gamma^{l} r_{t+l+1} \right] \\
A^{\pi}(s_t,a_t) & = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t) &\quad &\text{(Advantage Function)}
\end{aligned}
$$

- State-Action Value Function $Q^{\pi}(s,a)$: 상태 $s$에서 행동 $a$를 취했을 때 기대되는 누적 보상
- Value Function $V^{\pi}(s)$: 상태 $s$에서 기대되는 누적 보상 (평균적인 action 기준)
- Advantage Function $A^{\pi}(s,a)$: 특정 행동의 가치가 평균적인 행동에 비해 얼마나 좋은지

## Optimization Objective

![](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcRsxv4DO2c3w_6MfGlcdHXLtjcdQgod0E2h2_gRnGJU0T7JmJ4ahW2zNp4ASq_4MzN8JN3WSxxl4KT83NgXk8z6hWVOVvwuaio-OUis1rt-cMhZMTc)

강화학습의 목표는 policy $\pi_{\theta}$의 파라미터 $\theta$를 최적화하여 기대 누적 보상(expected cumulative reward)을 최대화하는 것이다. 즉, 다음과 같은 목적함수를 최적화한다.

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ G(\tau) \right]
$$




# PPO(Proximal Policy Optimization)

PPO는 제약 조건 하에서의 정책 최적화 알고리즘으로 볼 수 있다.

### Vanilla Policy Gradient

목적함수 $J(\theta)$의 gradient estimator는 다음과 같이 표현된다.

$$
\nabla_{\theta} J(\theta) = \hat{\mathbb{E}}_{t} \left[ \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \hat{A}_t \right]
$$


> $G_t$ 가 아닌 $A_t$ 를 사용하는 이유는 $G_t$가 가지는 높은 분산(variance)을 줄이기 위함이다. (highly stochastic reward 환경에서 특히 유용)

$$
A^\pi(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)
$$

## Trust Region Policy Optimization

TRPO는 policy 업데이트 시 KL divergence 제약 조건을 도입하여 급격한 변화를 방지한다.

$$
\begin{aligned}
\max_{\theta} \quad & \hat{\mathbb{E}}_{t} \left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right] \\
\text{subject to} \quad & \hat{\mathbb{E}}_{t} \left[ D_{KL} \left( \pi_{\theta_{old}}(\cdot|s_t) \| \pi_{\theta}(\cdot|s_t) \right) \right] \leq \delta
\end{aligned}
$$

이때 위 제약조건 문제는 penalized objective로 변환하여 해결할 수 있다.

$$
\max_\theta \hat{\mathbb{E}}_{t} \left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t - \beta D_{KL} \left( \pi_{\theta_{old}}(\cdot|s_t) \| \pi_{\theta}(\cdot|s_t) \right) \right]
$$

또한, 

### Clipped Surrogate Objective
	
PPO는 TRPO의 복잡성을 줄이기 위해 clipped surrogate objective를 도입한다. TRPO의 objective 내 비율 항목을 다음과 같이 정의한다.

$$
r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

그러면, CLIP된 목적함수는 다음과 같이 표현된다.

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_{t} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

여기서 $\epsilon$는 작은 양수 하이퍼파라미터로, 일반적으로 0.2로 설정된다.

> 만일 action이 advantage $\hat{A}_t$에 대해 긍정적이라면, 확률을 높인다 ($r_t(\theta) > 1$). 다만, $r_t(\theta)$가 $1 + \epsilon$를 초과하지 않도록 클리핑한다.


- Schulman, John, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. 2015. “Trust Region Policy Optimization.” Proceedings of the 32nd International Conference on Machine Learning, June 1, 1889–97. https://proceedings.mlr.press/v37/schulman15.html.

- Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347. Preprint, arXiv, August 28. https://doi.org/10.48550/arXiv.1707.06347.
