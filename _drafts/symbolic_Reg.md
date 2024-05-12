### Symbolic regression

간단한 예시로, 2D input을 갖는 $f(x,y)=\exp(\sin(\pi x)+ {y^2})$로부터 생성된 데이터들이 있다고 가정해봅시다. 함수 $f$를 사전에 알고 있다면 이는 $[2,1,1]$-KAN으로 표현된다는 것을 알 수 있지만, 당면한 상황은 이를 모르는 상황입니다. 이 경우 충분히 큰 KAN을 먼저 가정한 후, sparsity regularization을 바탕으로 모델을 학습하여 불필요한 노드와 엣지를 제거하고 (prune), 이를 원래 함수로 나타냅니다. (아래 그림 참고)

![](스크린샷%202024-05-12%20오후%201.46.54.png)
#### Sparsification

MLP에서는 [regularization](**2022-04-06-Regularization_on_DL**) 기법을 이용해 가중치에 sparsity를 부여하였습니다. 이와 유사하게 KAN에서는 다음과 같이 레이어의 L1 노음을 정의합니다.
$$
\left\vert \Phi\right\vert_{1}:=\sum_{i=1}^{n_\mathrm{in}}\sum_{j=1}^{n_\mathrm{out}}\left\vert \phi_{i,j}\right\vert_{1}
$$
또한, 레이어의 엔트로피<sup>entropy</sup>를 다음과 같이 정의하여
$$
S(\Phi) := \sum_{i=1}^{n_\mathrm{in}}\sum_{j=1}^{n_\mathrm{out}}\frac{\left\vert \phi_{i,j}\right\vert_{1}}{\left\vert \Phi\right\vert_{1}}\log \left(\frac{\left\vert \phi_{i,j}\right\vert_{1}}{\left\vert \Phi\right\vert_{1}}\right)
$$
손실함수(목적함수)에 다음과 같이 regularization term을 추가합니다.
$$
l_\mathrm{total} = l_\mathrm{pred} + \lambda \left(\mu_{1}\sum_{l=0}^{L-1}\left\vert \Phi_{l}\right\vert_{1}+ \mu_{2}\sum_{l=0}^{L-1}S(\Phi_{l})\right).
$$
#### Prune

Sparsification으로 모델을 학습한 후, 각 $\phi$의 노음을 계산하여 이를 다음과 같은 incoming / outgoing score으로 정의합니다.
$$
I_{l,i}=\max_{k}(\left\vert \phi_{l-1},k,i\right\vert_{1}),\quad O_{l,i}=\max_{j}(\left\vert \phi_{l+1},j,i\right\vert_{1})
$$
만일 두 score가 정해진 threshold $\theta=0.01$을 넘는다면 해당 노드를 필요한 것으로 간주하고, 그렇지 않은 노드를 제거합니다. 이 과정으로부터 간단한 형태의 네트워크를 얻을 수 있습니다.

#### Symbolic functions

만일 일부 activation function이 로그함수, 지수함수와 같이 특정 함수로 표현가능하다면 $\phi$를 해당 함수의 아핀 변환<sup>affine transformation</sup>으로 고정할 수 있습니다. 이를 통해 파라미터 수를 줄일 수 있으며, 이후 해당 파라미터를 재학습하기 위해 모델을 재차 훈련합니다.


