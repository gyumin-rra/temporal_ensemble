# temporal_ensemble
a simple tutorial for temporal ensemble

이 repository는 semi-supervised learning, 그 중에서도 temporal ensembling에 대한 내용을 중심으로 작성되었습니다. 우선 semi-suprevised learning의 개략적인 내용을 설명하고, 이 중 temporal ensembling의 내용에 대해 더 자세하게 살펴본 뒤 이를 실험하는 순서로 진행합니다. 이 repository의 이론적인 토대는 첨부드린 논문, 그리고 고려대학교 강필성 교수님의 [유튜브 강의](https://www.youtube.com/watch?v=vhitW3gsuhw&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW&index=30)를 참고하였음을 밝힙니다. 

## 목차

1. [Concepts of Semi-Supervised Learning](#concepts-of-semi-supervised-learning)
2. [Temporal Ensembling](#temporal-ensembling)
3. [Implementation](#implementation)
4. [Conclusion](#conclusion)

---

## Concepts of Semi-Supervised Learning
이름에서도 알 수 있든 semi-supervised learning(준지도학습)은 supervised learning과 unsupervised learning의 경계에 위치한 학습 알고리즘입니다. 왜 이런 이름인가 하면, 이 방법론에서는 labeled 된 데이터와 그렇지 않은 unlabeled 데이터를 동시에 활용하기 때문이라고 할 수 있겠습니다. 어떤 prediction task를 수행하는 경우에 대해, prediction 해야하는 feature에 대한 labeling이 되어 있으면 labeled data라 하고, 그렇지 않으면 unlabeled data라 합니다. 예를 들어 아래와 같은 데이터셋에서 우리가 나이, 성별, 몸무게를 이용해 키를 예측하는 모델을 개발하려고 한다면 키 feature가 결측치인 1, 2, 4번째 객체는 unlabeled data이고 3, 5번째 데이터 객체는 labeled data가 됩니다. 

| 나이 | 성별 | 몸무게 | 키 |
| --- | --- | --- | --- |
| 12  | M   | 35 | NA  |
| 15  | F   | 51 | NA  |
| 30  | M   | 80 | 178 |
| 40  | M   | 73 | NA  |
| 28  | F   | 65 | 165 |

일반적인 supervised learning에서는 labeled data만 활용하고 unsupervised learning에서는 모든 데이터를 활용할 수 있는 반면, semi-supervised learning은 모든 데이터를 활용하면서 data label 여부에 따라 서로 다른 학습 방식을 적용합니다. 어떤 경우에는 labeling된 데이터로 supervised decision boundary를 만들고 이를 기반으로 unlabeled data의 label을 결정하는 경우(pseudo-labeling)도 있고, data에 변형(perturbation)을 가하여 unlabeled data에 대한 모델 예측이 최대한 유지되도록 하는 방법도 있으며, graph를 활용하거나 GAN 기반의 모델을 활용하는 등등 정말 다양한 방법론들이 있습니다. 

그런데, 굳이 왜 unlabeled data를 활용해야 할까요? 많은 사람들이 왜 굳이 잘 정의된 labeled data에 unlabeled data를 더해서 활용하기 위해 노력하는 것을 보면, 분명 이유가 있을텐데 말입니다. 우선 여기에는 두 가지 배경이 있습니다. 우선 첫번째로는 labeling이 된 데이터를 구하기 쉽지 않다는 것을 꼽을 수 있습니다. 

<p align="center"><img src="https://user-images.githubusercontent.com/112034941/209565539-cf9a7a5a-4ea2-408e-986a-5eac1cfe26ac.png" height="780px" width="500px"></p>

위 사진은 구글에서 "data labeling job"으로 검색한 결과입니다. Data labeling에 관련해 굉장히 많은 일자리가 있다는 걸 알 수 있죠. 사실 우리가 어떤 예측 task를 위해 사용하는 대부분의 데이터들은 최초로 누군가가 labeling을 한 데이터입니다. 데이터가 범람하는 현대 사회에서는 이러한 labeling의 수요가 늘어날 수 밖에 없죠. 하지만 사실 모든 데이터를 labeling해서 데이터를 분석하는 것은 비용의 문제, 시간의 문제가 존재합니다. 그래서 지금 이 repository에서 다루는 준지도학습 방법론이 필요한 것이죠. 더불어 두 번째 이유로는 단순히 labeling된 데이터만 사용하는 것이 아니라 unlabeled data도 함께 사용했을 때 성능이 증가하는 경우에 대한 실험결과들이 많이 보고되었습니다. 이에 대해 다양한 추측들이 제기되어 semi-supervised learning이 잘 활용될만한 상황에 대한 가정 또한 만들어졌는데, 우선 이는 추후에 살펴보기로 하겠습니다. 어쨌든 요약하자면 labeling data가 구하기 어렵다는 점과 unlabeled data를 활용하는 것이 더 좋은 성능을 내는 경우가 많다는 것이 바로 semi-supervised learning이 필요한 이유라고 할 수 있겠습니다.

그렇다고, 상기 문단에서도 언급했지만, 항상 semi-supervised learning이 좋은 결과를 내는 것은 아닙니다. 어떤 경우에 semi-superivised learning을 활용하면 좋은 결과를 얻을 수 있는가가에 대해 많은 논의가 있었고, 이러한 논의를 종합한 결과 현재 제가 아는 바로는(배운 바로는) 총 세가지(혹은 네가지)의 가정이 성립해야 한다고 합니다. 아래 그림을 통해 각 가정에 대해 살펴보겠습니다. 아래 그림에서 원형의 점을 제외한 나머지 데이터는 labeled data이며 labeled data의 색이 바로 label입니다. 

![image](https://user-images.githubusercontent.com/112034941/209567208-94708694-abbf-4cce-9bb8-d8eba77b65a1.png)

1. Smoothness Assumption : 가장 왼쪽 그림과 같이, 어떤 labeled data의 주변에 위치하는 unlabeled data는 그 labeled data의 label을 공유해야 한다는 가정입니다. 
2. Low-density Assumption : 중앙의 그림과 같이, 어떤 classification을 위한 분류경계면은 데이터의 밀도가 작은 영역에 위치해야 한다는 가정입니다.
3. Manifold Assumption : 가장 우측 그림과 같이, 어떤 labeled data가 특정 manifold에 포함된다면 그 manifold의 데이터들은 그 labeled data의 label을 가진다는 가정입니다.

위와 같은 세가지의 가정이 만족되는 경우에, 일반적으로 semi-supervised learning이 좋은 성능을 낼 수 있다고 합니다. 물론 어떤 경우 한 가지 가정이 더 필요하다고도 할 수 있는데요, 그것은 현재 데이터의 분포 $p(x)$가 $p(y|x)$에 대한 정보를 담고 있다는 것입니다. 사실 이는 당연한 가정으로, 만약 현재 데이터의 분포로 $p(y|x)$를 전혀 유추할 수 없는 상황이라면 애초에 위 세 가지 가정의 성립이 어려울 것입니다. 따라서 위 세 가지 가정은 $p(x)$가 $p(y|x)$에 대한 정보를 어떤 방식으로 담고 있는가에 대한 더 세부적인 가정이라고 볼 수 있겠죠. 

### Taxonomy of Semi-Supervised Learning
이제부터 본격적으로 무슨 방법론이 semi-supervised learning에 존재하는지를 살펴보겠습니다.우선 [A survey on semi-supervised learning(van Engelen, 2020)](https://link.springer.com/content/pdf/10.1007/s10994-019-05855-6.pdf?pdf=button)에 따르면, semi-supervised learning의 분류는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/112034941/209569048-8ec94899-42bb-4439-9920-02ece846aa80.png)

우선 위의 taxonomy는 classification에 국한된 것으로써, 이 논문에 따르면 주로 semi-supervised learning이 classfication에 활용되었기 때문이라고 합니다.(8. related areas를 참조하시면 좋을 듯합니다.) 그래서 저도 여기서는 classification에 대한 semi-supervised learning을 위주로 살펴보려고 합니다. 우선 준지도학습을 나누는 가장 큰 기준은 inductive method VS transductive method 입니다. 전자의 경우에는 결과적으로 classification 모델을 산출해내는 반면 후자의 경우 곧바로 prediction을 산출한다는 차이가 있습니다. 그리고 iductive mehotds의 경우 wrapper methods, unsupervised processing, intrinsically semi-supervised methods로 나뉘며, transductive의 경우 대부분이 graph based의 방법론입니다.

Inductive method 각각의 방법론의 개략적인 특징만 빠르게 짚고 넘어가보겠습니다. 
1. Wrapper Method: 우선 wrapper method의 경우 가장 오래된 준지도학습 방법론으로써, labelled data로 모델을 학습하고 이 모델로 pseudo labeling을 하는 방식의 방법론입니다. 
2. Unsupervised Preprocssing: unsupervised preprocssing의 경우에는 unlabelled data를 적극적으로 활용하는 방법론인데요, 변수를 추출하거나(extraction) 군집화 후 데이터를 labelling하거나 아니면 learning을 위해 학습 parameter를 초기화하는 등 학습 이전 단계에 활용할 법한 task를 주로 수행하는 방법입니다. 
3. Intrinsically Semi-supervised Methods: 이 경우의 방법론은 최적화하는 objectvie function에 labelled, unlabelled data에 따른 component가 분리되어 포함되어 있습니다. 그리고 가장 최근의 딥러닝 계열의 방법론들이 다수 포함된 것도 바로 이 방법론의 계열입니다.

여기까지 개략적인 준지도학습의 개념에 대해 설명드렸습니다. 이 다음 섹션부터는, 이 repository에서 면밀히 살펴보려는 방법론인 temporal ensembling의 기반과 해당 방법론에 대해 설명하겠습니다. 

---

## Temporal Ensembling
앞서 taxonomy 속에서 이 repository에서 파고들고자 하는 부분은 바로 inductive method - perturbation based methods 중 temporal ensembling입니다. 때문에 이를 이해하기 위해 우선 perturbation method를 살펴보겠습니다.

### Perturbation-based Methods(Consistency Regularization)
![image](https://user-images.githubusercontent.com/112034941/209573331-bb454bb9-88d9-4415-80a4-a9767602605f.png)

Perturbation-based Methods란 앞서 살펴본 준지도학습의 세가지 가정에서 smoothness assumption에 기초하는 방법론으로, 만약 현재 input space가 smoothness assumption을 만족한다면 data point에 작은 변동(perturbation)을 가해도 동일한 prediction이 도출되어야 할 것입니다. 때문에 perturbation base methods에서는 데이터에 작은 노이즈를 가해도 모델이 비슷한 예측을 도출하도록 학습하며, 이 학습을 위해 노이즈를 가하기 전과 후의 차이를 목적함수에 추가합니다. 그리고 이러한 차이는 unsupervised loss라고 하며, 동시에 labelled data에 한해서는 prediction의 정확도도 측정할 수 있으므로 prediction error를 supervisde loss로써 목적함수에 추가합니다. 결과적으로 목적함수는 대체로 unsupervised loss 와 supervised loss의 가중합이 되겠죠. 이러한 방법론이 deep learning을 기반으로 구현된 경우 주로 consistnecy regularization이라고 일컫곤 합니다. 그리고 temporal ensembling은 이 consistency regualarization에 속한 방법론입니다. 

### $\Pi$-model and Temporal Ensembling
Temporal Ensembling은 $\Pi$-model의 확장으로써 제시된 모델입니다. 그러니 우선 $\Pi$-model 부터 살펴보겠습니다. 아래 그림을 보시죠.
![image](https://user-images.githubusercontent.com/112034941/209573421-8a1b3e42-5bf0-4c47-95f8-09d31453bc5c.png)

위 그림처럼, $\Pi$-model은 데이터의 augmentation 이후 같은 weight를 공유하지만 drop out이 다른 neural network 모델 두 가지를 이용해, 모든 데이터에 대해 두 모델의 output을 비교하여 그 비유사도(예를 들면 MSE)를 unsupervised loss로 사용하고, labelled data의 경우 추가적으로 supervised loss(예를 들면 cross entropy)를 사용하여 supervised loss를 산출하여 그 가중 합된 loss를 통해 학습하는 모델입니다. 모델의 구조가 간단하다 보니 다양한 사람들이 모델을 세개, 네개 혹은 그 이상을 사용해서 ensemble 해보고자 하는 시도를 이어 나갔습니다. 그리고 그 중 가장 성공적이었던 시도가 바로 temporal ensembling입니다. 그런데 흥미로운 점은, 사실 제 생각이긴 합니다만, temporal ensembling은 엄밀히 말해 앙상블은 아닙니다. 그냥 같은 모델을 drop out을 변경해가면서 지속적으로 학습시켜 나왔던 예측 값들의 EMA(exponetial moving average)를 통해 unsupervised loss를 계산하고, 이에 해당 epoch에서의 supervised loss를 가중합해서 학습하는 모델이죠. 그림으로 나타내면 아래와 같겠습니다.
![image](https://user-images.githubusercontent.com/112034941/209574494-27f1162b-124b-4480-a56d-7b65a8a10bfd.png)

제 생각에 temporal ensembling 모델의 재밌는 점은 오히려 간단한 방법론을 통해 더 좋은 성능을 냈다는 것입니다. 어떻게 보면 많이들 얘기하는 오컴의 면도날과도 유사한 느낌이 있죠. 이전 방법론들이 두 개 이상의 모델을 필요로 했던 것과는 다르게 오히려 하나의 모델만 사용해서 좋은 성능을 냈다는 점이 말입니다. 이 repository에서는 다루지 못하지만, 이후 perturbation method 방법론들의 발전사를 보면 좀 비슷한 부분이 있습니다. Adversarial training 계열의 방법론들도 보면 결국 더 쉬운 방향의 perturbation을 주는 것이 더 좋은 성능을 낸다는 식의 결론으로 이어지는 면이 있고, temporal ensembling 모델이 발전해나가는 방향도 모델 부분에서 멋지고 복잡한 해결책을 제시하기 보다는 기존 모델의 학습과정에서 얻은 내용을 더 활용하려는 방식으로 이어지는 것을 보면 더욱 그러합니다.

지금까지 temporal ensembling에 대해 살펴보았습니다. 이제 이 temporal ensembling을 직접 구현하고 실습해보겠습니다. 

---

## Implementation
우선 실험에 앞서 필요한 모듈 등의 버젼은 아래와 같습니다.
| env_name   | version |
|------------|---------|
| python     | 3.8.3   |
| numpy      | 1.19.2  |
| matplotlib | 3.5.2   |
| pandas     | 1.4.3   |
| pytorch    | 1.12.1  |

이번 실습을 위해 참조한 code는 https://github.com/ferretj/temporal-ensembling 의 코드입니다. 사용할 데이터셋은 [MNIST](https://yann.lecun.com/exdb/mnist/)입니다.

우선 실험에 앞서 temporal ensembling 논문에서 제시한 수도 코드를 살펴보시면 아래와 같습니다.
![image](https://user-images.githubusercontent.com/112034941/209666648-e8df95f9-2e62-4061-8445-dc81a30fb06d.png)

이 수도코드를 기반으로 아래와 같이 여러 함수 및 class를 정의하였습니다. 주석과 함께 코드를 살펴보시면 좋을 것 같습니다.

### 구현된 코드
1. 데이터로딩 관련 함수: 모델 학습을 위한 data normalize, loading을 위한 함수입니다. MNIST와 CIFAR10에 대해 진행하였습니다.
```python
# MNIST 데이터를 불러오고 정규화한 후 tensor화 하는 함수
def MNIST():
    normalizer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # -1 에서 1 사이로 픽셀값의 정규화
    
    # train dataset 불러오기
    train_dataset = datasets.MNIST(
        root = './MNIST/data', 
        train = True, 
        transform = normalizer,
        download = True)
    
    # test dataset 불러오기
    test_dataset = datasets.MNIST(
        root = './MNIST/data', 
        train = False, 
        transform = normalizer)

    return train_dataset, test_dataset # train, test 반환


# CIFAR10 데이터를 불러오고 정규화한 후 tensor화 하는 함수
def CIFAR10():
    normalizer = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # -1 에서 1 사이로 픽셀값 정규화, 채널이 세개이므로 맞추어 정규화함.

    # train dataset 불러오기
    train_dataset = datasets.CIFAR10(
        root = './CIFAR10/data', 
        train = True, 
        transform = normalizer,
        download = True)
    
    # test dataset 불러오기
    test_dataset = datasets.CIFAR10(
        root = './CIFAR10/data', 
        train = False, 
        transform = normalizer)

    return train_dataset, test_dataset # train, test 반환
```

2. 준지도학습을 위한 data loader: data 의 unlabeling 및 unlabeled data와 labeled data의 batch 단위 반환을 위한 함수.
```python
def SSL_loader(train_dataset, test_dataset, batch_size, n_labeled, n_classes, seed, shuffle_train=True, return_idxs=True):
    rnd = np.random.RandomState(seed) # seed 설정

    labeled_idxs = np.repeat(0, n_labeled)# labeling 된 채로 남겨둘 데이터의 인덱스
    unlabeled_idxs = np.repeat(0, (len(train_dataset) - n_labeled)) # unlabeled로 만들 데이터의 인덱스
    smpl_size = n_labeled // n_classes # 각 class별 labeling 된 채로 random하게 남겨둘 데이터의 수
    
    tmp = 0 # 이 변수는 unlabeled 데이터로 만들어줄 데이터의 index를 지정하기 위해 필요함.
    for label in range(n_classes): 
        # i label을 가진 데이터의 인덱스와 크기 저장
        tmp_class_idxs = (train_dataset.targets == label).nonzero()[:, 0]
        tmp_class_size = len(tmp_class_idxs)

        # 현재 label을 가진 데이터의 '인덱스 리스트의 인덱스'를 permutation함. 
        rnd = np.random.permutation(np.arange(tmp_class_size)) 

        # labeling 된 채로 남겨둘 데이터의 인덱스의 리스트에 smpl size 만큼 permutation된 인덱스에서 추가 
        labeled_idxs[label*smpl_size: (label+1)*smpl_size] = tmp_class_idxs[rnd[:smpl_size]]
        
        # unlabel로 만들 데이터의 인덱스의 리스트에 나머지 permutation된 인덱스 추가
        unlabeled_idxs[tmp: (tmp+tmp_class_size-smpl_size)] = tmp_class_idxs[rnd[smpl_size:]]

        # unlabel로 만들 데이터의 인덱스의 리스트에 나머지 permutation된 인덱스 추가
        tmp += tmp_class_size - smpl_size

    # unlabel로 만들 데이터의 target을 -1로 지정
    train_dataset.targets[unlabeled_idxs] = -1

    # data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               num_workers=10, shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              num_workers=10, shuffle=False)
    
    if return_idxs: # labeled data를 보고 싶은 경우
        return train_loader, test_loader, labeled_idxs 
    
    return train_loader, test_loader
```

3. 기반 모델 관련 코드: temporal ensembling을 진행하기 위해 MNIST와 CIFAR10에 맞추어 간단한 형태의 CNN 모델을 만듦.
```python
# gaussian noise의 class. 현 코드에서는 학습 이전의 텐서에 노이즈를 추가하기 위해 사용되는 class임.
class Gaussian_Noise(nn.Module):
    def __init__(self, batch_size, input_shape, std=0.15):
        super(Gaussian_Noise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = torch.zeros(self.shape).cuda() # GPU에 tensor를 loading
        self.std = std

    def forward(self, x):
        self.noise.data.normal_(mean=0, std=self.std) # 결과적으로, noise는 N(0, 0.15^2)의 정규분포를 따르는 input shape과 동일한 텐서
        return x + self.noise 

class CNN_for_MNIST(nn.Module):
    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32):
        super(CNN_for_MNIST, self).__init__()
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        self.gn    = Gaussian_Noise(batch_size, (1, 28, 28), std=self.std)
        self.act   = nn.LeakyReLU()
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding=1)) # batch normaliztion
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1)) # batch normaliztion 
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc    = nn.Linear(self.fm2 * 7 * 7, 10)
    
    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x


class CNN_for_CIFAR10(nn.Module):
    def __init__(self, batch_size, std, p=0.5, fm1=8, fm2=16):
        super(CNN_for_MNIST, self).__init__()
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        self.gn    = Gaussian_Noise(batch_size, (3, 32, 32), std=self.std)
        self.act   = nn.LeakyReLU()
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(3, self.fm1, 3, padding=1)) # batch normaliztion
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1)) # batch normaliztion 
        self.mp    = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc1 = nn.Linear(8*8*16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    
    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, 8*8*16)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        
        return x
```

4. training 준비에 관련된 코드: temporal ensemble training을 위해 weight 가 epoch에 따라 변경되도록 설정하는 함수, 현재 모델의 supervised loss(cross entropy), unsupervised loss(MSE), 전체 loss를 산출하는 함수, 마지막으로 evaluation을 진행하는 함수로 구성됨.
```python
# 논문에서 제시된 unsupervised와 supervised loss의 가중합을 위한 weight의 구현체
def wt_for_unsupl(epoch, max_ramp_up_epochs, w_max, n_labeled, n_samples):# 각 epoch에 따라 weight가 증가함.
    if epoch == 0:
        return 0. # 첫 epoch에는 unsupervised loss를 더 할 수가 없음.

    elif epoch >= max_ramp_up_epochs: # 설정된 최대 epoch 이후에는 최대 weight가 되도록 함.
        return w_max * (float(n_labeled) / n_samples) 

    else: # 원 논문에서 ramp up period에서는 가우시안 곡선을 따라 weight가 증가하도록 하기 위해 하기의 식을 사용함.
        return w_max * (float(n_labeled) / n_samples) * np.exp(-5. * (1. - float(epoch)/max_ramp_up_epochs)**2)

# 이번 epoch에서의 loss를 계산하기 위한 함수
def temporal_loss(out, z_tilda, w, labels):
    # 현 epoch에서의 unsupervised loss를 위한 MSE 계산
    def MSE(out, z_tilda):
        sqrd_diff = torch.sum((F.softmax(out, dim=1) - F.softmax(z_tilda, dim=1)) ** 2)
        return sqrd_diff / out.data.nelement()
    
    # 현 epoch에서의 supervised loss 계산을 위한 cross_entropy
    def cross_entropy_masked(out, labels):
        tmp_cond = (labels >= 0) # labeled data의 경우 label 값이 0 이상, 아니면 -1임.
        n_nz = torch.nonzero(tmp_cond) # labeled data의 인덱스를 가진 labeled data의 수 by 1의 텐서
        n_tmp_labeled_data = len(n_nz) # 현재 labeled data의 수

        # 현재 labeled data의 수가 0이 아니라면 하기 조건문 실행
        if n_tmp_labeled_data > 0:
            # labeled data에 대한 output 확률과 label 산출
            masked_outputs = torch.index_select(input=out, dim=0, index=n_nz.view(n_tmp_labeled_data))
            masked_labels = labels[tmp_cond]
            ce_loss = F.cross_entropy(masked_outputs, masked_labels) # cross entropy 계산
            return ce_loss, n_tmp_labeled_data 

        return torch.autograd.Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0 # labeled data가 없으면 loss 0
    
    sup_loss, n_tmp_labeled_data = cross_entropy_masked(out, labels) # 해당 함수는 cross entropy와 labeled data의 수 반환
    unsup_loss = MSE(out, z_tilda)
    loss = sup_loss + w*unsup_loss
    return loss, sup_loss, unsup_loss, n_tmp_labeled_data # 이번 epoch에서의 loss, supervised loss, unsupervised loss, labeled data의 수

# test set에서의 평가를 위한 함수
def evaluation(model, data_loader):
    correct = 0
    total = 0
    for _, (samples, labels) in enumerate(data_loader):
        with torch.no_grad(): # gradient 계산이 필요 없음
            samples = Tensor(samples).cuda()
            labels = Tensor(labels).cuda()
            outputs = model(samples) # 예측결과
            _, predicted = torch.max(outputs, 1) # batch의 데이터 객체 별 최대 확률을 나타낸 label을 predicted에 저장
            total += labels.size(0) # batch의 데이터 수
            correct += (predicted == labels.data.view_as(predicted)).sum() # batch 안의 데이터에 대해 예측이 맞은 개수 산출

    acc = 100 * float(correct) / total # 전체 정확도 산출
    return acc
```

5. training 코드: 정해진 epoch 동안 각 batch에 대해서 학습을 진행하고 진행한 후 test set을 통해 evaluation하는 코드.
```python
def train_and_eval(model, seed, dataset='MNIST', n_labeled=100, alpha=0.6, lr=0.002, max_epochs=300, 
                    batch_size=100, n_classes=10, max_ramp_up_epochs=80, w_max=300., n_samples=60000, print_res=True, **kwargs):

    # data loading 
    if dataset == 'MNIST':
        train_dataset, test_dataset = MNIST()
    elif dataset == 'CIFAR10':
        train_dataset, test_dataset = CIFAR10()
    
    n_train = len(train_dataset)
    n_samples = n_train

    # 모델을 GPU상에 load
    model.cuda()

    # semi-supervised learning을 위한 unlabeled data가 포함된 train, test batch 단위의 데이터 로더 생성
    train_loader, test_loader, total_labeled_idx = SSL_loader(train_dataset, test_dataset, batch_size, n_labeled, n_classes, 
                                                            seed, shuffle_train=False)
    c = n_train // batch_size

    # optimizer 생성
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # training 시작
    model.train()
    losses = []
    sup_losses = []
    unsup_losses = []
    best_loss = 25. # 그냥 내가 25살이라

    # 원 논문에서는 Z와 z를 0 벡터로 initialize함.
    Z = torch.zeros(n_train, n_classes).float().cuda() # 원 논문에서 Z에 해당(앙상블 예측 accumulation) 
    z = torch.zeros(n_train, n_classes).float().cuda() # 원 논문에서 z에 해당
    outputs = torch.zeros(n_train, n_classes).float().cuda() # model output

    for epoch in range(max_epochs):
        st = time.time() # 시작 시간 시간 측정
        
        # unsup loss 가중합을 위한 wt 계산
        w = wt_for_unsupl(epoch, max_ramp_up_epochs, w_max, n_labeled, n_samples)
        if (epoch + 1) % 10 == 0:
            print('현재 epoch: %d, w(t): %f' %(epoch+1, w))
    
        # gpu에 있는 tensor에 연산해줘야 하므로 텐서로 변환
        w = torch.autograd.Variable(torch.FloatTensor([w]).cuda(), requires_grad=False) 
        
        batch_loss = []
        batch_sup_loss = []
        batch_unsup_loss = []
        for i, (images, labels) in enumerate(train_loader):
            images = Tensor(images.cuda())
            labels = torch.autograd.Variable(labels.cuda(), requires_grad=False)

            # forward 연산 및 loss 산출
            optimizer.zero_grad()
            tmp_output = model(images)
            z_tilda = torch.autograd.Variable(z[i*batch_size: (i+1)*batch_size], requires_grad=False)
            loss, sup_loss, unsup_loss, n_tmp_labeled_data = temporal_loss(tmp_output, z_tilda, w, labels)

            # 현재 output 저장
            outputs[i*batch_size: (i+1)*batch_size] = tmp_output.data.clone()
            batch_loss.append(loss.item())
            batch_sup_loss.append(n_tmp_labeled_data * sup_loss.item())
            batch_unsup_loss.append(unsup_loss.item())

            # back propagation을 통한 네트워크 update
            loss.backward()
            optimizer.step()

            # 현재 epoch, step
            if (epoch + 1) % 10 == 0:
                if (i+1) == c:
                    print ('Epoch: %d /%d, Step: %d /%d, Loss: %.6f, Elapsed time: %.2f sec /epoch' 
                           %(epoch + 1, max_epochs, i + 1, len(train_dataset) // batch_size, np.mean(batch_loss), time.time() - st))
                elif (i+1) == (c//2):
                    print ('Epoch: %d /%d, Step: %d /%d, Loss: %.6f' 
                           %(epoch + 1, max_epochs, i + 1, len(train_dataset) // batch_size, np.mean(batch_loss)))

        # update temporal ensemble
        Z = alpha*Z + (1.-alpha)*outputs
        z = Z / (1. - alpha**(epoch+1))

        # loss 저장
        epoch_loss = np.mean(batch_loss)
        losses.append(epoch_loss)
        sup_losses.append((1. / n_labeled) * np.sum(batch_sup_loss)) 
        unsup_losses.append(np.mean(batch_unsup_loss))
        
        # 최고성능의 model save 
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({'state_dict': model.state_dict()}, 'model_best.pth.tar')

    # 최대 epoch
    # model evaluation
    model.eval() # drop out 그만하기 
    acc = evaluation(model, test_loader)
    if print_res:
        print('현재 모델의 test accuracy(on 10000개의 test image): %.2f %%' % (acc))
        
    # test best model
    model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])
    model.eval()
    acc_from_best = evaluation(model, test_loader)
    if print_res:
        print('지금까지 train error가 가장 낮은 모델의 test accuracy(on 10000개의 test image): %.2f %%' % (acc_from_best))
     
    return acc, acc_from_best, losses, sup_losses, unsup_losses, total_labeled_idx
```

6. 실험결과를 저장하는 코드: 해당 코드를 통해 여러 번의 실험을 통해 나온 결과를 저장함.
```python
# 결과 저장을 위한 함수. 여러 번의 실험 진행을 가정한 상황에서 구성됨.
# https://github.com/ferretj/temporal-ensembling의 코드를 거의 참조함.
def save_exp_result(time, losses, sup_losses, unsup_losses, accs, accs_best, idxs, **kwargs):
    labels = ['seed_'+str(sd) for sd in kwargs['seeds']]
    if not os.path.isdir('exp_result'):
        os.mkdir('exp_result')
    time_dir = os.path.join('exp_result', time)
    if not os.path.isdir(time_dir):
        os.mkdir(time_dir)
    
    fname_bst = os.path.join('exp_result', time, 'training_loss_plot_best.png')
    fname_all = os.path.join('exp_result', time, 'training_loss_plotall.png')
    fname_smr = os.path.join('exp_result', time, 'summary.txt')
    fname_lsp  = os.path.join('exp_result', time, 'labeled_samples')

    best = np.argmax(accs_best)
    save_loss_plot([losses[best]], [sup_losses[best]], [unsup_losses[best]], fname_bst) 
    save_loss_plot(losses, sup_losses, unsup_losses, fname_all, labels=labels)
    for seed, tmp_label_idx in zip(kwargs['seeds'], idxs):
        save_seed_samples(fname_lsp + '_seed' + str(seed) + '.png', tmp_label_idx)
    save_txt(fname_smr, accs, **kwargs)


def save_loss_plot(losses, sup_losses, unsup_losses, f_name, labels=None):
    plt.style.use('ggplot')

    # plot의 선 별로 색을 다르게 하기 위한 작업. 원래 코드에서는 randy olson의 pallete를 참조했다던데, 무슨 의미인지는 잘 모르겠음.
    colors = [
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14), 
        (255, 187, 120),    
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),    
        (148, 103, 189),
        (197, 176, 213), 
        (140, 86, 75),
        (196, 156, 148),    
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),    
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
        (158, 218, 229)]
    colors = [(float(c[0]) / 255, float(c[1]) / 255, float(c[2]) / 255) for c in colors]

    _, axs = plt.subplots(3, 1, figsize=(12, 18)) # plot 세개가 세로로 나타남.
    for i in range(3):
        axs[i].tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    for i in range(len(losses)):
        axs[0].plot(losses[i], color=colors[i])
        axs[1].plot(sup_losses[i], color=colors[i])
        axs[2].plot(unsup_losses[i], color=colors[i])
    axs[0].set_title('Overall loss', fontsize=14)
    axs[1].set_title('Supervised loss', fontsize=14)
    axs[2].set_title('Unsupervised loss', fontsize=14)
    
    if labels is not None:
        axs[0].legend(labels)
        axs[1].legend(labels)
        axs[2].legend(labels)
    plt.savefig(f_name)


def save_txt(f_name, accs, **kwargs):
    with open(f_name, 'w') as fp:
        fp.write('실험횟수: %d\n' %(kwargs['n_exp']))
        fp.write('labeling된 데이터의 수: %d\n' %(kwargs['n_labeled']))
        fp.write('dropout: %f \n' %(kwargs['drop']))
        fp.write('alpha: %f \n' %(kwargs['alpha']))
        fp.write('guasiisan noise std: %f \n' %(kwargs['std']))
        fp.write('ADAM lr: %f \n' %(kwargs['lr']))
        fp.write('ADAM beta2: %f \n' %(kwargs['beta2']))
        fp.write('최대 epoch: %f \n' %(kwargs['max_epochs']))
        fp.write('batch size: %f \n' %(kwargs['batch_size']))

        fp.write('\n실험결과\n')
        fp.write('best accuracy: {}\n'.format(np.max(accs)))
        fp.write('accuracy: {} (std={})\n'.format(np.mean(accs), np.std(accs)))
        fp.write('실험별 정확도: {}\n'.format(accs))


# 학습에 사용된 labeled data를 plot으로 저장
def save_seed_samples(fname, indices):
    train_dataset, test_dataset = MNIST()
    imgs = train_dataset.data[indices.astype(int)]

    plt.style.use('classic')
    fig = plt.figure(figsize=(30, 30))
    gs = gsp.GridSpec(10, 10, width_ratios=[1 for _ in range(10)], wspace=0.0, hspace=0.0)

    for ll in range(100):
        i = ll // 10
        j = ll % 10
        img = imgs[ll].numpy()
        ax = plt.subplot(gs[i, j])
        ax.tick_params(axis="both", which="both", bottom="off", top="off",
                        labelbottom="off", left="off", right="off", labelleft="off")
        ax.imshow(img)

    plt.savefig(fname)
```

7. 실험진행을 위한 ipynb 파일의 코드 
```python
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import config
from temporal_ensembling import train_and_eval
from utils import Gaussian_Noise, save_exp_result
import datetime

class CNN_for_MNIST(nn.Module):
    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32):
        super(CNN_for_MNIST, self).__init__()
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        self.gn    = Gaussian_Noise(batch_size, (1, 28, 28), std=self.std)
        self.act   = nn.LeakyReLU()
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding=1)) # batch normaliztion
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1)) # batch normaliztion 
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc    = nn.Linear(self.fm2 * 7 * 7, 10)
    
    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x


# MNIST 실험
accs = []
accs_best = []
losses = []
sup_losses = []
unsup_losses = []
idxs = []

st_for_exp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
cfg = vars(config)

dataset = 'MNIST'
for i in range(cfg['n_exp']):
    model = CNN_for_MNIST(cfg['batch_size'], cfg['std'])
    seed = cfg['seeds'][i]
    acc, acc_best, l, sl, usl, total_labeled_idx = train_and_eval(model, seed, dataset='MNIST', **cfg)
    accs.append(acc)
    accs_best.append(acc_best)
    losses.append(l)
    sup_losses.append(sl)
    unsup_losses.append(usl)
    idxs.append(total_labeled_idx)

save_exp_result(st_for_exp, losses, sup_losses, unsup_losses, accs, accs_best, idxs, **cfg)
```

### 실험진행 결과
실험에 사용한 dataset은 MNIST이고, 총 3회를 반복 실험하였습니다. 자세한 실험환경 및 파라미터 설정결과 및 실험 결과는 아래와 같습니다. 
- 윈도우 10, GPU: rtx 3080 
- 실험횟수: 3
- labeling된 데이터의 수: 100
- dropout: 0.500 
- alpha: 0.600
- guasiisan noise std: 0.150 
- ADAM lr: 0.003
- ADAM beta2: 0.990 
- 최대 epoch: 200.
- batch size: 100. 
>
- best accuracy: 98.46
- 전체 평균 accuracy: 97.72 (std=0.81)
- 실험별 평균 정확도: 98.11, 98.46, 96.59

실험 결과를 시드별로 정리한 plot은 아래와 같습니다. 
![training_loss_plotall](https://user-images.githubusercontent.com/112034941/209675736-21bfa5a8-ac84-4705-924b-037be1dba44c.png)

또한, 시드별로 사용된 labelled data의 plot은 아래와 같습니다.

1. seed 77
![labeled_samples_seed77](https://user-images.githubusercontent.com/112034941/209675870-8f361afe-413b-4ac7-a852-8cb2c8f480be.png)

2. seed 137
![labeled_samples_seed137](https://user-images.githubusercontent.com/112034941/209675876-3846b4f0-ba82-41ea-a31a-efb98533556d.png)

3. seed 195
![labeled_samples_seed195](https://user-images.githubusercontent.com/112034941/209675877-f5919af0-cb8e-4f51-8d6d-964d9808256e.png)

실험결과를 보면, labeling data를 100개만 사용한 후 매우 단순한 축에 속하는 모델을 활용했음에도 불구하고 정확도가 매우 높음(96% 이상)을 알 수 있습니다. 이에 대해 이유를 생각해보면 우선 MNIST의 input space에서의 representation을 고려할 필요가 있을 것입니다. MNIST를 t-SNE를 이용해 2차원으로 mapping한 결과를 보면 아래와 같습니다.
![image](https://user-images.githubusercontent.com/112034941/209679877-116179f2-b2c2-48b2-84e4-f7796a8883cc.png)

이를 보면 앞서 살펴본 semi-supervised learning의 세가지 가정, smoothness assumption, low density assumption, manifold assumption 모두가 잘 만족되는 데이터셋이라고 할 수 있을 것입니다. 이러한 특성이 반영된 결과로써 semi-supervised learning의 성능이 높을 수 있었던 것으로 생각할 수 있겠습니다.

---

## Conclusion
지금까지 temporal ensembling의 기반이 되는 다양한 개념들과 이를 실제로 구현한 결과 및 실험 결과에 대해서 살펴보았습니다.이번 repository commit에서는 우선 MNIST를 활용한 temporal ensembling의 구현에 집중하였고 추후에는 지금까지 진행한 실험에 더하여 추후 CIFAR10에 대해서도 동일한 실험을 진행하고 다양한 loss fucntion 변경 및 모델 구조 변경을 진행하는 것을 목표로 하고 있습니다. 

혹시 여기서 구현된 내용을 재현하고 싶으신 분들의 경우에는 우선 deep learning 용 GPU 설정을 진행하시는 것이 좋을 것이라 생각됩니다. 그리고 앞서 적어 둔바와 같이 환경 버젼을 통일하시고 올려둔 uitls.py, temporal_ensembling.py, config.py, temporal_ensemble_tutorial.ipynb 모두 다운받으신 후 이를 같은 폴더에 두고 temporal_ensemble_tutorial.ipynb를 실행하시면 됩니다. 그러면 실험 진행 과정이 아래처럼 나타날 것입니다. 진행된 실험 결과의 경우 동일한 폴더의 exp_result 폴더의 해당 실험 시작 시간 폴더 안에 저장됩니다. 
![image](https://user-images.githubusercontent.com/112034941/209676777-3bb54927-3a09-44c1-a79b-ed35f632cffd.png)

만약 이 repository를 참조하실 분들이 있다면 첨부드린 코드와 함께 꼭 readme file 서문에 써놓았듯 고려대학교 강필성 교수님의 [강의영상](https://www.youtube.com/watch?v=vhitW3gsuhw&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW&index=30)과 첨부논문을 꼭 참조하시길 바랍니다. 이 repository를 보신 분들이 temporal ensemble은 물론 다른 머신러닝 분야들도 더 재미있게 공부하실 수 있으면 좋겠습니다. 감사합니다.
