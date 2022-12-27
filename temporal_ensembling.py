import numpy as np
import time
import torch
from torch import Tensor
import torch.nn.functional as F
from utils import MNIST, CIFAR10, wt_for_unsupl, evaluation

def SSL_loader(train_dataset, test_dataset, batch_size, n_labeled, n_classes, seed, shuffle_train=True, return_idxs=True):
    rnd = np.random.RandomState(seed) # seed 설정

    labeled_idxs = np.repeat(0, n_labeled)# labeling 된 채로 남겨둘 데이터의 인덱스
    unlabeled_idxs = np.repeat(0, (len(train_dataset) - n_labeled)) # unlabeled로 만들 데이터의 인덱스
    smpl_size = n_labeled // n_classes # 각 class별 labeling 된 채로 random하게 남겨둘 데이터의 수
    
    tmp = 0 # 이 변수는 unlabeled 데이터로 만들어줄 데이터의 index를 지정하기 위해 필요함.
    for label in range(n_classes): 
        # i label을 가진 데이터의 인덱스와 크기 저장
        tmp_class_idxs = (Tensor(train_dataset.targets) == label).nonzero()[:, 0]
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