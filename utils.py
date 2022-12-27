from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gsp
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import transforms

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


# 논문에서 제시된 unsupervised와 supervised loss의 가중합을 위한 weight의 구현체
def wt_for_unsupl(epoch, max_ramp_up_epochs, w_max, n_labeled, n_samples):# 각 epoch에 따라 weight가 증가함.
    if epoch == 0:
        return 0. # 첫 epoch에는 unsupervised loss를 더 할 수가 없음.

    elif epoch >= max_ramp_up_epochs: # 설정된 최대 epoch 이후에는 최대 weight가 되도록 함.
        return w_max * (float(n_labeled) / n_samples) 

    else: # 원 논문에서 ramp up period에서는 가우시안 곡선을 따라 weight가 증가하도록 하기 위해 하기의 식을 사용함.
        return w_max * (float(n_labeled) / n_samples) * np.exp(-5. * (1. - float(epoch)/max_ramp_up_epochs)**2)


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


