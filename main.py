#############################################################################
# Import                                                                    #
#############################################################################
import os
import random
import PIL.Image as Image
from tqdm import tqdm

import numpy as np
import scipy.io

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#############################################################################
# Hyperparameters                                                           #
#############################################################################
opt = DotDict()

opt.nx = 3
opt.sizex = 256
opt.dataset = 'chairs'

# Convolution settings
opt.generator = 'dcgan'
opt.nResBlocs = 9
opt.nLayers = 4
opt.nf = 128
opt.nz = 128

# Hardward settings
opt.workers = 4               # workers data for preprocessing
opt.cuda = True               # use CUDA
opt.gpu = 1                   # GPU id

# Optimisation scheme
opt.batchSize = 20            # minibatch size
opt.nIteration = 1000001      # number of training iterations
opt.lrG = 2e-4
opt.lrD = 5e-5

# Save/Load networks
opt.checkpointDir = '.'
opt.load = 0                  # if > 0, load given checkpoint
opt.checkpointFreq = 1000     # frequency of checkpoints (in number of epochs)

#############################################################################
# Loading Weights                                                           #
#############################################################################
opt.netG = ''
opt.netT = ''
opt.netD1 = ''
opt.netD2 = ''
opt.netC = ''

if opt.load > 0:
    opt.netG = '%s/netG_%d.pth' % (opt.checkpointDir, opt.load)
    opt.netT = '%s/netT_%d.pth' % (opt.checkpointDir, opt.load)
    opt.netD1 = '%s/netD1_%d.pth' % (opt.checkpointDir, opt.load)
    opt.netD2 = '%s/netD2_%d.pth' % (opt.checkpointDir, opt.load)
    opt.netC = '%s/netC_%d.pth' % (opt.checkpointDir, opt.load)

#############################################################################
# RandomSeed                                                                #
#############################################################################   
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#############################################################################
# CUDA                                                                      #
#############################################################################   
if opt.cuda:
    cudnn.benchmark = True
    torch.cuda.set_device(opt.gpu)

#############################################################################
# Datasets                                                                  #
#############################################################################
class ChairsDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, labelFile, sets='train', transform=transforms.ToTensor()):
        super(ChairsDataset, self).__init__()
        self.dataPath = dataPath
        with open(labelFile, 'r') as f:
            self.files = np.array([p[:-1] for p in f.readlines()])
        self.sets = sets
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        filex = os.path.join(self.dataPath, self.files[idx])
        imgx = self.transform(Image.open(filex))
        if (imgx.size(0) == 1):
            imgx = imgx.repeat(3,1,1)
        return imgx, torch.LongTensor(1).fill_(0)
    
if opt.dataset == 'chairs':
    trainset = ChairsDataset("/local/data/rendered_chairs/train",
                             "/local/data/rendered_chairs/train.txt",
                             'train',
                             transforms.Compose([transforms.CenterCrop(400),
                                                 transforms.RandomCrop(300),
                                                 transforms.Resize(opt.sizex),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
elif opt.dataset == 'celebA':
    trainset = torchvision.datasets.ImageFolder("/local/data/celebA/",
                                                transforms.Compose([transforms.CenterCrop(160),
                                                                    transforms.RandomCrop(128),
                                                                    transforms.Resize(opt.sizex),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=4, drop_last=True)

#############################################################################
# Modules                                                                   #
#############################################################################
class _dcDecoder(nn.Module):
    def __init__(self, nIn=1024, nOut=3, nf=512, nLayer=4, sizeX=64):
        super(_dcDecoder, self).__init__()
        sizeX = sizeX // (2**nLayer)
        nf = nf * (2 ** (nLayer - 1))
        self.mods = nn.Sequential()
        self.mods.add_module("FC_%dx%dx%d" % (nf,sizeX,sizeX), nn.ConvTranspose2d(nIn, nf, sizeX, bias=False))
        self.mods.add_module("BN0", nn.BatchNorm2d(nf))
        self.mods.add_module("ReLU0", nn.ReLU(True))
        for i in range(1,nLayer):
            sizeX = sizeX * 2
            self.mods.add_module("ConvTr%d_%dx%dx%d" % (i, nf//2, sizeX, sizeX), nn.ConvTranspose2d(nf, nf//2, 4, 2, 1, bias=False))
            self.mods.add_module("BN%d"% i, nn.BatchNorm2d(nf//2))
            self.mods.add_module("ReLU%d" % i, nn.ReLU(True))
            nf = nf // 2
        self.mods.add_module("ConvTrO_%dx%dx%d" % (nf, sizeX, sizeX), nn.ConvTranspose2d(nf, nOut, 4, 2, 1, bias=False))
    def forward(self, x):
        return self.mods(x)

class _dcDiscriminator(nn.Module):
    def __init__(self, nIn=3, nOut=1024, nf=64, nLayer=4, sizeX=64):
        super(_dcDiscriminator, self).__init__()
        self.mods = nn.Sequential()
        sizeX = sizeX //2
        self.mods.add_module("Conv0_%dx%dx%d" % (nf, sizeX, sizeX), nn.Conv2d(nIn, nf, 4, 2, 1, bias=False))
        self.mods.add_module("LReLU0", nn.LeakyReLU(0.2))
        for i in range(1,nLayer):
            sizeX = sizeX //2
            self.mods.add_module("Conv%d_%dx%dx%d" % (i, nf*2, sizeX, sizeX), nn.Conv2d(nf, nf*2, 4, 2, 1, bias=False))
            self.mods.add_module("BN%d"% i, nn.BatchNorm2d(nf*2))
            self.mods.add_module("LReLU%d" % i, nn.LeakyReLU(0.2))
            nf = nf * 2
        self.mods.add_module("FC_%dx1x1" % nOut, nn.Conv2d(nf, nOut, sizeX, bias=False))
    def forward(self, x):
        return self.mods(x)

class _resEncoder(nn.Module):
    def __init__(self, nc=3, nf=128):
        super(_resEncoder, self).__init__()
        self.convs = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(nc,    nf//4, 7),
                                   nn.InstanceNorm2d(nf//4),
                                   nn.ReLU(),
                                   nn.Conv2d(nf//4, nf//2, 3, 2, 1),
                                   nn.InstanceNorm2d(nf//2),
                                   nn.ReLU(),
                                   nn.Conv2d(nf//2, nf, 3, 2, 1),
                                   nn.InstanceNorm2d(nf),
                                   nn.ReLU(),
                                   )   
    def forward(self, x):
        return self.convs(x)

class _resBloc(nn.Module):
    def __init__(self, nf=128):
        super(_resBloc, self).__init__()
        self.blocs = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1),
                                   nn.InstanceNorm2d(nf), #nn.BatchNorm2d(nf),
                                   nn.ReLU(),
                                   nn.Conv2d(nf, nf, 3, 1, 1),
                                  )
        self.activationF = nn.Sequential(nn.InstanceNorm2d(nf), #nn.BatchNorm2d(nf),
                                         nn.ReLU(),
        )
    def forward(self, x):
        return self.activationF(self.blocs(x) + x)

class _resDecoder(nn.Module):
    def __init__(self, nc=3, nf=128):
        super(_resDecoder, self).__init__()
        self.convs = nn.Sequential(nn.ConvTranspose2d(nf, nf//2, 3, 2, 1, 1),
                                   nn.InstanceNorm2d(nf//2),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(nf//2, nf//4, 3, 2, 1, 1),
                                   nn.InstanceNorm2d(nf//4),
                                   nn.ReLU(),
                                   nn.ReflectionPad2d(3),
                                   nn.Conv2d(nf//4, nc, 7),
                                   )
    def forward(self, x):
        return self.convs(x)

class _cResEncoder(nn.Module):
    def __init__(self, nc=3, nf=128, ns=10):
        super(_cResEncoder, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(nc,    nf//4, 7, bias=False),
                                    nn.Conv2d(nf//4, nf//2, 3, 2, 1, bias=False),
                                    nn.Conv2d(nf//2, nf, 3, 2, 1, bias=False)])
        self.norms = nn.ModuleList([nn.InstanceNorm2d(nf//4),
                                    nn.InstanceNorm2d(nf//2),
                                    nn.InstanceNorm2d(nf)])
        self.gammas = nn.ModuleList([nn.Conv2d(ns, nf//4, 1, bias=False),
                                     nn.Conv2d(ns, nf//2, 1, bias=False),
                                     nn.Conv2d(ns, nf, 1, bias=False)])
    def forward(self, x, s):
        x0 = F.pad(x, pad=(3,3,3,3), mode="reflect")
        x1 = F.relu(self.norms[0](self.convs[0](x0)) * self.gammas[0](s))
        x2 = F.relu(self.norms[1](self.convs[1](x1)) * self.gammas[1](s))
        x3 = F.relu(self.norms[2](self.convs[2](x2)) * self.gammas[2](s))
        return x3

class _cResBloc(nn.Module):
    def __init__(self, nf=128, ns=10):
        super(_cResBloc, self).__init__()
        self.blocs = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
                                   nn.InstanceNorm2d(nf),
                                   nn.ReLU(),
                                   nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
                                  )
        self.norm = nn.InstanceNorm2d(nf)
        self.gamma = nn.Conv2d(ns, nf, 1, bias=False)
    def forward(self, x, s):
        out = self.blocs(x) + x
        return F.relu(self.norm(out) * self.gamma(s))

class _cResDecoder(nn.Module):
    def __init__(self, nc=3, nf=128, ns=10):
        super(_cResDecoder, self).__init__()
        self.convs = nn.ModuleList([nn.ConvTranspose2d(nf, nf//2, 3, 2, 1, 1),
                                    nn.ConvTranspose2d(nf//2, nf//4, 3, 2, 1, 1),
                                    nn.Conv2d(nf//4, nc, 7)])        
        #         self.convs = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
        #                                                   nn.ReflectionPad2d(1),
        #                                                   nn.Conv2d(nf, nf//2, 3, 1, 0, bias=False)),
        #                                     nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
        #                                                   nn.ReflectionPad2d(1),
        #                                                   nn.Conv2d(nf//2, nf//4, 3, 1, 0, bias=False)),
        #                                     nn.Conv2d(nf//4, nc, 7, bias=False)])                                    
        self.norms = nn.ModuleList([nn.InstanceNorm2d(nf//2),
                                    nn.InstanceNorm2d(nf//4)])
        self.gammas = nn.ModuleList([nn.Conv2d(ns, nf//2, 1, bias=False),
                                     nn.Conv2d(ns, nf//4, 1, bias=False)])
    def forward(self, x0, s):
        x1 = F.relu(self.norms[0](self.convs[0](x0)) * self.gammas[0](s))
        x2 = F.relu(self.norms[1](self.convs[1](x1)) * self.gammas[1](s))
        x2 = F.pad(x2, pad=(3,3,3,3), mode="reflect")
        x3 = self.convs[2](x2)
        return x3

class _cResDec(nn.Module):
    def __init__(self, nRes=6, nf=128, nc=3, ns=10, activation=None):
        super(_cResDec, self).__init__()
        self.resBlocs = nn.ModuleList([_cResBloc(nf=nf, ns=ns) for i in range(nRes)])
        self.decoder = _cResDecoder(nf=nf, nc=nc, ns=ns)
        self.activation = activation
    def forward(self, x, s):
        for resBloc in self.resBlocs:
            x = resBloc(x, s)
        x = self.decoder(x, s)
        if self.activation:
            x = self.activation(x)
        return x

class _resEnc(nn.Module):
    def __init__(self, nRes=6, nf=128, nc=3, ns=10, activation=None):
        super(_resEnc, self).__init__()
        self.encoder = _resEncoder(nc=3, nf=nf)
        self.resBlocs = nn.ModuleList([_resBloc(nf=nf) for i in range(nRes)])
        self.activation = activation
    def forward(self, x):
        out = self.encoder(x)
        for resBloc in self.resBlocs:
            out = resBloc(out)
        if self.activation:
            out = self.activation(out)
        return out
    
netG = _dcDecoder(nIn=opt.nz, nOut=opt.nf, nf=opt.nf, nLayer=opt.nLayers, sizeX=opt.sizex//4)
netT = _cResDec(nRes=opt.nResBlocs, nf=opt.nf, nc=opt.nx, ns=opt.nz, activation=None)
netC = _resEnc(nRes=opt.nResBlocs, nf=opt.nf, nc=opt.nx)
netD2 = _dcDiscriminator(nIn=opt.nx, nOut=1, nf=opt.nf, nLayer=opt.nLayers, sizeX=opt.sizex)
netD1 = _dcDiscriminator(nIn=3, nOut=1, nf=opt.nf, nLayer=opt.nLayers, sizeX=64)
advLoss = nn.BCEWithLogitsLoss()

#############################################################################
# Placeholders                                                              #
#############################################################################
x = torch.FloatTensor()
c = torch.FloatTensor()
s = torch.FloatTensor()
labelPos = torch.FloatTensor(1,1,1,1).fill_(.9)
labelNeg = torch.FloatTensor(1,1,1,1).fill_(.1)

if opt.cuda:
    netD1.cuda()
    netD2.cuda()
    netG.cuda()
    netT.cuda()
    netC.cuda()
    x = x.cuda()
    c = c.cuda()
    s = s.cuda()
    labelPos = labelPos.cuda()
    labelNeg = labelNeg.cuda()

#############################################################################
# Test                                                                      #
#############################################################################
nTest = 10
c_test = torch.FloatTensor(nTest, 1, opt.nz).normal_()
s_test = torch.FloatTensor(1, nTest, opt.nz).normal_()

c_test = c_test.expand(-1, nTest, -1).contiguous().view(nTest*nTest,opt.nz,1,1)
s_test = s_test.expand(nTest, -1, -1).contiguous().view(nTest*nTest,opt.nz,1,1)


out = torch.FloatTensor(nTest, nTest, opt.nx, opt.sizex, opt.sizex)

if opt.cuda:
    c_test = c_test.cuda()
    s_test = s_test.cuda()
    out = out.cuda()

#############################################################################
# Optimizer                                                                 #
#############################################################################
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
optimizerT = optim.Adam(netT.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
optimizerC = optim.Adam(netC.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
optimizerD2 = optim.Adam(netD2.parameters(), lr=opt.lrD, betas=(0.5, 0.999))
optimizerD1 = optim.Adam(netD1.parameters(), lr=opt.lrD, betas=(0.5, 0.999))

#############################################################################
# Train                                                                     #
#############################################################################
iteration = opt.load

log_d1Pos = []
log_d1Neg = []
log_d2Pos = []
log_d2Neg = []
log_rec = []
netG.train()
netT.train()

netD2.train()
netD1.train()
netC.train()

genData = iter(trainloader)

pbar = tqdm(total=opt.checkpointFreq)
while iteration <= opt.nIteration:
    pbar.update(1)
    ######################### Get Batch ################################
    try:
        x_cpu, _ = next(genData)
    except StopIteration:
        genData = iter(trainloader)
        x_cpu, _ = next(genData)
    x.resize_(x_cpu.size(0), x_cpu.size(1), x_cpu.size(2), x_cpu.size(3)).copy_(x_cpu)
    c.resize_(x_cpu.size(0), opt.nz, 1, 1).normal_()
    s.resize_(x_cpu.size(0), opt.nz, 1, 1).normal_()
    ######################### Prepare Networks #########################
    netG.zero_grad()
    netT.zero_grad()
    netC.zero_grad()
    netD1.zero_grad()
    netD2.zero_grad()
    ######################### ##########################################
    xG = netG(Variable(c))
    xG = F.softmax(xG.view(xG.size(0), xG.size(1), xG.size(2)*xG.size(3)), dim=2).view(xG.size(0), xG.size(1), xG.size(2), xG.size(3))
    xT = F.tanh(netT(xG, Variable(s)))
    xS = F.avg_pool2d(xT, opt.sizex//64, opt.sizex//64)
    xC = netC(xT)
    xC = F.softmax(xC.view(xC.size(0), xC.size(1), xC.size(2)*xC.size(3)), dim=2).view(xC.size(0), xC.size(1), xC.size(2), xC.size(3))
    d1Gen = netD1(xS)
    d2Gen = netD2(xT)
    err_d1Gen = advLoss(d1Gen, Variable(labelPos.expand_as(d1Gen)))
    err_d2Gen = advLoss(d2Gen, Variable(labelPos.expand_as(d2Gen)))
    err_rec = (xC - xG).abs().mean()
    (err_d1Gen + err_d2Gen + err_rec).backward()
    ################### Discriminators step #############################
    netD1.zero_grad()
    netD2.zero_grad()
    d1Pos = netD1(F.avg_pool2d(Variable(x), opt.sizex//64, opt.sizex//64))
    d2Pos = netD2(Variable(x))
    d1Neg = netD1(xS.detach())
    d2Neg = netD2(xT.detach())
    err_d1Pos = advLoss(d1Pos, Variable(labelPos.expand_as(d1Pos)))
    err_d2Pos = advLoss(d2Pos, Variable(labelPos.expand_as(d2Pos)))
    err_d1Neg = advLoss(d1Neg, Variable(labelNeg.expand_as(d1Neg)))
    err_d2Neg = advLoss(d2Neg, Variable(labelNeg.expand_as(d2Neg)))
    (err_d1Pos + err_d1Neg + err_d2Pos + err_d2Neg).backward()
    #################### Updates ########################################
    optimizerG.step()
    optimizerT.step()
    optimizerC.step()
    optimizerD1.step()
    optimizerD2.step()
    ###################### Logging #####################################
    d1Pos = d1Pos.detach()
    d1Neg = d1Neg.detach()
    d1Pos.volatile = True
    d1Neg.volatile = True
    log_d1Pos.append(F.sigmoid(d1Pos).data.mean())
    log_d1Neg.append(F.sigmoid(d1Neg).data.mean())
    d2Pos = d2Pos.detach()
    d2Neg = d2Neg.detach()
    d2Pos.volatile = True
    d2Neg.volatile = True
    log_d2Pos.append(F.sigmoid(d2Pos).data.mean())
    log_d2Neg.append(F.sigmoid(d2Neg).data.mean())
    log_rec.append(err_rec.data.squeeze()[0])
    iteration += 1
    if iteration % opt.checkpointFreq == 0:
        pbar.close()
        with open('logs.dat', 'ab') as f:
            np.savetxt(f, np.vstack((
                np.array(log_d1Pos),
                np.array(log_d1Neg),
                np.array(log_d2Pos),
                np.array(log_d2Neg),
                np.array(log_rec),
            )).T)
        print(iteration,
              np.array(log_d1Pos).mean(),
              np.array(log_d1Neg).mean(),
              np.array(log_d2Pos).mean(),
              np.array(log_d2Neg).mean(),
              np.array(log_rec).mean(),
        )
        log_d1Pos = []
        log_d1Neg = []
        log_d2Pos = []
        log_d2Neg = []
        log_rec = []
        netG.eval()
        netT.eval()
        # netD1.eval()
        # netD2.eval()
        xG_test = netG(Variable(c_test, volatile=True))
        xG_test = F.softmax(xG_test.view(xG_test.size(0), xG_test.size(1), xG_test.size(2)*xG_test.size(3)), dim=2).view(xG_test.size(0), xG_test.size(1), xG_test.size(2), xG_test.size(3))
        xT_test = F.tanh(netT(xG_test, Variable(s_test, volatile=True)))
        try:
            vutils.save_image(xG_test.data.mean(1).unsqueeze(1).expand(-1,3,-1,-1), "%s/out_%d_G.png" % (opt.checkpointDir, iteration), normalize=True, nrow=nTest)
            vutils.save_image(xT_test.data, "%s/out_%d_T.png" % (opt.checkpointDir, iteration), normalize=True, range=(-1,1), nrow=nTest)
            torch.save(netG.state_dict(), '%s/netG_%d.pth' % (opt.checkpointDir, iteration))
            torch.save(netT.state_dict(), '%s/netT_%d.pth' % (opt.checkpointDir, iteration))
            torch.save(netD1.state_dict(), '%s/netD1_%d.pth' % (opt.checkpointDir, iteration)) 
            torch.save(netD2.state_dict(), '%s/netD2_%d.pth' % (opt.checkpointDir, iteration)) 
            torch.save(netC.state_dict(), '%s/netC_%d.pth' % (opt.checkpointDir, iteration))
        except:
            print("Cannot write in %s" % opt.checkpointDir)
        pbar = tqdm(total=opt.checkpointFreq)
        netG.train()
        netT.train()
        # netD1.train()
        # netD2.train()
