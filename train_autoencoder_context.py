from __future__ import print_function
import argparse
import os
import sys
import shutil
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import time
from model import _netlocalD, _netG
from model_context_encoder import Generator, Discriminator, SmallGenerator_16x_plus, Autoencoder_SED
import utils
from my_utils import logprint
pjoin = os.path.join

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='dataset/train/paris', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--less_nef',type=int,default=16,help='of encoder filters in first conv layer')
parser.add_argument('--less_ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_false', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--wtl2',type=float,default=0.999,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')
parser.add_argument('-m', '--mode', type=str)
parser.add_argument('--e1', type=str, default=None)
parser.add_argument('--e2', type=str, default=None)
parser.add_argument('--ploss_weight', type=float, default=1.0)
parser.add_argument('-p', '--project_name', type=str)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--test_data', type=str, default="/home3/wanghuan/Dataset/paris_test_data")
args = parser.parse_args()

# Set up directories and logs etc
project_path = pjoin("../KD/Experiments", args.project_name)
rec_img_path = pjoin(project_path, "reconstructed_images")
weights_path = pjoin(project_path, "weights") # to save torch model
cropped_img_path = pjoin(rec_img_path, "cropped")
real_img_path    = pjoin(rec_img_path, "real")
recon_img_path   = pjoin(rec_img_path, "recon")

if not args.resume:
  if os.path.exists(project_path):
    respond = "Y" # input("The appointed project name has existed. Do you want to overwrite it (everything inside will be removed)? (y/n) ")
    if str.upper(respond) in ["Y", "YES"]:
      shutil.rmtree(project_path)
    else:
      exit(1)
  if not os.path.exists(rec_img_path):
    os.makedirs(rec_img_path)
  if not os.path.exists(weights_path):
    os.makedirs(weights_path)
  if not os.path.exists(cropped_img_path):
    os.makedirs(cropped_img_path)
  if not os.path.exists(real_img_path):
    os.makedirs(real_img_path)
  if not os.path.exists(recon_img_path):
    os.makedirs(recon_img_path)
    
TIME_ID = os.environ["SERVER"] + time.strftime("-%Y%m%d-%H%M")
log_path = pjoin(weights_path, "log_" + TIME_ID + ".txt")
log = open(log_path, "w+")
logprint(str(args._get_kwargs()), log)


if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
logprint("Random Seed: %s" % args.manualSeed, log)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    logprint("WARNING: You have a CUDA device, so you should probably run with --cuda", log)

if args.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(args.imageSize),
                                   transforms.CenterCrop(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif args.dataset == 'lsun':
    dataset = dset.LSUN(db_path=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(args.imageSize),
                            transforms.CenterCrop(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif args.dataset == 'streetview':
    transform = transforms.Compose([transforms.Scale(args.imageSize),
                                    transforms.CenterCrop(args.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=args.dataroot, transform=transform )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers))

ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
nc = 3
nef = int(args.nef)
nBottleneck = int(args.nBottleneck)
wtl2 = float(args.wtl2)
overlapL2Weight = 10

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch=0

# netG = _netG(opt)
netG = Autoencoder_SED(args, args.e1, args.e2) # replace the original network with my net
netG.apply(weights_init)
# if args.netG != '': # resume
    # netG.load_state_dict(torch.load(args.netG,map_location=lambda storage, location: storage)['state_dict'])
    # resume_epoch = torch.load(args.netG)['epoch']
# logprint(str(netG), log)


# netD = _netlocalD(opt)
netD = Discriminator(args)
netD.apply(weights_init)
# if args.netD != '': # resume
    # netD.load_state_dict(torch.load(args.netD,map_location=lambda storage, location: storage)['state_dict'])
    # resume_epoch = torch.load(args.netD)['epoch']
# logprint(str(netD), log)


criterionCLS = nn.BCELoss()
criterionMSE = nn.MSELoss()


input_real    = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
input_cropped = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
label = torch.FloatTensor(args.batchSize)
real_label = 1
fake_label = 0
real_center = torch.FloatTensor(args.batchSize, 3, int(args.imageSize/2), int(args.imageSize/2))

if args.cuda:
  netD.cuda()
  netG.cuda()
  criterionCLS.cuda()
  criterionMSE.cuda()
  input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
  real_center = real_center.cuda()

input_real    = Variable(input_real)
input_cropped = Variable(input_cropped)
label         = Variable(label)
real_center   = Variable(real_center)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

for epoch in range(resume_epoch, args.niter):
  for i, data in enumerate(dataloader, 0):
    real_cpu, _ = data
    real_center_cpu = real_cpu[:,:,int(int(args.imageSize/4)):int(int(args.imageSize/4))+int(int(args.imageSize/2)),
                                   int(int(args.imageSize/4)):int(int(args.imageSize/4))+int(int(args.imageSize/2))]
    input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
    real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
    # initialized to the mean value and make the range in [-1, 1]
    input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
    input_cropped.data[:, 0, int(int(args.imageSize/4)+args.overlapPred) : int(int(args.imageSize/4)+int(args.imageSize/2)-args.overlapPred),
                             int(int(args.imageSize/4)+args.overlapPred) : int(int(args.imageSize/4)+int(args.imageSize/2)-args.overlapPred)] = 2*117.0/255.0 - 1.0
    input_cropped.data[:, 1, int(int(args.imageSize/4)+args.overlapPred) : int(int(args.imageSize/4)+int(args.imageSize/2)-args.overlapPred),
                             int(int(args.imageSize/4)+args.overlapPred) : int(int(args.imageSize/4)+int(args.imageSize/2)-args.overlapPred)] = 2*104.0/255.0 - 1.0
    input_cropped.data[:, 2, int(int(args.imageSize/4)+args.overlapPred) : int(int(args.imageSize/4)+int(args.imageSize/2)-args.overlapPred),
                             int(int(args.imageSize/4)+args.overlapPred) : int(int(args.imageSize/4)+int(args.imageSize/2)-args.overlapPred)] = 2*123.0/255.0 - 1.0
    
    # -------------------------------------------------
    # (1) Update D network
    ### train with real
    netD.zero_grad()
    batch_size = real_cpu.size(0)
    label.data.resize_(batch_size).fill_(real_label)
    output = netD(real_center)
    errD_real = criterionCLS(output, label) ##!!!!!!!
    errD_real.backward()
    prob_real = output.data.mean() # the average output prob for real images

    ### train with fake
    # noise.data.resize_(batch_size, nz, 1, 1)
    # noise.data.normal_(0, 1)
    # -----------------------------
    # replace
    # fake = netG(input_cropped)
    feats, feats2 = netG(input_cropped); fake = feats2[-1]
    # -----------------------------
    label.data.fill_(fake_label)
    output = netD(fake.detach()) # detach, so that the loss only be considered for discriminator
    errD_fake = criterionCLS(output, label) ##!!!!!!!
    errD_fake.backward()
    prob_fake = output.data.mean() # the average output prob for fake images
    
    errD = errD_real + errD_fake # total loss for discriminator
    optimizerD.step()

    # -------------------------------------------------
    # (2) Update G network: maximize log(D(G(z)))
    netG.zero_grad()
    label.data.fill_(real_label)  # fake labels are real for generator cost
    output = netD(fake)
    errG_D = criterionCLS(output, label) ##!!!!!!!
    # errG_D.backward(retain_variables=True)

    # errG_l2 = criterionMSE(fake, real_center)
    # Instead of using the provided MSE loss, use different weights for the overlap part and the center part in the image
    wtl2Matrix = real_center.clone()
    wtl2Matrix.data.fill_(wtl2 * overlapL2Weight)
    wtl2Matrix.data[:, :, int(args.overlapPred) : int(int(args.imageSize/2) - args.overlapPred),
                          int(args.overlapPred) : int(int(args.imageSize/2) - args.overlapPred)] = wtl2
    errG_l2 = (fake - real_center).pow(2)
    errG_l2 = errG_l2 * wtl2Matrix
    errG_l2 = errG_l2.mean()

    # --------------------------------------------------------------
    # feature reconstruction loss
    ploss1 = criterionMSE(feats2[0], feats[0].data) * args.ploss_weight
    ploss2 = criterionMSE(feats2[1], feats[1].data) * args.ploss_weight
    ploss3 = criterionMSE(feats2[2], feats[2].data) * args.ploss_weight
    ploss4 = criterionMSE(feats2[3], feats[3].data) * args.ploss_weight
    ploss5 = criterionMSE(feats2[4], feats[4].data) * args.ploss_weight
    ploss6 = criterionMSE(feats2[5], feats[5].data) * args.ploss_weight
    ploss = ploss1 + ploss2 + ploss3 + ploss4 + ploss5 + ploss6
    # --------------------------------------------------------------
    
    # ------------------------------
    # replace
    # errG = (1 - wtl2) * errG_D + wtl2 * errG_l2
    errG = (1 - wtl2) * errG_D + wtl2 * errG_l2 + ploss
    # ------------------------------
    
    errG.backward()
    D_G_z2 = output.data.mean()
    optimizerG.step()
    
    # Logging
    format_str = "[%d/%d][%d/%d] | lossD=%.4f lossG=%.4f-%.4f | ploss1=%.4f ploss2=%.4f ploss3=%.4f ploss4=%.4f ploss5=%.4f ploss6=%.4f | prob_real=%.4f prob_fake=%.4f"
    logstr = format_str % (epoch,  args.niter, i, len(dataloader),
        errD.data[0], errG_D.data[0], errG_l2.data[0],
        ploss1.cpu().data.float(), ploss2.cpu().data.float(), ploss3.cpu().data.float(), ploss4.cpu().data.float(), ploss5.cpu().data.float(), ploss6.cpu().data.float(),
        prob_real, prob_fake)
    logprint(logstr, log)
    
    if i % 100 == 0:
      vutils.save_image(real_cpu, pjoin(real_img_path, 'real_samples_epoch_%03d.png' % (epoch)))
      vutils.save_image(input_cropped.data, pjoin(cropped_img_path, 'cropped_samples_epoch_%03d.png' % (epoch)))
      recon_image = input_cropped.clone()
      recon_image.data[:,:,int(int(args.imageSize/4)):int(int(args.imageSize/4)+int(args.imageSize/2)),
                           int(int(args.imageSize/4)):int(int(args.imageSize/4)+int(args.imageSize/2))] = fake.data
      vutils.save_image(recon_image.data, pjoin(recon_img_path, 'recon_center_samples_epoch_%03d.png' % (epoch)))

      # check reconstruction results
      test_imgs = [pjoin(args.test_data, i) for i in os.listdir(args.test_data)]
      for img_path in test_imgs:
        try:
          img = Image.open(img_path).convert("RGB")
        except:
          continue
        img = transforms.ToTensor()(img).unsqueeze(0).cuda(args.gpu)
        _, feats2 = netG(img); decoded = feats2[-1]
        out_img_path = pjoin(rec_img_path, "%s_%s_E%sS%s.jpg" % (TIME_ID, os.path.splitext(os.path.basename(img_path))[0], epoch, i))
        vutils.save_image(decoded.data.cpu().float(), out_img_path)
      
      # save model
      torch.save(netG.e2.state_dict(), pjoin(weights_path, "%s_%s_E%sS%s_G.pth" % (TIME_ID, args.mode, epoch, i)))
      torch.save(netD.state_dict(),    pjoin(weights_path, "%s_%s_E%sS%s_D.pth" % (TIME_ID, args.mode, epoch, i)))