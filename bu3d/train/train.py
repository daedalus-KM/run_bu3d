import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import h5py
from parse_args import args, update_args
os.sys.path.append("../")

# load unroll model
from unroll.model.vanilla import Model
from unroll.dataset import dataset

from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(args.dresult)

# some settings
torch.backends.cudnn.benchmark=True
torch.manual_seed(1)
np.random.seed(1)

#------------------------------------------------
# load data and the model
#------------------------------------------------
#middlebury
ds = dataset.TofDataset(f"../data/middlebury/train_{args.fdata}.h5", nscale=args.nscale, pnorm=args.pnorm, userefl=args.userefl)
print(f"@ data size: {len(ds)}")

print("make ds_loader")
ds_loader = torch.utils.data.DataLoader(ds, args.b, shuffle=True, pin_memory=True, drop_last=True)

model = Model(ds.nscale, attention_type=args.att, nblock=args.nblock, K=args.K, alpha=args.alpha)
model.cuda()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
nparams = sum([np.prod(p.size()) for p in model_parameters])
print(f"nparams: {nparams}")

if args.resume > 0:
    model.load_state_dict(torch.load(args.dresult+f"model_{nparams}_{args.resume}.pth"))
    if args.resume > 100:
        args.lr *= 0.5
    # args.nepoch += 21

nblock = model.nblock

# for simplicity, I omitted the validation part

#------------------------------------------------
# run
#------------------------------------------------
params = model.parameters()
# betas=(0.9, 0.99)
opt = torch.optim.Adam(params, args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)

# opt = torch.optim.SGD(params, args.lr, momentum=0.9)
if args.b == 0:
    args.b = len(ds)

# weights for the loss terms
if args.losstype == 0:
    loss_weights = torch.ones(nblock, dtype=torch.float32)
else:
    loss_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

hfepoch = args.nepoch // 2

for epoch in range(args.resume, args.nepoch):
    print(epoch)
    for i, ds_batch in enumerate(ds_loader):
        depth, _, d_gt, _ = ds_batch[0], ds_batch[1], ds_batch[2], ds_batch[3]
            
        xgt = d_gt.cuda()
        opt.zero_grad()
        x, dd, m, rr = model(depth.cuda(), None)

        loss = 0.0
        for ii in range(nblock):
            loss += loss_weights[ii] * F.l1_loss(xgt, x[ii], reduction='mean')

        gstep = epoch*len(ds_loader) + i
        loss.backward()
        opt.step()

        ############### log
        writer.add_scalar('training loss', loss, gstep)

    # for simplicity, visualizations are omitted

    if epoch == hfepoch:
        for g in opt.param_groups:
            g['lr'] = g['lr'] * 0.5

    if (epoch+1) % 30 == 0:
        torch.save(model.state_dict(), args.dresult+f"model_{nparams}_{epoch}.pth")

#------------------------------------------------
# save output
#------------------------------------------------
writer.add_hparams( {'lr':args.lr,'b':args.b}, {'hparams/loss': loss, 'hparams/nparams':nparams})
writer.close()

torch.save(model.state_dict(), args.dresult+f"model_{nparams}.pth")