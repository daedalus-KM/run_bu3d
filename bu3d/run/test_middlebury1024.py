import mat73
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
import time
# os.sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.sys.path.append("../")
from unroll.dataset import dataset
from unroll.model.vanilla import Model
from gen_initial import gen_initial_multiscale, shift_h
from compute_uncertainty import compute_uncertainty

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False

use_cuda = True if torch.cuda.is_available() else False
def main():
    L = 12
    T = 1024
    ds = dataset.TofDataset(f"../data/middlebury/test_Art_T=1024_ppp=64.0_sbr=64.0.h5", nscale=12, pnorm=0, userefl=0)
    # ds.add_ds(dataset.TofDataset(f"../data/middlebury/test_Reindeer_T=1024_ppp=64.0_sbr=64.0.h5", nscale=12, pnorm=0, userefl=0))
    ds_loader = torch.utils.data.DataLoader(ds, 1, shuffle=False, drop_last=False)

    with torch.no_grad():
        model = Model(L)
        if use_cuda:
            torch.cuda.empty_cache()
            model.cuda()
            model.load_state_dict(torch.load("../result/mpi/T=1024_ppp=64.0_sbr=64.0/ours_K=3_alpha=2.0_att=pa_b=16_datamode=2_losstype=0_lr=0.0001_nblock=4_nepoch=50_nscale=12_tau=10.0_userefl=0_zmemo=x/model_53136.pth"))
        else:
            model.load_state_dict(torch.load("../result/mpi/T=1024_ppp=64.0_sbr=64.0/ours_K=3_alpha=2.0_att=pa_b=16_datamode=2_losstype=0_lr=0.0001_nblock=4_nepoch=50_nscale=12_tau=10.0_userefl=0_zmemo=x/model_53136.pth", map_location=torch.device('cpu')))

        for i, ds_batch in enumerate(ds_loader):
            t0 = time.time()

            depth, _, _, _ = ds_batch[0], ds_batch[1], ds_batch[2], ds_batch[3]

            if use_cuda:
                out = model(depth.cuda(), debug = True)
            else:
                out = model(depth, debug = True)
            
            depth_final = out[0][-1].cpu().numpy()

            t1 = time.time()
            print("@ elapsed time:", t1 - t0)

            if os.path.isdir("../result/") == False:
                os.makedirs("../result/")

            to_meter = T * 0.003
            img = np.flipud(depth_final[0,0,:,:]*to_meter)
            plt.imsave(f'depth.png', img)

            uncertainty = compute_uncertainty(out)
            img = np.flipud(uncertainty)
            
            plt.imsave('uncertainty.png', img, cmap = 'gray')
    

if __name__ == "__main__":
    main()
    # main()
