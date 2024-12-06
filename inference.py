"""
Multiple step denoising inference of AirECG.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from models import AirECG_model
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 17, 15
import numpy as np
from scipy.stats import pearsonr
import pickle

import os


def extract_model(model_name):
    """
    load a pre-trained AirECG model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find AirECG checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]
    return checkpoint

def sample_images(ref, y, x, samples, batchIdx, resultPath, isVal=True):
    """Saves generated signals from the validation set"""

    current_img_dir = resultPath + '/%s_Val.png' % (batchIdx)

    fig, axes = plt.subplots(y.shape[0], 4)
    axes[0][0].set_title('Ref')
    axes[0][1].set_title('mmWave')
    axes[0][2].set_title('ECG GroundTruth')
    axes[0][3].set_title('Generated ECG')

    
    for idx, signal in enumerate(y):
        axes[idx][0].plot(ref[idx], color='c')
        axes[idx][1].plot(y[idx], color='c')
        axes[idx][2].plot(x[idx], color='m')
        axes[idx][3].plot(samples[idx], color='y')

    fig.canvas.draw()
    fig.savefig(current_img_dir)
    plt.close(fig)

from torch.utils.data import DataLoader,Dataset
def DataLoader_example(bs, personID=0, fold_idx=0, splitChannel = False, mmWaveNorm = True):
    #Load your data here
    test_mmwave = torch.randn(16, 8, 1024)
    test_ecg = torch.randn(16, 1024)
    ref_ecg = torch.randn(16, 1024)

    def patchingmmWave(inputSignal):
        B = inputSignal.shape[0]
        C = inputSignal.shape[1]
        inputSignal = inputSignal.reshape(B,C,32,32)
        return inputSignal
    
    def patchingECG(inputSignal):
        B = inputSignal.shape[0]
        inputSignal = inputSignal.reshape(B,32,32)
        inputSignal = inputSignal.unsqueeze(1)
        return inputSignal
    
    test_mmwave = patchingmmWave(test_mmwave)
    test_ecg = patchingECG(test_ecg)
    ref_ecg = patchingECG(ref_ecg)

    class DataSet(Dataset):
        def __init__(self, x : torch.Tensor, y : torch.Tensor, y_ref : torch.Tensor):
            self.x = x
            self.y = y
            self.y_ref = y_ref
        
        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, index):
            return [self.x[index], self.y[index], self.y_ref[index]] 

    testECGDataSet = DataSet(test_mmwave.float(), test_ecg.float(), ref_ecg.float())
    test_ecg_loader = DataLoader(testECGDataSet, batch_size=bs, shuffle=False)

    return test_ecg_loader



def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Load model:
    latent_size = 32
    model = AirECG_model(
        input_size=latent_size,
        mm_channels = args.mmWave_channels,
    ).to(device)

    ckpt_path = args.ckpt
    state_dict = extract_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    in_channels = args.mmWave_channels

    testPersonList = [0]

    for personIdx in testPersonList:
        # Modify your data loader here to replace the following random data
        if in_channels == 8:
            test_mmwave_loader = DataLoader_example(bs = int(args.global_batch_size), personID=personIdx, splitChannel=False, mmWaveNorm = True)
        else:
            pass
        
        import pandas as pd
        result_df = pd.DataFrame(columns = ['BatchIdx','Corr','CrossCorr', 'MSE'], dtype=object)
        resultPath = ckpt_path.rsplit('/',2)[0] + '/eval/' + ckpt_path.split('/')[-1] + '/OnePerson/' + str(personIdx)+'/'
        os.makedirs(resultPath, exist_ok=True)
        corrList = [] 
        crossCorrList = []
        mseList = []

        ecgList = []
        sampleList = []
        
        idx = 0
        for batch_idx, (mmwave, ecg, ref) in enumerate(tqdm(test_mmwave_loader)):
            n = ecg.shape[0]
            mmwave = mmwave.to(device)
            ref = ref.to(device)

            z = torch.randn(n, 1, latent_size, latent_size, device=device)
            # Setup guidance:
            
            model_kwargs = dict(y1=mmwave, y2=ref)
            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )

            mmwave = mmwave.reshape(n,in_channels,1024)[:,0,:]
            ecg = ecg.reshape(n, 1024)
            samples = samples.reshape(n, 1024)
            ref = ref.reshape(n, 1024).cpu().numpy()
            mmwave = mmwave.cpu().numpy()
            ecg = ecg.cpu().numpy()
            samples = samples.cpu().numpy()
            
            ecgList.append(ecg)
            sampleList.append(samples)
            
            if batch_idx < 3:
                sample_images(ref, mmwave, ecg, samples, batch_idx, resultPath)

            for i in range(n):
                corr = pearsonr(samples[i],ecg[i])
                tempSample = samples[i]
                # tempSample = tempSample - tempSample.mean()
                tempECG = ecg[i]
                # tempECG = tempECG - tempECG.mean()
                crossCorr = (np.correlate(tempSample,tempECG,mode='valid') / np.correlate(tempECG,tempECG,mode='valid'))[0]
                mse = np.mean((samples[i,:] - ecg[i,:]) ** 2)
                result_df.loc[idx] = [idx, corr[0],crossCorr, mse]
                idx += 1
                corrList.append(abs(corr[0]))
                crossCorrList.append(crossCorr)
                mseList.append(mse)
                
        corrList = np.array(corrList)
        mseList = np.array(mseList)
        result_df.loc[idx] = ['AVG', np.mean(corrList),np.mean(crossCorrList), np.mean(mseList)]

        ecgList = np.vstack(ecgList)
        sampleList = np.vstack(sampleList)
        pickle.dump(ecgList, open(resultPath + '/ecg.pkl','wb'), protocol=4)
        pickle.dump(ecgList, open(resultPath + '/sample.pkl','wb'), protocol=4)
        result_df.to_csv(resultPath + '/resultCross.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmWave-channels", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='') #Define your model weights here
    args = parser.parse_args()
    main(args)
