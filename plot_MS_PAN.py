import random
import cv2
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_visual_comp(data_path, save_dir, num_imgs=10, fused_present=False, filename='plot.png', show=False):
    '''
    Input:
        scipy io input for in *.mat format containing 'PAN', 'MS_out' and 'MS' labels
        mat['MS'] --> B, H, W, C
        mat['PAN'] --> B, H, W
    Output:
        Row 1: MS original   
        Row 2: PAN original
        Raw 3: MS generated
    '''
    data = io.loadmat(data_path)
    PAN = data['PAN']
    MS_out = data['MS_out']
    MS = data['MS']
    nrows = 3
    if fused_present:
        Fu = data['Fused']
        nrows = 4
    
    num_samples = PAN.shape[0]
    if num_samples < num_imgs:
        num_imgs = num_samples

    idxs = random.sample(range(num_samples), num_imgs)
    fig, axs = plt.subplots(nrows=nrows, ncols=num_imgs, figsize=(15, 12))  # Change 3 -> 4
    RGB = (2, 1, 0)   # Can change maybe add to arguments
    # RGB = (4, 2, 1)   # WV2
    
    for i, idx in enumerate(idxs):
        # Computing quantile for normalisation
        Q_MS = np.quantile(MS[idx], (0.02, 0.98), (0, 1), keepdims=True)
        Q_PAN = np.quantile(PAN[idx], (0.02, 0.98), (0, 1), keepdims=True)
        if fused_present:
            Q_Fu = np.quantile(Fu[idx], (0.02, 0.98), (0, 1), keepdims=True)    # #
            Fu_norm = (Fu[idx] - Q_Fu[0, :, :]) / (Q_Fu[1, :, :] - Q_Fu[0, :, :])

        # Plotting images
        PAN_norm = (PAN[idx] - Q_PAN[0, :, :]) / (Q_PAN[1, :, :] - Q_PAN[0, :, :])
        MS_norm = (MS[idx] - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
        MS_norm = np.clip(MS_norm, 0, 1)
        MS_out_norm = (MS_out[idx] - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
        MS_out_norm = np.clip(MS_out_norm, 0, 1)
        axs[0,i].imshow(MS_norm[:, :, RGB])
        axs[1,i].imshow(PAN_norm, cmap='gray')
        axs[2,i].imshow(MS_out_norm[:, :, RGB])
        if fused_present:
            axs[3,i].imshow(Fu_norm[:, :, RGB])

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f'{save_dir}/{filename}')
    if show:
        plt.show()
    
    # Add both in this order for keeping memory usage low
    plt.clf()   
    plt.close()

    return idxs


def plot_visual_comp_senven(data_path, save_dir, num_imgs=10, fused_present=False, disc_present=True, abs_diff=False, filename='plot.png', show=False):
    '''
    Input:
        scipy io input for in *.mat format containing 'PAN', 'MS_out' and 'MS' labels
        mat['MS'] --> B, H, W, C
        mat['PAN'] --> B, H, W
    Output:
        Row 1: MS original   
        Row 2: PAN original
        Raw 3: MS generated
    '''
    data = io.loadmat(data_path)
    MS_out = data['MS_out']
    MS = data['MS']
    nrows = 2
    if disc_present or abs_diff:
        Disc = data['Disc']
        print(Disc.shape)
        nrows += 1
    
    if fused_present:
        Fu = data['Fused']
        nrows += 1
    
    num_samples = MS.shape[0]
    if num_samples < num_imgs:
        num_imgs = num_samples
    # print()
    idxs = random.sample(range(num_samples), num_imgs)
    fig, axs = plt.subplots(nrows=nrows, ncols=num_imgs, figsize=(15, 12))  # Change 3 -> 4
    RGB = (2, 1, 0)   # Can change maybe add to arguments
    # RGB = (4, 2, 1)   # WV2
    
    for i, idx in enumerate(idxs):
        # Computing quantile for normalisation
        Q_MS = np.quantile(MS[idx], (0.02, 0.98), (0, 1), keepdims=True)
        Q_MS_out = np.quantile(MS_out[idx], (0.02, 0.98), (0, 1), keepdims=True)
        if fused_present:
            Q_Fu = np.quantile(Fu[idx], (0.02, 0.98), (0, 1), keepdims=True)    # #
            Fu_norm = (Fu[idx] - Q_Fu[0, :, :]) / (Q_Fu[1, :, :] - Q_Fu[0, :, :])
            Fu_norm = np.clip(Fu_norm, 0, 1)
            
        if disc_present and (Disc.shape[-1] <= 4):
            Q_Disc = np.quantile(Disc[idx], (0.02, 0.98), (0, 1), keepdims=True)    # #
            Disc_norm = (Disc[idx] - Q_Disc[0, :, :]) / (Q_Disc[1, :, :] - Q_Disc[0, :, :])
            Disc_norm = np.clip(Disc_norm, 0, 1)

        # Plotting images
        # PAN_norm = (PAN[idx] - Q_PAN[0, :, :]) / (Q_PAN[1, :, :] - Q_PAN[0, :, :])
        MS_norm = (MS[idx] - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
        MS_norm = np.clip(MS_norm, 0, 1)
        MS_out_norm = (MS_out[idx] - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
        MS_out_norm = np.clip(MS_out_norm, 0, 1)
        axs[0,i].imshow(MS_norm[:, :, RGB])
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        # print(np.moveaxis(MS_out[idx, :, :, RGB], 0, -1).min(), np.moveaxis(MS_out[idx, :, :, RGB], 0, -1).max())
        # print(np.moveaxis(MS[idx, :, :, RGB], 0, -1).min(), np.moveaxis(MS[idx, :, :, RGB], 0, -1).max())
        # axs[0,i].imshow(np.moveaxis(MS[idx, :, :, RGB], 0, -1)/(2**11))
        axs[1,i].imshow(MS_out_norm[:, :, RGB])
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])
        
        if disc_present and (Disc[idx].shape[-1] > 4):
            axs[2,i].imshow(Disc[idx], cmap='gray', vmin=0, vmax=1)
            axs[2,i].set_xticks([])
            axs[2,i].set_yticks([])
        elif disc_present:
            axs[2,i].imshow(Disc_norm[:, :, RGB])
            axs[2,i].set_xticks([])
            axs[2,i].set_yticks([])
            
        if abs_diff:
            # Divide existing axes and create
            # new axes at right side of image
            divider = make_axes_locatable(axs[2, i])
            cax = divider.append_axes("right", size="5%", pad=0.15)
            
            im = axs[2,i].imshow(Disc[idx], cmap='turbo', vmin=0, vmax=0.5)
            axs[2,i].set_xticks([])
            axs[2,i].set_yticks([])
            
            plt.colorbar(im, cax=cax)  #ax=axs[2, i])
        # axs[1,i].imshow(np.moveaxis(MS_out[idx, :, :, RGB], 0, -1)/(2**11))
        if fused_present:
            axs[nrows-1,i].imshow(Fu_norm[:, :, RGB])
            axs[nrows-1,i].set_xticks([])
            axs[nrows-1,i].set_yticks([])

    # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(f'{save_dir}/{filename}')
    plt.savefig(f'{save_dir}/{filename}')
    if show:
        plt.show()
    
    # Add both in this order for keeping memory usage low
    plt.clf()   
    plt.close()

    return idxs



def plot_histogram_comp(data_path, save_dir, num_imgs=5, idxs=None, filename='plot_hist.png', show=False):
    '''
    Row 1: Red histogram of MS_out and MS
    Row 2: Green histogram of MS_out and MS
    Row 3: Blue histogram of MS_out and MS
    Row 4: NIR histogram of MS_out and MS
    '''
    data = io.loadmat(data_path)
    # PAN = data['PAN']
    MS_out = data['MS_out']
    MS = data['MS']
    norm = data['NORM']
    RGB = (2, 1, 0)
    num_samples = MS_out.shape[0]
    if num_samples < num_imgs:
        num_imgs = num_samples
    # if idxs is None:
    idxs = random.sample(range(num_samples), num_imgs)

    fig, axs = plt.subplots(nrows=5, ncols=num_imgs, figsize=(20, 12),
                            gridspec_kw = {'height_ratios':[2,1,1,1,1]})
    window_size = 5
    for i, idx in enumerate(idxs):
        Q_MS = np.quantile(MS_out[idx], (0.02, 0.98), (0, 1), keepdims=True)
        MS_norm = (MS_out[idx] - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
        MS_norm = np.clip(MS_norm, 0, 1)
        axs[0,i].imshow(MS_norm[:, :, RGB])
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        
        for j in range(1, 5):
            lb, ub = norm[j]
            lb = int(lb)
            ub = int(ub)
            histr = cv2.calcHist([MS[idx]],[j-1],None,[ub-lb],[lb, ub])
            # Smooth the histogram using a moving average
            shistr = np.convolve(histr.flatten(), np.ones(window_size) / window_size, mode='same')
            histr2 = cv2.calcHist([MS_out[idx]],[j-1],None,[ub-lb],[lb, ub])
            shistr2 = np.convolve(histr2.flatten(), np.ones(window_size) / window_size, mode='same')
            axs[j, i].plot(shistr)
            axs[j, i].plot(shistr2)

    axs[0, 0].set_ylabel("Output")
    axs[1, 0].set_ylabel("Blue")
    axs[2, 0].set_ylabel("Green")
    axs[3, 0].set_ylabel("Red")
    axs[4, 0].set_ylabel("NIR")
    fig.tight_layout()
    plt.savefig(f'{save_dir}/{filename}')
    if show:
        plt.show()

    # Add both in this order for keeping memory usage low
    plt.clf()   
    plt.close()
    
    return idxs


if __name__ == '__main__':
    data_path = './data_out_results.mat'
    
    idxs = plot_visual_comp(data_path, num_imgs=5, show=True)
    plot_histogram_comp(data_path, num_imgs=5)
