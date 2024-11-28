import torch
from diffusion_utils.diffusion_multinomial import sum_except_batch
import numpy as np

#these functions have to return the log of the ratios !

def no_constraint(log_x,t):
    return torch.zeros_like(log_x)

def dummy_score_white(log_x,t):
    #returns the score of the distribution r(x) = 1/Z * 2^{number of non zero pixels of x}

    #the scores have the same shape than the images
    log_score = torch.zeros_like(log_x)

    #calculate the scores

    index = torch.ones_like(log_x)
    index[:,0,...]=0
    b1 = (index>0) & (log_x < 0)
    b2 = (index == 0) & (log_x < 0)
    device = log_score.get_device()
    
    log_score = torch.where(b1, torch.log(torch.tensor(2.)).to(device), torch.where(b2, torch.log(torch.tensor(0.5)).to(device), torch.tensor(1.).to(device)))

    return log_score

def dummy_score_black(log_x,t):
    #returns 1/r(x) where r(x) is the distribution defined in dummy_score_white
    log_score = torch.zeros_like(log_x)

    index = torch.ones_like(log_x)
    index[:,0,...]=0
    b1 = (index>0) & (log_x < 0)
    b2 = (index == 0) & (log_x < 0)
    device = log_score.get_device()
    
    log_score = torch.where(b1, torch.log(torch.tensor(0.5)).to(device), torch.where(b2, torch.log(torch.tensor(2.0)).to(device), torch.tensor(1.).to(device)))

    return log_score

def test_score(log_x,t):
    log_score = torch.zeros_like(log_x)

    #calculate the scores
    t = t[:, None,None,None,None].expand(log_x.shape)
    index = torch.ones_like(log_x)
    index[:,0,...]=0
    b1 = (index>0) & (log_x < 0) & (t <= 100)
    b2 = (index == 0) & (log_x < 0) & (t <= 100)
    device = log_score.get_device()
    
    log_score = torch.where(b1, torch.log(torch.tensor(2.)).to(device), torch.where(b2, torch.log(torch.tensor(0.5)).to(device), torch.tensor(1.).to(device)))

    return log_score


def circle_constraint(log_x, t):
    #find the radius r such that the image best corresponds to C(center image, r)
    #for that, argmax on the number of pixels in C(center, r)

    #log_x is of size (B, C, H, W)
    #so the r_counts have to be of size (B,R)
    B = log_x.size()[0]
    rmin, rmax = 8, 30
    H, W = 64, 64
    x,y=(H-1)/2,(W-1)/2
    rcounts = np.zeros((B,rmax-rmin+1))
    for b in range(B):
        for i in range(H):
            for j in range(W):
                if log_x[b,1,0,i,j] == 0:
                    r = int(np.sqrt((i-x)**2 + (j-y)**2))
                    if r>=rmin and r<=rmax:
                        rcounts[b,r-rmin]+=1
    R = np.argmax(rcounts,axis=1)+rmin

    log_score = torch.zeros_like(log_x)
    for b in range(B):
        for i in range(H):
            for j in range(W):
                r = int(np.sqrt((i-x)**2 + (j-y)**2))
                if r == R[b]:
                    #this means we should put this pixel to one
                    log_score[b,1,0,i,j] = np.log(2) if log_x[b,1,0,i,j] < 0 else 0
                    log_score[b,0,0,i,j] = -np.log(2) if log_x[b,0,0,i,j] < 0 else 0
                else:
                    log_score[b,1,0,i,j] = -np.log(2) if log_x[b,1,0,i,j] < 0 else 0
                    log_score[b,0,0,i,j] = np.log(2) if log_x[b,0,0,i,j] < 0 else 0
    return log_score

    *dims, H, W = log_x.shape
    xcoords = torch.arange(H)
    ycoords = torch.arange(W)
    xgrid, ygrid = torch.meshgrid(xcoords, ycoords, indexing='ij')

    # Dynamically unsqueeze the coordinate grids
    for _ in range(len(dims)):
        xgrid = xgrid.unsqueeze(0)
        ygrid = ygrid.unsqueeze(0)

    # Expand to match the shape of the input tensor
    expanded_shape = list(log_x.shape)
    xs = xgrid.expand(expanded_shape)
    ys = ygrid.expand(expanded_shape)

    rs = torch.round(torch.sqrt(xs**2 + ys**2))

    x = log_x.exp()
    rs = x * rs
    rs_white = rs[:,1,:,:,:]
    unique, counts = torch.unique(rs_white,return_counts=True)
