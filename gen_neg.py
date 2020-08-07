import random
import torch
import torch.nn.functional as F

def adjacent_shuffle(x):
    # (C X T x H x W)
    tmp = torch.chunk(x, 4, dim=1)
    order = [0,1,2,3]
    ind1 = random.randint(0,3)
    ind2 = (ind1 + random.randint(0,2) + 1) % 4
    order[ind1], order[ind2] = order[ind2], order[ind1]
    x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),1)
    return x_new

def spatial_permutation(x):
    c, t, h, w = x.shape
    hm = h // 2
    wm = w // 2
    slices = []
    slices.append(x[:,:,:hm,:wm]) # A
    slices.append(x[:,:,:hm,wm:]) # B
    slices.append(x[:,:,hm:,:wm]) # C
    slices.append(x[:,:,hm:,wm:]) # D
    order = [1,2,3,4]
    while order == [1,2,3,4]:
        random.shuffle(order)
    #order = [3,2,1,0]
    x_new = torch.cat((torch.cat((slices[order[0]], slices[order[1]]), 3), torch.cat((slices[order[2]], slices[order[3]]), 3)), 2)
    return x_new


def repeating(x):
    c, t, h, w = x.shape
    ind = random.randint(0,t-1)
    one_frame = x[:,ind,:,:] # c, h, w
    one_frame = torch.unsqueeze(one_frame, 1)# -> c, 1, h, w
    x_new = one_frame.repeat(1,t,1,1)
    return x_new

def blur_clip(x):
    c, t, h, w = x.shape
    kernel = torch.ones(t,1,5,5) / 25
    kernel = kernel.cuda()
    x_new = F.conv2d(x, kernel, padding=2, groups=t)
    return x_new


def preprocess(inputs, option='repeat'):
    b, c, t, h, w = inputs.shape
    # blur should be positive
    # repeating should be negtive
    new_in = []
    # origin, rotation, spatial permtation, temporal shuffling, remote clip
    for i in range(b):
        one_sample = inputs[i,:,:,:,:]
        if option == 'repeat':
            new_in.append(repeating(one_sample))
        if option == 'shuffle':
            new_in.append(adjacent_shuffle(one_sample))
    return torch.stack(new_in)