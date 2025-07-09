import pdb
import torch
import config as c



means = torch.tensor([0.,0.,0.,0.], device = c.device, dtype= torch.float32)
stds = torch.tensor([1.,1.,1.,1.], device = c.device, dtype= torch.float32)
#stds = torch.tensor([0.25, 0.5, 0.5, 0.5], device = c.device, dtype= torch.float32)


def MMD_multiscale(x, y, kind = None):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device))
    
    if kind:
       for C, a in [(0.2, 2), (1.5, 2), (3.0, 2)]:
            XX += C**a * ((C + dxx) / a)**-a
            YY += C**a * ((C + dyy) / a)**-a
            XY += C**a * ((C + dxy) / a)**-a

    else:
        for C, a in [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]:
            XX += C**a * ((C + dxx) / a)**-a
            YY += C**a * ((C + dyy) / a)**-a
            XY += C**a * ((C + dxy) / a)**-a

    
    return torch.mean(XX + YY - 2.*XY)
    

def NLL(z, det_log_j):
    
    return torch.mean(0.5/(c.y_noise_scale) * torch.sum((z**2), dim=-1) - det_log_j)

  


def uniform_prior_x_loss(x, a=-2.5, b=2.5, log_px = torch.log(torch.tensor(5)), lambda_ = 5.0, device = c.device):
    
    loss_vals = len(x[x>b]) + len(x[x<a])
    #pdb.set_trace()
    return torch.tensor(loss_vals * lambda_, dtype = torch.float32, requires_grad = True).to(c.device)
    

def normal_prior_x_loss(x_gt, x, mu = means, sigma = stds):
    
    #nll = torch.sum(torch.log(sigma * torch.sqrt(torch.tensor(2 * torch.pi))) + (x_gt - x)**2 / (2 * sigma**2))
    nll = torch.mean(torch.sum(torch.log(sigma) + (x_gt - x)**2 / (2 * sigma**2), dim=-1))
    return torch.mean(nll)

    

def fit_l2(inp, target):
    return torch.sqrt(torch.mean((inp - target) ** 2))
    
def fit_l1(inp, target):
    return torch.mean(torch.abs(inp - target))

def fit_huber(input, target):
    return torch.mean(torch.where(torch.abs(input - target) < 1, 0.5 * (input - target)**2, torch.abs(input - target)))


def uniform_prior_loss(x, low=-2.5, high=2.5):
    return torch.mean((torch.max(torch.tensor(0.0, device=c.device), x - high) + 
                      torch.max(torch.tensor(0.0, device=c.device), low - x)))
    
    
def f_star_kl(t, clip_min=-10.0, clip_max=10.0):
    t_clamped = torch.clamp(t, min=clip_min, max=clip_max)
    return torch.exp(t_clamped - 1)

