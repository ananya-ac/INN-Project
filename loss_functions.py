import pdb
import torch
import config as c
import torch.nn as nn
from geomloss import SamplesLoss



means = torch.tensor([0.,0.,0.,0.], device = c.device, dtype= torch.float32)
stds = torch.tensor( [0.25, 0.5, 0.5, 0.5], device = c.device, dtype= torch.float32)



def sinkhorn_divergence(x, y, blur = 0.01):
  loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=True)
  return loss(x, y)

def sinkhorn_approx(x, y, blur=0.01):
  loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur,debias=False)
  return loss(x, y)


def MMD_multiscale_geom(x,y, sigma = 0.05):
  loss = SamplesLoss(loss="energy", p=2, blur=sigma)
  return loss(x, y)

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
    

# def NLL(z, det_log_j):
    
#     return torch.mean(0.5/(c.y_noise_scale) * torch.sum((z**2), dim=-1) - det_log_j)

def NLL(z, det_log_j):
    
    return torch.mean(torch.sum((z**2), dim=-1) - det_log_j)
  


def uniform_prior_x_loss(x, a=-2.5, b=2.5, log_px = torch.log(torch.tensor(5)), lambda_ = 5.0, device = c.device):
    
    loss_vals = len(x[x>b]) + len(x[x<a])
   
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


# def uniform_prior_loss(x, low=-2.5, high=2.5):
#     return torch.mean((torch.max(torch.tensor(0.0, device=c.device), x - high) + 
#                       torch.max(torch.tensor(0.0, device=c.device), low - x)))


def uniform_prior_loss(x, low=-2.5, high=2.5, lambda_bd=10.0):
    v = torch.clamp_min(x - high, 0) + torch.clamp_min(low - x, 0)
    per_sample = v.sum(dim=-1)              # sum over dims
    return lambda_bd * per_sample.mean()    # mean over batch   

def f_star_kl(t, clip_min=-10.0, clip_max=10.0):
    t_clamped = torch.clamp(t, min=clip_min, max=clip_max)
    return torch.exp(t_clamped - 1)


class Activation_g(nn.Module):
  def __init__(self,divergence="GAN"):
    super(Activation_g,self).__init__()
    self.divergence =divergence
  def forward(self,v):
    divergence = self.divergence
    if divergence == "KLD":
      return v
    elif divergence == "RKL":
      return -torch.exp(-v)
    elif divergence == "CHI":
      return v
    elif divergence == "SQH":
      return 1-torch.exp(-v)
    elif divergence == "JSD":
      return torch.log(torch.tensor(2.))-torch.log(1.0+torch.exp(-v))
    elif divergence == "GAN":
      return -torch.log(1.0+torch.exp(-v)) # log sigmoid



class Conjugate_f(nn.Module):
  def __init__(self,divergence="GAN"):
    super(Conjugate_f,self).__init__()
    self.divergence = divergence
  def forward(self,t):
    divergence= self.divergence
    if divergence == "KLD":
      return torch.exp(t-1)
    elif divergence == "RKL":
      return -1 -torch.log(-t)
    elif divergence == "CHI":
      return 0.25*t**2+t
    elif divergence == "SQH":
      return t/(torch.tensor(1.)-t)
    elif divergence == "JSD":
      return -torch.log(2.0-torch.exp(t))
    elif divergence == "GAN":
      return  -torch.log(1.0-torch.exp(t))


class VLOSS(nn.Module):
    def __init__(self, divergence="KLD"):
        super(VLOSS, self).__init__()
        self.activation = Activation_g(divergence)

    def forward(self, v):
        return torch.mean(self.activation(v))

class QLOSS(nn.Module):
    def __init__(self, divergence="KLD"):
        super(QLOSS, self).__init__()
        self.conjugate = Conjugate_f(divergence)
        self.activation = Activation_g(divergence)

    def forward(self, v):
        t = self.activation(v)
        return torch.mean(-self.conjugate(t))

def f_star_rkl(t):
    t_clamped = torch.clamp(t, min=-10.0, max=10.0)
    return -torch.exp(1 + torch.log(-t_clamped))




