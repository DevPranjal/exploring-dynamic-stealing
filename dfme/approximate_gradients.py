import numpy as np
import torch
import torch.nn.functional as F

# from cifar10_models import *


def estimate_gradient_objective(args, victim_model, clone_model, x, epsilon = 1e-7, m = 5, verb=False, num_classes=10, device = "cpu", pre_x=False):
    # Sampling from unit sphere is the method 3 from this website:
    #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    #x = torch.Tensor(np.arange(2*1*7*7).reshape(-1, 1, 7, 7))
    
    if pre_x and args.generator_activation is None:
        raise ValueError(args.generator_activation)

    clone_model.eval()
    victim_model.eval()
    with torch.no_grad():
        # Sample unit noise vector
        N = x.size(0)
        C = x.size(1)
        S = x.size(2)
        dim = S**2 * C

        u = np.random.randn(N * m * dim).reshape(-1, m, dim) # generate random points from normal distribution

        d = np.sqrt(np.sum(u ** 2, axis = 2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim = 1) # Shape N, m + 1, S^2

            

        u = u.view(-1, m + 1, C, S, S)

        evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
        if pre_x: 
            evaluation_points = args.generator_activation(evaluation_points) # Apply args.G_activation function

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        pred_clone = []
        max_number_points = 32*156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU
        
        for i in (range(N * m // max_number_points + 1)): 
            pts = evaluation_points[i * max_number_points: (i+1) * max_number_points]
            pts = pts.to(device)

            pred_victim_pts = victim_model(pts).detach()
            pred_clone_pts = clone_model(pts)

            pred_victim.append(pred_victim_pts)
            pred_clone.append(pred_clone_pts)


        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_clone = torch.cat(pred_clone, dim=0).to(device)

        u = u.to(device)

        loss_fn = F.l1_loss
        pred_victim = F.log_softmax(pred_victim, dim=1).detach()
        pred_victim -= pred_victim.min(dim=1).values.view(-1, 1).detach()

        loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').mean(dim = 1).view(-1, m + 1)
            
        
        # print(f'LOSS SHAPE: {loss_values.size()}')

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        gradient_estimates *= dim            

        gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, S, S) / (num_classes * N) 

        clone_model.train()
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


def compute_gradient(args, victim_model, clone_model, x, pre_x=False, device="cpu"):
    if pre_x and args.generator_activation is None:
        raise ValueError(args.generator_activation)

    clone_model.eval()
    N = x.size(0)
    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device)


    if pre_x:
        x_ = args.generator_activation(x_)


    pred_victim = victim_model(x_)
    pred_clone = clone_model(x_)

    loss_fn = F.l1_loss
    pred_victim_no_logits = F.log_softmax(pred_victim, dim=1)
    pred_victim = pred_victim_no_logits - pred_victim_no_logits.mean(dim=1).view(-1, 1)

    loss_values = -loss_fn(pred_clone, pred_victim, reduction='mean')
    # print("True mean loss", loss_values)
    loss_values.backward()

    clone_model.train()
    
    return x_copy.grad, loss_values
