# from DFME: https://github.com/cake-lab/datafree-model-extraction

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from .utils import get_dataloader
from .approximate_gradients import *


def train_step(args, teacher, student, generator, device, student_optimizer, generator_optimizer):
    teacher.eval()
    student.train()

    pbar = tqdm(range(args.epoch_iter))
    for i in pbar:
        ########################## GENERATOR STEP ##########################
        for _ in range(args.g_iter):
            z = torch.randn((args.batch_size, args.random_noise_size)).to(device)
            generator_optimizer.zero_grad()
            generator.train()
            fake = generator(z, pre_x=True)
            approx_grad, generator_loss = estimate_gradient_objective(
                args, teacher, student, fake, 
                epsilon=args.grad_epsilon, m=args.grad_m,
                num_classes=args.num_classes, device=device, pre_x=True)

            fake.backward(approx_grad)
            generator_optimizer.step()

        ########################## CLONE STEP ##########################
        for _ in range(args.s_iter):
            z = torch.randn((args.batch_size, args.random_noise_size)).to(device)
            fake = generator(z).detach()
            student_optimizer.zero_grad()

            with torch.no_grad(): 
                t_logit = teacher(fake)
            t_logit = F.log_softmax(t_logit, dim=1).detach()
            t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            s_logit = student(fake)

            student_loss = F.l1_loss(s_logit, t_logit.detach())
            student_loss.backward()
            student_optimizer.step()

        ########################## UPDATES AND LOGGING ##########################
        pbar.set_postfix(
            {
                'S Loss': f"{student_loss.item():.3f}",
                'G Loss': f"{generator_loss.item():.3f}",
            }
        )

        args.current_budget += args.cost_per_iteration
        if args.current_budget > args.query_budget:
            return

def test(student, device, test_loader):
    student.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = student(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.4f}%)")

    return accuracy / 100.

def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloader(args)

    teacher = args.teacher
    teacher.load_state_dict(torch.load(args.teacher_load_path))
    teacher.to(device)
    print('\n########################################################')
    print(f"Loaded teacher")
    test(teacher, device, test_loader)
    print('########################################################')

    student = args.student.to(device)
    generator = args.generator.to(device)

    if args.student_load_path:
        student.load_state_dict( torch.load( args.student_load_path ) )
        print('\n########################################################')
        print(f"Loaded student")
        acc = test(student, device, test_loader)
        print('########################################################')

    if args.generator_load_path:
        generator.load_state_dict( torch.load( args.generator_load_path ) )
        print('\n########################################################')
        print(f"Loaded generator")
        print('########################################################')

    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m+1) + args.s_iter)
    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_iter) + 1

    print ("\nTotal number of epochs: ", number_epochs)

    optimizer_S = torch.optim.SGD( student.parameters(), lr=args.student_lr, weight_decay=args.weight_decay, momentum=args.momentum )
    optimizer_G = torch.optim.Adam( generator.parameters(), lr=args.generator_lr )

    steps = sorted([int(step * number_epochs) for step in args.steps])

    scheduler_S = torch.optim.lr_scheduler.MultiStepLR(optimizer_S, steps, 0.3)
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, steps, 0.3)

    best_acc = 0

    for epoch in range(1, int(number_epochs) + 1):
        print(f"\nEpoch {epoch}:")
        train_step(args, teacher, student, generator, device, optimizer_S, optimizer_G)
        
        scheduler_S.step()
        scheduler_G.step()

        acc = test(student, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), f"{args.student_save_folder}/{args.student_name}_{args.teacher_name}.pt")
            torch.save(generator.state_dict(), f"{args.generator_save_folder}/{args.student_name}_{args.teacher_name}.pt")

    print(f"\n########## BEST ACCURACY = {best_acc} ##########")

    return best_acc
