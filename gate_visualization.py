import torch
from dfme.train import train, test
from dfme.utils import get_dynamic_connections, get_dataloader
from models.resnet import *
from models.gated_resnet import DynamicResNet18_8x
from models.gan import GeneratorA

class ARGS:
    epoch_iter = 50
    g_iter = 1
    s_iter = 5

    num_classes = 10
    dataset = 'cifar10'
    data_root = './data/'
    image_size = 32
    image_channels = 3

    batch_size = 256
    student_lr = 0.1
    generator_lr = 1e-4
    weight_decay = 5e-4
    steps = [0.1, 0.3, 0.5]
    momentum = 0.9

    random_noise_size = 256
    grad_epsilon = 1e-3
    grad_m = 1

    current_budget = 0
    cost_per_iteration = 0 # will be calculated within the program
    query_budget = 10e6

    seed = 4

    generator_activation = torch.tanh
    
    teacher = resnet34()
    teacher_name = 'resnet34'
    student = DynamicResNet18_8x(num_classes)
    student_name = 'dynamic_resnet_18'
    generator = GeneratorA(nz=random_noise_size, nc=image_channels, img_size=image_size, activation=generator_activation)

    teacher_load_path = './checkpoints/teacher/resnet34.pt'
    student_load_path = './checkpoints/student/dynamic_resnet_18_resnet34.pt'   # fill after saving atleast once
    generator_load_path = './checkpoints/generator/dynamic_resnet_18_resnet34.pt' # fill after saving atleast once

    student_save_folder = './checkpoints/student/'
    generator_save_folder = './checkpoints/generator'


args = ARGS()
acc = {}
f = open('results.txt', "a")


args.student_load_path = ''
args.generator_load_path = ''

print('\n\n\n')

# VISUALIZING DYNAMIC GATES
torch.cuda.empty_cache()
device=torch.device('cuda')
train_loader, test_loader = get_dataloader(args)
student = args.student.to(device)
student.load_state_dict(torch.load('./checkpoints/student/dynamic_resnet_18_resnet18.pt'))
test(student, device, test_loader=test_loader)
for i, (X, y) in enumerate(train_loader):
    print(get_dynamic_connections(student, X[0].to(device)))
    if (i == 5):
        break

print('\n\n\n')

torch.cuda.empty_cache()
device=torch.device('cuda')
train_loader, test_loader = get_dataloader(args)
student = args.student.to(device)
student.load_state_dict(torch.load('./checkpoints/student/dynamic_resnet_18_resnet34.pt'))
test(student, device, test_loader=test_loader)
for i, (X, y) in enumerate(train_loader):
    print(get_dynamic_connections(student, X[0].to(device)))
    if (i == 5):
        break

print('\n\n\n')

torch.cuda.empty_cache()
device=torch.device('cuda')
train_loader, test_loader = get_dataloader(args)
student = args.student.to(device)
student.load_state_dict(torch.load('./checkpoints/student/dynamic_resnet_18_resnet50.pt'))
test(student, device, test_loader=test_loader)
for i, (X, y) in enumerate(train_loader):
    print(get_dynamic_connections(student, X[0].to(device)))
    if (i == 5):
        break

print('\n\n\n')


torch.cuda.empty_cache()
device=torch.device('cuda')
train_loader, test_loader = get_dataloader(args)
student = args.student.to(device)
student.load_state_dict(torch.load('./checkpoints/student/dynamic_resnet_18_googlenet.pt'))
test(student, device, test_loader=test_loader)
for i, (X, y) in enumerate(train_loader):
    print(get_dynamic_connections(student, X[0].to(device)))
    if (i == 5):
        break

print('\n\n\n')