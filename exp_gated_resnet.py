from dfme.train import train, test
from models.resnet import *
from models.googlenet import *
from models.inception import *
from models.mobilenetv2 import *
from models.resnet_8x import ResNet18_8x
from models.gated_resnet import DynamicResNet18_8x
from models.gan import GeneratorA
from dfme.utils import get_dynamic_connections, get_dataloader
import torch

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


# # DYNAMIC WITH DFME: TEACHER RESNET34
# acc[args.teacher_name] = train(args)
# f.write("\n"); f.write(f'{args.teacher_name}: {acc[args.teacher_name]}')


# # DYNAMIC WITH DFME: TEACHER RESNET18
# args.teacher = resnet18()
# args.teacher_name = 'resnet18'
# args.teacher_load_path = './checkpoints/teacher/resnet18.pt'
# acc[args.teacher_name] = train(args)
# f.write("\n"); f.write(f'{args.teacher_name}: {acc[args.teacher_name]}')


# # DYNAMIC WITH DFME: TEACHER RESNET50
# args.teacher = resnet50()
# args.teacher_name = 'resnet50'
# args.teacher_load_path = './checkpoints/teacher/resnet50.pt'
# acc[args.teacher_name] = train(args)
# f.write("\n"); f.write(f'{args.teacher_name}: {acc[args.teacher_name]}')


# DYNAMIC WITH DFME: TEACHER GOOGLENET
args.teacher = googlenet()
args.teacher_name = 'googlenet'
args.teacher_load_path = './checkpoints/teacher/googlenet.pt'
acc[args.teacher_name] = train(args)
f.write("\n"); f.write(f'{args.teacher_name}: {acc[args.teacher_name]}')


# DYNAMIC WITH DFME: TEACHER INCEPTION V3
args.teacher = inception_v3()
args.teacher_name = 'inception_v3'
args.teacher_load_path = './checkpoints/teacher/inception_v3.pt'
acc[args.teacher_name] = train(args)
f.write("\n"); f.write(f'{args.teacher_name}: {acc[args.teacher_name]}')


# DYNAMIC WITH DFME: TEACHER MOBILENET
args.teacher = mobilenet_v2()
args.teacher_name = 'mobilenet_v2'
args.teacher_load_path = './checkpoints/teacher/mobilenet_v2.pt'
acc[args.teacher_name] = train(args)
f.write("\n"); f.write(f'{args.teacher_name}: {acc[args.teacher_name]}')


# # VISUALIZING DYNAMIC GATES
# torch.cuda.empty_cache()
# device=torch.device('cuda')
# train_loader, test_loader = get_dataloader(args)
# student = args.student.to(device)
# student.load_state_dict(torch.load('./checkpoints/student/dynamic_resnet_18_resnet34.pt'))
# for i, (X, y) in enumerate(train_loader):
#     test(student, device, test_loader=test_loader)
#     print(get_dynamic_connections(student, X[0]))
#     break
