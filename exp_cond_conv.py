from dfme.train import train
from models.resnet import resnet34
from models.cond_conv import cond_mobilenetv2
from models.gan import GeneratorA
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
    student = cond_mobilenetv2(num_classes=num_classes)
    student_name = 'cond_conv_mobilenetv2'
    generator = GeneratorA(nz=random_noise_size, nc=image_channels, img_size=image_size, activation=generator_activation)

    teacher_load_path = './checkpoints/teacher/resnet34.pt'
    student_load_path = './checkpoints/student/cond_conv_mobilenetv2_resnet34'
    generator_load_path = './checkpoints/generator/cond_conv_mobilenetv2_resnet34'

    student_save_folder = './checkpoints/student/'
    generator_save_folder = './checkpoints/generator'

args = ARGS()
train(args)