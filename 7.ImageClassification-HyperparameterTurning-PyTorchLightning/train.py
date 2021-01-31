# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, datasets
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from azureml.core import Run

print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)
print("PyTorch Lightning Vision version:", pl.__version__)

# ------------
# args
# ------------
torch.manual_seed(0)
pl.seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--epoch', type=int, dest='epoch', default=10, help='epoch size for training')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, dest='momentum', default=0.9, help='momentum')
parser.add_argument('--model-name', type=str, dest='model_name', default='resnet', help='Fine Turning model name')
parser.add_argument('--optimizer', type=str, dest='optimizer', default='SGD', help='Optimzers to use for training.')
parser.add_argument('--criterion', type=str, dest='criterion', default='cross_entropy', help='Loss Function to use for training.')
parser.add_argument('--gpus', type=int, dest='gpus', default=1, help='The count of GPU')
parser.add_argument('--feature_extract', type=bool, dest='feature_extract', default=True, help='Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params')

args = parser.parse_args()

args.num_workers=8

data_folder = args.data_folder
print('training dataset is stored here:', data_folder)
print('GPUs: ', args.gpus)

input_size = 224
if args.model_name == "inception":
    input_size = 299
# ---------------------------
# Azure Machnie Learning
# 1) get Azure ML run context and log hyperparameters
run = Run.get_context()
run.log('model_name', args.model_name)
run.log('optimizer', args.optimizer)
run.log('criterion', args.criterion)

run.log('lr', np.float(args.learning_rate))
run.log('momentum', np.float(args.momentum))
# ---------------------------


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        if feature_extract == True:
            set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        if feature_extract == True:
            set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        if feature_extract == True:
            set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        if feature_extract == True:
            set_parameter_requires_grad(model_ft)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        if feature_extract == True:
            set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        if feature_extract == True:
            set_parameter_requires_grad(model_ft)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# ------------
# data
# ------------
transform = transforms.Compose([
                # Augmentation
#                transforms.RandomHorizontalFlip(),
#                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)),
                transforms.RandomRotation(degrees=10),
                # Resize
                transforms.Resize(int(input_size * 1.3)),
                transforms.CenterCrop(input_size),
                # Tensor
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.ImageFolder(args.data_folder, transform)
args.num_classes = len(dataset.classes)

n_train = int(len(dataset) * 0.7)
n_val = int(len(dataset) * 0.15)
n_test = len(dataset) - n_train - n_val

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size)

# ------------
# model
# ------------
class FineTurningModel(pl.LightningModule):

    def __init__(self, hparams, model):
        super().__init__()
        
        self.hparams = hparams
        self.model = model

        # --- Metrics
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
       
    def forward(self, x):
        h = self.model(x)
        return h

    def training_step(self, batch, batch_idx):

        inputs, labels = batch
        outputs = self(inputs)

        if self.hparams.model_name == 'inception':
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            outputs, aux_outputs = self.model(inputs)
            loss1 = self.configure_criterion(outputs, labels)
            loss2 = self.configure_criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
        else:
            outputs = self.model(inputs)
            loss = self.configure_criterion(outputs, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('tran_acc', self.train_acc(outputs, labels), on_step=True, on_epoch=True)

        # ---------------------------
        # Azure Machnie Learning
        # 2) send log a value repeated which creates a list
        run.log('Loss', np.float(loss))
        run.log('Accuracy', np.float(self.train_acc(outputs, labels)))
        # ---------------------------
        return loss
    
    def validation_step(self, batch, batch_idx):

        inputs, labels = batch
        outputs = self(inputs)
        loss = self.configure_criterion(outputs, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc(outputs, labels), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):

        inputs, labels = batch
        outputs = self(inputs)
        loss = self.configure_criterion(outputs, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc(outputs, labels), on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)

        if self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)
        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate)
        elif self.hparams.optimizer == "Adadelta":
            optimizer = torch.optim.Adadelta(self.parameters(), self.hparams.learning_rate)
        elif self.hparams.optimizer == "Adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), self.hparams.learning_rate)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), self.hparams.learning_rate)
        elif self.hparams.optimizer == "SparseAdam":
            optimizer = torch.optim.SparseAdam(self.parameters(), self.hparams.learning_rate)
        elif self.hparams.optimizer == "Adamax":
            optimizer = torch.optim.Adamax(self.parameters(), self.hparams.learning_rate)
        elif self.hparams.optimizer == "ASGD":
            optimizer = torch.optim.ASGD(self.parameters(), self.hparams.learning_rate)
        elif self.hparams.optimizer == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), self.hparams.learning_rate)
        elif self.hparams.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), self.hparams.learning_rate)
        elif self.hparams.optimizer == "Rprop":
            optimizer = torch.optim.Rprop(self.parameters(), self.hparams.learning_rate)

        return optimizer
    
    def configure_criterion(self, y, t):

        criterion = F.cross_entropy(y, t)

        if self.hparams.criterion == "cross_entropy":
            criterion = F.cross_entropy(y, t)
        elif self.hparams.criterion == "binary_cross_entropy":
            criterion = F.binary_cross_entropy(y, t)
        elif self.hparams.criterion == "binary_cross_entropy_with_logits":
            criterion = F.binary_cross_entropy_with_logits(y, t)
        elif self.hparams.criterion == "poisson_nll_loss":
            criterion = F.poisson_nll_loss(y, t)
        elif self.hparams.criterion == "hinge_embedding_loss":
            criterion = F.hinge_embedding_loss(y, t)
        elif self.hparams.criterion == "kl_div":
            criterion = F.kl_div(y, t)
        elif self.hparams.criterion == "l1_loss":
            criterion = F.l1_loss(y, t)
        elif self.hparams.criterion == "mse_loss":
            criterion = F.mse_loss(y, t)
        elif self.hparams.criterion == "margin_ranking_loss":
            criterion = F.margin_ranking_loss(y, t)
        elif self.hparams.criterion == "multilabel_margin_loss":
            criterion = F.multilabel_margin_loss(y, t)
        elif self.hparams.criterion == "multilabel_soft_margin_loss":
            criterion = F.multilabel_soft_margin_loss(y, t)
        elif self.hparams.criterion == "multi_margin_loss":
            criterion = F.multi_margin_loss(y, t)
        elif self.hparams.criterion == "nll_loss":
            criterion = F.nll_loss(y, t)
        elif self.hparams.criterion == "smooth_l1_loss":
            criterion = F.smooth_l1_loss(y, t)
        elif self.hparams.criterion == "soft_margin_loss":
            criterion = F.soft_margin_loss(y, t)

        return criterion


# Initialize the model for this run
model_ft, input_size = initialize_model(args.model_name, args.num_classes, feature_extract=args.feature_extract , use_pretrained=True)

model = FineTurningModel(args, model_ft)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print('use Multiple GPU.')

# ------------
# training
# ------------
trainer = pl.Trainer(max_epochs=args.epoch, gpus=args.gpus)
trainer.fit(model, train_loader, val_loader)

# ------------
# Test (Not Validation)
# ------------
test_result = trainer.test(test_dataloaders=test_loader)
test_result

# ------------
# save model
# ------------
# TODO: Done -- Comment out for job performance
outputdir = './outputs/model'
os.makedirs(outputdir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(outputdir, 'model.dict'))
torch.save(model, os.path.join(outputdir, 'model.pt'))
