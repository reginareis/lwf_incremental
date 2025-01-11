"""Contains various network definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

#=========================================================================================================
class task_block(nn.Module):
    
    def __init__(self ):
        super().__init__()
        #cria bloco relacionado a determinada tarefa, que vem após as camadas compartilhadas
        self.paths = []
        self.common_path = []
        self.npaths = 0
        

    def add_path( self, out_c ): #adiciona dataset dentro de uma tarefa já definida antes
        # Get the pretrained model para pegar camada fc7 pré-treinada
        vgg16 = models.vgg16(pretrained=True)

        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 7:
                    fc7 = module
                idx += 1        
        
        path = []
        torch.manual_seed(3)
        path.append( fc7.to('cuda') )
        path.append( nn.ReLU(inplace=True).to('cuda') )
        path.append( nn.Dropout().to('cuda') )
        # path.append( nn.Linear( 4096, 64 ).to('cuda') )
        # path.append( nn.Linear( 4096, 1024 ).to('cuda') )
        # path.append( nn.ReLU(inplace=True).to('cuda') )
        # path.append( nn.Dropout().to('cuda') )
        # path.append( nn.Linear( 1024, 256 ).to('cuda') )
        # path.append( nn.ReLU(inplace=True).to('cuda') )
        # path.append( nn.Dropout().to('cuda') )
        # path.append( nn.Linear( 256, 64 ).to('cuda') )
        # path.append( nn.ReLU(inplace=True).to('cuda') )
        # path.append( nn.Dropout().to('cuda') )
        self.paths.append( path )
        self.npaths = len( self.paths )
        self.common_path = []
        self.common_path.append( nn.Linear( self.npaths * 4096, out_c ).to('cuda') )
        #self.common_path.append( nn.Sigmoid().to('cuda') )

    
    def forward(self, inputs):        
        x_concat = ''
        for path in self.paths:
            x = inputs
            for layer in path:  
                x = layer( x )
            if x_concat == '':
                x_concat = x
            else:
                x_concat = torch.cat((x_concat, x))

        x = x_concat
        x = x.view(x.size(0), -1)
        for layer in self.common_path:
            x = layer( x )

        return x    

class ModifiedVGG16Novo(nn.Module):
    """VGG16 with different classifiers."""

    ntasks = 0
    def __init__(self, make_model=True, pretrain = True):
        super(ModifiedVGG16Novo, self).__init__()
        if make_model:
            self.make_model( pretrain = pretrain )

    def make_model(self, pretrain = True ):
        """Creates the model."""
        # Get the pretrained model.
        vgg16 = models.vgg16(pretrained=pretrain)
        print( "pretrain = " + str(pretrain))
        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1
        features = list(vgg16.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # fc7,
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
        ])
        #retira dataset e classifiers se não houver pre treino
        if not pretrain:
            self.datasets, self.classifiers = [], nn.ModuleList()

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None


        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)

    def add_dataset(self, dataset, num_outputs, itask ):
        """Adds a new dataset to the classifier."""
        torch.manual_seed(3)
        if dataset not in self.datasets:
            self.datasets.append( dataset )
            if ( itask == self.ntasks ): #this is a new task
                self.classifiers.append(task_block())
                self.ntasks += 1
                print("Adicionando novo bloco para a nova tarefa. Total de tarefas:")
            self.classifiers[ itask + 1 ].add_path( num_outputs )
       
        print("numero classifiers = " + str( len(self.classifiers)))
        print("numero datasets = " + str( len(self.datasets)))
        print("numero paths =  " + str( len(self.classifiers[ itask + 1 ].paths)))

    def set_classifier(self, itask ):
        """Change the active classifier."""
        print("Classifier ativo é " + str(itask+1))
        self.classifier = self.classifiers[itask + 1]

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#=========================================================================================================
#=========================================================================================================

class ModifiedVGG16(nn.Module):
    """VGG16 with different classifiers."""

    def __init__(self, make_model=True, pretrain = True):
        super(ModifiedVGG16, self).__init__()
        if make_model:
            self.make_model( pretrain = pretrain )

    def make_model(self, pretrain = True ):
        """Creates the model."""
        # Get the pretrained model.
        vgg16 = models.vgg16(pretrained=pretrain)
        print( "pretrain = " + str(pretrain))
        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1
        features = list(vgg16.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])
        #retira dataset e classifiers se não houver pre treino
        if not pretrain:
            self.datasets, self.classifiers = [], nn.ModuleList()

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)

    def add_dataset(self, dataset, num_outputs, ilayer = -1 ):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            if ( ilayer == -1 ):
                torch.manual_seed(3)
                self.classifiers.append(nn.Linear(4096, num_outputs))
                #print( self.classifiers[-1].weight)
        print("numero classifiers = " + str( len(self.classifiers)))
        print("numero datasets = " + str( len(self.datasets)))

    def set_dataset(self, dataset, ilayer = -1 ):
        """Change the active classifier."""
        assert dataset in self.datasets
        if ( ilayer == -1 ):
            self.classifier = self.classifiers[self.datasets.index(dataset)]
        else:
            self.classifier = self.classifiers[ ilayer ]

    # def set_dataset(self, dataset, ilayer ):
    #     """Change the active classifier."""
    #     assert dataset in self.datasets
    #     self.classifier = self.classifiers[self.datasets.index(dataset)]

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16, self).train(mode)

    def check_correctness(self, vgg16):
        """Ensures that conversion of fc layers to conv is correct."""
        # Test to make sure outputs match.
        vgg16.eval()
        self.shared.eval()
        self.classifier.eval()

        rand_input = Variable(torch.rand(1, 3, 224, 224))
        fc_output = vgg16(rand_input)
        print(fc_output)

        x = self.shared(rand_input)
        x = x.view(x.size(0), -1)
        conv_output = self.classifier[-1](x)
        print(conv_output)

        print(torch.sum(torch.abs(fc_output - conv_output)))
        assert torch.sum(torch.abs(fc_output - conv_output)).data[0] < 1e-8
        print('Check passed')
        raw_input()


class ModifiedVGG16BN(ModifiedVGG16):
    """VGG16 with batch norm."""

    def __init__(self, make_model=True):
        super(ModifiedVGG16BN, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16BN, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.children():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Get classifiers.
        idx = 6
        for module in vgg16_bn.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1

        features = list(vgg16_bn.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)


class ModifiedResNet(ModifiedVGG16):
    """ResNet-50."""

    def __init__(self, make_model=True):
        super(ModifiedResNet, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedResNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        resnet = models.resnet50(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(resnet.fc)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(2048, num_outputs))


class ModifiedDenseNet(ModifiedVGG16):
    """DenseNet-121."""

    def __init__(self, make_model=True):
        super(ModifiedDenseNet, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedDenseNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        densenet = models.densenet121(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = densenet.features

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(densenet.classifier)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def forward(self, x):
        features = self.shared(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(1024, num_outputs))

class Philnet(nn.Module):
    
    def __init__(self, make_model=True):
        super( Philnet, self).__init__()
        if make_model:
            self.make_model()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        self.datasets, self.classifiers = [], nn.ModuleList()

        features = nn.Sequential( 
        nn.Conv2d(3, 32, kernel_size=3),
        nn.ReLU( inplace = True ),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU( inplace = True ),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU( inplace = True ),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 128, kernel_size=3),
        nn.ReLU( inplace = True ),
        nn.MaxPool2d(2),
        nn.Flatten() )

        self.classifier = nn.Sequential( 
            nn.Linear(128, 128),
            nn.ReLU( inplace = True ),
            nn.Dropout( p=0.5 ),
            nn.Linear(128, 2),
            nn.Sigmoid() )
        

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)

    def add_dataset(self, dataset, num_outputs, ilayer = -1 ):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            if ( ilayer == -1 ):
                #self.classifiers.append(nn.Linear(4096, num_outputs))
                self.classifiers.append(nn.Linear(18432, num_outputs))

    def set_dataset(self, dataset, ilayer = -1 ):
        """Change the active classifier."""
        assert dataset in self.datasets
        if ( ilayer == -1 ):
            self.classifier = self.classifiers[self.datasets.index(dataset)]
        else:
            self.classifier = self.classifiers[ ilayer ]

    # def set_dataset(self, dataset, ilayer ):
    #     """Change the active classifier."""
    #     assert dataset in self.datasets
    #     self.classifier = self.classifiers[self.datasets.index(dataset)]

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16, self).train(mode)

    def check_correctness(self, vgg16):
        """Ensures that conversion of fc layers to conv is correct."""
        # Test to make sure outputs match.
        vgg16.eval()
        self.shared.eval()
        self.classifier.eval()

        rand_input = Variable(torch.rand(1, 3, 224, 224))
        fc_output = vgg16(rand_input)
        print(fc_output)

        x = self.shared(rand_input)
        x = x.view(x.size(0), -1)
        conv_output = self.classifier[-1](x)
        print(conv_output)

        print(torch.sum(torch.abs(fc_output - conv_output)))
        assert torch.sum(torch.abs(fc_output - conv_output)).data[0] < 1e-8
        print('Check passed')
        raw_input()

class ModifiedVGG16_Ori(nn.Module):
    """VGG16 with different classifiers."""

    def __init__(self, make_model=True):
        super(ModifiedVGG16, self).__init__()
        if make_model:
            self.make_model()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        vgg16 = models.vgg16(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1
        features = list(vgg16.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)

    def add_dataset(self, dataset, num_outputs, ilayer = -1 ):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            if ( ilayer == -1 ):
                self.classifiers.append(nn.Linear(4096, num_outputs))

    def set_dataset(self, dataset, ilayer = -1 ):
        """Change the active classifier."""
        assert dataset in self.datasets
        if ( ilayer == -1 ):
            self.classifier = self.classifiers[self.datasets.index(dataset)]
        else:
            self.classifier = self.classifiers[ ilayer ]

    # def set_dataset(self, dataset, ilayer ):
    #     """Change the active classifier."""
    #     assert dataset in self.datasets
    #     self.classifier = self.classifiers[self.datasets.index(dataset)]

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16, self).train(mode)

    def check_correctness(self, vgg16):
        """Ensures that conversion of fc layers to conv is correct."""
        # Test to make sure outputs match.
        vgg16.eval()
        self.shared.eval()
        self.classifier.eval()

        rand_input = Variable(torch.rand(1, 3, 224, 224))
        fc_output = vgg16(rand_input)
        print(fc_output)

        x = self.shared(rand_input)
        x = x.view(x.size(0), -1)
        conv_output = self.classifier[-1](x)
        print(conv_output)

        print(torch.sum(torch.abs(fc_output - conv_output)))
        assert torch.sum(torch.abs(fc_output - conv_output)).data[0] < 1e-8
        print('Check passed')
        raw_input()