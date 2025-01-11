from PIL import Image, ImageOps
import pandas as pd
import os, sys
import random
import pickle
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torchvision
from torch.utils.data import DataLoader,random_split
from torch import nn
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.optim as optim
import torch.utils.data as data
from collections import OrderedDict
import plotly.express as px

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. Loggin with wandb will result in error.')


wandb_run = wandb.init(id='aeconv_0033', project='gan_latente', name='unet_tsne_imagemamento_cycle_eval',
                            entity='uruk') if not wandb.run else wandb.run


torch.cuda.empty_cache()


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#Importing Images

#Tranformation util functions
def get_transform(params=None, grayscale=False, method=Image.BICUBIC, convert = True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in params:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in params:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)



class BaseDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, fileguide, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #TO-DO create a task for reading by filepath
        #if root_csv_file is not None: 
        #    self.fileguide = pd.read_csv(root_csv_file)
        self.fileguide = fileguide['image_path'].tolist()
        self.fileguide_size = len(self.fileguide)
        self.transform = transform

    def __len__(self):
        return len(self.fileguide)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fileguide[idx % self.fileguide_size]
        image = Image.open(img_name).convert('RGB')
        image = ImageOps.exif_transpose(image)
        if self.transform:
            im = self.transform(image)
        #sample = {'image': im, 'path': img_name}
        return im

#### Unet-256 architeture ####

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.05):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size = 2, stride = 2, padding = 0)
        self.conv = conv_block(out_c + out_c, out_c)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x)
        return x
        
        

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        
        """ Bottleneck """
        self.b = conv_block(256, 512)
        
        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)
        
        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size = 1, padding = 0)
        
    def forward(self, inputs, is_tsne = False):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        
        """ Bottleneck """
        b = self.b(p3)
        
        """ Decoder """
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        
        """ Classifier """
        if is_tsne:
            outputs = b
        else:
            outputs = self.outputs(d3)
        return outputs
        
### Training function
def train_epoch(model, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.float().to(device)
        # unet data
        unet_data = model(image_batch)
        # Evaluate loss
        loss = loss_fn(unet_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)    

### Testing function
def val_epoch(model, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    model.eval()
    val_loss = []
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.float().to(device)
            # unet data
            unet_data = model(image_batch)
            v_loss = loss_fn(unet_data, image_batch)
            val_loss.append(v_loss.detach().cpu().numpy())
        # Evaluate global loss
        
    return np.mean(val_loss)

def encode2tsne(model, device, dataloader, isreal, isShenzhen = False):
    # Set evaluation mode for encoder and decoder
    model.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        encoded_samples = []
        for image_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.float().to(device)
            # Encode data
            encoded_data = model(image_batch, is_tsne = True)
            # Append the network output and the original image to the lists
            encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_data.flatten().cpu().numpy())}
            if isreal:
                if isShenzhen:
                    encoded_sample['label'] = 2
                else:
                    encoded_sample['label'] = 1
            else:
                encoded_sample['label'] = 0
            encoded_samples.append(encoded_sample)

    return pd.DataFrame(encoded_samples)

def plot_ae_outputs(model, data_eval, wandb_, n=10):
    plt.figure(figsize=(16,4.5))
    #t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    t_idx = np.random.randint(len(data_eval) - 1, size=n)
    imgs_ = [im for i, im in enumerate(data_eval) if i in t_idx]
    for i, img in enumerate(imgs_):
      ax = plt.subplot(2,n,i+1)
      #img = data_eval[t_idx[i]][0].unsqueeze(0).to(device)
      model.eval()
      with torch.no_grad():
         rec_img  = model(img.float().to(device))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()
    wandb_.log({"Reconstructed Images": plt})
     

#Defining raw path
#raw_path_imageamento = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/'

#Defining partition from imageamento raw
path_imageamento_anonimizado_valid = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/splits.pkl'

#Defining partition from imageamento raw
path_shenzhen = '/home/brics/public/brics_data/Shenzhen/china/raw/splits.pkl'


#Defining a given fold to work

test = 0

for sort in [0]:
    
    #Openning imageamento partition from pickle file
    partition_imageamento_anonimizado_valid = open(path_imageamento_anonimizado_valid, "rb")
    partition_iltbi = pickle.load(partition_imageamento_anonimizado_valid)
    partition_imageamento_anonimizado_valid.close()
    splits_imageamento = partition_iltbi[test]

    #Openning shenzhen partition from pickle file
    partition_shenzhen = open(path_shenzhen, "rb")
    partition_china = pickle.load(partition_shenzhen)
    partition_shenzhen.close()
    splits_shenzhen = partition_china[test]
    
    params = 'resize_and_scale_width'
    load_size = 256
    crop_size = 256

    #Importing fake data
    df_fake = pd.read_csv('/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/fake_images/user.otto.tavares.task.SantaCasa_imageamento_anonimizado_valid.cycle_v1_tb.r1.Shenzhen_to_SantaCasa.samples.csv')
    df_fake.drop("Unnamed: 0", axis=1, inplace=True)

    #Importing real data from imageamento anonimizado valid
    df_real_imageamento = pd.read_csv('/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/SantaCasa_imageamento_anonimizado_valid_table_from_raw.csv')
    df_real_imageamento.drop("Unnamed: 0", axis=1, inplace=True)
        
    #Importing real data from shenzhen TB
    df_real_shenzhen = pd.read_csv('/home/brics/public/brics_data/Shenzhen/china/raw/Shenzhen_china_table_from_raw.csv')
    df_real_shenzhen.drop("Unnamed: 0", axis=1, inplace=True)

    #splitting in real and fake data in partitions
    training_real = df_real_imageamento.iloc[splits_imageamento[sort][0]]
    validation_real = df_real_imageamento.iloc[splits_imageamento[sort][1]]
    
    #Importing fake data
    training_fake = df_fake.loc[(df_fake.type == 'train') & (df_fake.test == test) & (df_fake.sort == sort)]
    validation_fake = df_fake.loc[(df_fake.type == 'val') & (df_fake.test == test) & (df_fake.sort == sort)]
   
    #Importing shenzhen validation for tsne map
    validation_shenzhen = df_real_shenzhen.iloc[splits_shenzhen[sort][1]]
    #train_fake_imgs = []
    #train_real_imgs = [] 
    #transform_F = get_transform(params, grayscale=True)
    #for f_img in training_fake['image_path']:
    #    F_img = Image.open(f_img).convert('RGB')
        #A_img = ImageOps.exif_transpose(A_img)
    #    train_fake_imgs.append(transform_F(F_img))

    #transform_R = get_transform(params, grayscale=True)
    #for r_img in training_real['image_path']:
    #    R_img = Image.open(r_img).convert('RGB')
    #    R_img = ImageOps.exif_transpose(R_img)
    #   train_real_imgs.append(transform_R(R_img))

    #val_fake_imgs = []
    #val_real_imgs = [] 
    #for f_img in validation_fake['image_path']:
    #    F_img = Image.open(f_img).convert('RGB')
    #    #A_img = ImageOps.exif_transpose(A_img)
    #    val_fake_imgs.append(transform_F(F_img))

    #transform_R = get_transform(params, grayscale=True)
    #for r_img in validation_real['image_path']:
    #    R_img = Image.open(r_img).convert('RGB')
    #    val_real_imgs.append(transform_R(R_img))

    print("\n------Setting train params------\n")    
    
    batch_size = 8
    learning_rate = 0.0002
    num_epoch = 20
        
    print("\n------Importing Real Data------\n")
    
    transform_R = get_transform(params, grayscale=True)
    train_real_imgs = BaseDataset(training_real, transform_R)
    val_real_imgs = BaseDataset(validation_real, transform_R)
    
    print("\n------Creating Real Dataloader------\n")
    train_loader = torch.utils.data.DataLoader(train_real_imgs, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_real_imgs, batch_size=1)        
    
    print("\n------Importing Fake Data------\n")
    
    transform_F = get_transform(params, grayscale=True)
    #train_fake_imgs = BaseDataset(training_fake, transform_F)
    val_fake_imgs = BaseDataset(validation_fake, transform_F)
    #all_fake_imgs = BaseDataset(pd.concat([training_fake, validation_fake]), transform_F)
    
    print("\n------Creating Fake Dataloader------\n")

    #fake_loader = torch.utils.data.DataLoader(all_fake_imgs, batch_size=1)
    fake_loader = torch.utils.data.DataLoader(val_fake_imgs, batch_size=1)        
  
    print("\n------Importing Shenzhen Val. Data------\n")
    val_shenzhen_imgs = BaseDataset(validation_shenzhen, transform_R)
    shenzhen_val_loader = torch.utils.data.DataLoader(val_shenzhen_imgs, batch_size=1)
    
    print("\n------Training Unet------\n")

    #model = AutoEncoderConv().cuda()

    print("Training Start")

    #aec_train_images = train_real_imgs + val_fake_imgs
    #aec_val_images = val_real_imgs + val_fake_imgs
    #train_loader = torch.utils.data.DataLoader(aec_train_images, batch_size=batch_size)
    #valid_loader = torch.utils.data.DataLoader(aec_val_images, batch_size=batch_size)        

    ### Define the loss function
    #loss_fn = torch.nn.MSELoss()
    loss_fn = nn.MSELoss() 

    ### Define an optimizer (both for the encoder and the decoder!)
    #lr= 0.001
    #lr= 0.0001

    ### Set the random seed for reproducible results
    torch.manual_seed(512)

    ### Initialize the two networks
    #d = 4
    #d = 32 * 32

    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    #encoder = Encoder(encoded_space_dim=d,fc2_input_dim=256)
    #decoder = Decoder(encoded_space_dim=d,fc2_input_dim=256)
    model = build_unet()
    params_to_optimize = [
        {'params': model.parameters()}
    ]

    optim_ = torch.optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    model.to(device)
    
    #Define early stopping
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.05)
    
    diz_loss = {'train_loss':[],'val_loss':[]}
    losses_wandb = OrderedDict()
    for epoch in range(num_epoch):
        train_loss = train_epoch(model, device, train_loader, loss_fn, optim_)
        val_loss = val_epoch(model,device, valid_loader,loss_fn)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epoch,train_loss,val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        losses_wandb['epoch'] = float(epoch)
        losses_wandb['train_loss'] = float(train_loss)
        losses_wandb['val_loss'] = float(val_loss)
        wandb_run.log(losses_wandb)
        plot_ae_outputs(model, valid_loader, wandb_run, n=10)
        early_stopping(float(train_loss), float(val_loss))
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break
    
    print("\n------Starting TSNE------\n")
    #TO - DO Plot the tsne latent space controlling by label (TB (Synth) and NTB (Normal))
    #valid_eval = torch.utils.data.DataLoader(aec_val_images, batch_size=1)        
    print("\n------Encoding Real------\n")
    encoded_real_data = encode2tsne(model, device, valid_loader, isreal = True)
    print("\n------Encoding Fake------\n")
    encoded_fake_data = encode2tsne(model, device, fake_loader, isreal = False)
    print("\n------Encoding Shenzhen Data------\n")
    encoded_shenzhen_data = encode2tsne(model, device, shenzhen_val_loader, isreal = True, isShenzhen = True)
    print("\n------Dimension Reduction by PCA------\n")
    pca_model = PCA(n_components=0.99,random_state=512)
    encoded_samples = pd.concat([encoded_real_data, encoded_fake_data, encoded_shenzhen_data], ignore_index=True)
    #print(encoded_samples)
    #encoded_reduced = pca_model.fit_transform(encoded_samples.drop(['label'], axis=1))
    #print(encoded_reduced)
    print("\n------TSNE Projection------\n")
    tsne_model = TSNE(n_components=2, init='pca',random_state=512)
    tsne_results = pd.DataFrame(tsne_model.fit_transform(encoded_samples), columns = ['Comp_1','Comp_2'])
    
    

    # Create a table
    table = wandb.Table(columns = ["tsne_figure"])

    # Create path for Plotly figure
    path_to_plotly_html = "./tsne_figure.html"

    # Example Plotly figure
    fig = px.scatter(tsne_results, x="Comp_1", y="Comp_2",
                 color=encoded_samples.label.astype(str))

    # Write Plotly figure to HTML
    # Set auto_play to False prevents animated Plotly charts 
    # from playing in the table automatically
    fig.write_html(path_to_plotly_html, auto_play = True) 

    # Add Plotly figure as HTML file into Table
    table.add_data(wandb.Html(path_to_plotly_html))

    # Log Table
    wandb_run.log({"Latent Space Analysis": table})
    wandb.finish()  
    
    print("\n------TSNE Done------\n")

    