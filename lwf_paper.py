
from __future__ import division, print_function
from optparse import Values
import os

import torch
import utils as utils
import pickle
import json
import numpy as np
from lwf_functions_paper import *
from loaddata_paper import *

#user parameters
user_parameters = {}
user_parameters['load_path'] = '' #'/home/regina.alves/results/lwf_res/SENS90/2WganSoFC_3ManausWganSoFC/' #path of the previous model, from which the new task will be learned. if first task, = ''
user_parameters['save_path'] = '/home/regina.alves/results/lwf_res/PAPER/stanford/' #Path to save the trained models
user_parameters['lastTaskName'] = '' #'Shenzhen' #'Manaus'#'Stanford' - if first one, = ''
user_parameters['taskName'] = 'Stanford' #'Shenzhen' #'Manaus'#'Stanford'
user_parameters['inputCycle'] = '' # 'AllToSantaCasa' "ShenzhenToSantaCasa" #"ShenzhenToSantaCasa" - if doesn't apply,  = ''
#Defines whether it will be a finetuning of the last existing layer
user_parameters['ilayer'] = -1 #used in output = pred_logits[ ilayer ] - if = -1, add new layer
user_parameters['runmode'] = 'newtask'  #'finetuning' ,'newtask'
user_parameters['finetune_layers'] = '' #'all', 'fc', 'classifier'
#Choose the data reading mode
user_parameters['loadmode'] = 'real' #'real' #'pix2pix' #'wgan' #'wgan_p2p' 
user_parameters['trainmode'] = 'real' #alltogether, allfake, real
user_parameters['fakeData'] = ''#'Manaus'
user_parameters['sens90'] = False #Defines whether the operating point should be with sensitivity > 90
user_parameters['histogramEqualization'] = False
user_parameters['nkfolds'] = 3
user_parameters['imgwidth'] = 224
user_parameters['ifoldbegin'] = 0
user_parameters['lr'] = 1e-4 #5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5 - learning rate
user_parameters['lr_decay_factor'] = 0.5 #0.5, 0.1
user_parameters['wd'] = 0.00025 #[0.00005, 0.0001, 0.0005, 0.001 ] - weight decay
user_parameters['temp'] = 2.0 #[0.5, 1.0, 1.5, 2.5, 3.0] #2.0 - temperature


#-----------------------------------------------------------------------------------------------------
save_path = user_parameters['save_path']
lr = user_parameters['lr']
wd = user_parameters['wd']
temp = user_parameters['temp']
lr_decay_factor = user_parameters['lr_decay_factor']


def Run_LwF( lr, wd, temp, lr_decay_factor, save_path, user_parameters ):
  load_path = user_parameters['load_path']
  lastTaskName = user_parameters['lastTaskName']
  taskName = user_parameters['taskName']
  inputCycle = user_parameters['inputCycle']
  ilayer = user_parameters['ilayer'] #usado em output = pred_logits[ ilayer ] - se = -1, acrescenta nova camada
  runmode = user_parameters['runmode'] 
  finetune_layers = user_parameters['finetune_layers']
  loadmode = user_parameters['loadmode']
  trainmode = user_parameters['trainmode']
  fakeData = user_parameters['fakeData']
  SENS90 = user_parameters['sens90']
  nkfolds = user_parameters['nkfolds']
  histogramEqualization = user_parameters['histogramEqualization']
  imgwidth = user_parameters['imgwidth']
  ifoldbegin = user_parameters['ifoldbegin']

  if taskName == 'Stanford':
    #tasks other than TB - pickle with data and pickle with partition--------------------
    dataPartition = "/home/regina.alves/dados_radiografias/infoStanford.pkl"
    partitionPickle = "/home/regina.alves/info/test_folds_stanford.pkl" 
  
  elif taskName == 'Shenzhen':
    #TB task with real data - pickle with data and Shenzhen partition-----------------------------
    dataPartition = "/home/regina.alves/dados_radiografias/infoShenPartitionOtto.pkl"
    partition_path = "/home/regina.alves/info/partition.pkl" #pickle com partition do Otto
  
  elif taskName == 'MC':
    dataPartition = "/home/regina.alves/dados_radiografias/fold_data_MC.pkl"

  elif taskName == "Manaus":
    csv_path = '/home/brics/public/brics_data/Manaus/c_manaus/raw/Manaus_c_manaus_table_from_raw.csv'
    partition_path = "/home/brics/public/brics_data/Manaus/c_manaus/raw/splits.pkl"

  #task with real data from Santa Casa - load from folders and read partition in csv----------------------------
  elif taskName == 'SantaCasa':
    if inputCycle == 'ShenzhenToSantaCasa':
        fake_path_tb1 = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/fake_images/user.otto.tavares.task.SantaCasa_imageamento_anonimizado_valid.cycle_v1_tb.r1.Shenzhen_to_SantaCasa.samples/job.test_'
        fake_path_tb2 = ''
    elif inputCycle == 'ManausToSantaCasa':
        fake_path_tb1 = '/home/brics/public/brics_data/Manaus/c_manaus/fake_images/user.otto.tavares.Manaus.c_manaus.cycle_v1_tb.r5.Manaus_to_SantaCasa.samples/job.test_'
        fake_path_tb2 = ''
    elif inputCycle == 'AllToSantaCasa':
        fake_path_tb1 = '/home/brics/public/brics_data/Manaus/c_manaus/fake_images/user.otto.tavares.Manaus.c_manaus.cycle_v1_tb.r5.Manaus_to_SantaCasa.samples/job.test_'
        fake_path_tb2 = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/fake_images/user.otto.tavares.task.SantaCasa_imageamento_anonimizado_valid.cycle_v1_tb.r1.Shenzhen_to_SantaCasa.samples/job.test_'
    path_img = '/home/brics/public/brics_data/SantaCasa/imageamento/raw/images/'
    csv_path = '/home/brics/public/brics_data/SantaCasa/imageamento/raw/user.joao.pinto_SantaCasa_imageamento_table_from_raw_splitted.csv'

  #task with fake data - paths to load the data during training, partition is in the folders where images are saved
  if fakeData == "Shenzhen" and loadmode == 'wgan': 
    fake_path_notb = '/home/brics/public/brics_data/Shenzhen/china/fake_images/user.joao.pinto.task.Shenzhen_china.wgan.v2_notb.r1.samples/job.test_' #0.sort_0/p2p_NTB/'
    fake_path_tb = '/home/brics/public/brics_data/Shenzhen/china/fake_images/user.joao.pinto.task.Shenzhen_china.wgan.v2_tb.r1.samples/job.test_' #0.sort_0/p2p_TB/'
  elif fakeData == "Shenzhen" and loadmode == 'pix2pix':
    fake_path_notb = '/home/brics/public/brics_data/Shenzhen/china/fake_images/user.otto.tavares.task.Shenzhen_china.pix2pix.v1_notb.r1.samples/job.test_'
    fake_path_tb = '/home/brics/public/brics_data/Shenzhen/china/fake_images/user.otto.tavares.task.Shenzhen_china.pix2pix.v1_tb.r1.samples/job.test_'
  elif fakeData == "Shenzhen" and loadmode == 'wgan_p2p':
    fake_path_notb = '/home/brics/public/brics_data/Shenzhen/china/fake_images/user.joao.pinto.task.Shenzhen_china.wgan.v2_notb.r1.samples/job.test_' #0.sort_0/p2p_NTB/'
    fake_path_tb = '/home/brics/public/brics_data/Shenzhen/china/fake_images/user.joao.pinto.task.Shenzhen_china.wgan.v2_tb.r1.samples/job.test_' #0.sort_0/p2p_TB/'
    fake_path_notb_2 = '/home/brics/public/brics_data/Shenzhen/china/fake_images/user.otto.tavares.task.Shenzhen_china.pix2pix.v1_notb.r1.samples/job.test_'
    fake_path_tb_2 = '/home/brics/public/brics_data/Shenzhen/china/fake_images/user.otto.tavares.task.Shenzhen_china.pix2pix.v1_tb.r1.samples/job.test_'
  elif fakeData == "Manaus" and loadmode == 'pix2pix':
    fake_path_notb = '/home/brics/public/brics_data/Manaus/c_manaus/fake_images/user.otto.tavares.Manaus.c_manaus.pix2pix_v1.notb.r1.samples/job.test_'
    fake_path_tb = '/home/brics/public/brics_data/Manaus/c_manaus/fake_images/user.otto.tavares.Manaus.c_manaus.pix2pix_v1.tb.r1.samples/job.test_'
  elif fakeData == "Manaus" and loadmode == 'wgan':
    fake_path_tb = '/home/brics/public/brics_data/Manaus/c_manaus/fake_images/user.otto.tavares.task.Manaus.c_manaus.wgan_v2_tb/job.test_'
    fake_path_notb = '/home/brics/public/brics_data/Manaus/c_manaus/fake_images/user.otto.tavares.task.Manaus.c_manaus.wgan_v2_notb/job.test_'

  print("Starting to run...")

  #Fix GPU and GPU-Large
  device = -1
  gpu_ids = 0
  try:
    device = int(gpu_ids)
  except Exception as ex:
    print(ex)
    print('getting cuda visible devices')
    device = int(os.environ['CUDA_VISIBLE_DEVICES'])
    
  print('Using {} device'.format(device))
  if device>=0:
      torch.cuda.set_device(device)
      torch.cuda.empty_cache()
      gpu_ids = [device]
  else:
      gpu_ids = []

      
  """# Load Data SESSION

  OPEN INFORMATION ABOUT ALL TASKS
  """
  #loading data, names, labels for tasks that are not TB
  if taskName == 'Stanford':
    datataskaux, labelsaux, namesImg = Read_pickle_tasksPaths( dataPartition )
    datatask = datataskaux.numpy() #.transpose(0,3,2,1) )
    labels = labelsaux.numpy() 
    Id_folds = np.array( Read_Id_Test( partitionPickle, namesImg ) )
    print("Loaded " + str(len(labels)) + " real images from Stanford")

  #------ loading data, names, labels for tasks that are real TB
  elif taskName == 'Shenzhen':
    datataskaux, labelsaux, namesImgaux = Read_pickle_tasksPaths( dataPartition )
    datatask = datataskaux.numpy()
    labels = labelsaux.numpy() 
    namesImg = namesImgaux 
    a_file = open(partition_path, "rb")
    partition = pickle.load(a_file)
    a_file.close()
    print("Loaded " + str(len(labels)) + " real images from Shenzhen")

  elif taskName == 'MC':
    datatask, labels, namesImg = Read_pickle_others( dataPartition )

  elif taskName == 'Manaus':
    datatask, labels, namesImg = Load_Names_Manaus( csv_path )
    a_file = open(partition_path, "rb")
    partition = pickle.load(a_file)
    a_file.close()

  """# Run LwF"""

  if ( runmode == 'newtask' ):
    #Establishes the operating model path of the previous task, if it is not the first task learned
    if lastTaskName != '':
      loadname = findfilebest( lastTaskName, load_path )
      print("Starting to learn new task from model " + loadname )
    else:
      loadname = 'null'
      print("Starting to learn first task" )
    
    # Loop for each test-----------------------------------------------------------------------------------------
    datalwf, labellwf, namelwf = {}, {}, {}
    for ifold in range( ifoldbegin, nkfolds ): #test fold. Test fold won't be used for anything.    
      auxResults = Results( 1 )

      for jfold in range( nkfolds ): #validation fold
        datalwf, labellwf, namelwf = {}, {}, {}
        if( ( jfold == ifold and ( taskName == 'Stanford' or taskName == 'Stanford_X' )) or ( jfold == nkfolds - 1 and taskName != 'Stanford' )   ): #can't use the same fold for test and validation
          #Put fake values in FN, FP so that sensitivity and SP are equal to 0
          auxResults.VP[ 0 ].append( [ ] )
          auxResults.VN[ 0 ].append( [ ] )  
          auxResults.FP[ 0 ].append( [ 0 ] )
          auxResults.FN[ 0 ].append( [ 0 ] )       
          continue
        #Save validation in temporary file. In the end, only one model per test fold will exist.
        save_val = save_path + 'aux_lr' + str(lr)

        print("Begin calculus for test = " + str(ifold) +" , validation = " + str(jfold) )
                                      
        if loadmode == 'real' and ( taskName == 'Stanford' or taskName == 'Stanford_X' ):
          datalwf, labellwf, namelwf = Define_Data_Fold_Stanford( ifold, jfold, Id_folds, datatask, labels, namesImg )
                                                      
        elif loadmode == 'real' and taskName == 'MC':
          datalwf, labellwf, namelwf = Define_Data_Fold_Others( ifold, jfold, datatask, labels, namesImg )
        
        elif taskName == 'Manaus':
          datalwf, labellwf, namelwf = Define_Data_Fold_Partition_Manaus( ifold, jfold, partition, datatask, labels, namesImg )                                      
                                  
        if fakeData == 'Shenzhen' and trainmode == 'alltogether':
          print("Loading fake data from Shenzhen - alltogether")
          iFoldOMTask, jFoldOMTask = findOperModelIJ( fakeData, load_path )
          print("i=" + str(iFoldOMTask))
          print("j=" + str(jFoldOMTask))
          Define_Data_Fold_Fake_Alltogether_Only2( iFoldOMTask, jFoldOMTask, datalwf, labellwf, namelwf,
                                      histogramEqualization, imgwidth, fake_path_notb, fake_path_tb, loadmode = loadmode )
        
        elif loadmode == 'real' and taskName == 'Shenzhen':
          print("Loading real data from Shenzhen")
          datalwf, labellwf, namelwf = Define_Data_Fold_Partition( ifold, jfold, partition, datatask, labels, namesImg )                                      
                                  
        elif(taskName == 'Shenzhen' and ( loadmode == 'wgan' or loadmode == 'pix2pix') and trainmode != 'alltogether' ):
          print("Loading fake data from " + loadmode + " ,train only with fake data")
          datalwf, labellwf, namelwf = Define_Data_Fold_Fake( ifold, jfold, partition, datatask, labels, namesImg,
                                                              histogramEqualization, imgwidth, fake_path_notb, fake_path_tb, loadmode = loadmode )
          
        elif(taskName == 'Shenzhen' and loadmode == 'wgan_p2p' and trainmode != 'alltogether' ):
          print("Loading fake data from " + loadmode + " ,train only with fake data")
          datalwf, labellwf, namelwf = Define_Data_Fold_Wgan_P2P( ifold, jfold, partition, datatask, labels, namesImg, 
                                                                 fake_path_notb, fake_path_tb, fake_path_notb_2, fake_path_tb_2 )
        
        elif( taskName == 'Shenzhen' and ( loadmode == 'wgan' or loadmode == 'pix2pix') and trainmode == 'alltogether' ):
          print("Loading fake data from " + loadmode + " ,train alltogether")
          datalwf, labellwf, namelwf = Define_Data_Fold_Fake_Alltogether( ifold, jfold, partition, datatask, labels, namesImg,
                                                              histogramEqualization, imgwidth, fake_path_notb, fake_path_tb, loadmode = loadmode )
              
        elif( taskName == 'SantaCasa'  ):
          datalwf, labellwf, namelwf = Define_Data_Fold_SantaCasa( path_img, csv_path, fake_path_tb1, fake_path_tb2, ifold, jfold )
        
            
        lwf_function('finetune', taskName, save_val, loadname, datalwf, labellwf, namelwf, SENS90 = SENS90, 
                    ifold = ifold, jfold = jfold,
                      lr = lr, wd = wd, temp = temp, lr_decay_factor = lr_decay_factor  )
         
       
  if ( runmode == 'finetuning' ):  
      ifoldbegin = 0   
      print("Running in runmode finetuning")
      iFoldOMTask, jFoldOMTask = findOperModelIJ( taskName, load_path )
      print("i=" + str(iFoldOMTask))
      print("j=" + str(jFoldOMTask))
      # Loop for each test-----------------------------------------------------------------------------------------
      for ifold in range( ifoldbegin, nkfolds ): #test fold. Test fold won't be used for anything.
        for jfold in range( nkfolds-1 ): #test fold. Test fold won't be used for anything.
          filename, jfold  = findfile( taskName, load_path,  ifold )          
          loadname = load_path + filename          
          #Save validation in temporary file. In the end, only one model per test fold will exist.
          save_val = save_path + 'aux_test' + str( ifold ) + '_val' + str( jfold )
          
          print("Begin calculus for test = " + str(ifold) +" , validation = " + str(jfold) )
        
          if loadmode == 'real' and ( taskName == 'Stanford' or taskName == 'Stanford_X' ):
            datalwf, labellwf, namelwf = Define_Data_Fold_Stanford( ifold, jfold, Id_folds, datatask, labels, namesImg )
                                                        
          elif loadmode == 'real' and taskName == 'MC':
            datalwf, labellwf, namelwf = Define_Data_Fold_Others( ifold, jfold, datatask, labels, namesImg )
          
          elif loadmode == 'real' and taskName == 'Manaus':
            datalwf, labellwf, namelwf = Define_Data_Fold_Partition_Manaus( ifold, jfold, partition, datatask, labels, namesImg )                                      
                                    
          elif loadmode == 'real' and taskName == 'Shenzhen':
            datalwf, labellwf, namelwf = Define_Data_Fold_Partition( ifold, jfold, partition, datatask, labels, namesImg )                                      
                                    
          elif( ( loadmode == 'wgan' or loadmode == 'pix2pix') and trainmode != 'alltogether' ):
            print("Carregando dado fake de origem " + loadmode + " ,treino só fake")
            datalwf, labellwf, namelwf = Define_Data_Fold_Fake( ifold, jfold, partition, datatask, labels, namesImg,
                                                                histogramEqualization, imgwidth, fake_path_notb, fake_path_tb, loadmode = loadmode )
          
          elif( ( loadmode == 'wgan' or loadmode == 'pix2pix') and trainmode == 'alltogether' ):
            print("Loading fake data from " + loadmode + " ,train alltogether")
            datalwf, labellwf, namelwf = Define_Data_Fold_Fake_Alltogether( ifold, jfold, partition, datatask, labels, namesImg,
                                                                histogramEqualization, imgwidth, fake_path_notb, fake_path_tb, loadmode = loadmode )
                
          elif(taskName == 'Shenzhen' and loadmode == 'wgan_p2p' and trainmode != 'alltogether' ):
            print("Loading fake data from " + loadmode + " ,train only with fake data")
            datalwf, labellwf, namelwf = Define_Data_Fold_Wgan_P2P( ifold, jfold, partition, datatask, labels, namesImg, 
                                                                  fake_path_notb, fake_path_tb, fake_path_notb_2, fake_path_tb_2 )
          
          elif( taskName == 'SantaCasa'  ):
            datalwf, labellwf, namelwf = Define_Data_Fold_SantaCasa( path_img, csv_path, fake_path_tb1, fake_path_tb2, ifold, jfold )                          
                    
          #Train model to finetune
          lwf_function('finetune', taskName, save_val, loadname, datalwf, labellwf, namelwf, 
                      ilayer = ilayer, SENS90 = SENS90, ifold = ifold, 
                      jfold = jfold, finetune_layers = finetune_layers,
                      lr = lr,  wd = wd, temp = temp, lr_decay_factor = lr_decay_factor )

#save parameters information in folder
params = { 'lr': lr, 'wd':wd, 'temp':temp, 'lr_decay_factor':lr_decay_factor }
with open(save_path + '/file.txt', 'w') as file: 
  file.write( json.dumps(params))
#run lwf
Run_LwF( lr, wd, temp, lr_decay_factor, save_path, user_parameters )