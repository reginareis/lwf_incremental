from __future__ import division, print_function

import argparse
import copy
import json
import warnings

from loaddata_paper import Load_Data_Batch
import networks as net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchnet as tnt
import utils as utils
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import StratifiedKFold
import glob
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sys import exit
import pickle


#Function definitions-------------------------------------------------




def Create_train( Id_folds, itest, ivalidation, itask ):
  #Create train joining the kfolds that are not in test nor validation
  # Id_folds - folds in which data was divided. Id_fold[ntasks][nkfolds][n elements per fold]
  # itest - fold which is being used as test, so won't be part of the train
  # ivalidation - fold which is being used as validation, so won't be part of the train
  # itask - current task

  nkfolds = len( Id_folds[ itask ] )
  Id_train = []
  for ikf in range( nkfolds ):
    if ( ikf != itest and ikf != ivalidation ):
      for iimg in range( len( Id_folds[ itask ][ ikf ] ) ):
        Id_train.append( Id_folds[ itask ][ ikf ][ iimg ] )

  return Id_train

def Create_trainval_tb( partition ):
  #Create train in Otto's partition joining the train and validation sets
  # partition - specific partition to join train and validation sets.
  #            it has lenght = 3, partition[0] = train, partition[1] = val, partition[3] = test. 

  Id_trainval = []
  for iimg in range( len( partition[ 0 ] ) ):
    Id_trainval.append( partition[ 0 ][ iimg ] )
  for iimg in range( len( partition[ 1 ] ) ):
    Id_trainval.append( partition[ 1 ][ iimg ] )

  return Id_trainval

def sensitivity( nVP, nFN ):
  # calculates sensitivity
  # nVP - number of true positives
  # nFN - number of false negatives
  sens = nVP/( nVP + nFN )
  return sens

def specificity( nVN, nFP ):
  # calculates specificity
  # nVN - number of true negatives
  # nFP - number of false positives
  spec = nVN/( nVN + nFP )
  return spec

def sp_index(nVP, nVN, nFP, nFN):
  # calculates sp index
  # nVN - number of true negatives
  # nFP - number of false positives
  # nVP - number of true positives
  # nFN - number of false negatives
  import math

  sens = nVP/( nVP + nFN )
  spec = nVN/( nVN + nFP )
  arit = ( sens + spec )/2
  geom = math.sqrt( sens * spec )
  sp = math.sqrt( arit * geom )
  return sp

def eval_test( usecuda, modelfile, test_data, test_label, test_names, VP, VN, FP, FN, 
          features_shared, output_last, delta = 0 ):
      # evaluates adjusted model
      # test_data = input of test set
      # test_label = labels of test set
      # test_names = names of images in test set
      # VP, VN.. = classification results. VP[ nkfolds][ n true positives ]

    #load adjusted model
    ckpt = torch.load( modelfile )
    model = ckpt['model']
    model.eval()

    print('Performing eval...')
    print('Size test is ' + str(len(test_label)))
    """Performs evaluation."""
  
    print("Evaluating in test")        

    #get batches
    batch_size = 32
  
    #Initialize VPs, VNs, FPs, FNs
    VP_aux = []
    VN_aux = []
    FP_aux = []
    FN_aux = []
    
    for i in range(0,test_label.size()[0], batch_size):
      batch_data, batch_label, batch_names = test_data[i:i+batch_size], test_label[i:i+batch_size], test_names[i:i+batch_size]
      if usecuda:
          batch = batch_data.cuda()
      else:
          batch = batch_data
      batch = Variable(batch, volatile=True)    
      
      x = model.shared(batch)
      pred_logits = [classifier(x) for classifier in model.classifiers]
      output = pred_logits[-1]

      for ifeat in range( len( output ) ):
        features_shared[ batch_names[ ifeat ] ] = x[ ifeat ].cpu().detach().numpy() 
        output_last[ batch_names[ ifeat ] ] = output[ ifeat ].cpu().detach().numpy() 

      #Calculate VPs, VNs, FPs, FNs for last task
      count = 0
      for outvalue in output:
        #print( "outvalue: " + str(outvalue) + " label: " + str(batch_label[ count ]) )
        if( outvalue[ 0 ] > outvalue[ 1 ] + delta ): #output = 0
          if( batch_label[ count ] == 0):
            VN_aux.append( batch_names[ count ] )
          else:
            FN_aux.append( batch_names[ count ] )
        else: #output = 1
          if( batch_label[ count ] == 1):
            VP_aux.append( batch_names[ count ] )
          else:
            FP_aux.append( batch_names[ count ] )
        count += 1
    
    VP.append( VP_aux )
    VN.append( VN_aux )
    FP.append( FP_aux )
    FN.append( FN_aux )
  
    sp = sp_index(len(VP_aux), len(VN_aux), len(FP_aux), len(FN_aux))
    sens = sensitivity( len(VP_aux), len(FN_aux) )
    spec = specificity( len(VN_aux), len(FP_aux) )
    print( "SP index for eval  is " + str( sp ) )
    print( "Sensitivity for eval  is " + str( sens ) )
    print( "Specificity for eval  is " + str( spec ) )
    print(" VP is " + str(len(VP_aux)) + " FP is " + str(len(FP_aux)) + " VN is " + str(len(VN_aux)) + " FN is " + str(len(FN_aux)) )
    print("Sum VP, VN etc is " + str(len(VP_aux)+len(VN_aux)+len(FP_aux)+len(FN_aux)) )

def distillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    # y: classification output
    # teacher_scores: labels
    # T: logit temperature


    return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * scale

def Define_Operation_Point( label, output ):
      print( "iniciando ordenacao")
      orderedDelta = []
      for i in range( len( label ) ):
        for j in range( len( label[ i ] ) ):
          if label[ i ][ j ] == True:
            pos = len( orderedDelta )
            delta = output[ i ][ j ][ 1 ] - output[ i ][ j ][ 0 ]
            for iOrd in range( len( orderedDelta ) ):              
              if delta > orderedDelta[ iOrd ]:
                pos = iOrd 
                break
            orderedDelta.insert( pos, delta )
      print("len ondered delta = " + str( len( orderedDelta )))
      ind = int( 9 * len( orderedDelta )/10 )
      print( "ind  = " + str(ind))
      delta90 = orderedDelta[ ind ]
      #print( orderedDelta )
      
      #print("Delta corte 90% = " + str( delta90 ) )
      
      return delta90

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, original_model, model, weights = '' ):
        self.args = args
        self.cuda = args.cuda
        self.original_model = original_model
        self.model = model

        if args.mode != 'check':
            # Set up data loader, criterion, and pruner.
       #     if 'cropped' in args.train_path:
       #         train_loader = dataset.train_loader_cropped
       #         test_loader = dataset.test_loader_cropped
       #     else:
       #         train_loader = dataset.train_loader
       #         test_loader = dataset.test_loader
       #     self.train_data_loader = train_loader(
       #         args.train_path, args.batch_size, pin_memory=args.cuda)
       #     self.test_data_loader = test_loader(
       #         args.test_path, args.batch_size, pin_memory=args.cuda)
            if weights == '':
              self.criterion = nn.CrossEntropyLoss()   
            else:
              self.criterion = nn.CrossEntropyLoss( weights )  
    
    
    def eval( self, datalwf, labellwf, namelwf, VP = 0, VN = 0, FP = 0, FN = 0, mode = 'finetune', 
             getFeatures = False, features_shared = 0, output_last = 0, ilayer = -1, SENS90 = False  ):
      # evaluates adjusted model
      # test_data = input of test set
      # test_label = labels of test set
      # test_names = names of images in test set
      # VP, VN.. = classification results. VP[ nkfolds][ n true positives ]
        if mode == 'finetune' or mode == 'evalval':
          test_data = datalwf['val']
          test_label = labellwf['val']
          test_names = namelwf['val']
          test_data2 = ""
          test_label2 = ""
          test_names2 = ""
        
        elif mode == 'evaltrain':
          print("Mode evaltrain")
          test_data = datalwf['train']
          test_label = labellwf['train']
          test_names = namelwf['train']
          test_data2 = ""
          test_label2 = ""
          test_names2 = ""
        
        elif mode == 'evaltest':
          test_data = datalwf['test']
          test_label = labellwf['test']
          test_names = namelwf['test']
          test_data2 = ""
          test_label2 = ""
          test_names2 = ""

        elif mode == 'evaltrainval':
          test_data = datalwf['trainval']
          test_label = labellwf['trainval']
          test_names = namelwf['trainval']
          test_data2 = ""
          test_label2 = ""
          test_names2 = ""

        elif mode == 'evalall':
          test_data = datalwf['all']
          test_label = labellwf['all']
          test_names = namelwf['all']
          test_data2 = ""
          test_label2 = ""
          test_names2 = "" 

        
        print('Performing eval...')
        print('Size test or val is ' + str(len(test_label)))
        """Performs evaluation."""
      
        print("Evaluating in test")        
        self.model.eval()
        error_meter = None

        #get batches
        batch_size = 32
        
        #Initialize VPs, VNs, FPs, FNs
        VP_aux = []
        VN_aux = []
        FP_aux = []
        FN_aux = []
        features_shared = []
        output_last = []
        output_all = [] #estrutura necessária para mudar ponto de operação
        label_all = []
        x = []
        
        #avalia teste 1
        print("Avalia conjunto de teste 1")
        if type( test_label ) == np.ndarray:
          testsize = test_label.size
        else:
          testsize = test_label.size()[0]
        print("TESTSIZE " + str(testsize))
        for i in range(0,testsize, batch_size):
          #print("i= " + str(i))
          batch_data, batch_label, batch_names = test_data[i:i+batch_size], test_label[i:i+batch_size], test_names[i:i+batch_size]
          if type( batch_data[0] ) == str or type( batch_data[0] ) == np.str_:
              print("Importando imagens na hora do batch")
              batch_data = Load_Data_Batch( batch_data )

          if type( batch_label ) == np.ndarray:
              batch_label = torch.from_numpy( batch_label ).long()
          if type( batch_data ) == np.ndarray:
              batch_data = torch.from_numpy( batch_data ).float()
          
          if self.cuda:
              batch = batch_data.cuda()
          else:
              batch = batch_data
          batch = Variable(batch, volatile=True)    
          
          torch.cuda.empty_cache()
          x = self.model.shared(batch)
          
          pred_logits = [classifier(x) for classifier in self.model.classifiers]
          #print("ilayer in eval is " + str(ilayer))
          output = pred_logits[ilayer]
          output_cpu, batch_label_cpu = output.cpu(), batch_label.cpu()
          output_all.append( output_cpu.detach().numpy() )
          label_all.append( batch_label_cpu.detach().numpy() )

          if( getFeatures ):
            for ifeat in range( len( output ) ):
              features_shared.append( x[ ifeat ].cpu().detach().numpy() )
              output_last.append( output[ ifeat ].cpu().detach().numpy() )

        #Define operation point
        if ( SENS90 ):
          deltaop = Define_Operation_Point(label_all,output_all)
        else:
          deltaop = 0.0
        
        #Calculate VPs, VNs, FPs, FNs for last task
        # print("outputs:")
        # print( output_all )
        # print("labels:")
        # print( label_all )
        count = 0
        for i in range( len( output_all ) ):
          for j in range( len( output_all[ i ] ) ):
            if( output_all[ i ][ j ][ 0 ] > output_all[ i ][ j ][ 1 ] - deltaop ): #output = 0
              if( label_all[ i ][ j ] == 0):
                VN_aux.append( label_all[ i ][ j ] )
              else:
                FN_aux.append( label_all[ i ][ j ] )
            else: #output = 1
              if( label_all[ i ][ j ] == 1):
                VP_aux.append( label_all[ i ][ j ] )
              else:
                FP_aux.append( label_all[ i ][ j ] )
            count += 1

          # Init error meter.
          # if error_meter is None:
          #     topk = [1]
          #     if output.size(1) > 5:
          #         topk.append(5)
          #     error_meter = tnt.meter.ClassErrorMeter(topk=topk)
          # error_meter.add( output_all[ i ].data, label_all[ i ] )

        #avalia teste 2
        if test_label2 != "":
          print("Avalia conjunto de teste 2")
          print("tipo teste 2 " + str(type(test_label2)))
          print(test_label2)
          if type( test_label2 ) == np.ndarray:
            testsize2 = test_label2.size
          else:
            testsize2 = test_label2.size()[0]
          for i in range(0,testsize2, batch_size):
            batch_data, batch_label, batch_names = test_data2[i:i+batch_size], test_label2[i:i+batch_size], test_names2[i:i+batch_size]
																				
																	  
														
            if type( batch_label ) == np.ndarray:
                batch_data, batch_label = torch.from_numpy( batch_data ).float(), torch.from_numpy( batch_label ).long()
            if self.cuda:
                batch = batch_data.cuda()
            else:
                batch = batch_data
            batch = Variable(batch, volatile=True)      
									
            x = self.model.shared(batch)
            pred_logits = [classifier(x) for classifier in self.model.classifiers]
            #print("ilayer in eval is " + str(ilayer))
            output = pred_logits[ilayer]																

            if( getFeatures ):
              for ifeat in range( len( output ) ):
                features_shared.append( x[ ifeat ].cpu().detach().numpy() )
                output_last.append( output[ ifeat ].cpu().detach().numpy() )

            #Calculate VPs, VNs, FPs, FNs for last task
            count = 0
            for outvalue in output:
              if( outvalue[ 0 ] > outvalue[ 1 ]): #output = 0
                if( batch_label[ count ] == 0):
                  VN_aux.append( batch_names[ count ] )
                else:
                  FN_aux.append( batch_names[ count ] )
              else: #output = 1
                if( batch_label[ count ] == 1):
                  VP_aux.append( batch_names[ count ] )
                else:
                  FP_aux.append( batch_names[ count ] )
              count += 1

            # Init error meter.
            # if error_meter is None:
            #     topk = [1]
            #     if output.size(1) > 5:
            #         topk.append(5)
            #     error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            # error_meter.add(output.data, batch_label)

        
        sp = sp_index(len(VP_aux), len(VN_aux), len(FP_aux), len(FN_aux))
        sens = sensitivity( len(VP_aux), len(FN_aux) )
        print( "Sensitivity is " + str( sens ) )
        # errors = error_meter.value()
        # print('Error: ' + ', '.join('@%s=%.2f' %
        #                             t for t in zip(topk, errors)))
        if( mode != 'finetune' and mode != 'evaltrain'):
          VP.append( VP_aux )
          VN.append( VN_aux )
          FP.append( FP_aux )
          FN.append( FN_aux )
        
          sp = sp_index(len(VP_aux), len(VN_aux), len(FP_aux), len(FN_aux))
          print( "SP index for eval  is " + str( sp ) )
          print(" VP is" + str(len(VP_aux)) + " FP is" + str(len(FP_aux)) + " VN is" + str(len(VN_aux)) + " FN is" + str(len(FN_aux)) )
          print("Sum VP, VN etc is " + str(len(VP_aux)+len(VN_aux)+len(FP_aux)+len(FN_aux)) )

        #self.model.train() ->linha do codigo original, não sei pq tinha isso
        errors = 0 
        
        return errors, sp

    def do_batch( self, optimizer, batch, label, epoch_idx, ilayer = -1, weight = 1.0, train = True ):
        """Runs model for one batch."""
        batch_original = batch.clone()
        if self.cuda:
            batch_original = batch_original.cuda(0)
            batch = batch.cuda()
            label = label.cuda()
        batch_original = Variable(batch_original, requires_grad=False)
        batch = Variable(batch)
        label = Variable(label)

        # Get targets using original model.
        if (train):
          self.original_model.eval()
          x = self.original_model.shared(batch_original)
          target_logits = [classifier(x).data.cpu()
                          for classifier in self.original_model.classifiers]
          # Move to same GPU as current model.
          target_logits = [Variable(item.cuda(), requires_grad=False)
                          for item in target_logits]
          scale = [item.size(-1) for item in target_logits]

          # Work with current model.
          # Set grads to 0.
          self.model.zero_grad()

        # Do forward.
        x = self.model.shared(batch)
        pred_logits = [classifier(x) for classifier in self.model.classifiers]
        # Compute loss.
        dist_loss = 0
        
        if (train):
          # Apply distillation loss to all old tasks.
          for idx in range(len(target_logits)):
              dist_loss += distillation_loss(
                  pred_logits[idx], target_logits[idx], self.args.temperature, scale[idx])
        
        # Apply cross entropy for current task.
        #print("ilayer in batch is " + str(ilayer))
        output = pred_logits[ ilayer ]
        new_loss = self.criterion(output, label)

        if new_loss != new_loss:
          exit()
        loss = weight * ( dist_loss + new_loss ) #/20
        
        if (train):
          # Do backward.
          loss.backward()

          if epoch_idx <= self.args.ft_shared_after:
              # Set shared layer gradients to 0 if early epochs.
              for module in self.model.shared.modules():
                if isinstance(module, nn.Conv2d):
                #if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)
              # Set old classifier gradients to 0 if early epochs.
              for idx in range(len(self.model.classifiers) - 1):
                  module = self.model.classifiers[idx]
                  module.weight.grad.data.fill_(0)
                  module.bias.grad.data.fill_(0)

          # Update params.
          optimizer.step()
        loss, new_loss = loss.cpu().detach().numpy(), new_loss.cpu().detach().numpy()

        return loss/weight, new_loss

    def do_epoch( self, epoch_idx, optimizer, datalwf, labellwf, inds=[], ilayer = -1 ):
											  
        """Trains model for one epoch."""
        #get batches
        batch_size = 32
        razao_fontes = 1
        n_fontes = 1
        #soma da perda de todos os batches
        total_loss, new_loss, val_loss, size_menor = 0, 0, 0, 0
        #Batches alternados de imagens de 2 fontes
        train_data_maior, train_label_maior, size_maior = datalwf['train'], labellwf['train'], labellwf['train'].size#(dim=0)
        if labellwf['train2'] != "" :   
          n_fontes = 2  
          if labellwf['train'].size > labellwf['train2'].size:  
            train_data_menor, train_label_menor, size_menor = datalwf['train2'], labellwf['train2'], labellwf['train2'].size#(dim=0)
          else:
            train_data_menor, train_label_menor, size_menor = datalwf['train'], labellwf['train'], labellwf['train'].size#(dim=0)
            train_data_maior, train_label_maior, size_maior = datalwf['train2'], labellwf['train2'], labellwf['train2'].size#(dim=0)
          
          razao_fontes = int( size_maior / size_menor )
          print( "razao_fontes " + str(razao_fontes))
          print( "size_maior " + str(size_maior))
          print( "size_menor " + str(size_menor))
        
        for i in range(0,size_maior, batch_size):
          #load data if not loaded yet
          if type( train_data_maior[0] ) == str or type( train_data_maior[0] ) == np.str_:
            #print("Importando imagens na hora do batch")
            batch_label = train_label_maior[i:i+batch_size]
            batch_data = Load_Data_Batch( train_data_maior[i:i+batch_size] )
          else:
            batch_data, batch_label = train_data_maior[inds[i:i+batch_size]], train_label_maior[inds[i:i+batch_size]]
          
          if type( batch_data ) == np.ndarray:
            batch_data = torch.from_numpy( batch_data ).float()
          if type( batch_label ) == np.ndarray:
            batch_label = torch.from_numpy( batch_label ).long()
          
          #for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
          #print("Treinando batch maior")
          auxtotal, auxnew = self.do_batch( optimizer, batch_data, batch_label, epoch_idx, ilayer = ilayer,  weight = 1 / ( razao_fontes * n_fontes ) ) # / ( 2 * razao_fontes ) )
          #print("Calcula loss e total loss no batch de treino")
          total_loss += auxtotal
          new_loss += auxnew
          #print( total_loss )

          #Treina batch real alternado             
          if ( labellwf['train2'] != "" ) and ( i % ( batch_size * razao_fontes ) ) == 0 and ( i / razao_fontes < size_menor ):
            ini = int( i / razao_fontes )
            #print( "ini= " + str( i / razao_fontes ))
            fim = ini + batch_size
            #print("Treinando batch menor")
            #load data if not loaded yet
            if type( train_data_menor[0] ) == str or type( train_data_menor[0] ) == np.str_:
              #print("Importando imagens na hora do batch")
              batch_label2 = train_label_menor[ini:fim]
              batch_data2 = Load_Data_Batch( train_data_menor[ini:fim] )
            else:
              batch_data2, batch_label2 = train_data_menor[ini:fim], train_label_menor[ini:fim]
            if type( batch_label2 ) == np.ndarray:
              batch_data2, batch_label2 = torch.from_numpy( batch_data2 ).float(), torch.from_numpy( batch_label2 ).long()
          
            auxtotal, auxnew = self.do_batch( optimizer, batch_data2, batch_label2, epoch_idx, ilayer = ilayer,  weight = 1 / ( n_fontes ) )
            #"Calcula loss e total loss no batch de treino")
            total_loss += auxtotal
            new_loss += auxnew

        print("Calcula no conjunto de validação")
        batch_data, batch_label = [], []
        size_val = len( labellwf['val'] )
        for i in range(0, size_val, batch_size ):  
          #load data if not loaded yet
          if type( datalwf['val'][0] ) == str or type( datalwf['val'][0] ) == np.str_:
            #print("Importando imagens na hora do batch")
            batch_label = labellwf['val'][i:i+batch_size]
            batch_data = Load_Data_Batch( datalwf['val'][i:i+batch_size])
          else:
            batch_data, batch_label = datalwf['val'][i:i+batch_size], labellwf['val'][i:i+batch_size]
          #print( batch_label )
          if type( batch_label ) == np.ndarray:
            batch_label = torch.from_numpy( batch_label ).long()
          if type( batch_data ) == np.ndarray:
            print("type batch data")
            print(type( batch_data ))
            batch_data = torch.from_numpy( batch_data ).float()
          auxtotal, auxnew = self.do_batch( optimizer, batch_data, batch_label, epoch_idx, ilayer = ilayer, train = False )
          val_loss += auxnew
        
        return total_loss/( size_maior + size_menor ), new_loss/size_maior, val_loss/size_val
      

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        base_model = self.model
        if isinstance(self.model, nn.DataParallel):
            base_model = self.model.module

        # Prepare the ckpt.
        ckpt = {
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'args': self.args,
            'model': base_model,
        }

        # Save to file.
        torch.save(ckpt, savename + '.pt')

    def train( self, epochs, optimizer, datalwf, labellwf, namelwf, 
              inds=[], save=True, savename='', best_accuracy=0, best_sp=0, ilayer = -1, SENS90 = False ):
											  
        """Performs training."""
        best_accuracy = best_accuracy
        best_sp = best_sp
        sp_history = []
        sp_train_history = []
        losses = []
        new_losses_val = []
        new_losses_train = []

        
        #Numero de epocas toleráveis sem aumento no SP de validação
        patience = 10

        if self.args.cuda:
            self.model = self.model.cuda()
        
        #iEpoch = self.epoch
        #print("Saved epoch is " + str( iEpoch ))

        count_patience = 0
        for idx in range(epochs):
            if count_patience > patience:
               break
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))

            optimizer = utils.step_lr(epoch_idx, self.args.lr, self.args.lr_decay_every,
                                      self.args.lr_decay_factor, optimizer)
            self.model.train()
            total_loss, new_loss, loss_val = self.do_epoch( epoch_idx, optimizer, datalwf, labellwf, inds=inds, ilayer = ilayer )
            print("Total loss = " + str(total_loss) + ", new loss = " + str(new_loss) + ", loss val = " + str(loss_val))
            losses.append( total_loss )  
            new_losses_val.append( loss_val ) 
            new_losses_train.append( new_loss )

            errors, sp = self.eval( datalwf, labellwf, namelwf, ilayer = ilayer, SENS90 = SENS90 )
            errors, sp_train = self.eval( datalwf, labellwf, namelwf, ilayer = ilayer, SENS90 = SENS90, mode = 'evaltrain' )
            
            sp_history.append( sp )
            sp_train_history.append( sp_train )
            #error_history.append(errors)
            #accuracy = 100 - errors[0]  # Top-1 accuracy.

            # Save best model, if required.
            if save and sp > best_sp and idx > 0:
              count_patience = 0
              print('Best model so far, SP: %0.2f%% -> %0.2f%%' %
                    (100 * best_sp, 100 * sp))
              best_sp = sp
              self.save_model(epoch_idx, best_accuracy, errors, savename)
              
            else:
              count_patience += 1
              print('SP:  %0.2f%%' %
                    (100 * sp))
            print('Sp train:  %0.2f%%' %
                    (100 * sp_train))
           # if save and accuracy > best_accuracy:
                #print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                #      (best_accuracy, accuracy))
                #best_accuracy = accuracy
               # self.save_model(epoch_idx, best_accuracy, errors, savename)

        
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)
        # Save performance history and stats.
        with open(savename + '.json', 'w') as fout:
            json.dump({
                'sp_history': sp_history,
                'sp_train_history': sp_train_history,
                'args': vars(self.args),
            }, fout)
        
        #Plota progresso do indice sp pelas epocas
        fig, ax1 = plt.subplots()

        color1 = 'green'
        color2 = 'blue'
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel("Índice SP", color='black')
        ax1.plot( [i for i in range( len( sp_train_history ) )], sp_train_history, color=color1)
        ax1.plot( [i for i in range( len( sp_history ) )], sp_history, color=color2)
        ax1.set_ylim([0.5, 1.0])
        ax1.tick_params(axis='y', labelcolor='black')
        plt.legend(["Índice SP - Treino", "Índice SP - Validação"], loc ="upper right")
        plt.xticks(np.arange(0, idx, 1.0))
        

        color3 = 'red'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis        
        ax2.set_ylabel('Perda média por imagem', color='black')  # we already handled the x-label with ax1
        ax2.plot(  [i for i in range( len( losses ) )], losses, color=color3)
        #ax2.plot(  [i for i in range( len( new_losses_train ) )], new_losses_train, color='darkred')
        ax2.plot(  [i for i in range( len( new_losses_val ) )], new_losses_val, color='pink')
        ax2.tick_params(axis='y', labelcolor='black')

        plt.legend(["Total Loss - Treino", "Loss - Validação"], loc ="upper left")
        #plt.legend(["Total Loss - Treino" ], loc ="lower left")
        fig.tight_layout() 
  
        print("Salvando progresso em " + savename  )
        plt.savefig( savename + '_epochs.png' )
        plt.clf()
      



    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))

"""# Main LwF Function"""

def lwf_function( mode, dataset_name, save_prefix, loadname, datalwf, labellwf, namelwf,
                  VP = 0, VN = 0, FP = 0, FN = 0, ilayer = -1, SENS90 = False, shared_after = 0, lr = 1e-4,
                  wd = 0.00025, temp = 2.0, lr_decay_factor = 0.5, finetune_layers = 'all',
                  ifold = -1, jfold = -1, pretrain = True, itask = 0, novarede = False ):
					  


  # To prevent PIL warnings.
  warnings.filterwarnings("ignore")

  FLAGS = argparse.ArgumentParser()
  FLAGS.add_argument('--mode',
                  choices=['finetune', 'eval'], default = 'finetune',
                  help='Run mode')
  FLAGS.add_argument('--finetune_layers',
                  choices=['all', 'fc', 'classifier'], default=finetune_layers,
                  help='Which layers to finetune')
  FLAGS.add_argument('--num_outputs', type=int, default=2,
                  help='Num outputs for dataset')
  # Optimization options.
  FLAGS.add_argument('--lr', type=float, default = lr,
                  help='Learning rate')
  FLAGS.add_argument('--lr_decay_every', type=int, default = 10,
                  help='Step decay every this many epochs')
  FLAGS.add_argument('--lr_decay_factor', type=float, default = lr_decay_factor,
                  help='Multiply lr by this much every step of decay')
  FLAGS.add_argument('--finetune_epochs', type=int, default = 40,
                  help='Number of initial finetuning epochs')
  FLAGS.add_argument('--batch_size', type=int, default=32,
                  help='Batch size')
  FLAGS.add_argument('--dropout', type=float, default=0.5,
                  help='Dropout ratio')
  FLAGS.add_argument('--weight_decay', type=float, default=wd,
                  help='Weight decay')
  FLAGS.add_argument('--temperature', type=float, default= temp,
                  help='LwF logit temperature')
  FLAGS.add_argument('--ft_shared_after', type=int, default=shared_after,
                  help='Finetune shared layers after this epoch')
  # Paths.
  FLAGS.add_argument('--dataset', type=str, default='SHENZHEN',
                    help='Name of dataset')
 # FLAGS.add_argument('--train_path', type=str, default='',
 #                 help='Location of train data')
 # FLAGS.add_argument('--test_path', type=str, default='',
 #                 help='Location of test data')
  FLAGS.add_argument('--save_prefix', type=str, default=save_prefix,
                  help='Location to save model')
  FLAGS.add_argument('--loadname', type=str, default='',
                  help='Location to save model')
  # Other.
  FLAGS.add_argument('--cuda', action='store_true', default=True,
                  help='use CUDA')

  args = FLAGS.parse_args(['--loadname', ''])


  # Set default train and test path if not provided as input.
  args.save_prefix = save_prefix
  args.dataset = dataset_name
  args.mode = mode
  #args.train_path = train_path
  #args.test_path = test_path
  args.loadname = loadname
  #create index to shuffle data - NÃO ESTÁ DANDO SHUFFLE NA DATA2
  inds = ''
  a = np.array([i for i in range(len( labellwf['train'] ))])
  np.random.shuffle(a)
  inds = np.argsort(a)

  print("Carregando vgg") 

  # Load the required model.
  if 'finetune' in args.mode and args.loadname == 'null':
    print("foi para o if")
    if novarede:
      model = net.ModifiedVGG16Novo( pretrain = pretrain )
    else:
      model = net.ModifiedVGG16( pretrain = pretrain )
    #model = net.Philnet()

  else:
    print("foi para o else")
    ckpt = torch.load(args.loadname ) #map_location=torch.device('cpu') )
    model = ckpt['model']

  original_model = copy.deepcopy(model)

  print("incluindo dataset")

  # Add and set the model dataset.
  if novarede:
    print("Treinando com nova rede")
    model.add_dataset(args.dataset, args.num_outputs, itask )
    model.set_classifier( itask )
  else:
    print("Treinando com velha rede")
    model.add_dataset(args.dataset, args.num_outputs, ilayer )
    model.set_dataset(args.dataset, ilayer )

  if args.cuda: #cpu
      model = model.cuda(0)
      if args.mode == 'finetune':
          original_model = original_model.cuda(0)

  #calculate weights to balance classes
  nfalse = np.count_nonzero(labellwf['train'] == 0) 
  ntrue = np.count_nonzero(labellwf['train'] == 1) 
  print("n false = " + str(nfalse))
  print("n true = " + str(ntrue))
  ntimes = 1
  if nfalse > 0 and ntrue > 0:
    ntimes = nfalse/ntrue
  print("ntimes: " + str(ntimes))
  norm_factor = 2/(1 + ntimes )
  weight_class_0 = norm_factor
  weight_class_1 = ntimes * norm_factor
  
  class_weights = torch.tensor([weight_class_0, weight_class_1]).cuda()
  # Create the manager object.
  manager = Manager(args, original_model, model, class_weights )

  # Perform necessary mode operations.
  if args.mode == 'finetune':
      # Get optimizer with correct params.
      if args.finetune_layers == 'all':
          params_to_optimize = model.parameters()
      elif args.finetune_layers == 'classifier':
          for param in model.shared.parameters():
              param.requires_grad = False
          params_to_optimize = model.classifier.parameters()
      elif args.finetune_layers == 'fc':
          params_to_optimize = []
          # Add fc params.
          for param in model.shared.parameters():
              if param.size(0) == 4096:
                  param.requires_grad = True
                  params_to_optimize.append(param)
              else:
                  param.requires_grad = False
          # Add classifier params.
          for param in model.classifier.parameters():
              params_to_optimize.append(param)
          params_to_optimize = iter(params_to_optimize)
      optimizer = optim.SGD(params_to_optimize, lr=args.lr,
                            momentum=0.9, weight_decay=args.weight_decay)
      # Perform finetuning.
      manager.train( args.finetune_epochs, optimizer, datalwf, labellwf, namelwf, 
                     inds=inds, save=True, savename=args.save_prefix + "_test" + str( ifold ) + "_val" + str( jfold), ilayer = ilayer, SENS90 = SENS90 )
										  

  elif args.mode == 'check':
      # Load model and make sure everything is fine.
      manager.check(verbose=True)
  elif args.mode != 'finetune' and args.mode != 'check' : 
      # Just run the model on the eval set.
																						  
      manager.eval( datalwf, labellwf, namelwf, VP, VN, FP, FN, args.mode, ilayer = ilayer)

class Results:
  def __init__( self, ndatasets ):
    self.VP = []
    self.VN = []
    self.FP = []
    self.FN = []
    for i in range( ndatasets + 1 ):
      self.VP.append([])
      self.VN.append([])
      self.FP.append([])
      self.FN.append([])