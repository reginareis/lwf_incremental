import argparse
import copy
import json
import warnings

import dataset
from loaddata_paper import *
import networks as net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchnet as tnt
import utils as utils
from prune import SparsePruner
from torch.autograd import Variable
from tqdm import tqdm
import collections
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
import math
import matplotlib.pyplot as plt

import os
from ELLA import ELLA

def Define_Data_Fold_Fake( ifold, jfold, histogramEqualization, imgwidth, 
                           fake_path_notb, fake_path_tb ):
  #return 3 dictionaires: data with images arrays, label with labels true or false,
  #names with images names. Each dictionaire has entries 'test', 'val' and 'train'.
  #ifold = index for test
  #jfold = index for val
  #histogramEqualization = if true, use hist equalization when images are loaded
  #imgwidth =  dimensions for image
  #fake_path_notb = path in cluster with no tb images
  
  datalwf, labellwf, namelwf = {}, {}, {}
  #Load test data
  pathFalse = fake_path_notb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_NTB/TEST/*.png"
  pathTrue = fake_path_tb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_TB/TEST/*.png"
  sizePathFalse = 135
  sizePathTrue = 132
  datalwf['test'], labellwf['test'], namelwf['test'] = Load_Dataset( True, histogramEqualization, imgwidth, 
                                                          sizePathFalse, pathFalse, sizePathTrue, 
                                                          pathTrue )
  #Load train data
  pathFalse = fake_path_notb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_NTB/TRAIN/*.png"
  pathTrue = fake_path_tb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_TB/TRAIN/*.png"
  sizePathFalse = 136
  sizePathTrue = 133
  datalwf['train'], labellwf['train'], namelwf['train'] = Load_Dataset( True, histogramEqualization, imgwidth, 
                                                              sizePathFalse, pathFalse, sizePathTrue, 
                                                              pathTrue )
  #Data
  #Load val data
  pathFalse = fake_path_notb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_NTB/VAL/*.png"
  pathTrue = fake_path_tb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_TB/VAL/*.png"
  sizePathFalse = 134
  sizePathTrue = 131
  datalwf['val'], labellwf['val'], namelwf['val'] = Load_Dataset( True, histogramEqualization, imgwidth, 
                                                          sizePathFalse, pathFalse, sizePathTrue, 
                                                          pathTrue )
  return datalwf, labellwf, namelwf




"""# Functions to create train sets """


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

"""# Function that applies model to the data for ELLA"""

def Calculate_ELLA_Results( modelname, itask, data, labels, names, fold  ):
  
  a_file = open( modelname, "rb")
  model = pickle.load(a_file)
  a_file.close()
  
  y_pred = model.predict( data, itask )
 

  for i in range( y_pred.shape[0] ):
    if( labels[ i ] == True ):
      if( y_pred[ i ] == labels[ i ] ):
        fold.VP.append( names[ i ])
      else:
        fold.FN.append( names[ i ])
    else:
      if( y_pred[ i ] == labels[ i ] ):
        fold.VN.append( names[ i ])
      else:
        fold.FP.append( names[ i ])

  fold.sp = sp_index(len(fold.VP), len(fold.VN), len(fold.FP), len(fold.FN))
  fold.sens = sensitivity( len( fold.VP ), len( fold.FN ) )
  fold.spec = sensitivity( len( fold.VN ), len( fold.FP ) )
  print( "SP index for eval  is " + str( fold.sp ) )
  print(" VP is " + str(len(fold.VP)) + " FP is " + str(len(fold.FP)) + " VN is " + str(len(fold.VN)) + " FN is " + str(len(fold.FN)) )
  print("Sum VP, VN etc is " + str(len(fold.VP)+len(fold.VN)+len(fold.FP)+len(fold.FN)) )

def Classification_ELLA_All( y_pred, idimg ):
  probs = []
  for i in range( len( y_pred ) ):
    probs.append( 1.0 - y_pred[ i ][ idimg ] ) #prob associated to class false, task i
    probs.append( y_pred[ i ][ idimg ] ) #prob associated to class true, task i

  idmax = np.where( probs == np.max( probs ) )
  idmax = idmax[0][0]   

  outtask = np.floor( idmax/2 )
  outlabel = ( np.mod( idmax, 2 ) == 1 )
  
  return outtask, outlabel
  


def Calculate_ELLA_Results_All( modelname, ntasks, idataset, data, labels, names_test, fold  ):
  a_file = open( modelname, "rb")
  model = pickle.load(a_file)
  a_file.close()

  y_pred = []
  for itask in range( ntasks ):
    y_pred.append( model.predict_probs( data, itask ) )

  for idimg in range( y_pred[ 0 ].shape[0] ):
    outtask, outlabel = Classification_ELLA_All( y_pred, idimg )
    #Right Task
    if ( outtask == idataset ):
      #Right classifications
      if ( outlabel == labels[ idimg ] ):
        if outlabel == False:
          #print("Right task, right neg label")
          fold.VN.append( names_test[ idimg ] )
        else:
          #print("Right task, right pos label")
          fold.VP.append( names_test[ idimg ] )
      #Wrong classifications
      else:
        if outlabel == 0:
          #print("Right task, wrong neg label")
          fold.FN.append( names_test[ idimg ] )
        else:
          #print("Right task, wrong pos label")
          fold.FP.append( names_test[ idimg ] )
    #Wrong Task
    else:
        #"Right" classifications (wrong when is positive because it is other task)
      if ( outlabel == labels[ idimg ]  ):
        if outlabel == 0:
          #print("Wrong task, right neg label")
          fold.VN_other.append( names_test[ idimg ] )
        else:
          #print("Wrong task, right pos label")
          fold.VP_other.append( names_test[ idimg ] )
      #Wrong classifications
      else:
        if outlabel == 0:
          #print("Wrong task, wrong neg label")
          fold.FN_other.append( names_test[ idimg ] )
        else:
          #print("Wrong task, wrong pos label")
          fold.FP_other.append( names_test[ idimg ] )

  fold.sp = sp_index(len(fold.VP), len(fold.VN), len(fold.FP), len( fold.FN ) + len( fold.FN_other ) + len( fold.VP_other ) )
  fold.sens = sensitivity( len( fold.VP ), len( fold.FN ) + len( fold.FN_other ) + len( fold.VP_other )  )
  fold.spec = sensitivity( len( fold.VN ) + len( fold.VN_other ), len( fold.FP ) + len( fold.FP_other ) )
  print( "SP index for eval  is " + str( fold.sp ) )
  print(" VP is " + str(len(fold.VP)) + " FP is " + str(len(fold.FP)) + " VN is " + str(len(fold.VN)) + " FN is " + str(len(fold.FN)) )
  print(" VP other is " + str(len(fold.VP_other)) + " FP other is " + str(len(fold.FP_other)) + " VN other is " + str(len(fold.VN_other)) + " FN other is " + str(len(fold.FN_other)) )
  print("Sum VP, VN etc is " + str(len(fold.VP)+len(fold.VN)+len(fold.FP)+len(fold.FN)) )

"""# Function that applies model to the data for LwF"""

def Calcula_Features( usecuda, modelfile, test_data, test_label, test_names ):
   #load adjusted model
  ckpt = torch.load( modelfile )
  model = ckpt['model']
  model.eval()

  #print('Performing eval...')
  #print('Size test is ' + str(len(test_label)))
  """Performs evaluation."""

  #print("Evaluating in test")        

  #get batches
  batch_size = 32
  feat_true, feat_false, names_true, names_false = [], [], [], []
  
  for i in range(0,len(test_label), batch_size):
    batch_data, batch_label, batch_names = test_data[i:i+batch_size], test_label[i:i+batch_size], test_names[i:i+batch_size]
    if usecuda:
        batch = batch_data.cuda()
    else:
        batch = batch_data
    batch = Variable(batch, volatile=True)      
    
    feat_aux = model.shared(batch)
    feat_aux = feat_aux.cpu().detach().numpy()
    for i in range( len(batch_names) ):
      if batch_label[ i ]:
        feat_true.append( feat_aux[ i ] )
        names_true.append( batch_names [ i ] )
      else:
        feat_false.append( feat_aux[ i ] )
        names_false.append( batch_names [ i ] )    

  return feat_true, feat_false, names_true, names_false

def eval_test( usecuda, modelfile, test_data, test_label, test_names, 
               ilayer = -1, save_shared = False ):
  # evaluates adjusted model
  # test_data = input of test set
  # test_label = labels of test set
  # test_names = names of images in test set

  #load adjusted model
  ckpt = torch.load( modelfile )
  model = ckpt['model']
  model.eval()

  """Performs evaluation."""   

  #get batches
  batch_size = 32
  output_all, labels_all, names_all, output_shared = [], [], [], []
  
  for i in range(0,len(test_label), batch_size):
    batch_data, batch_label, batch_names = test_data[i:i+batch_size], test_label[i:i+batch_size], test_names[i:i+batch_size]
    if type( batch_data[0] ) == str or type( batch_data[0] ) == np.str_:
      print("Importando imagens na hora do batch")
      batch_data = Load_Data_Batch( batch_data )
    if type( batch_data ) == np.ndarray:
      batch_data = torch.from_numpy( batch_data ).float()         
    if usecuda:
        batch = batch_data.cuda()
    else:
        batch = batch_data
    batch = Variable(batch, volatile=True)      
    
    x = model.shared(batch)
    #print("Tamanho de x = " + str(len(x)))
    #print("Tamanho de x[0] = " + str(len(x[0])))
    if save_shared:
      output_shared.append( x.cpu().detach().numpy())
    pred_logits = [classifier(x) for classifier in model.classifiers]
    #print( "Len pred logits is " + str( len( pred_logits )))
    output = pred_logits[ ilayer ]
    output_all.append( output.cpu().detach().numpy() )
    if type(batch_label) != np.ndarray:
      labels_all.append( batch_label.cpu().detach().numpy() )
    else:
      labels_all.append( batch_label )
    names_all.append( batch_names )
    
  return output_all, labels_all, names_all, output_shared
        
def Calculate_Classifications_eval( output_all, labels_all, names_all, fold, delta ):
  #Calculate VPs, VNs, FPs, FNs for last task
  for i in range( len( output_all) ):
    for j in range( len( output_all[ i ] ) ): 
      fold.Output0[ names_all[ i ][ j ] ] = output_all[ i ][ j ][ 0 ]
      fold.Output1[ names_all[ i ][ j ] ] = output_all[ i ][ j ][ 1 ]
      if( output_all[ i ][ j ][ 0 ] > output_all[ i ][ j ][ 1 ] - delta ): #output = 0
        if( labels_all[ i ][ j ] == 0):
          fold.VN.append( names_all[ i ][ j ] )
        else:
          fold.FN.append( names_all[ i ][ j ] )
      else: #output = 1
        if( labels_all[ i ][ j ] == 1 ) :
          fold.VP.append( names_all[ i ][ j ] )
        else:
          fold.FP.append( names_all[ i ][ j ] )

  if len(fold.VP) + len(fold.FN) > 0 and len(fold.VN) + len(fold.FP) > 0:
    fold.sp = sp_index(len(fold.VP), len(fold.VN), len(fold.FP), len(fold.FN))
  else:
    fold.sp = 0
  if len(fold.VP) + len(fold.FN) > 0:
    fold.sens = sensitivity( len( fold.VP ), len( fold.FN ) )
  else:
    fold.sens = 0
  if len(fold.VN) + len(fold.FP) > 0:
    fold.spec = sensitivity( len( fold.VN ), len( fold.FP ) )
  else:
    fold.spec = 0
  print( "SP index is " + str( fold.sp ) )
  print( "Sens index is " + str( fold.sens ) )
  print( "Spec index is " + str( fold.spec ) )
  print(" VP is " + str(len(fold.VP)) + " FP is " + str(len(fold.FP)) + " VN is " + str(len(fold.VN)) + " FN is " + str(len(fold.FN)) )
  #print(" VP[0] is " + str(fold.VP[0]) + " FP[0] is " + str(fold.FP[0]) + " VN[0] is " + str(fold.VN[0]) + " FN[0] is " + str(fold.FN[0]) )
  #print("Sum VP, VN etc is " + str(len(fold.VP)+len(fold.VN)+len(fold.FP)+len(fold.FN)) )

def cudamax( cudarray ):
  maxval = cudarray[ 0 ]
  maxid = 0
  for id in range( len( cudarray ) ):
    if ( cudarray[ id ] > maxval ):
      maxval = cudarray[ id ]
      maxid = id 
  #print( "maxid: " + str( maxid ) )
  return maxid

def eval_all( usecuda, modelfile, test_data, test_label, test_names, fold, itask ):
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
    
    for i in range(0,len(test_label), batch_size):
      batch_data, batch_label, batch_names = test_data[i:i+batch_size], test_label[i:i+batch_size], test_names[i:i+batch_size]
      if usecuda:
          batch = batch_data.cuda()
      else:
          batch = batch_data
      batch = Variable(batch, volatile=True)      
      
      x = model.shared(batch)
      pred_logits = [classifier(x) for classifier in model.classifiers]
      
      for ix in range( len( batch_label ) ):
        outx = []
        for iclass in range( 1, len( pred_logits ) ):
          outx.append( pred_logits[ iclass ][ ix ][ 0 ] )
          outx.append( pred_logits[ iclass ][ ix ][ 1 ] )
        
        idmax = cudamax( outx )
        outtask = np.floor( idmax/2 )
        outlabel = np.mod( idmax, 2 )
        #Right Task
        if ( outtask == itask ):
          #Right classifications
          if ( outlabel == batch_label[ ix ] ):
            if outlabel == 0:
              #print("Right task, right neg label")
              fold.VN.append( batch_names[ ix ] )
            else:
              #print("Right task, right pos label")
              fold.VP.append( batch_names[ ix ] )
          #Wrong classifications
          else:
            if outlabel == 0:
              #print("Right task, wrong neg label")
              fold.FN.append( batch_names[ ix ] )
            else:
              #print("Right task, wrong pos label")
              fold.FP.append( batch_names[ ix ] )
        #Wrong Task
        else:
           #"Right" classifications (wrong when is positive because it is other task)
          if ( outlabel == batch_label[ ix ] ):
            if outlabel == 0:
              #print("Wrong task, right neg label")
              fold.VN_other.append( batch_names[ ix ] )
            else:
              #print("Wrong task, right pos label")
              fold.VP_other.append( batch_names[ ix ] )
          #Wrong classifications
          else:
            if outlabel == 0:
              #print("Wrong task, wrong neg label")
              fold.FN_other.append( batch_names[ ix ] )
            else:
              #print("Wrong task, wrong pos label")
              fold.FP_other.append( batch_names[ ix ] )

    fold.sp = sp_index(len(fold.VP), len(fold.VN), len(fold.FP), len( fold.FN ) + len( fold.FN_other ) + len( fold.VP_other ) )
    fold.sens = sensitivity( len( fold.VP ), len( fold.FN ) + len( fold.FN_other ) + len( fold.VP_other )  )
    fold.spec = sensitivity( len( fold.VN ) + len( fold.VN_other ), len( fold.FP ) + len( fold.FP_other ) )
    print( "SP index for eval  is " + str( fold.sp ) )
    print(" VP is " + str(len(fold.VP)) + " FP is " + str(len(fold.FP)) + " VN is " + str(len(fold.VN)) + " FN is " + str(len(fold.FN)) )
    print(" VP other is " + str(len(fold.VP_other)) + " FP other is " + str(len(fold.FP_other)) + " VN other is " + str(len(fold.VN_other)) + " FN other is " + str(len(fold.FN_other)) )
    print("Sum VP, VN etc is " + str(len(fold.VP)+len(fold.VN)+len(fold.FP)+len(fold.FN)) )

def Contar_acertos_por_imagem ( modeltask, acertos_imagem, tipo_fold  ):
 
  nfolds = len( modeltask.folder )
  print("Nfolds = " + str(nfolds))
  
  for ifold in range( nfolds ):
    set_vp = set( modeltask.folder[ ifold ].VP )
    set_vn = set( modeltask.folder[ ifold ].VN )
    set_fp = set( modeltask.folder[ ifold ].FP )
    set_fn = set( modeltask.folder[ ifold ].FN )
    for img in set_vp:
      acertos_imagem[img][tipo_fold + ' C'] += 1
    for img in set_vn:
      acertos_imagem[img][tipo_fold + ' C'] += 1
    for img in set_fp:
      acertos_imagem[img][tipo_fold + ' E'] += 1
    for img in set_fn:
      acertos_imagem[img][tipo_fold + ' E'] += 1
    total = len(set_vp) + len(set_vn) + len(set_fp) + len(set_fn)
    print("Total de imagens para ifold = " + str(ifold) + " é " + str(total))

"""# Function to join VP, FP, VN, FN from all folds"""

def Create_Sets_with_VPFPVNFN_from_all_folds ( modeltask ):
  VP_set = []
  FP_set = []
  VN_set = []
  FN_set = []
  VP_other_set = []
  FP_other_set = []
  VN_other_set = []
  FN_other_set = []
  for ifold in range( len( modeltask.folder ) ):
    VP_other_set += modeltask.folder[ ifold ].VP_other
    FP_other_set += modeltask.folder[ ifold ].FP_other
    VN_other_set += modeltask.folder[ ifold ].VN_other
    FN_other_set += modeltask.folder[ ifold ].FN_other
    VP_set += modeltask.folder[ ifold ].VP
    FP_set += modeltask.folder[ ifold ].FP
    VN_set += modeltask.folder[ ifold ].VN
    FN_set += modeltask.folder[ ifold ].FN
  
  print("Quantidade de VP, FP, VN, FN = " + str( len( VP_set ) + len( FP_set ) + len( VN_set ) + len( FN_set ) ))
  modeltask.VP_all_other = set( VP_other_set ) 
  modeltask.FP_all_other = set( FP_other_set )
  modeltask.VN_all_other = set( VN_other_set )
  modeltask.FN_all_other = set( FN_other_set )
  modeltask.VP_all = set( VP_set ) 
  modeltask.FP_all = set( FP_set )
  modeltask.VN_all = set( VN_set )
  modeltask.FN_all = set( FN_set )
  print("Quantidade de VP_all = " + str( len( modeltask.VP_all ) ) )
  print("Quantidade de FP_all = " + str( len( modeltask.FP_all ) ) )
  print("Quantidade de VN_all = " + str( len( modeltask.VN_all ) ) )
  print("Quantidade de FN_all = " + str( len( modeltask.FN_all ) ) )
 

"""# Functions to calculate sens, spec and SP"""

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

"""# Defining class to store all the results (models from tasks applied to old tasks)"""

class Folders:
  #class that stores the results of vp, vn etc for a model that
  #was originated from a specific folder, with all classifiers being presented
  #to the data
  def __init__( self ):
    self.VP = []
    self.VN = []
    self.FP = []
    self.FN = []
    self.FP_other = []
    self.VP_other = []
    self.FN_other = []
    self.VN_other = []
    self.sens = 0
    self.spec = 0
    self.sp = 0
    self.Output0 = {}
    self.Output1 = {}

class Modeltasks:
  #class that stores the results of the application of all models that 
  #were trained during a specific task. each fold from the dataset has
  #originated a different model.
  #format: alldata.folder[0].VP[0]
  def __init__( self, nfolders ):
    self.sens_mean = 0
    self.sens_std = 0
    self.spec_mean = 0
    self.spec_std = 0
    self.sp_mean = 0
    self.sp_std = 0
    self.folder = []
    self.VP_all = []
    self.VN_all = []
    self.FP_all = []
    self.FN_all = []
    self.FP_all_other = []
    self.VP_all_other = []
    self.FN_all_other = []
    self.VN_all_other = []
    for i in range( nfolders ):
      self.folder.append( Folders() )

class Task:
  #class that stores the results of all models that need to be applied to an
  #specific task, to see if the model has lost knowledge about this task
  #format: alldata.aftertask[0].folder[0].VP[0]
  def __init__( self, nTasks, nfolders, difNfolders = False  ):
    self.modeltask = []
    for n in range( nTasks ):
      if( difNfolders ):
        self.modeltask.append( Modeltasks( nfolders[ n ] ) )
      else:
        self.modeltask.append( Modeltasks( nfolders ) )

class Alltasks:
  #class that stores the results of all tasks.
  #format: alldata.task[0].aftertask[0].folder[0].VP[0]
  def __init__( self, nTasks, nfolders, difNfolders = False  ):
    self.datatask = []
    for n in range( nTasks ):
      self.datatask.append( Task( nTasks, nfolders, difNfolders ))  

def Intersections( list1, list2, name, value ):
  intersect = set( list1 ).intersection( set( list2 ) )
 # print( name + " size is " + str( len( intersect ) ) )
  value[0] = value[0] + len( intersect ) 
  print( name + " sum is " + str( value[0] ) )

def Intersect_results( class1, allclass ):
  print( "Comparações entre 1 classificador ou varios classificadores" )
  VP_VP_other = [0]
  VP_FN_other = [0]
  VP_VP = [0]
  VN_VN_other = [0]
  VN_FP_other = [0]
  VN_VN = [0]
  FP_FP_other = [0]
  FP_VN_other = [0]
  FP_FP = [0]
  FN_FN_other = [0]
  FN_VP_other = [0]
  FN_FN = [0]

  Intersections( class1.VP_all, allclass.VP_all_other, "VP_VP_other", VP_VP_other )
  Intersections( class1.VP_all, allclass.FN_all_other, "VP_FN_other", VP_FN_other )
  Intersections( class1.VP_all, allclass.VP_all, "VP_VP", VP_VP )
  Intersections( class1.VN_all, allclass.VN_all_other, "VN_VN_other", VN_VN_other )
  Intersections( class1.VN_all, allclass.FP_all_other, "VN_FP_other", VN_FP_other )
  Intersections( class1.VN_all, allclass.VN_all, "VN_VN", VN_VN )
  Intersections( class1.FP_all, allclass.FP_all_other, "FP_FP_other", FP_FP_other )
  Intersections( class1.FP_all, allclass.VN_all_other, "FP_VN_other", FP_VN_other )
  Intersections( class1.FP_all, allclass.FP_all, "FP_FP", FP_FP )
  Intersections( class1.FN_all, allclass.FN_all_other, "FN_FN_other", FN_FN_other )
  Intersections( class1.FN_all, allclass.VP_all_other, "FN_VP_other", FN_VP_other )
  Intersections( class1.FN_all, allclass.FN_all, "FN_FN", FN_FN )

def Calculate_indexes_mean_std( modeltask ):
  nkfolds = len( modeltask.folder )
  sp_aux = []
  sens_aux = []
  spec_aux = []
  max_sp = 0.0
  id_maxsp = -1
  for imodel in range( nkfolds ):
    sp_aux.append( modeltask.folder[ imodel ].sp )
    sens_aux.append( modeltask.folder[ imodel ].sens )
    spec_aux.append( modeltask.folder[ imodel ].spec )
    if modeltask.folder[ imodel ].sp > max_sp:
      max_sp = modeltask.folder[ imodel ].sp
      id_maxsp = imodel
  modeltask.sens_mean = np.mean( sens_aux )
  modeltask.sens_std = np.std( sens_aux )
  modeltask.spec_mean = np.mean( spec_aux )
  modeltask.spec_std = np.std( spec_aux )
  modeltask.sp_mean = np.mean( sp_aux )
  modeltask.sp_std = np.std( sp_aux )

  print(" Maximum value for SP is " + str(max_sp) + " for imodel = " + str(id_maxsp) )
  sens = str( round( 100 * modeltask.sens_mean, 1 ) ) + " +- " + str( round( 100 * modeltask.sens_std, 1 ) )
  print("Sensitivity is " + sens )
  spec = str( round( 100 * modeltask.spec_mean, 1 ) ) + " +- " + str( round( 100 * modeltask.spec_std, 1 ) )
  print("Specificity is " + spec )
  sp = str( round( 100 * modeltask.sp_mean, 1 ) ) + " +- " + str( round( 100 * modeltask.sp_std, 1 ) )
  print("SP Index is " + sp )
  
  return sens, spec, sp

def Define_Data_Fold_Fake( ifold, jfold, histogramEqualization, imgwidth, 
                           fake_path_notb, fake_path_tb ):
  
  #Recebe caminhos dos dados gerados pela GAN, carrega imagens 
  #retorna data, label e name
  
  datalwf, labellwf, namelwf = {}, {}, {}
  #Load test data
  pathFalse = fake_path_notb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_NTB/TEST/*.png"
  pathTrue = fake_path_tb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_TB/TEST/*.png"
  sizePathFalse = 135
  sizePathTrue = 132
  datalwf['test'], labellwf['test'], namelwf['test'] = Load_Dataset( True, histogramEqualization, imgwidth, 
                                                          sizePathFalse, pathFalse, sizePathTrue, 
                                                          pathTrue )
  #Load train data
  pathFalse = fake_path_notb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_NTB/TRAIN/*.png"
  pathTrue = fake_path_tb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_TB/TRAIN/*.png"
  sizePathFalse = 136
  sizePathTrue = 133
  datalwf['train'], labellwf['train'], namelwf['train'] = Load_Dataset( True, histogramEqualization, imgwidth, 
                                                              sizePathFalse, pathFalse, sizePathTrue, 
                                                              pathTrue )
  #Data
  #Load val data
  pathFalse = fake_path_notb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_NTB/VAL/*.png"
  pathTrue = fake_path_tb + str( ifold ) + ".sort_" + str( jfold ) + "/p2p_TB/VAL/*.png"
  sizePathFalse = 134
  sizePathTrue = 131
  datalwf['val'], labellwf['val'], namelwf['val'] = Load_Dataset( True, histogramEqualization, imgwidth, 
                                                          sizePathFalse, pathFalse, sizePathTrue, 
                                                          pathTrue )
  return datalwf, labellwf, namelwf


def Load_partition_task( infoTestFold, namesImg ):
  #load partition from tasks that have been previously loaded and partitioned.
  
  a_file = open( infoTestFold, "rb")
  testNames = pickle.load(a_file)
  a_file.close()

  #Divide in train and test as the previously saved
  print( "Len testNames = " + str( len( testNames ) ))
  Id_test_task = []
  for iFold in range( len( testNames ) ):
    Id_test_task_fold = []
    for name in testNames[ iFold ] :
      idImg = namesImg.tolist().index( name )
      Id_test_task_fold.append( idImg )
    Id_test_task.append( Id_test_task_fold )

  return Id_test_task 

def Load_Partition_Otto( partition_path ):
  a_file = open(partition_path, "rb")
  partition = pickle.load(a_file)
  a_file.close()

  return partition

def idSpec( valueSpec, deltaRes ):
#Dado o array de especificidades obtidas com os deltas, retorna indice do valor de especificidade logo abaixo do
#que está sendo procurado
  for id in range( len( deltaRes.folder ) ):
    if deltaRes.folder[ id ].spec > valueSpec:
      return max(id - 1, 0)
  return len( deltaRes.folder ) - 1

def interpola( targvalue, spec1, spec2, sens1, sens2 ):
#Dado o valor de especificidade procurado, faz uma interpolação da sensibilidade entre 2 valores de especificidades que foram obtidos.
#Exemplo: Se, com os deltas que foram usados, obtive os seguintes valores (entre vários outros):
# espec 0.73 e sens 0.88
# espec 0.77 e sens 0.86
# Mas quero saber quanto é a sensibilidade quando a especificidade é de 0.75. Essa interpolacao calcula e retorna o valor de 0.87 neste exemplo.
  if spec2 > spec1:
    sensInterp = sens1 + (sens2 - sens1) * ( targvalue - spec1) / ( spec2 - spec1 )
  else:
    sensInterp = sens2
  return sensInterp

def Calcula_Delta_Operacao( CTB_Model, deltas, sens90 ):
  sens = []
  sp = []
  for idelta in range( len( CTB_Model.folder ) ):
    sens.append( CTB_Model.folder[ idelta ].sens )
    sp.append( CTB_Model.folder[ idelta ].sp )
    
  ideltamax, maxsp = 0, 0    
  for idelta in range( len( sens ) ):
    if ((sens90 and sens[ idelta ] > 0.9) or not sens90)  and sp[ idelta ] > maxsp:
        ideltamax, maxsp = idelta, sp[ idelta ]
  
  delta_op = deltas[ ideltamax ]
  #print("No ponto de operacao, sens = " + str( sensMean[ ideltamax ] ) + ", sp = " + str( spMean[ ideltamax]) + " e idelta = " + str( ideltamax))
  return delta_op

def Calcula_Delta_Operacao_Output( label, output, value, mode = 'sens' ):
      orderedDelta = []
      pos, delta = '', ''
      for i in range( len( label ) ):
        for j in range( len( label[ i ] ) ):
          pos = -1
          if mode == 'sens' and label[ i ][ j ] == True:
            pos = len( orderedDelta )
            delta = output[ i ][ j ][ 1 ] - output[ i ][ j ][ 0 ]
            #print("delta = " + str(delta))
          
          elif mode == 'spec' and label[ i ][ j ] == False:
            pos = len( orderedDelta )
            delta = output[ i ][ j ][ 0 ] - output[ i ][ j ][ 1 ]

          if pos >= 0: #significa que entrou em uma das 2 condições acima  
            for iOrd in range( len( orderedDelta ) ):              
              if delta > orderedDelta[ iOrd ]:
                pos = iOrd 
                break
            #print("pos = " + str(pos))
            orderedDelta.insert( pos, delta )
            #print("Ordered delta:")
            #print( orderedDelta )

      if len( orderedDelta ) > 0:
        ind = int( value * len( orderedDelta ) )
        delta = orderedDelta[ ind ]
      else:
        delta = 0.0
      
      print("Delta corte = " + str( delta ) )
      
      return delta

def Calcula_ROC( modelname, specDiscrete,
                 datalwf, labellwf, namelwf, ndeltas, ilayer = -1  ):
  
  sp_aux = []
  sens_aux = []
  spec_aux = []
  deltas = []
  CTB = Modeltasks ( ndeltas )
   
  #Varia valores de delta que fazem uma imagem ser classificada como TB. Tenho 2 neurônios, um para TB e outro para não TB. Chamando a saída do
  # neurônio de TB de SaidaTB e do neurônio de não TB de SaidaNTB, se SaidaTB > SaidaNTB + delta, classifico a imagem como TB. Faço para 120 deltas.
  output_all, labels_all, names_all, aux = eval_test( True, modelname, datalwf[ 'val' ] , labellwf[ 'val' ],  
                namelwf[ 'val' ], ilayer = ilayer )
  count = 0
  for spec in specDiscrete:
    delta = -1 * Calcula_Delta_Operacao_Output( labels_all, output_all, spec, mode = 'spec' )
    Calculate_Classifications_eval( output_all, labels_all, names_all, CTB.folder[ count ], delta)
    deltas.append( delta )
    count += 1

  #Obtenho aqui valores de sens, espec e SP que são consequências dos deltas que escolhi. Não controlo diretamente.
  
  
  #Definicao dos valores de sensibilidade correspondentes às especificidades discretas, através de interpolação. Para cada um dos 10 modelos.
  sensAux = []
  idmax = ndeltas - 1
  for value in specDiscrete:
    id = idSpec( value, CTB )
    #print("spec value = " + str( value ) )
    #print("spec before = " + str( CTB_Model[ imodel ].folder[ id ].spec ))
    #print("spec after= " + str(CTB_Model[ imodel ].folder[ min( id + 1, idmax) ].spec))
    sensValue = interpola( value, CTB.folder[ id ].spec, CTB.folder[ min( id + 1, idmax) ].spec, CTB.folder[ id ].sens, CTB.folder[ min( id + 1, idmax) ].sens )
    #print("sens value= " + str(sensValue))
    sensAux.append( sensValue )
  
  return sensAux, CTB, deltas
    
def Plota_ROC( sensDiscrete, specDiscrete, namegraf,  linhaoms = True ):
  specDiscrete = np.array( specDiscrete )
  sensDiscrete = np.array( sensDiscrete )
  sensRocMean = np.mean( sensDiscrete, 0 )
  sensStdMean = np.std( sensDiscrete, 0 )
  sensMax = sensRocMean + sensStdMean
  sensMin = sensRocMean - sensStdMean

  ones = np.array( [ 1 ] * len( specDiscrete ) )
  plt.clf()
  plt.fill_between( ones - specDiscrete, sensMin, sensMax, color="grey"  )
  plt.plot( ones - specDiscrete, sensRocMean, 'b-')
  plt.axis([0, 1, 0, 1])
  #Plot das linhas que definem os limites de sens e espec definidos pela OMS
  if ( linhaoms ):
    plt.plot( [ 0, 1 ], [ 0.9, 0.9 ], 'k--')
    plt.plot( [ 0.3, 0.3 ], [ 0, 1 ], 'k--')
  plt.savefig( namegraf )
  
def Show_img( name,  X_cxr, namesImg, img_file ):
    id = list(namesImg).index( name )
    img_array = X_cxr[id]
    img_array = img_array.transpose(2,1,0)
    plt.close()
    plt.imshow(img_array)
    plt.savefig(img_file)

def Calcula_Quadrante( V1, F1, nome1, V2, F2, nome2, namegraf, classe, nomes_ranking = False ):
  
  Right_common = V1.intersection(V2)
  Right_only1 = V1 - Right_common
  Right_only2 = V2 - Right_common
  Wrong_common = F1.intersection(F2)
  if nomes_ranking:
    Right_common = Right_common.intersection( nomes_ranking )
    Right_only1 = Right_only1.intersection( nomes_ranking )
    Right_only2 = Right_only2.intersection( nomes_ranking )
    Wrong_common = Wrong_common.intersection( nomes_ranking )

  len1 = len(nome1)
  len2 = len(nome2)
  if classe:
    V = 'VP'
    F = 'FN'
    print("Classe verdadeiro")
  else:
    V = 'VN'
    F = 'FP'
    print("Classe false")

  plt.clf()
  #linhas verticais
  plt.plot( [ 0, 0 ], [ 0, 1 ], 'k-')
  plt.plot( [ 0.2, 0.2 ], [ 0, 1 ], 'k-')
  plt.plot( [ 0.5, 0.5 ], [ 0, 1.5 ], 'k-')
  plt.plot( [ 1.0, 1.0 ], [ 0, 1.3 ], 'k-')
  plt.plot( [ 1.5, 1.5 ], [ 0, 1.5 ], 'k-')
  #linhas horizontais
  plt.plot( [ 0, 1.5 ], [ 0, 0 ], 'k-')
  plt.plot( [ 0.2, 1.5 ], [ 0.5, 0.5 ], 'k-')
  plt.plot( [ 0.5, 0.5 ], [ 0.2, 1.5 ], 'k-')
  plt.plot( [ 0, 1.5 ], [ 1.0, 1.0 ], 'k-')
  plt.plot( [ 0.5, 1.5 ], [ 1.3, 1.3 ], 'k-')
  plt.plot( [ 0.5, 1.5 ], [ 1.5, 1.5 ], 'k-')
  plt.axis('off')
  #Escreve textos
  plt.text( 1- 0.055 * len1 / 2, 1.35, nome1, fontsize=18 )
  plt.text(0.7, 1.1, V, fontsize=18 )
  plt.text(1.2, 1.1, F, fontsize=18 )
  plt.text(0.05, 0.5 - 0.055 * len2 / 2, nome2, fontsize=16, rotation = 'vertical'  )
  plt.text( 0.3, 0.7, V, fontsize=18 )
  plt.text(0.3, 0.2, F, fontsize=18 )
  plt.text(0.7, 0.7, str(len(Right_common)), fontsize=16 )
  plt.text(1.2, 0.7, str(len(Right_only2)), fontsize=16 )
  plt.text(0.7, 0.2, str(len(Right_only1)), fontsize=16 )
  plt.text(1.2, 0.2, str(len(Wrong_common)), fontsize=16 )
  print("Nome 1: " + str(nome1))
  print("Certos apenas 1: " + str(len(Right_only1)))
  print("Nome 2: " + str(nome2))
  print("Certos apenas 2: " + str(len(Right_only2)))
  plt.savefig( namegraf + str(classe) + '.png' )

  return Right_common, Right_only1, Right_only2, Wrong_common

def Salva_Imagens_Quadrantes_em_Pastas( Right_common, Right_only1, Right_only2, Wrong_common, datatask, namesimg, imgs_quadrante_path, reports ):
  
  from io import StringIO
  
  if not os.path.exists(imgs_quadrante_path + 'Imgs_Acerto_Ambos/'): 
    os.makedirs(imgs_quadrante_path + 'Imgs_Acerto_Ambos/') 
  if not os.path.exists(imgs_quadrante_path + 'Imgs_Acerto1_Erro2/'): 
    os.makedirs(imgs_quadrante_path + 'Imgs_Acerto1_Erro2/') 
  if not os.path.exists(imgs_quadrante_path + 'Imgs_Acerto2_Erro1/'): 
    os.makedirs(imgs_quadrante_path + 'Imgs_Acerto2_Erro1/')
  if not os.path.exists(imgs_quadrante_path + 'Imgs_Erro_Ambos/'): 
    os.makedirs(imgs_quadrante_path + 'Imgs_Erro_Ambos/')  
  
  for nameimg in Right_common:
    Show_img( nameimg,  datatask, namesimg, imgs_quadrante_path + 'Imgs_Acerto_Ambos/' + nameimg )
  for nameimg in Right_only1:
    Show_img( nameimg,  datatask, namesimg, imgs_quadrante_path + 'Imgs_Acerto1_Erro2/' + nameimg )
  for nameimg in Right_only2:
    Show_img( nameimg,  datatask, namesimg, imgs_quadrante_path + 'Imgs_Acerto2_Erro1/' + nameimg )
  for nameimg in Wrong_common:
    Show_img( nameimg,  datatask, namesimg, imgs_quadrante_path + 'Imgs_Erro_Ambos/' + nameimg )

  df = pd.read_csv(reports, sep=';')  

  Right_common_sem_png = [name.replace('.png', '') for name in Right_common]
  mask = df['image'].apply(lambda x: any(substring in x for substring in Right_common_sem_png))
  filtered_df = df[mask]
  filtered_df.to_csv(imgs_quadrante_path + 'Imgs_Acerto_Ambos/report_Acerto_Ambos.csv', index=False)

  Right_only1_sem_png = [name.replace('.png', '') for name in Right_only1]
  mask = df['image'].apply(lambda x: any(substring in x for substring in Right_only1_sem_png))
  filtered_df = df[mask]
  filtered_df.to_csv(imgs_quadrante_path + 'Imgs_Acerto1_Erro2/report_Acerto1_Erro2.csv', index=False)

  Right_only2_sem_png = [name.replace('.png', '') for name in Right_only2]
  mask = df['image'].apply(lambda x: any(substring in x for substring in Right_only2_sem_png))
  filtered_df = df[mask]
  filtered_df.to_csv(imgs_quadrante_path + 'Imgs_Acerto2_Erro1/report_Acerto2_Erro1.csv', index=False)

  Wrong_common_sem_png = [name.replace('.png', '') for name in Wrong_common]
  mask = df['image'].apply(lambda x: any(substring in x for substring in Wrong_common_sem_png))
  filtered_df = df[mask]
  filtered_df.to_csv(imgs_quadrante_path + 'Imgs_Erro_Ambos/report_Erro_Ambos.csv', index=False)


def Salva_Imagens_em_Pastas( lista_imgs_na_pasta, nome_csv, datatask, namesimg, save_path, reports ):
  
  if not os.path.exists( save_path ): 
    os.makedirs( save_path ) 

  for nameimg in lista_imgs_na_pasta:
    Show_img( nameimg,  datatask, namesimg, save_path + nameimg )

  df = pd.read_csv(reports, sep=';')  

  lista_imgs_sem_png = [name.replace('.png', '') for name in lista_imgs_na_pasta]
  mask = df['image'].apply(lambda x: any(substring in x for substring in lista_imgs_sem_png ))
  filtered_df = df[mask]
  filtered_df.to_csv(save_path + nome_csv, index=False)