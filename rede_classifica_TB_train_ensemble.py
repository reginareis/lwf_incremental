import csv
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from functions_apply_lwf import *


#Lê dados de output, nomes e label
#namecase = '1StanfordColab'
#namecase = '1Real_v3'
#namecase = '1Real2Pix_allshared_v2'
#namecase = '1Real2Pix_soFC_v2'
#namecase = '1Real2Wgan_allshared'
#namecase = '1Real2Wgan_soFC'
#namecase = '1AlltogetherWGAN'
#namecase = '1AlltogetherPix'
#namecase = '1StanfordBloco'
#namecase = '2ShenzhenBloco'
#namecase = '2Real_3Manaus'
#namecase = '1Stanford'
#namecase = '1Stanford2Shenzhen'
#namecase = '1Stanford_lr1e4'
#namecase = '2WganSoFC_3Manaus'
#namecase = '2WganSoFC_3Manaus_4SCasa'
#namecase = '2Real_3Manaus_4SCasaCycleShenzhen'
#namecase = '2Real_3Manaus_4SCasaCycleManaus'
#namecase = '2Real_3Manaus_4SCasaCycleAll'



def Le_Picke_Outputs( taskData, ilayer, namecase ):
    path = '/home/regina.alves/results/lwf_res/SENS90/'      
    sufixo = ""
    modelpath = path + namecase + '/'
    results_file = modelpath + 'resultados_data' + taskData + 'caso' + namecase + '_ilayer' + str(ilayer) + sufixo + '.pkl'
    #print("Abrindo " + results_file)
    r_file = open( results_file, "rb")
    info = pickle.load(r_file)
    r_file.close()

    return info['output_last_train'], info['labels_last_train'], info['names_last_train'], info['output_last_val'], info['labels_last_val'], info['names_last_val'], info['output_last_test'], info['labels_last_test'], info['names_last_test'] 

#Carrega dados
def Preenche_Outputs( outputs, labels, names, id_task, itask, output_ilayers, labels_ilayer, names_ilayer, itest, fracao = 1.0 ):
    nlayers = len(output_ilayers)
    #print("nlayers = " + str(nlayers))
    #print( "len( labels_ilayer[itest] ) = " + str(len( labels_ilayer[itest] ) ))
    #print( "len( labels_ilayer[itest][0] ) = " + str(len( labels_ilayer[itest][0] ) ))
    #print( "len( output_ilayers[ ilay=0 ][ itest ][0] ) = " + str(len(output_ilayers[ 0 ][ itest ][0] ) ))
    #calcula total de imagens
    total = 0
    for j in range( len( labels_ilayer[ itest ] ) ): 
        for k in range( len( labels_ilayer[ itest ][ j ] ) ): 
            total += 1
    #calcula qtde de imagens para pegar
    n_subamostra = total * fracao
    #balanceia por classe
    n_sub_classe = n_subamostra/2
    count_true = 0
    count_false = 0
    #print("total = " + str(total))
    #print("n_subamostra = " + str(n_subamostra))
    for j in range( len( labels_ilayer[ itest ] ) ): 
        for k in range( len( labels_ilayer[ itest ][ j ] ) ): 
            if labels_ilayer[ itest ][ j ][ k ] == 0:
                if count_false > n_sub_classe:
                    continue
                count_false += 1
            elif labels_ilayer[ itest ][ j ][ k ] == 1:
                if count_true > n_sub_classe:
                    continue
                count_true += 1

            labels.append( labels_ilayer[ itest ][ j ][ k ] )
            names.append( names_ilayer[ itest ][ j ][ k ] )
            output_aux = np.zeros( 2*nlayers )
            id_task.append( itask )
            for ilay in range(nlayers):
                output_aux[ 2*ilay ] = output_ilayers[ ilay ][ itest ][ j ][ k ][ 0 ] 
                output_aux[ 2*ilay + 1 ] = output_ilayers[ ilay ][ itest ][ j ][ k ][ 1 ] 
            outputs.append( output_aux )


def calculate_accuracy(predictions, labels):
    # Convert the predictions to binary values (0 or 1)
    binary_predictions = torch.round(predictions)
    # Compare with ground truth labels
    correct = (binary_predictions == labels).float()
    # Calculate accuracy
    accuracy = correct.sum() / len(correct)
    return accuracy.item()

def calculate_indexes_only( VP, VN, FP, FN ):
    if len(VP) + len(FN) > 0:
        sens = len(VP) / (len(VP) + len(FN))
    else:
        sens = 0
    if len(VN) + len(FP) > 0:
        spec = len(VN) / (len(VN) + len(FP))
    else:
        spec = 0
    arit = ( sens + spec )/2
    geom = np.sqrt( sens * spec )
    sp = np.sqrt( arit * geom )

    return sens, spec, sp

def calculate_indexes_general( Class_itask, itest, nimgs_itask ):
    ntasks = len(nimgs_itask) 
    VP, VN, FP, FN = 0, 0, 0, 0
    for itask in range(ntasks):
        VP += len(Class_itask[itask].folder[itest].VP)/nimgs_itask[itask]
        VN += len(Class_itask[itask].folder[itest].VN)/nimgs_itask[itask]
        FP += len(Class_itask[itask].folder[itest].FP)/nimgs_itask[itask]
        FN += len(Class_itask[itask].folder[itest].FN)/nimgs_itask[itask]

    if VP + FN > 0:
        sens = VP / (VP + FN)
    else:
        sens = 0
    if VN + FP > 0:
        spec = VN / (VN + FP)
    else:
        spec = 0
    arit = ( sens + spec )/2
    geom = np.sqrt( sens * spec )
    sp = np.sqrt( arit * geom )


    return sens, spec, sp


def calculate_indexes_ensemble(predictions, labels, names, deltas, id_task_val = False ):
    # Convert the predictions to binary values (0 or 1)
    binary_predictions = []
    ntasks = len( deltas )
    n_imgs = len(labels)
    for iimg in range(n_imgs):
        if id_task_val and id_task_val[iimg] < ntasks -1:
            continue
        count_predictions = 0 #add the predictions of each model, which are 0 or 1
        for itask in range( ntasks ):
            if predictions[ iimg ][ itask *2 ] <=  predictions[ iimg ][ itask *2 + 1 ]+ deltas[ itask ]: #model classified as positive
                count_predictions += 1
        if count_predictions >= ntasks/2:
            binary_predictions.append( 1 ) #majority of models voted for positive
        else:
            binary_predictions.append( 0 )
    
    # Compare with ground truth labels
    VP, VN, FP, FN = [], [], [], []
    i = -1
    for iimg in range(n_imgs):
        if id_task_val and id_task_val[iimg] < ntasks -1:
            continue
        i+=1
        if int(labels[ iimg ]) == 0: #false
            if binary_predictions[ i ] == labels[ iimg ]: #VN
                VN.append( names[iimg] )
            else: #FP
                FP.append( names[iimg] )
        elif int(labels[ iimg ]) == 1: #true
            if binary_predictions[ i ] == labels[ iimg ]: #VP
                VP.append( names[iimg] )
            else: #FN
                FN.append( names[iimg] )
    sens, spec, sp = calculate_indexes_only( VP, VN, FP, FN )
    
    return sens, spec, sp, VP, VN, FP, FN

def calculate_indexes(predictions, labels, names, delta):
    # Convert the predictions to binary values (0 or 1)
    binary_predictions = torch.round(predictions + delta)
    # Compare with ground truth labels
    ndata = len(labels)
    VP, VN, FP, FN = [], [], [], []
    for i in range( ndata ):
        if int(labels[ i ]) == 0: #false
            if binary_predictions[ i ] == labels[ i ]: #VN
                VN.append( names[i] )
            else: #FP
                FP.append( names[i] )
        elif int(labels[ i ]) == 1: #true
            if binary_predictions[ i ] == labels[ i ]: #VP
                VP.append( names[i] )
            else: #FN
                FN.append( names[i] )
    sens, spec, sp = calculate_indexes_only( VP, VN, FP, FN )
    
    return sens, spec, sp, VP, VN, FP, FN



    correct = (binary_predictions == labels).float()
    # Calculate accuracy
    accuracy = correct.sum() / len(correct)
    return accuracy.item()

def Treina_para_1_Conjunto( itest, fold_train, fold_val, fold_test, taskDatas, ilayer_tasks, namecase ):

    #Number of taskDatas
    ntasks = len(taskDatas)

    #estruturas que receberão os dados de shenzhen, manaus e santacasa
    outputs_train, labels_train, names_train, id_task_train = [], [], [], []
    outputs_val, labels_val, names_val, id_task_val = [], [], [], []
    outputs_test, labels_test, names_test, id_task_test = [], [], [], []
    #estruturas para dados de cada task
    for itask in range(ntasks):
        output_ilayers_train, output_ilayers_val, output_ilayers_test = [], [], []
        taskData = taskDatas[itask]
        for ilayer in ilayer_tasks:
            output_last_train, labels_last_train, names_last_train, output_last_val, labels_last_val, names_last_val, output_last_test, labels_last_test, names_last_test = Le_Picke_Outputs( taskData, ilayer, namecase )

            output_ilayers_train.append(output_last_train)
            output_ilayers_val.append(output_last_val)
            output_ilayers_test.append(output_last_test)
        Preenche_Outputs( outputs_train, labels_train, names_train, id_task_train, itask, output_ilayers_train, labels_last_train, names_last_train, itest, fracao = 0.1 )
        Preenche_Outputs( outputs_val, labels_val, names_val, id_task_val, itask, output_ilayers_val, labels_last_val, names_last_val, itest )    
        Preenche_Outputs( outputs_test, labels_test, names_test, id_task_test, itask, output_ilayers_test, labels_last_test, names_last_test, itest )
    

    #define deltas para cada task

    deltas_aux = []
    delta = 0.0
    for i in range( 50 ):
        deltas_aux.append( delta )
        delta += 0.1
    best_sens, best_spec, best_sp, best_delta = 0.0, 0.0, 0.0, 0.0
    best_sens_train, best_spec_train, best_sp_train = 0.0, 0.0, 0.0
    best_sens_test, best_spec_test, best_sp_test = 0.0, 0.0, 0.0

    for delta1 in deltas_aux:
        for delta2 in deltas_aux:
            for delta3 in deltas_aux:
                deltas = [ delta1, delta2, delta3 ]

 
                #sens_onlylast, spec, sp_onlylast, VP, VN, FP, FN = calculate_indexes_ensemble( outputs_val, labels_val, names_val, deltas, id_task_val )
                sens, spec, sp, VP, VN, FP, FN = calculate_indexes_ensemble( outputs_val, labels_val, names_val, deltas )
                
                sens_train, spec_train, sp_train, VP_train, VN_train, FP_train, FN_train = calculate_indexes_ensemble( outputs_train, labels_train, names_train, deltas )
                sens_test, spec_test, sp_test, VP_test, VN_test, FP_test, FN_test = calculate_indexes_ensemble( outputs_test, labels_test, names_test, deltas )
                print("Para delta = " + str(delta)  )
                print("Sens = " + str(sens) + " Spec = " + str(spec) + " SP = " + str(sp))
                print(sens)
                if sens > 0.9 and sp > best_sp:
                    best_sens, best_spec, best_sp, best_delta, fold_val.VP, fold_val.VN, fold_val.FP, fold_val.FN  = sens, spec, sp, deltas, VP, VN, FP, FN 
                    best_sens_train, best_spec_train, best_sp_train, fold_train.VP, fold_train.VN, fold_train.FP, fold_train.FN = sens_train, spec_train, sp_train, VP_train, VN_train, FP_train, FN_train 
                    best_sens_test, best_spec_test, best_sp_test, fold_test.VP, fold_test.VN, fold_test.FP, fold_test.FN = sens_test, spec_test, sp_test, VP_test, VN_test, FP_test, FN_test 
    #print(f' Sens: {best_sens:.4f}, Spec: {best_spec:.4f}, SP: {best_sp:.4f}, delta1: {best_delta[0]:.4f}, delta2: {best_delta[1]:.4f}, delta3: {best_delta[2]:.4f}')
    

    print("Best sp val = " + str(best_sp))
    print("Best sp train = " + str(best_sp_train))
    print("Best sp test = " + str(best_sp_test))
    print("Deltas = " + str(best_delta))

    return fold_train, fold_val, fold_test, names_train, names_val, names_test, id_task_train, id_task_val, id_task_test

def Insere_imagens_fold_task( folder_set_geral, folder_set_task, names, id_task, itask ):
    for iimg in range( len( folder_set_geral ) ):
            #acha indice na lista de imagens
            name = folder_set_geral[ iimg ]
            id = names.index( name )
            if id_task[ id ] == itask:
                folder_set_task.append( name )

def Roda( namecase, taskDatas, id ):
    if id > 0:
        csvOld_medias = path + 'RMediasTB_EnsembleWgan' + str(id) + '.csv'
    else:
        csvOld_medias = ''
    csvNew_medias = path + 'RMediasTB_EnsembleWgan' + str(id+1) + '.csv'
    pickle_model_path = path + namecase + '/'
    Lista_csv = []
    Results_header = []
    Class_Train = Modeltasks( nkfolds )
    Class_Val = Modeltasks( nkfolds )
    Class_Test = Modeltasks( nkfolds )
    ntasks = len( taskDatas )
    Class_Train_itask = []
    Class_Val_itask = []
    Class_Test_itask = []
    
    for itask in range(ntasks):
        Class_Train_itask.append( Modeltasks( nkfolds ) )
        Class_Val_itask.append( Modeltasks( nkfolds ) )
        Class_Test_itask.append( Modeltasks( nkfolds ) )
        

    for itest in range( nkfolds ):
        print("Itest = " + str(itest))
        Class_Train.folder[ itest ], Class_Val.folder[ itest ], Class_Test.folder[ itest ], names_train, names_val, names_test, id_task_train, id_task_val, id_task_test  = Treina_para_1_Conjunto( itest, Class_Train.folder[ itest ], Class_Val.folder[ itest ], Class_Test.folder[ itest ], taskDatas, ilayer_tasks, namecase )
        nimgs_train_itask, nimgs_val_itask, nimgs_test_itask = [], [], [] 

        for itask in range(ntasks):
            nimgs_train_itask.append( id_task_train.count(itask))
            nimgs_val_itask.append( id_task_val.count(itask))
            nimgs_test_itask.append( id_task_test.count(itask))

        #Separa resultados por task
        for itask in range(ntasks):
            Insere_imagens_fold_task( Class_Train.folder[ itest ].VP, Class_Train_itask[ itask ].folder[ itest ].VP, names_train, id_task_train, itask )
            Insere_imagens_fold_task( Class_Train.folder[ itest ].VN, Class_Train_itask[ itask ].folder[ itest ].VN, names_train, id_task_train, itask )
            Insere_imagens_fold_task( Class_Train.folder[ itest ].FP, Class_Train_itask[ itask ].folder[ itest ].FP, names_train, id_task_train, itask )
            Insere_imagens_fold_task( Class_Train.folder[ itest ].FN, Class_Train_itask[ itask ].folder[ itest ].FN, names_train, id_task_train, itask )
            Insere_imagens_fold_task( Class_Val.folder[ itest ].VP, Class_Val_itask[ itask ].folder[ itest ].VP, names_val, id_task_val, itask )
            Insere_imagens_fold_task( Class_Val.folder[ itest ].VN, Class_Val_itask[ itask ].folder[ itest ].VN, names_val, id_task_val, itask )
            Insere_imagens_fold_task( Class_Val.folder[ itest ].FP, Class_Val_itask[ itask ].folder[ itest ].FP, names_val, id_task_val, itask )
            Insere_imagens_fold_task( Class_Val.folder[ itest ].FN, Class_Val_itask[ itask ].folder[ itest ].FN, names_val, id_task_val, itask )
            Insere_imagens_fold_task( Class_Test.folder[ itest ].VP, Class_Test_itask[ itask ].folder[ itest ].VP, names_test, id_task_test, itask )
            Insere_imagens_fold_task( Class_Test.folder[ itest ].VN, Class_Test_itask[ itask ].folder[ itest ].VN, names_test, id_task_test, itask )
            Insere_imagens_fold_task( Class_Test.folder[ itest ].FP, Class_Test_itask[ itask ].folder[ itest ].FP, names_test, id_task_test, itask )
            Insere_imagens_fold_task( Class_Test.folder[ itest ].FN, Class_Test_itask[ itask ].folder[ itest ].FN, names_test, id_task_test, itask )

            #Calcula indices
            Class_Train_itask[ itask ].folder[ itest ].sens, Class_Train_itask[ itask ].folder[ itest ].spec, Class_Train_itask[ itask ].folder[ itest ].sp = calculate_indexes_only( Class_Train_itask[ itask ].folder[ itest ].VP, Class_Train_itask[ itask ].folder[ itest ].VN, Class_Train_itask[ itask ].folder[ itest ].FP, Class_Train_itask[ itask ].folder[ itest ].FN )
            Class_Val_itask[ itask ].folder[ itest ].sens, Class_Val_itask[ itask ].folder[ itest ].spec, Class_Val_itask[ itask ].folder[ itest ].sp = calculate_indexes_only( Class_Val_itask[ itask ].folder[ itest ].VP, Class_Val_itask[ itask ].folder[ itest ].VN, Class_Val_itask[ itask ].folder[ itest ].FP, Class_Val_itask[ itask ].folder[ itest ].FN )
            Class_Test_itask[ itask ].folder[ itest ].sens, Class_Test_itask[ itask ].folder[ itest ].spec, Class_Test_itask[ itask ].folder[ itest ].sp = calculate_indexes_only( Class_Test_itask[ itask ].folder[ itest ].VP, Class_Test_itask[ itask ].folder[ itest ].VN, Class_Test_itask[ itask ].folder[ itest ].FP, Class_Test_itask[ itask ].folder[ itest ].FN )

        #Calcula índice geral, balanceando tamanho dos conjuntos das tasks
        Class_Train.folder[ itest ].sens, Class_Train.folder[ itest ].spec, Class_Train.folder[ itest ].sp = calculate_indexes_general( Class_Train_itask, itest, nimgs_train_itask ) 
        Class_Val.folder[ itest ].sens, Class_Val.folder[ itest ].spec, Class_Val.folder[ itest ].sp = calculate_indexes_general( Class_Val_itask, itest, nimgs_val_itask ) 
        Class_Test.folder[ itest ].sens, Class_Test.folder[ itest ].spec, Class_Test.folder[ itest ].sp = calculate_indexes_general( Class_Test_itask, itest, nimgs_test_itask ) 
    
    Lista_Medias_csv = []
    Medias_csv = {}

    if csvOld_medias != '':
        with open( csvOld_medias, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                Lista_Medias_csv.append( row )

    Medias_csv["Caso"] = namecase
    for itask in range( ntasks ):
        print("1 CALCULA INDICES MEDIOS PARA CONJUNTO TRAIN DA TAREFA " + taskDatas[ itask ])
        sens, spec, sp = Calculate_indexes_mean_std( Class_Train_itask[ itask ] )
        Medias_csv["Train_sens_" + taskDatas[ itask ] ] = sens
        Medias_csv["Train_spec_" + taskDatas[ itask ] ] = spec
        Medias_csv["Train_sp_" + taskDatas[ itask ] ] = sp
        print("2 CALCULA INDICES MEDIOS PARA CONJUNTO VAL DA TAREFA " + taskDatas[ itask ])
        sens, spec, sp = Calculate_indexes_mean_std( Class_Val_itask[ itask ] )
        Medias_csv["Val_sens_" + taskDatas[ itask ] ] = sens
        Medias_csv["Val_spec_" + taskDatas[ itask ] ] = spec
        Medias_csv["Val_sp_" + taskDatas[ itask ] ] = sp
        print("3 CALCULA INDICES MEDIOS PARA CONJUNTO TEST DA TAREFA " + taskDatas[ itask ])
        sens, spec, sp = Calculate_indexes_mean_std( Class_Test_itask[ itask ] )
        Medias_csv["Test_sens_" + taskDatas[ itask ] ] = sens
        Medias_csv["Test_spec_" + taskDatas[ itask ] ] = spec
        Medias_csv["Test_sp_" + taskDatas[ itask ] ] = sp

        Medias_csv["Test_sens_all" + taskDatas[ itask ]], Medias_csv["Test_spec_all" + taskDatas[ itask ]], Medias_csv["Test_sp_all" + taskDatas[ itask ]] = [], [], []
        for imodel in range( nkfolds ):
            Medias_csv["Test_sp_all" + taskDatas[ itask ]] = Class_Test_itask[ itask ] .folder[ imodel ].sp 
            Medias_csv["Test_sens_all" + taskDatas[ itask ]] = Class_Test_itask[ itask ] .folder[ imodel ].sens
            Medias_csv["Test_spec_all" + taskDatas[ itask ]] = Class_Test_itask[ itask ] .folder[ imodel ].spec  

    print("1 CALCULA INDICES MEDIOS PARA CONJUNTO TRAIN GERAL")
    sens, spec, sp = Calculate_indexes_mean_std( Class_Train )
    Medias_csv["Train_sens_geral" ] = sens
    Medias_csv["Train_spec_geral" ] = spec
    Medias_csv["Train_sp_geral" ] = sp
    print("2 CALCULA INDICES MEDIOS PARA CONJUNTO VAL GERAL")
    sens, spec, sp = Calculate_indexes_mean_std( Class_Val )
    Medias_csv["Val_sens_geral" ] = sens
    Medias_csv["Val_spec_geral" ] = spec
    Medias_csv["Val_sp_geral" ] = sp
    print("3 CALCULA INDICES MEDIOS PARA CONJUNTO TEST GERAL")
    sens, spec, sp = Calculate_indexes_mean_std( Class_Test )
    Medias_csv["Test_sens_geral" ] = sens
    Medias_csv["Test_spec_geral" ] = spec
    Medias_csv["Test_sp_geral" ] = sp
        
    Lista_Medias_csv.append( Medias_csv )
    Medias_header = list(Medias_csv.keys())


    #Salva resultados em pickle
    resultados = {'Class_Train': Class_Train, 'Class_Val': Class_Val, 'Class_Test': Class_Test }

    a_file = open( pickle_file, "wb" )
    pickle.dump( resultados, a_file )
    a_file.close()


    with open( csvNew_medias, 'w') as file:
        # Create a CSV dictionary writer and add the student header as field names
        writer = csv.DictWriter(file, fieldnames=Medias_header)
        # Use writerows() not writerow()
        writer.writeheader()
        writer.writerows( Lista_Medias_csv )

namecases =[]
# namecases.append( '2Real_3Manaus_4SCasaCycleManaus' )
# namecases.append( '2Real_3Manaus_4SCasaCycleShenzhen' )
# namecases.append( '2Real_3Manaus_4SCasaCycleAll' )
namecases.append( '2WganSoFC_3ManausWganSoFC_4SCasaCycleManaus' )
namecases.append( '2WganSoFC_3ManausWganSoFC_4SCasaCycleShenzhen' )
namecases.append( '2WganSoFC_3ManausWganSoFC_4SCasaCycleAll' )

id = 0
path = '/home/regina.alves/results/lwf_res/SENS90/'
taskDatas = [ "Shenzhen", "Manaus", "SantaCasa" ] 
ilayer_tasks = [ 2, 3, 4] 
nkfolds = 10

for namecase in namecases:
    csvNew = path + 'ResultsTB_EnsembleWGan_' + namecase + '.csv'
    pickle_file = path + 'ResultsTB_EnsembleWgan_' + namecase + '.pkl'

    print("Roda para namecase = " + namecase)
    Roda( namecase, taskDatas, id )
    id += 1