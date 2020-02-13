import autograd.numpy as np
import pyriemann as riemann
from pymanopt.manifolds import Rotations
from pymanopt.solvers import SteepestDescent
from pymanopt import Problem
from functools import partial
import sklearn as learn
import time

"""
Structure des données : 
data = {'covs': n_trials x (n_features x n_features),
        'labels' : n_trials ('right', 'left', None si non étiquetté)
}
"""

def geometric_mean(covs) :
    return riemann.utils.mean.mean_riemann(covs)

def recenter(cov, mean) :
    inv_mean = riemann.utils.base.invsqrtm(mean)
    return np.dot(inv_mean, np.dot(cov, inv_mean))

def dispersion(covs) :
    disp = 0
    for i in range(covs.shape[0]) :
        disp += np.power(riemann.utils.distance.distance_riemann(covs[i], np.eye(covs.shape[1])), 2)
    return disp/covs.shape[0]
        

def stretch(cov, dispersion) :
    return riemann.utils.base.powm(cov, dispersion)


def cost_function(cov_source, cov_target, U):
    t1 = cov_target
    t2 = np.dot(U, np.dot(cov_source, U.T))
    return np.linalg.norm(t1 - t2)**2

def cost_function_full(mean_source, mean_target_train, U) :
    C = 0
    for M1, M2 in zip(np.array(mean_source), np.array(mean_target_train)) :
        C += cost_function(M1, M2, U)
    return np.array(C)
    
def rotation_matrix(mean_source, mean_target_train) :
    manifold = Rotations(mean_source[0].shape[0])
    cost = partial(cost_function_full, mean_source, mean_target_train)
    problem = Problem(manifold, cost)
    solver = SteepestDescent(mingradnorm = 1e-3)
    U = solver.solve(problem)
    return U


   
def RPA_recenter(source, target_train, target_test) :
    source_rct, target_train_rct, target_test_rct = {}, {}, {}
    source_rct['labels'] = source['labels']
    target_train_rct['labels'] = target_train['labels']
    target_test_rct['labels'] = target_test['labels']
    
    mean_source = geometric_mean(source['covs'])
    source_rct['covs'] = np.array([recenter(cov, mean_source) for cov in source['covs']])
    mean_target_train = geometric_mean(target_train['covs'])
    target_train_rct['covs'] = np.array([recenter(cov, mean_target_train) for cov in target_train['covs']])
    mean_target_test = geometric_mean(target_test['covs'])
    target_test_rct['covs'] = np.array([recenter(cov, mean_target_test) for cov in target_test['covs']])
    return(source_rct, target_train_rct, target_test_rct)

def RPA_stretch(source, target_train, target_test) :
    target_train_str, target_test_str = {}, {}
    target_train_str['labels'] = target_train['labels']
    target_test_str['labels'] = target_test['labels']

    disp_source = dispersion(source['covs'])
    disp_target = dispersion(target_train['covs'])
    target_train_str['covs'] = np.array([stretch(cov, np.sqrt(disp_source/disp_target)) for cov in target_train['covs']])
    target_test_str['covs'] = np.array([stretch(cov,  np.sqrt(disp_source/disp_target)) for cov in target_test['covs']])
    return(source, target_train_str, target_test_str)

    
    


def RPA_rotate(source, target_train, target_test) :
    class_labels = np.unique(source['labels'])
    mean_source = []
    mean_target_train = []
    for i in class_labels :
        mean_source.append(geometric_mean(source['covs'][source['labels'] == i]))
        mean_target_train.append(geometric_mean(target_train['covs'][target_train['labels'] == i]))
    
    R = rotation_matrix(mean_source, mean_target_train)
    
    target_train_rot, target_test_rot = {}, {}
    target_train_rot['labels'] = target_train['labels']
    target_test_rot['labels'] = target_test['labels']
    
    target_train_rot['covs'] = np.array([np.dot(R.T, np.dot(cov, R)) for cov in target_train['covs']])
    target_test_rot['covs'] = np.array([np.dot(R.T, np.dot(cov, R)) for cov in target_test['covs']])
    return(source, target_train_rot, target_test_rot)
    
def RPA(source, target_train, target_test) :
    t = time.time()
    source_rct, target_train_rct, target_test_rct = RPA_recenter(source, target_train, target_test)
    print('Recentering took ', time.time() - t, 's.')
    t = time.time()
    source_str, target_train_str, target_test_str = RPA_stretch(source, target_train, target_test)
    print('Stretching took ', time.time() - t, 's.')
    t = time.time()
    source_rot, target_train_rot, target_test_rot = RPA_rotate(source, target_train, target_test)
    print('Rotating took ', time.time() - t, 's.')
    return(source_rot, target_train_rot, target_test_rot)

def get_dataset(raw_data, label, id_patient, older_labeled_size = 0.2 , session_labeled_size = 0.2) :
    id_patient = id_patient - 1
    
    epochs_source, epochs_target, source_label, target_label = learn.model_selection.train_test_split(
        raw_data[id_patient], label[id_patient], train_size = older_labeled_size)
    epochs_target_train, epochs_target_test, target_train_label, target_test_label = learn.model_selection.train_test_split(
        epochs_target, target_label, train_size = session_labeled_size)
    
    source, target_train, target_test = {}, {}, {}
    source['labels'] = source_label
    target_train['labels'] = target_train_label
    target_test['labels'] = target_test_label
    
    source['covs'] = np.array([np.cov(epoch) for epoch in epochs_source])
    target_train['covs'] = np.array([np.cov(epoch) for epoch in epochs_target_train])
    target_test['covs'] = np.array([np.cov(epoch) for epoch in epochs_target_test])
    
    return(source, target_train, target_test)  
    
