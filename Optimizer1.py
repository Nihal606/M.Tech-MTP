import scipy.optimize
import numpy as np
import tensorflow as tf

class Opt_lbfgsb:

    def __init__(self, pinn,num, disp, factr=0, pgtol=0, m=50, maxls=50, maxiter=30000):
        # set options
        self.pinn = pinn   
        self.num=num
        self.disp=disp
        # self.traction=traction

        self.factr = factr
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.his_iter = []
        self.his_loss = []


    def set_weights(self, flat_weights):
        # get model weights
        shapes = [ w.shape for w in self.pinn.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.pinn.set_weights(weights)


    def Loss(self, weights):
        # update weights
        self.set_weights(weights)
        
        self.iteration_counter += 1  # Increment the iteration counter
        loss_domain , loss_boundary_D1, grads = self.pinn.loss_grad( self.num,self.disp)  # compute loss and gradients for weights
        # print('Iteration:', self.iteration_counter, 'L_dom =', loss_domain.numpy() , 'L_1 =', loss_boundary_D1.numpy() , 'L_2 =', loss_boundary_D2.numpy() , \
        #       'L_3 =', loss_boundary_D3.numpy() , 'L_4 =', loss_boundary_D4.numpy() , 'L_5 =', loss_boundary_D5.numpy() , 'L_6 =', loss_boundary_D6.numpy() , '')
        
        self.his_iter.append(self.iteration_counter) 
        loss=  loss_domain +  loss_boundary_D1
        print('Iteration:', self.iteration_counter, 'Total_loss =', loss.numpy() , '')
        
        self.his_loss.append(loss)
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')
        return loss, grads

    def fit(self):
        # get initial weights as a flat vector
        initial_weights = np.concatenate([ w.flatten() for w in self.pinn.get_weights() ])
        # optimize the weight vector
        # print('Optimizer: L-BFGS-B (Provided by Scipy package)')
        print('Initializing the framework ...')
        self.iteration_counter = 0  # Initialize the iteration counter
        result = scipy.optimize.fmin_l_bfgs_b(func=self.Loss, x0=initial_weights,line_search='strong_wolfe',
            factr=self.factr, pgtol=self.pgtol, m=self.m, maxls=self.maxls, maxiter=self.maxiter)
        return result , [np.array(self.his_iter), np.array(self.his_loss)]