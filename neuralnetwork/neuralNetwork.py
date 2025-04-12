import numpy as np

class NeuralNetWork:
    
    def __init__(self, input_shape):
        
        # samples x features
        self.input_shape = input_shape
        
        self.weights = dict()
        self.bs = dict()
        self.z = dict()
        self.acts = {'act_0': lambda x:x}
    
        self.layer_shapes = {'0':(self.input_shape[1], None)}
        
        # Number of layers
        self.N_l = 1

        # Number of times the fit method is called
        self.fit_calls = 0
        
    
    #===============================================================================#
    #                setting weights, biases and activation functions               #
    #===============================================================================#
    def _set_weights(self, w):
        
        if w:
            self.weights['w_' + str(self.N_l)] = w
        else:
            self.weights['w_' + str(self.N_l)] = np.random.uniform(-0.1, 0.1, 
                                                                    size=self.layer_shapes[str(self.N_l)])

        
    def _set_biases(self, b):
        
        if b:
            self.bs['b_' + str(self.N_l)] = b
        else:
            self.bs['b_' + str(self.N_l)] = np.ones(shape=(self.layer_shapes[str(self.N_l)][0], 1))
    
    
    def _set_acts(self, act='identity'):
        
        if act!='identity':
            self.acts['act_' + str(self.N_l)] = act
            
        else:
            self.acts['act_'  + str(self.N_l)] = lambda x: x
            self.acts['dact_' + str(self.N_l)] = lambda x: np.ones_like(x)
        
            
    #===============================================================================#
    #                                      add layers                               #
    #===============================================================================#
    def add(self,
            units,
            act='identity',
            w=None,
            b=None):
    
        self.layer_shapes[str(self.N_l)] = (units, self.layer_shapes[str(self.N_l-1)][0])

        # weights for the layer
        #np.random.seed(0 + self.N_l)
        self._set_weights(w)
            
        # bias for the layer
        self._set_biases(b)
            
        # activation functions
        self._set_acts(act)
        
        # update current numbe of layers
        self.N_l += 1
        
        
    #===============================================================================#
    #                                    compile                                    #
    #===============================================================================#
    def compile_model(self,
                      loss,
                      optimizer,
                      metrics=1):
        
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        
    
    #===============================================================================#
    #                               forward propagation                             #
    #===============================================================================#
    def forward_prop(self, X):
        
        self.a = {'a_0': X.reshape(-1,1)}
        
        for l in range(1, self.N_l):
            # weighted sum z_l = W.a(l-1) + b
            self.z['z_' + str(l)] = np.dot(self.weights['w_' + str(l)], self.a['a_' + str(l-1)]) + self.bs['b_' + str(l)]
            
            # a_l = g(z_l) (output for layer l)
            self.a['a_' + str(l)] = self.acts['act_' + str(l)].act(self.z['z_' + str(l)])
            
            
    #===============================================================================#
    #                                  backpropagation                              #
    #===============================================================================#
    def backpropagation(self, y):
        
        # dL -> error for the output layer
        L  = self.N_l-1
        dL = self.loss.dcda(y_true=y, y_pred=self.a['a_' + str(L)])*self.acts['act_' + str(L)].dadz(self.z['z_' + str(L)])

        # dl -> errors in the hidden layers
        self.dl = {f'dl_{L}': dL}
        for l in range(L-1, 0, -1):
            self.dl['dl_' + str(l)] = np.dot(self.weights['w_' + str(l+1)].T, 
                                             self.dl['dl_' + str(l+1)])\
                                            *self.acts['act_' + str(l)].dadz(self.z['z_' + str(l)])
               
        # gradient dCdW and dCdb
        self.gradC = {'dCdW':dict(), 'dCdb':dict()}
        for l in range(1, self.N_l):
            self.gradC['dCdW']['dCdW_' + str(l)] = np.dot(self.dl['dl_' + str(l)], self.a['a_' + str(l-1)].T)
            self.gradC['dCdb']['dCdb_' + str(l)] = self.dl['dl_' + str(l)]
            
            
    #===============================================================================#
    #                                   update weights                              #
    #===============================================================================#
    def update_weights(self):
            self.weights, self.bs = self.optimizer.optimizer(w=self.weights,
                                                             b=self.bs,
                                                             grad=self.gradC)
            
            
    #===============================================================================#
    #                                     set batches                               #
    #===============================================================================#
    def _set_batches(self, N, batch_size=32):
        
        self.batch_size = batch_size
        
        # number of batches
        self.n_batch    = int(N/self.batch_size)
        
        # ids dor every batch
        if self.n_batch==1:   
            self.batches = np.array(list(range(N))*self.n_batch).reshape(-1, N)          
        else:                                                                                   
            self.batches = np.random.choice(range(N), (self.n_batch, self.batch_size))   
        
            
    #===============================================================================#
    #                                         Fit                                   #
    #===============================================================================#
    def fit(self,
            X,
            y, 
            batch_size=32,
            ephocs=1):
        
        self.ephocs = ephocs
        self.losses = []
        
        if self.fit_calls != 0:
            raise RuntimeWarning(f"Fit has already been called")
        
        # creating the batches
        self._set_batches(N=X.shape[0], batch_size=batch_size)
        L = self.N_l-1
        
        # training        
        for ephoc in range(self.ephocs):
            for batch in range(self.n_batch):
                gradsW = []
                gradsb = []
                
                # training samples for each batch
                xs = X[self.batches[batch]]
                ys = y[self.batches[batch]]
                loss_batch = 0
                
                for i in range(self.batch_size):
                    self.forward_prop(xs[i])
                    self.backpropagation(ys[i])
                    
                    gradsW.append(np.array(list(self.gradC['dCdW'].values()), dtype=object))
                    gradsb.append(np.array(list(self.gradC['dCdb'].values()), dtype=object))
                    
                    loss_batch += self.loss.loss(ys[i], self.a['a_' + str(L)])
                
                # mean of the gradients for the batch and                           
                # updating the gradient 
                self.gradC['dCdW'] = dict(zip(self.gradC['dCdW'].keys(), np.mean(gradsW, axis=0)))
                self.gradC['dCdb'] = dict(zip(self.gradC['dCdb'].keys(), np.mean(gradsb, axis=0)))
                
                # losses
                self.losses.append(loss_batch)
                                
                # update weights
                self.update_weights()
                
            #print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss_batch:.8f}")
            print(f"Epoch {ephoc + 1}/{self.epochs}, Loss: {np.mean(self.losses[-self.batch_size:]):.8f}")
            
        self.fit_calls += 1
                
    #===============================================================================#
    #                                     Predict                                   #
    #===============================================================================#
    def predict(self, X):
        
        ps = []
        
        for x in X:
            self.forward_prop(x)
            ps.append(self.a['a_' + str(self.N_l-1)][0])
            
        return np.array(ps)
