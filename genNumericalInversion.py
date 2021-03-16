import numpy as np
import settings as settings

class genNumericalInverter():
  '''
  A tool for performing generalized numerical inversion for jet calibration with arbitrary number of features. 
  See https://indico.cern.ch/event/691745/contributions/2893042/attachments/1599321/2536326/GenNI_2.13.18.pdf for details.
  '''
  #======
  def __init__(self,learned_model,inversion_model):
    '''
    genNumericalInverter(learned_model,inversion_model)
    learned_model - previously trained model learning reco pTs given true pTs and other_features
    inversion_model - untrained model architecture for learning inversion (not necessarily same architecture as learned_model)
    '''
    self.lm = learned_model
    if not hasattr(self.lm,'predict'): raise AttributeError('*** AttributeError: learned_model must be a model with a predict() function')

    self.model = inversion_model
    if not hasattr(self.model,'fit'): raise AttributeError('*** AttributeError: inversion_model must be a model with a fit() function')
    if not hasattr(self.model,'predict'): raise AttributeError('*** AttributeError: inversion_model must be a model with a predict() function')

    self.fitted = False
    return
  #========

  #========
  def fit(self,true_values, other_features, truthLabel, eventWeights, doEventWeights): 
    '''
    genNumericalInverter.fit(true_values,other_features)
    true_values - array of true pT values (can be row or column vector)
    other_features - matrix of other features (should be horizontally stacked array of column vectors)
    Does the inversion.
    Returns fitted inversion model, which is also accessible as genNumericalInverter.model.
    '''
    #if len(true_values.shape)==1: pass #true_values is a row vector
    #elif len(true_values.shape)==2 and true_values.shape[1]==1: #true_values is a column vector
    #  true_values = true_values #true_values is a row vector
    #else: raise AssertionError('true_values has to be a row or column vector of values to learn from')

    print "fitting stuff"
    self.maxx = max(true_values) #needed for linear extrapolation
    self.minx = min(true_values) #needed for linear extrapolation
    self.p99x = np.percentile(true_values,99.) #needed for linear extrapolation
    self.p01x = np.percentile(true_values,1.) #needed for linear extrapolation

    features = other_features #not including reco pT
    if not len(features.shape)==len(true_values.reshape(-1,1).shape): raise AssertionError('other_features has to be horizontally stacked column vectors of other features')
    if not len(true_values.reshape(-1,1))==len(features): raise AssertionError('true_values and other_features must be same length!')
    # create pseudo-data
    try: recoVals = self.lm.predict(np.hstack([true_values.reshape(-1,1),features]))
    #try: recoVals = self.lm.predict(np.hstack([true_values.reshape(-1,1),features, truthLabel.reshape(-1,1)]))
    #try: recoVals = self.lm.predict(np.hstack([true_values.reshape(-1,1)]))
    except: raise RuntimeError('Tried to predict using learned_model on true_values and other_features and failed')
    recoVals = recoVals.reshape(-1,1) #column vector
    if settings.doRatio:
      recoVals = recoVals.reshape(-1) * true_values.reshape(-1) #only if using the response to learn

    print 'Inverting...'
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    #es = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3), ModelCheckpoint(filepath='best_model2.h5', monitor='val_loss', save_best_only=True)]
    es = [EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3), ModelCheckpoint(filepath='best_model2.h5', monitor='loss', save_best_only=True)]

    #self.model.fit(np.hstack([recoVals,features]),true_values, batch_size=10000, epochs=1000, shuffle=True, validation_split=settings.validationSplit, sample_weight=eventWeights, callbacks=es) 
    self.model.fit(np.hstack([recoVals.reshape(-1,1),features]),true_values, batch_size=1000, epochs=1000, shuffle='batch', validation_split=settings.validationSplit, sample_weight=eventWeights, callbacks=es) 
    #self.model.fit(np.hstack([newRecoVals.reshape(-1,1),features]),true_values, batch_size=10000, epochs=1000, shuffle=True, validation_split=settings.validationSplit, sample_weight=eventWeights, callbacks=es) 
    if hasattr(self.model,'loss_'): print 'Done fitting inversion! Loss: ',self.model.loss_
    
    self.fitted = True
    print "Predicting the reco values"

    pred = self.predict(recoVals,features)
    print "Done making the predictions"

    #print pred.shape, truth.shape
    closure = pred/(true_values.reshape(-1))
    case1 = np.where(closure<=0)
    case2 = np.where(closure>2)
    case3 = np.where(closure<2)
    case4 = np.argwhere(np.isnan(closure))
    mean = np.mean(closure)
    std = np.std(closure)
    print 'Technical closure:',mean,std

    return self.model
  #========

  #======= 
  def predict(self,values,other_features):
    '''
    genNumericalInverter.predict(values,other_features)
    After inversion, makes predictions with new calibration function. Uses a linear extrapolation for pT values outside its training set.
    values - jet pTs to calibrate (can be row or column vector)
    other_features - horizontally stacked column vectors of other features
    Returns predicted calibrated values.
    '''
    if not self.fitted: raise RuntimeError('Have to fit first before predicting!')

    lm = self.lm
    lm2 = self.model

    try: assert(len(values)==len(other_features))
    except AssertionError: print '*** AssertionError: values and other_features must be same length!'
    #if len(values.shape)==1: pass #values is a row vector
    #elif len(values.shape)==2 and values.shape[1]==1: #values is a column vector
    #  values = values.reshape(-1) #values is a row vector
    #else: raise AssertionError('values has to be a row or column vector of values to invert')
    valuesArray = np.asarray(values, dtype=np.float32)
    valuesArray = valuesArray.reshape(-1)
    recoVals = valuesArray.reshape(-1,1)
    features = other_features
    if not (len(features.shape)==len(recoVals.shape) and len(features)==len(recoVals)): raise AssertionError('other_features has to be horizontally stacked column vectors of other features')

    #miny and maxy are predictive limits of learned_model
    # note: if test set has features  that were never seen in training, then not good behavior
    #maxrecoVals = lm.predict(np.hstack([self.maxx*np.ones_like(recoVals),features, self.maxx*np.ones_like(recoVals)])).reshape(-1,1) #will just crash if this doesn't work
    #p99recoVals = lm.predict(np.hstack([self.p99x*np.ones_like(recoVals),features, self.maxx*np.ones_like(recoVals)])).reshape(-1,1) #will just crash if this doesn't work
    #minrecoVals = lm.predict(np.hstack([self.minx*np.ones_like(recoVals),features, self.maxx*np.ones_like(recoVals)])).reshape(-1,1) #will just crash if this doesn't work
    #p01recoVals = lm.predict(np.hstack([self.p01x*np.ones_like(recoVals),features, self.maxx*np.ones_like(recoVals)])).reshape(-1,1) #will just crash if this doesn't work
    maxrecoVals = lm.predict(np.hstack([self.maxx*np.ones_like(recoVals), features])).reshape(-1,1) #will just crash if this doesn't work
    p99recoVals = lm.predict(np.hstack([self.p99x*np.ones_like(recoVals), features])).reshape(-1,1) #will just crash if this doesn't work
    minrecoVals = lm.predict(np.hstack([self.minx*np.ones_like(recoVals), features])).reshape(-1,1) #will just crash if this doesn't work
    p01recoVals = lm.predict(np.hstack([self.p01x*np.ones_like(recoVals), features])).reshape(-1,1) #will just crash if this doesn't work
    
    #derivatives given features at limits of learned_model
    H_max = lm2.predict(np.hstack([maxrecoVals,features])).reshape(-1)
    H_p99 = lm2.predict(np.hstack([p99recoVals,features])).reshape(-1)
    dHdy_up = (H_max-H_p99)/(maxrecoVals-p99recoVals).reshape(-1)
    H_min = lm2.predict(np.hstack([minrecoVals,features])).reshape(-1)
    H_p01 = lm2.predict(np.hstack([p01recoVals,features])).reshape(-1)
    dHdy_down = (H_p01-H_min)/(p01recoVals-minrecoVals).reshape(-1)


    # to-do: just return inds with positive derivative
    minrecoVals = minrecoVals.reshape(-1) #row vector
    maxrecoVals = maxrecoVals.reshape(-1) #row vector

    #linear extrapolation outside of predictive limits
    case1 = np.where(valuesArray>maxrecoVals.reshape(-1))
    case2 = np.where(valuesArray<minrecoVals.reshape(-1))
    case3 = np.where(np.logical_not(np.any([valuesArray>maxrecoVals.reshape(-1),valuesArray<minrecoVals.reshape(-1)],axis=0)))
    
    result = np.zeros_like(valuesArray) #row vector

    if settings.doRatio:
      if len(case1[0])>0: result[case1] = lm2.predict(np.hstack([recoVals[case1],features[case1]])).reshape(-1)
      if len(case2[0])>0: result[case2] = lm2.predict(np.hstack([recoVals[case2],features[case2]])).reshape(-1)
      if len(case3[0])>0: result[case3] = lm2.predict(np.hstack([recoVals[case3],features[case3]])).reshape(-1)
    else:
      if len(case1[0])>0: result[case1] = H_max[case1]+dHdy_up[case1]*(valuesArray[case1]-maxrecoVals[case1])
      if len(case2[0])>0: result[case2] = H_min[case2]+dHdy_down[case2]*(valuesArray[case2]-minrecoVals[case2])
      if len(case3[0])>0: result[case3] = lm2.predict(np.hstack([recoVals[case3],features[case3]])).reshape(-1)

    return result
