#include <Rcpp.h>
using namespace Rcpp;

// Loss function and its derivative

double closs(NumericVector y, NumericVector yhat) {
  double loss = 0.0;
  int nY = y.length();
  for(int i = 0; i < nY; i++) loss += (y[i] - yhat[i])*(y[i] - yhat[i]);
  return loss;
}

NumericVector cdLoss(NumericVector y, NumericVector yhat) {
  return -2.0 * (y - yhat);
}

// Activation functions and their derivatives

double cActivation(double z, int actFun = 0) {
  if( z > 20.0) z = 20.0;
  if( z < -20.0) z = -20.0;
  switch( actFun ) {
  case 1:
    return 1.0 / (1.0 + exp(-z));
  case 2:
    return 1.0 / (1.0 + exp(-z)) - 0.5;
  default:
    return z;
  }
}

double cdActivation(double v, int actFun = 0) {
  switch( actFun ) {
  case 1:
    return v * (1.0 - v);
  case 2:
    return (0.5 + v) * (0.5 - v);
  default:
    return 1.0;
  }
}


// Number of inputs to each node
// inputs
//    design as returned by cprepare_nn()
// output
//    L integer vector of number of inputs
//
// [[Rcpp::export]]
IntegerVector Lin(List design) {
  // unpack the design
  IntegerVector nPtr = design["nPtr"];
  IntegerVector wPtr = design["wPtr"];
  NumericVector weight = design["weight"];
  
  IntegerVector L (weight.length());
  int k = 0;
  int nLayers = nPtr.length() - 1;
  for( int i = 1; i < nLayers; i++) {
    int size = nPtr[i] - nPtr[i-1];
    for(int j = wPtr[i-1]; j < wPtr[i]; j++) {
      L[k++] = size;
    }
  }
  return L;
}

// function to return pointers and initial values for an arbitrary NN
// inputs
//    arch  the architecture of the network
// output
//    a List of pointers and initial values
//
// [[Rcpp::export]]
List cprepare_nn(IntegerVector arch) {
  // Number of layers
  int nLayers = arch.length();
  // Number of Nodes
  int nNodes = 0;
  for(int j = 0; j < nLayers; j++) nNodes += arch[j];
  // Pointer to first Node of each layer
  IntegerVector nPtr (nLayers+1);
  int h = 0;
  for(int j = 0; j < nLayers; j++) {
    nPtr[j] = h;
    h += arch[j];
  }
  nPtr[nLayers] = nNodes;
  // number of weights for a fully connected NN
  int nWt = 0;
  for(int j = 1; j < nLayers; j++) nWt += arch[j-1] * arch[j];
  // origin and destination of each node
  IntegerVector from (nWt);
  IntegerVector to (nWt);
  int q1 = 0;
  int q2 = 0;
  h = 0;
  for(int j = 1; j < nLayers; j++) {
    q1 += arch[j-1];
    for(int f = 0; f < arch[j-1]; f++ ) {
      for(int t = 0; t < arch[j]; t++ ) {
        from[h] = f + q2;
        to[h] = t + q1;
        h++;
      }
    }
    q2 = q1;
  }
  // Pointer to the first weight of each layer
  IntegerVector wPtr (nLayers);
  for(int j = 1; j < nLayers - 1; j++) {
    wPtr[j] = wPtr[j-1] + arch[j-1] * arch[j];
  }
  wPtr[nLayers - 1] = nWt;
  // Random Starting Values
  NumericVector bias = Rcpp::runif(nNodes, -1.0, 1.0);
  NumericVector weight = Rcpp::runif(nWt, -1.0, 1.0);
  // return the design
  List L = List::create(Named("bias") = bias , 
                        _["weight"]   = weight,
                        _["from"]     = from,
                        _["to"]       = to,
                        _["nPtr"]     = nPtr,
                        _["wPtr"]     = wPtr,
                        _["arch"]     = arch);
  return L;
}



// function to make a forward pass through a NN 
// inputs
//    v the values of each node (input nodes must be set before calling this function)
//    bias the bias of each node 
//    weight the weights of the network 
//    from the source of each weight (from cprepare_nn)
//    to the destination of each weight (from cprepare_nn)
//    nPtr pointer to first node of each layer (from cprepare_nn)
//    wPtr pointer to first weight of each layer (from cprepare_nn)
//    actFun coded activation function choice
// output
//    v the updated values of each node
// requires
//    cActivation() calculates selected activation function
//
// [[Rcpp::export]]
NumericVector cforward_nn(NumericVector v,
                          NumericVector bias,
                          NumericVector weight,
                          IntegerVector from,
                          IntegerVector to,
                          IntegerVector nPtr,
                          IntegerVector wPtr,
                          IntegerVector actFun) {
  // Size of network
  int nLayers = nPtr.length() - 1;
  int nNodes  = v.length();
  // z = linear combinations inputs to each node
  // v = value of each node v = activation(z)
  NumericVector z(nNodes);
  for(int i = 0; i < nNodes; i++) {
    z[i] = bias[i];
  }
  for(int i = 1; i < nLayers; i++) {
    for(int k = nPtr[i]; k < nPtr[i+1]; k++ ) {
      for(int h = wPtr[i-1]; h < wPtr[i]; h++) {
        if( to[h] == k ) {
          z[k] += weight[h] * v[from[h]];
        }
      }
      // apply activation function
      v[k] = cActivation( z[k], actFun[i-1]);
    }
  }
  return v;
}

// function to make a backward pass through a NN calculating the derivatives of the 
// weights and biases corresponding to a single observation yi
// inputs
//    y the single observation
//    v the current values of each node 
//    bias the bias of each node 
//    weight the weights of the network 
//    from the source of each weight (from cprepare_nn)
//    to the destination of each weight (from cprepare_nn)
//    nPtr pointer to first node of each layer (from cprepare_nn)
//    wPtr pointer to first weight of each layer (from cprepare_nn)
//    actFun coded activation function choice
// output
//    List containing the derivatives of the loss wrt the weights and biases
// requires
//    cdActivation() derivative of activation function
//    cdloss() derivative of the loss 
//
List cbackprop_nn(NumericVector y, 
                  NumericVector v,
                  NumericVector bias,
                  NumericVector weight,
                  IntegerVector from,
                  IntegerVector to,
                  IntegerVector nPtr,
                  IntegerVector wPtr,
                  IntegerVector actFun) {
  
  // Size of network
  int nLayers = nPtr.length() - 1;
  int nNodes  = bias.length();
  int nWts    = weight.length();
  int nY      = y.length();
  // define structures 
  double wjk  = 0.0;
  NumericVector dbias(nNodes);
  NumericVector dweight(nWts);
  NumericVector dv(nNodes);
  NumericVector df(nNodes);
  NumericVector yhat(nY);
  NumericVector dLoss(nY);
  // df = derivatives of activation functions
  for(int i = 1; i < nLayers; i++) {
    for(int j = nPtr[i]; j < nPtr[i+1]; j++) {
      df[j] = cdActivation(v[j], actFun[i-1]);
    }
  }
  // yhat = predicted network outputs
  for(int j = nPtr[nLayers-1]; j < nPtr[nLayers]; j++) {
    yhat[j-nPtr[nLayers-1]] = v[j];
  }
  // derivative of loss wrt to yhat
  dLoss = cdLoss(y, yhat);
  // dv derivatives of loss wrt each nodal value 
  // dbias derivatives of loss wrt the bias
  // dweight derivatives of loss wrt the weights
  for(int j = nPtr[nLayers-1]; j < nPtr[nLayers]; j++) {
    dv[j] = dLoss[j-nPtr[nLayers-1]];
  }
  for(int i = nLayers-1; i > 0; i-- ) {
    for(int j = nPtr[i]; j < nPtr[i+1]; j++) {
      dbias[j] = dv[j] * df[j];
      for(int h = wPtr[i-1]; h < wPtr[i]; h++) {
        if( to[h] == j) {
          dweight[h] = dv[j] * df[j] * v[from[h]];
        }
      }
    }
    for(int j = nPtr[i-1]; j < nPtr[i]; j++) {
      dv[j] = 0.0;
      for(int k = nPtr[i]; k < nPtr[i+1]; k++) {
        for(int h = wPtr[i-1]; h < wPtr[i]; h++) {
          if( (from[h] == j) & (to[h] == k) ) wjk = weight[h];
        }
        dv[j] += dv[k] * df[k] * wjk;
      }
    }
  }
  // return derivatives as a named list
  List L = List::create(Named("dbias") = dbias , 
                        _["dweight"] = dweight);
  return L;
}

// Fit a NN by Full Gradient Descent
// inputs
//   X training data predictors
//   Y training data responses
//   design as returned by cprepare_nn
//   actFun coded activation functions
//   eta the initial step length
//   etaAuto whether eta is automatically adjusted (1) or fixed (0)
//   etaDrop gap between scheduled reductions in eta (0=No drop)
//   epsEta smallest allowable step length .. stops early if eta < epsEta
//   epsLoss smallest allowable percent change in loss .. stops early if %change < epsLoss
//   nIter number of iterations
//   trace number of iterations before reporting progress
//
// outputs
//   List containing
//     bias   the biases with min training loss
//     weight the weights with min training loss
//     lossHistory loss after each iteration
//     earlyStop the number of the final iteration
//     eta the final step length
//     finalbias   the biases on completion
//     finalweight the weights on completion
//     dbias  derivative of bias on completion
//     dweight derivative of the weight on completion
//
// [[Rcpp::export]]
List cfit_nn( NumericMatrix X, 
              NumericMatrix Y, 
              List design, 
              IntegerVector actFun,
              double eta = 0.1, 
              int etaAuto = 0,
              int etaDrop = 0,
              double epsEta = 1e-8,
              double epsLoss = 1e-8,
              int nIter = 1000, 
              int trace = 0 ) {
  // unpack the design
  IntegerVector from   = design["from"];
  IntegerVector to     = design["to"];
  IntegerVector nPtr   = design["nPtr"];
  IntegerVector wPtr   = design["wPtr"];
  NumericVector bias   = design["bias"];
  NumericVector weight = design["weight"];
  // size of the problem
  int nr = X.nrow();
  int nX = X.ncol();
  int nY = Y.ncol();
  int nNodes  = bias.length();
  int nWts    = weight.length();
  // working variables
  NumericVector v (nNodes);
  NumericVector yhat (nY);
  NumericVector y (nY);
  NumericVector lossHistory (nIter);
  NumericVector dw (nWts);
  NumericVector db (nNodes);
  NumericVector fweight (nWts);
  NumericVector fbias (nNodes);
  double tloss = 0.0;
  double minLoss = 0.0;
  double delta = 0.0;
  // reset trace
  if( trace == 0  ) trace   = nIter + 1;
  if( etaDrop == 0) etaDrop = nIter + 1;
  // iterate nIter times
  int earlyStop = 0;
  int iter = 0;
  while( (iter < nIter) & (earlyStop == 0)  ) {
    // update the weights and biases
    for(int i = 0; i < nWts; i++)    weight[i] -= eta * dw[i] / nr;
    for(int i = nX; i < nNodes; i++) bias[i]   -= eta * db[i] / nr;
    // reset derivatives & loss to zero
    for(int i = 0; i < nWts; i++)   dw[i] = 0.0;
    for(int i = 0; i < nNodes; i++) db[i] = 0.0;
    tloss = 0.0;
    // iterate over the rows of the training data
    for( int d = 0; d < nr; d++) {
      // set the predictors into v
      for(int i = 0; i < nX; i++) v[i] = X(d, i);
      // forward pass
      v = cforward_nn(v, bias, weight, from, to, nPtr, wPtr, actFun);
      // extract the predictions
      for(int i = 0; i < nY; i++) {
        yhat[i] = v[nNodes - nY + i];
        y[i]    = Y(d, i);
      }
      // calculate the loss 
      tloss += closs(y, yhat);
      // back-propagate
      List deriv = cbackprop_nn(y, v, bias, weight, from, to, nPtr, wPtr, actFun);
      NumericVector dweight = deriv["dweight"];
      NumericVector dbias   = deriv["dbias"];
      // sum the derivatives
      for(int i = 0; i < nWts; i++)   dw[i] += dweight[i];
      for(int i = 0; i < nNodes; i++) db[i] += dbias[i];
    }
    // save loss 
    lossHistory[iter] = tloss / nr;
    // save parameters if loss is an improvement
    if( (iter == 0) | (lossHistory[iter] < minLoss) ) {
      minLoss = lossHistory[iter];
      for(int i = 0; i < nWts; i++) fweight[i] = weight[i];
      for(int i = 0; i < nNodes; i++) fbias[i] = bias[i];
    }
    // automatic step reduction 
    if( (iter > 0) & (etaAuto == 1) ) {
      if( lossHistory[iter] > lossHistory[iter-1] ) {
        // reject the last change & reduce the step length
        for(int i = 0; i < nWts; i++)    weight[i] += eta * dw[i] / nr;
        for(int i = nX; i < nNodes; i++) bias[i]   += eta * db[i] / nr;
        eta*= 0.5;
      }
      // Percentage change in loss
      delta = 100.0 * abs(lossHistory[iter-1] - lossHistory[iter]) / abs(lossHistory[iter-1]);
      if( delta < 1e-3 ) eta += eta;
      if( delta < epsLoss ) earlyStop = 1;
    } 
    // increment iteration count
    iter++;
    // reduce learning rate if appropriate
    if( iter % etaDrop == 0)  eta *= 0.9;
    // test for early stopping due to step length
    if( (eta < epsEta) ) earlyStop = 1;
    // report progress if appropriate
    if( (iter % trace == 0) | (earlyStop == 1) ) Rprintf("%i %f %f %f %f\n", 
        iter, tloss / nr, delta, minLoss, eta);
  }
  // return the results
  List  L = List::create(Named("bias")     = fbias , 
                          _["weight"]      = fweight,
                          _["lossHistory"] = lossHistory,
                          _["earlyStop"]   = iter,
                          _["eta"]         = eta,
                          _["finalbias"]   = bias,
                          _["finalweight"] = weight,
                          _["dbias"]       = db / nr,
                          _["dweight"]     = dw / nr);
    
  return L;
}

// predictions for a fitted NN
// inputs
//    X matrix of test data (predictors)
//    design list as returned by cpreprare_nn()
//    actFun coded activation functions for each layer
// returns 
//    Y a matrix of predictions
//
// [[Rcpp::export]]
NumericMatrix cpredict_nn( NumericMatrix X, 
                           List design,
                           IntegerVector actFun) {
  // unpack the design
  IntegerVector from   = design["from"];
  IntegerVector to     = design["to"];
  IntegerVector nPtr   = design["nPtr"];
  IntegerVector wPtr   = design["wPtr"];
  NumericVector bias   = design["bias"];
  NumericVector weight = design["weight"];
  // size of the test data
  int nr = X.nrow();
  int nX = X.ncol();
  int nLayers = nPtr.length() - 1;
  int nNodes = bias.length();
  int nY = nPtr[nLayers] - nPtr[nLayers-1];
  NumericVector v (nNodes);
  NumericMatrix Y (nr, nY);
  // iterate over the rows of the test data
  for( int d = 0; d < nr; d++) {
    // set the predictors into v
    for(int i = 0; i < nX; i++) v[i] = X(d, i);
    // forward pass
    v = cforward_nn(v, bias, weight, from, to, nPtr, wPtr, actFun);
    // extract the predictions
    for(int i = 0; i < nY; i++) Y(d, i) = v[nNodes - nY + i];
  }
  // return the predictions
  return Y;
}

// Fit a NN by Full Gradient Descent + validation loss
// inputs
//   X training data predictors
//   Y training data responses
//   design as returned by cprepare_nn
//   XV validation data predictors
//   YV validation data responses
//   actFun coded activation functions
//   eta the initial step length
//   etaAuto whether eta is automatically adjusted (1) or fixed (0)
//   etaDrop gap between scheduled reductions in eta (0=No drop)
//   epsEta smallest allowable step length .. stops early if eta < epsEta
//   epsLoss smallest allowable percent change in loss .. stops early if %change < epsLoss
//   validGap interval between evaluations of the test loss
//   nIter number of iterations
//   trace number of iterations before reporting progress
//
// outputs
//   List containing
//     bias   the biases for minimum validation loss
//     weight the weights for minimum validation loss
//     lossHistory training loss after each iteration
//     validHistory validation loss after each iteration
//     earlyStop the number of the final iteration
//     eta the final step length
//     finalbias   the biases on completion
//     finalweight the weights on completion
//     dbias  derivative of bias on completion
//     dweight derivative of the weight on completion
//
// [[Rcpp::export]]
List cfit_valid_nn( NumericMatrix X, 
              NumericMatrix Y, 
              List design, 
              NumericMatrix XV, 
              NumericMatrix YV,
              IntegerVector actFun,
              double eta = 0.1, 
              int etaAuto = 0,
              int etaDrop = 0,
              double epsEta = 1e-8,
              double epsLoss = 1e-8,
              int validGap = 1,
              int nIter = 1000, 
              int trace = 0 ) {
  // unpack the design
  IntegerVector from   = design["from"];
  IntegerVector to     = design["to"];
  IntegerVector nPtr   = design["nPtr"];
  IntegerVector wPtr   = design["wPtr"];
  NumericVector bias   = design["bias"];
  NumericVector weight = design["weight"];
  // size of the problem
  int nr = X.nrow();
  int nX = X.ncol();
  int nY = Y.ncol();
  int nv = XV.nrow();
  int nNodes  = bias.length();
  int nWts    = weight.length();
  // working variables
  NumericVector v (nNodes);
  NumericVector yhat (nY);
  NumericVector y (nY);
  NumericVector lossHistory (nIter);
  NumericVector validHistory (nIter);
  NumericVector dw (nWts);
  NumericVector db (nNodes);
  NumericVector fbias (nNodes);
  NumericVector fweight (nWts);
  double tloss = 0.0;
  double vloss = 0.0;
  double floss = 0.0;
  double delta = 0.0;
  int count = 0;
  // reset trace
  if( trace == 0  ) trace   = nIter + 1;
  if( etaDrop == 0) etaDrop = nIter + 1;
  // iterate nIter times
  int earlyStop = 0;
  int iter = 0;
  while( (iter < nIter) & (earlyStop == 0)  ) {
    // update the weights and biases
    for(int i = 0; i < nWts; i++)    weight[i] -= eta * dw[i] / nr;
    for(int i = nX; i < nNodes; i++) bias[i]   -= eta * db[i] / nr;
    // reset derivatives & loss to zero
    for(int i = 0; i < nWts; i++)   dw[i] = 0.0;
    for(int i = 0; i < nNodes; i++) db[i] = 0.0;
    tloss = 0.0;
    // iterate over the rows of the training data
    for( int d = 0; d < nr; d++) {
      // set the predictors into v
      for(int i = 0; i < nX; i++) v[i] = X(d, i);
      // forward pass
      v = cforward_nn(v, bias, weight, from, to, nPtr, wPtr, actFun);
      // extract the predictions
      for(int i = 0; i < nY; i++) {
        yhat[i] = v[nNodes - nY + i];
        y[i]    = Y(d, i);
      }
      // calculate the loss 
      tloss += closs(y, yhat);
      // back-propagate
      List deriv = cbackprop_nn(y, v, bias, weight, from, to, nPtr, wPtr, actFun);
      NumericVector dweight = deriv["dweight"];
      NumericVector dbias   = deriv["dbias"];
      // sum the derivatives
      for(int i = 0; i < nWts; i++)   dw[i] += dweight[i];
      for(int i = 0; i < nNodes; i++) db[i] += dbias[i];
    }
    // save loss and update the parameters
    lossHistory[iter] = tloss / nr;
    // iterate over the rows of the validation data
    if( iter % validGap == 0 ) {
      vloss = 0.0;
      for( int d = 0; d < nv; d++) {
        // set the predictors into v
        for(int i = 0; i < nX; i++) v[i] = XV(d, i);
        // forward pass
        v = cforward_nn(v, bias, weight, from, to, nPtr, wPtr, actFun);
        // extract the predictions
        for(int i = 0; i < nY; i++) {
          yhat[i] = v[nNodes - nY + i];
          y[i]    = YV(d, i);
        }
        // calculate the loss
        vloss += closs(y, yhat);
      }
      validHistory[iter] = vloss / nv;
      // save parameters if an improvement
      if( (count == 0) | (validHistory[iter] < floss) ) {
        count++;
        floss = validHistory[iter];
        for(int i = 0; i < nWts; i++) fweight[i] = weight[i];
        for(int i = 0; i < nNodes; i++) fbias[i] = bias[i];
      }
    } else {
      validHistory[iter] = validHistory[iter-1];
    }
    // automatic step reduction 
    if( (iter > 0) & (etaAuto == 1) ) {
      if( lossHistory[iter] > lossHistory[iter-1] ) {
        // reject the last change & reduce the step length
        for(int i = 0; i < nWts; i++)    weight[i] += eta * dw[i] / nr;
        for(int i = nX; i < nNodes; i++) bias[i]   += eta * db[i] / nr;
        eta*= 0.5;
      }
      // Percentage change in loss
      delta = 100.0 * abs(lossHistory[iter-1] - lossHistory[iter]) / abs(lossHistory[iter-1]);
      if( delta < 1e-3 ) eta += eta;
      if( delta < epsLoss ) earlyStop = 1;
    } 
    // increment iteration count
    iter++;
    // reduce learning rate if appropriate
    if( iter % etaDrop == 0)  eta *= 0.9;
    // test for early stopping due to step length
    if( (eta < epsEta) ) earlyStop = 1;
    // report progress if appropriate
    if( (iter % trace == 0) | (earlyStop == 1) ) Rprintf("%i %f %f %f %f\n", 
        iter, lossHistory[iter-1], delta, validHistory[iter-1], eta);
  }
  // return the results
  List  L = List::create(Named("fittedbias")    = fbias , 
                         _["fittedweight"]      = fweight,
                         _["lossHistory"] = lossHistory,
                         _["validHistory"]  = validHistory,
                         _["earlyStop"]   = iter,
                         _["eta"]         = eta,
                         _["finalbias"]   = bias,
                         _["finalweight"] = weight,
                         _["dbias"]       = db / nr,
                         _["dweight"]     = dw / nr);
  
  return L;
}

// Values of each node for a fitted NN
// inputs
//    X matrix of test data (predictors)
//    design list as returned by cpreprare_nn()
//    actFun coded activation functions for each layer
// returns 
//    Y a matrix of nodal values
//
// [[Rcpp::export]]
NumericMatrix cNodalValues_nn( NumericMatrix X, 
                           List design,
                           IntegerVector actFun) {
  // unpack the design
  IntegerVector from = design["from"];
  IntegerVector to   = design["to"];
  IntegerVector nPtr = design["nPtr"];
  IntegerVector wPtr = design["wPtr"];
  NumericVector bias = design["bias"];
  NumericVector weight = design["weight"];
  // size of the test data
  int nr = X.nrow();
  int nX = X.ncol();
  int nNodes = bias.length();
  int nY = nNodes - nX;
  NumericVector a (nNodes);
  NumericMatrix Y (nr, nY);
  // iterate over the rows of the test data
  for( int d = 0; d < nr; d++) {
    // set the predictors into a
    for(int i = 0; i < nX; i++) a[i] = X(d, i);
    // forward pass
    a = cforward_nn(a, bias, weight, from, to, nPtr, wPtr, actFun);
    // extract the predictions
    for(int i = 0; i < nY; i++) Y(d, i) = a[nX + i];
  }
  return Y;
}

