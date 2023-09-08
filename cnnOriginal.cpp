#include <Rcpp.h>
using namespace Rcpp;

// Problem specific C functions

double closs(NumericVector y, NumericVector yhat) {
  double loss = 0.0;
  int nY = y.length();
  for(int i = 0; i < nY; i++) loss += (y[i] - yhat[i])*(y[i] - yhat[i]);
  return loss;
}

NumericVector cdLoss(NumericVector y, NumericVector yhat) {
  return -2.0 * (y - yhat);
}

double cActHidden(double z) {
  return 1.0 / (1.0 + exp(-z));
}

double cActOutput(double z) {
  return z;
}

double cdActHidden(double a) {
  return a * (1.0 - a);
}

double cdActOutput(double a) {
  return 1.0;
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
// output
//    v the updated values of each node
// requires
//    cActHidden() activation function to be applied to hidden nodes
//    cActOutput() activation to be applied to output nodes
//
// [[Rcpp::export]]
NumericVector cforward_nn(NumericVector v,
                          NumericVector bias,
                          NumericVector weight,
                          IntegerVector from,
                          IntegerVector to,
                          IntegerVector nPtr,
                          IntegerVector wPtr) {
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
      if( i < nLayers - 1 )
        v[k] = cActHidden( z[k]);
      else 
        v[k] = cActOutput( z[k]);
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
// output
//    List containing the derivatives of the loss wrt the weights and biases
// requires
//    cdActHidden() derivative of activation function to be applied to hidden nodes
//    cdActOutput() derivative of activation to be applied to output nodes
//    cdloss() derivative of the loss 
//
List cbackprop_nn(NumericVector y, 
                  NumericVector v,
                  NumericVector bias,
                  NumericVector weight,
                  IntegerVector from,
                  IntegerVector to,
                  IntegerVector nPtr,
                  IntegerVector wPtr) {
  
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
  // yhat = predicted network outputs
  for(int j = nPtr[1]; j < nPtr[nLayers-1]; j++) {
    df[j] = cdActHidden(v[j]);
  }
  for(int j = nPtr[nLayers-1]; j < nPtr[nLayers]; j++) {
    df[j] = cdActOutput(v[j]);
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


// fits a neural network by gradient descent
// inputs
//    X matrix of training data (predictors)
//    Y matrix of training data (responses)
//    design list as returned by cpreprare_nn()
//    eta the learning rate (step length)
//    nIter number of iterations of the algorithm
//    trace whether to report progress 1=yes 0=no
// returns list of results containing
//    bias the biases of the final model
//    weight the weights of the final model
//    lossHistory the loss after each iteration
//    dbias derivatives of loss wrt the bias after the final iteration
//    dweight derivative of loss wrt the weights after the final iteration
//  requires
//    closs() calculates the loss function
//
// [[Rcpp::export]]
List cfit_nn( NumericMatrix X, 
              NumericMatrix Y, 
              List design, 
              double eta = 0.1, 
              int nIter  = 1000, 
              int trace  = 1 ) {
  // unpack the design
  IntegerVector from   = design["from"];
  IntegerVector to     = design["to"];
  IntegerVector nPtr   = design["nPtr"];
  IntegerVector wPtr   = design["wPtr"];
  NumericVector bias   = design["bias"];
  NumericVector weight = design["weight"];
  // size of the training data
  int nr = X.nrow();
  int nX = X.ncol();
  int nY = Y.ncol();
  // problem size and working variables
  int nNodes   = bias.length();
  int nWts     = weight.length();
  double tloss = 0.0;
  NumericVector v (nNodes);
  NumericVector yhat (nY);
  NumericVector y (nY);
  NumericVector lossHistory (nIter);
  NumericVector dw (nWts);
  NumericVector db (nNodes);
  // iterate nIter times
  for( int iter = 0; iter < nIter; iter++ ) {
    // set derivatives & loss to zero
    for(int i = 0; i < nWts;   i++) dw[i] = 0.0;
    for(int i = 0; i < nNodes; i++) db[i] = 0.0;
    tloss = 0.0;
    // iterate over the rows of the training data
    for( int d = 0; d < nr; d++) {
      // set the predictors into v
      for(int i = 0; i < nX; i++) v[i] = X(d, i);
      // forward pass
      v = cforward_nn(v, bias, weight, from, to, nPtr, wPtr);
      // extract the predictions
      for(int i = 0; i < nY; i++) {
        yhat[i] = v[nNodes - nY + i];
        y[i]    = Y(d, i);
      }
      // calculate the loss 
      tloss += closs(y, yhat);
      // back-propagate and unpack
      List deriv = cbackprop_nn(y, v, bias, weight, from, to, nPtr, wPtr);
      NumericVector dweight = deriv["dweight"];
      NumericVector dbias   = deriv["dbias"];
      // sum the derivatives
      for(int i = 0; i < nWts;   i++) dw[i] += dweight[i];
      for(int i = 0; i < nNodes; i++) db[i] += dbias[i];
    }
    // save loss and update the parameters
    lossHistory[iter] = tloss / nr;
    for(int i = 0; i < nWts; i++) weight[i] -= eta * dw[i] / nr;
    for(int i = 0; i < nNodes; i++) bias[i] -= eta * db[i] / nr;
    // report loss every 100 iterations
    if( (trace == 1) & (iter % 100 == 0) ) {
      Rprintf("%i %f \n", iter, tloss / nr);
    }
  }
  // return the results
  List L = List::create(Named("bias")    = bias , 
                        _["weight"]      = weight,
                        _["lossHistory"] = lossHistory,
                        _["dbias"]       = db / nr,
                        _["dweight"]     = dw / nr);
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
                        _["wPtr"]     = wPtr);
  return L;
}

// predictions for a fitted NN
// inputs
//    X matrix of test data (predictors)
//    design list as returned by cpreprare_nn()
// returns 
//    Y a matrix of predictions
//
// [[Rcpp::export]]
NumericMatrix cpredict_nn( NumericMatrix X, 
                           List design) {
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
  NumericVector a (nNodes);
  NumericMatrix Y (nr, nY);
  // iterate over the rows of the test data
  for( int d = 0; d < nr; d++) {
    // set the predictors into a
    for(int i = 0; i < nX; i++) a[i] = X(d, i);
    // forward pass
    a = cforward_nn(a, bias, weight, from, to, nPtr, wPtr);
    // extract the predictions
    for(int i = 0; i < nY; i++) Y(d, i) = a[nNodes - nY + i];
  }
  // return the predictions
  return Y;
}
