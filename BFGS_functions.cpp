//[[Rcpp::depends("RcppArmadillo")]]

#include <RcppArmadillo.h>
using namespace Rcpp;


// [[Rcpp::export]]
arma::mat d1_logli(arma::mat X, arma::vec y, arma::vec beta){
        // First derivative of log-likelihood: t(X) * (Y- pi)
        // pi = e^(X*beta)/(1+e^(X*beta))
        return X.t() * (y-(1-1/(1+exp(X*beta))));
}

// [[Rcpp::export]]
double logli(arma::mat X, arma::vec Y, arma::vec beta){
        // log-likelihood: sum(y_i * log(pi/(1-pi)) + log(1-pi))
        // pi = e^(X*beta)/(1+e^(X*beta))
        return arma::as_scalar(Y.t()*(X*beta)-sum(log(1+exp(X*beta))));
}

// [[Rcpp::export]]
arma::mat Hk_f(arma::mat sk, arma::mat yk, arma::mat Hk){
        int n = Hk.n_rows;
        arma::mat I = arma::eye<arma::mat>(n, n);
        return  (I-(sk*yk.t())/ as_scalar(sk.t()*yk)) * Hk * (I-yk*sk.t()/as_scalar(sk.t()*yk)) +
                sk*sk.t()/as_scalar(sk.t()*yk);
}


// [[Rcpp::export]]
arma::vec beta_se(arma::mat X, arma::vec Y, arma::vec beta){
        // calculate e^(X*beta)
        arma::vec exp_eta_t = exp(X*beta);
        
        // w = e^(X*beta)/(1+e^(X*beta))^2
        arma::vec w = exp_eta_t/pow(1+exp_eta_t, 2);
        
        // (t(X)*W*X)^-1
        arma::mat cov = inv(X.t() * (X.each_col() % w));
        
        // return updated beta and its se
        return arma::sqrt(cov.diag());
}

// [[Rcpp::export]]
List optim_bfgs(arma::mat X,
                arma::vec Y,
                arma::vec beta,
                double tol=1e-5,
                double maxit=10000) {
        
        // Iteration setting
        int iter = 0;
        
        // Set the initial value
        arma::vec xk = beta;        
        
        
        // Define the identity matrix
        arma::mat I = arma::eye<arma::mat>(xk.n_rows, xk.n_rows);
        
        // Set the initial H matrix
        arma::mat Hk = I;
        
        // Set res to infinity
        double res = std::numeric_limits<double>::infinity();
        
        
        while (res > tol && iter <= maxit) {
                // save the previous value
                arma::mat gk = -d1_logli(X,Y,xk);
                
                // calculate the new direction of descent
                arma::mat dk = -Hk*gk;
                
                // Line Search with Wolfe condition
                
                
                // set rho
                double rho = 0.0001;
                
                // set sigma
                double sigma = 0.9;
                
                // set initial step length
                double ak = 1;
                double a = 0;
                double b = 12345;
                // Object function at current value
                double f = -logli(X,Y,xk);
                
                while(a < b){
                        // if not satisfy Armijo Condition
                        if(-logli(X,Y,(xk + ak*dk)) > f + rho*ak* as_scalar(gk.t()*dk)){
                                
                                b = ak;
                                
                                ak = 0.5*(a+b);
                                
                                // if not satisfy Wolfe condition   
                                
                        }else if( as_scalar(-d1_logli(X,Y,(xk + ak*dk)).t()*dk) < sigma* as_scalar(gk.t()*dk)){
                                a = ak;
                                if(b == 12345){
                                        ak = 2*a;
                                }
                                else{
                                        
                                        ak = 0.5*(a+b);
                                }
                        }else{break;}
                }    
                
                // then ak is the step length
                
                // update xk
                arma::vec x_next = xk + ak*dk;
                
                // update sk
                arma::vec sk = x_next-xk;
                
                // update yk
                arma::vec yk = -d1_logli(X,Y,x_next) + d1_logli(X,Y,xk);
                
                // update Hk
                Hk = (I-(sk*yk.t())/ as_scalar(sk.t()*yk)) * Hk * (I-yk*sk.t()/as_scalar(sk.t()*yk)) +
                        sk*sk.t()/as_scalar(sk.t()*yk);
                
                // update iteration
                iter = iter + 1;
                
                // update fun
                double f_next = -logli(X,Y,x_next);  
                
                //calculate residual
                res = abs(f-f_next);
                
                // update current value
                xk = x_next;
                
                // update gradient
                gk = -d1_logli(X,Y,xk);
                
                // terminate if iter hits maxit
                if (iter == maxit)
                        warning("Iteration limit reached without convergence");
                
                // print out info to keep track
                Rprintf("Iter: %d ak: %8f Loglik: %.8f norm(gk): %.8f etc. eps:%.8f\n",
                        iter,ak,f_next, norm(gk), res);
        }
        // calculate se of beta
        
        // calculate e^(X*beta)
        arma::vec es = exp(X*xk);
        
        // w = e^(X*beta)/(1+e^(X*beta))^2
        arma::vec w = es/pow(1+es, 2);
        
        // (t(X)*W*X)^-1
        arma::mat cov = inv(X.t() * (X.each_col() % w));
        
        arma::vec se =  arma::sqrt(cov.diag());
        
        //return beta, se, log-likelihood, iteration, and eps
        return List::create(Named("Estimate")=DataFrame::create(Named("Beta")=xk, Named("se")=se), Named("Function_min")=logli(X,Y,xk), Named("Iteration")=iter, Named("eps")=res);
}
