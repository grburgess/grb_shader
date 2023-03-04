functions{
    real partial_sum(array[] real log10t90_true_slice,
                   int start, int end, 
                   real theta, vector mu, array[] real sigma) {
                       return log_mix(theta,
                        normal_lpdf(log10t90_true_slice | mu[1], sigma[1]),
                        normal_lpdf(log10t90_true_slice | mu[2], sigma[2]));
                   }
}

data {
    int<lower=0> N;
    array[N] real<lower=0> t90;
    array[N] real t90_err;
}

parameters {
    ordered[2] mu;
    array[2] real<lower=0> sigma;
    real<lower=0, upper=1> theta;
    array[N] real log10t90_true;
}

transformed parameters {
    array[N] real<lower=0> t90_true;
    t90_true = pow(10,log10t90_true); 
}

model {
    sigma ~ normal(0.5, 0.2);
    //SGRBs
    mu[1] ~ normal(-0.04, 0.3);
    //LGRBs
    mu[2] ~ normal(1.5, 0.4); 
    //mixture
    theta ~ beta(5, 5);

    int grainsize = 1;
    target += reduce_sum(partial_sum,log10t90_true, grainsize, theta, mu, sigma);
    
    t90 ~ normal(t90_true,t90_err);
     
}

generated quantities {
    //for prior predictive
    array[2] real mu_prior;
    array[2] real sigma_prior;
    real  theta_prior;

    //generate random numbers for prior predictive
    sigma_prior[1] = normal_rng(0.5, 0.2);
    sigma_prior[2] = normal_rng(0.5, 0.2);
    //SGRBs
    mu_prior[1] = normal_rng(0, 0.1);
    //LGRBs
    mu_prior[2] = normal_rng(1.5, 0.4); 
    //mixture
    theta_prior = beta_rng(5, 5);

    //sample prior predictive data
   // array[N] real<lower=0> y_prior_pred;
   // for (n in 1:N){
   //     real u = uniform_rng(0,1);//

   //     //sample prior predictive
   //     if (u < theta_prior){
   //         //sample from SGRB distribution
   //         real log10t90_true_prior = normal_rng(mu_prior[1], sigma_prior[1]);
   //         y_prior_pred[n] = normal_rng(pow(10,log10t90_true_prior),t90_err[n]);
   //     }
   //     else {
   //         // sample from LGRB distribution
   //         real log10t90_true_prior = normal_rng(mu_prior[2], sigma_prior[2]);
   //         y_prior_pred[n] = normal_rng(pow(10,log10t90_true_prior),t90_err[n]);
   //     }
   //}//

   // //sample posterior predictive data
   array[N] real t90_post_pred;
   //array[N] real log10t90_post_pred;
   for (n in 1:N){
        real t90_post=-1.;
        while (t90_post<=0.){
            t90_post = normal_rng(t90_true[n], t90_err[n]);
        }
        t90_post_pred[n] = t90_post;
       //log10t90_post_pred[n] = log10(t90_post_pred[n]);
   }
}