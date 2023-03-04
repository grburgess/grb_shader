data {
    int<lower=0> N;
    array[N] real y;
}

parameters {
  ordered[2] mu;
  array[2] real<lower=0> sigma;
  real<lower=0, upper=1> theta;
}

model {
    sigma ~ normal(0.5, 0.2);
    //SGRBs
    mu[1] ~ normal(-0.04, 0.3);
    //LGRBs
    mu[2] ~ normal(1.5, 0.4); 
    //mixture
    theta ~ beta(5, 5);
    for (n in 1:N)
        target += log_mix(theta,
                        normal_lpdf(y[n] | mu[1], sigma[1]),
                        normal_lpdf(y[n] | mu[2], sigma[2]));
     
}

generated quantities {
    //for prior predictive distribution
    array[2] real mu_prior;
    array[2] real sigma_prior;
    real  theta_prior;

    // generate random numbers for prior predictive
    sigma_prior[1] = normal_rng(0.5, 0.2);
    sigma_prior[2] = normal_rng(0.5, 0.2);
    //SGRBs
    mu_prior[1] = normal_rng(-0.04, 0.3);
    //LGRBs
    mu_prior[2] = normal_rng(1.5, 0.4); 
    //mixture
    theta_prior = beta_rng(5, 5);

    array[N] real y_post_pred;
    array[N] real y_prior_pred;
    for (n in 1:N){
        real u = uniform_rng(0,1);

        //sample prior predictive
        if (u < theta_prior){
            //sample from SGRB distribution
            y_prior_pred[n] = normal_rng(mu_prior[1], sigma_prior[1]);
        }
        else {
           // sample from LGRB distribution
           y_prior_pred[n] = normal_rng(mu_prior[2], sigma_prior[2]);
        }

        //sample posterior predictive
        if (u < theta){
            //sample from SGRB distribution
            y_post_pred[n] = normal_rng(mu[1], sigma[1]);
        }
        else {
           // sample from LGRB distribution
           y_post_pred[n] = normal_rng(mu[2], sigma[2]);
        }

    }
}