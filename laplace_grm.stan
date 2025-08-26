// grm_team_laplace_probit.stan
data {
  int<lower=1> N_obs;
  int<lower=1> L;
  int<lower=1> J;
  int<lower=2> K;
  array[N_obs] int<lower=1,upper=L> leader;
  array[N_obs] int<lower=1,upper=J> item;
  array[N_obs] int<lower=1,upper=K> y;
}
parameters {
  vector[L] theta_team;                     // team abilities
  vector[J] log_a;                          // item discriminations
  array[J] vector[K-1] kappa_raw;           // free thresholds
  real<lower=0> sigma_rater;                // SD of rater RE (marginalized)
}
transformed parameters {
  vector<lower=0>[J] a = exp(log_a);

  // monotone thresholds on probit scale
  array[J] ordered[K-1] kappa;
  {
    real eps = 1e-3;
    for (j in 1:J) {
      vector[K-1] tmp;
      tmp[1] = kappa_raw[j,1];
      for (k in 2:(K-1))
        tmp[k] = tmp[k-1] + eps + exp(kappa_raw[j,k]);
      kappa[j] = sort_asc(tmp); // already increasing, sort is safe
    }
  }
}
model {
  theta_team ~ normal(0, 0.5);
  log_a      ~ normal(0, 0.35);
  for (j in 1:J) kappa_raw[j] ~ normal(0, 1.5);
  sigma_rater ~ normal(0, 1);

  {
    real s = sqrt(1 + square(sigma_rater)); // variance inflation from marginalized raters
    for (n in 1:N_obs) {
      real mu = a[item[n]] * theta_team[leader[n]];
      // scale both mean and cutpoints by s (probit link only!)
      target += ordered_probit_lpmf(y[n] | mu / s, kappa[item[n]] / s);
    }
  }
}
generated quantities {
  array[N_obs] int y_rep;
  {
    real s = sqrt(1 + square(sigma_rater));
    for (n in 1:N_obs) {
      real mu = a[item[n]] * theta_team[leader[n]];
      // simulate on probit scale with inflated variance
      y_rep[n] = ordered_probit_rng(mu / s, kappa[item[n]] / s);
    }
  }
}
