data {
  int<lower=1> O;                 // number of organizations
  int<lower=1> T;                 // number of teams
  int<lower=1> I;                 // number of items
  int<lower=2> K;                 // number of response categories
  int<lower=1> N;                 // number of responses
  int<lower=1> N_person;          // number of persons

  // identifiers
  array[T] int<lower=1,upper=O> org_id;           // org of each team
  array[N] int<lower=1,upper=T> team_id;          // (unused; ok to pass)
  array[N] int<lower=1,upper=I> item_id;          // item for each response
  array[N] int<lower=1,upper=N_person> person_id; // person for each response
  array[N_person] int<lower=1,upper=T> team_of_person; // team for each person

  // observed responses
  array[N] int<lower=1,upper=K> Y;
}

parameters {
  // latent traits
  vector[O]        theta_org;
  vector[T]        theta_team;
  vector[N_person] theta_ind;

  // variance partitioning: proportions must sum to 1 (org, team, indiv)
  simplex[3] var_share;

  // item parameters
  vector<lower=0>[I] a;            // discriminations
  array[I] ordered[K-1] kappa;     // step params (κ’s)
}

transformed parameters {
  real<lower=0> sigma_org;
  real<lower=0> sigma_team;
  real<lower=0> sigma_ind;

  // FIX total variance = 1 by construction (simplex sums to 1)
  sigma_org  = sqrt(var_share[1]);
  sigma_team = sqrt(var_share[2]);
  sigma_ind  = sqrt(var_share[3]);
}

model {
  // ----- Priors -----
  var_share ~ dirichlet(rep_vector(2, 3));

  // soft anchoring of means
  mean(theta_org)  ~ normal(0, 0.1);
  mean(theta_team) ~ normal(0, 0.1);
  mean(theta_ind)  ~ normal(0, 0.1);

  a ~ lognormal(0, 0.5);
  for (i in 1:I) kappa[i] ~ normal(0, 2);

  // ----- Hierarchy -----
  theta_org  ~ normal(0, sigma_org);
  theta_team ~ normal(theta_org[org_id], sigma_team);
  theta_ind  ~ normal(theta_team[team_of_person], sigma_ind);

  // ----- Likelihood -----
  for (n in 1:N) {
    int p = person_id[n];
    int i = item_id[n];
    real eta = a[i] * theta_ind[p];

    if (Y[n] == 1) {
      target += log1m_inv_logit(eta - kappa[i][1]);
    } else if (Y[n] == K) {
      target += log_inv_logit(eta - kappa[i][K-1]);
    } else {
      target += log_diff_exp(
        log_inv_logit(eta - kappa[i][Y[n]-1]),
        log_inv_logit(eta - kappa[i][Y[n]])
      );
    }
  }
}
generated quantities {
  // θ-scale thresholds
  array[I] vector[K-1] beta;
  for (i in 1:I) beta[i] = kappa[i] / a[i];

  // variance proportions
  real prop_org  = var_share[1];
  real prop_team = var_share[2];
  real prop_ind  = var_share[3];

  // sigma values for inspection
  real sigma_org_out  = sigma_org;
  real sigma_team_out = sigma_team;
  real sigma_ind_out  = sigma_ind;

  real var_org   = square(sigma_org);
  real var_team  = square(sigma_team);
  real var_ind   = square(sigma_ind);
  real var_total = var_org + var_team + var_ind;

  // pairwise org differences (max size O*(O-1)/2)
  int n_org_pairs = O * (O - 1) / 2;
  vector[O * (O - 1) / 2] org_diffs;   // fixed max size
  {
    int idx = 1;
    for (i in 1:(O-1)) {
      for (j in (i+1):O) {
        org_diffs[idx] = theta_org[i] - theta_org[j];
        idx += 1;
      }
    }
    // fill remaining slots with 0 if any (safety, though here idx == n_org_pairs+1)
    for (k in idx:(O * (O - 1) / 2)) org_diffs[k] = 0;
  }

  // team differences vs their org mean
  vector[T] team_vs_org;
  for (t in 1:T)
    team_vs_org[t] = theta_team[t] - theta_org[org_id[t]];

  // org differences vs 0
  vector[O] org_vs_zero = theta_org;

  // org differences vs grand mean
  vector[O] org_vs_mean;
  {
    real grand_mean = mean(theta_org);
    for (o in 1:O)
      org_vs_mean[o] = theta_org[o] - grand_mean;
  }

  // ICCs
  real ICC_org  = sigma_org^2 / (sigma_org^2 + sigma_team^2 + sigma_ind^2);
  real ICC_team = (sigma_org^2 + sigma_team^2) / (sigma_org^2 + sigma_team^2 + sigma_ind^2);

  // posterior check
  real mean_theta_ind = mean(theta_ind);
}
