import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmdstanpy
import os, glob
import scipy.stats as st   # ✅ add this
import seaborn as sns

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def logistic(x): return 1 / (1 + np.exp(-x))

def grm_probs(theta, a, thresholds):
    K = len(thresholds) + 1
    cum = logistic(np.outer(theta, a) - a * thresholds)
    p_ge = np.concatenate([np.ones((theta.size,1)), cum, np.zeros((theta.size,1))], axis=1)
    probs = p_ge[:,:K] - p_ge[:,1:K+1]
    return probs

def simulate_response(theta, a, thresholds, rng):
    probs = grm_probs(np.array([theta]), a, thresholds)[0]
    return rng.choice(len(probs), p=probs) + 1

# ------------------------------------------------------------
# Item bank (20 items, fixed)
# ------------------------------------------------------------
items = [
    dict(a=1.2, thresholds=[-2.5,-1.5,-0.5,0.5,1.5,2.5]),
    dict(a=0.9, thresholds=[-2.2,-1.2,-0.2,0.6,1.4,2.4]),
    dict(a=1.5, thresholds=[-2.0,-1.0,0.0,1.0,2.0,3.0]),
    dict(a=0.7, thresholds=[-2.8,-1.8,-0.8,0.2,1.0,1.8]),
    dict(a=1.8, thresholds=[-2.3,-1.3,-0.5,0.7,1.5,2.3]),
    dict(a=1.0, thresholds=[-2.6,-1.6,-0.6,0.4,1.2,2.0]),
    dict(a=1.3, thresholds=[-2.1,-1.1,-0.3,0.5,1.3,2.1]),
    dict(a=1.6, thresholds=[-2.4,-1.4,-0.4,0.6,1.4,2.4]),
    #dict(a=0.8, thresholds=[-2.7,-1.7,-0.7,0.3,1.1,1.9]),
    #dict(a=1.4, thresholds=[-2.2,-1.2,-0.2,0.8,1.6,2.5]),
    #dict(a=1.1, thresholds=[-2.5,-1.5,-0.5,0.5,1.5,2.5]),
    #dict(a=1.9, thresholds=[-2.0,-1.0,0.0,0.9,1.7,2.6]),
    #dict(a=0.95, thresholds=[-2.3,-1.3,-0.3,0.7,1.5,2.2]),
    #dict(a=1.25, thresholds=[-2.6,-1.6,-0.6,0.4,1.2,2.1]),
    #dict(a=1.7, thresholds=[-2.1,-1.1,-0.1,0.9,1.7,2.7]),
    #dict(a=0.85, thresholds=[-2.4,-1.4,-0.4,0.6,1.4,2.2]),
    #dict(a=1.45, thresholds=[-2.2,-1.2,-0.2,0.8,1.6,2.4]),
    #dict(a=1.6, thresholds=[-2.5,-1.5,-0.5,0.5,1.3,2.2]),
    #dict(a=1.0, thresholds=[-2.3,-1.3,-0.3,0.7,1.5,2.3]),
    #dict(a=1.8, thresholds=[-2.0,-1.0,0.0,1.0,2.0,3.0]),
]

# ------------------------------------------------------------
# Simulation setup
# ------------------------------------------------------------
rng = np.random.default_rng(42)

L = 30           # number of teams
J = len(items)    # items
K = 7             # categories

# Draw team sizes uniformly between 10 and 30, for example
team_sizes = rng.integers(low=5, high=20, size=L)

# True team thetas
# variance split
var_teams = 0.25
var_within = 0.75
sigma_between = np.sqrt(var_teams)     # sqrt(0.25)
sigma_within  = np.sqrt(var_within)  # sqrt(0.75)

# Team-level means
theta_team_true = rng.normal(0, sigma_between, L)

# Individual thetas nested in teams
theta_ind = []
team_id   = []
for t, n_members in enumerate(team_sizes):
    ind_theta = rng.normal(theta_team_true[t], sigma_within, n_members)
    theta_ind.extend(ind_theta)
    team_id.extend([t+1] * n_members)

theta_ind = np.array(theta_ind)
team_id   = np.array(team_id)

# quick check
print("Empirical mean:", theta_ind.mean())
print("Empirical sd:", theta_ind.std())


theta_ind = np.array(theta_ind)
team_id   = np.array(team_id)
N_ind     = len(theta_ind)

print("Team sizes:", team_sizes)
print("Total individuals:", N_ind)

# Generate responses (long format)
rows = []
for i in range(N_ind):
    for j,it in enumerate(items):
        y = simulate_response(theta_ind[i], it["a"], np.array(it["thresholds"]), rng)
        rows.append((i+1, team_id[i], j+1, y))

df = pd.DataFrame(rows, columns=["ind","leader","item","y"])
print(df.head())

# Stan long format inputs
stan_data = dict(
    N_obs=len(df),
    L=L,
    J=J,
    K=K,
    leader=df["leader"].tolist(),
    item=df["item"].tolist(),
    y=df["y"].tolist()
)

# ------------------------------------------------------------
# Fit or reload Stan
# ------------------------------------------------------------


stan_file = "laplace_grm.stan"
model = cmdstanpy.CmdStanModel(stan_file=stan_file)

output_dir = "stan_output_team"
os.makedirs(output_dir, exist_ok=True)
csv_files = glob.glob(os.path.join(output_dir, "laplace_grm-*.csv"))

if csv_files:
    print("🔄 Reloading existing Stan fit...")
    fit = cmdstanpy.from_csv(csv_files)
else:
    print("⚡ Running Stan fit...")
    fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=800,
        iter_sampling=800,
        output_dir=output_dir,
        save_warmup=False,
        show_console=True
    )

posterior = fit.draws_pd()

# ============================================================
# 2. Extract posterior medians for item params
# ============================================================

# -----------------
# Data generator helpers (used after calibration)
# -----------------
def logistic(x):
    return 1 / (1 + np.exp(-x))

def grm_probs(theta, a, thresholds):
    K = len(thresholds) + 1
    cum = logistic(np.outer(theta, a) - a * thresholds)
    p_ge = np.concatenate([np.ones((theta.size,1)), cum, np.zeros((theta.size,1))], axis=1)
    probs = p_ge[:,:K] - p_ge[:,1:K+1]
    return probs / probs.sum(axis=1, keepdims=True)

def simulate_item(theta, a, thresholds, rng):
    probs = grm_probs(theta, a, thresholds)
    cum = np.cumsum(probs, axis=1)
    r = rng.random((theta.size,1))
    return (r > cum).sum(axis=1)  # returns 0..K-1


a_hats, b_hats = [], []
for j in range(1, J+1):
    a_hat = posterior[f"a[{j}]"].median()
    cols = [c for c in posterior.columns if c.startswith(f"kappa[{j},")]
    cols = sorted(cols, key=lambda x: int(x.split(",")[1][:-1]))
    kappas = [posterior[c].median() for c in cols]
    b_hat = [k / a_hat for k in kappas]   # probit thresholds -> divide by a
    a_hats.append(a_hat)
    b_hats.append(b_hat)

calibrated_items = [dict(a=a, thresholds=np.array(b)) for a,b in zip(a_hats,b_hats)]

# ============================================================
# 3. Simulate *new* teams and individuals with variable sizes
# ============================================================
team_sizes_new = rng.integers(low=5, high=30, size=L)   # e.g. 5–30 members
theta_team_new = rng.normal(0, 0.5, L)

theta_ind_new = []
team_id_new   = []
for t, n_members in enumerate(team_sizes_new):
    ind_theta = rng.normal(theta_team_new[t], 0.7, n_members)
    theta_ind_new.extend(ind_theta)
    team_id_new.extend([t] * n_members)

theta_ind_new = np.array(theta_ind_new)
team_id_new   = np.array(team_id_new)

Y_new = np.zeros((len(theta_ind_new), J), dtype=int)
for j,it in enumerate(calibrated_items):
    Y_new[:,j] = simulate_item(theta_ind_new, it["a"], it["thresholds"], rng) + 1

# ============================================================
# 4. Bayesian updating for new teams (fixed a,b)
# ============================================================
theta_grid = np.linspace(-3, 3, 121)

def team_loglike(theta_val, responses, items):
    ll = 0.0
    for y_resp, it in zip(responses, items):
        probs = grm_probs(np.array([theta_val]), it["a"], it["thresholds"])
        ll += np.log(probs[0, y_resp-1] + 1e-12)
    return ll

def team_posterior(team_resps, items, theta_grid, prior_sd=0.5):
    log_post = np.zeros_like(theta_grid)
    for g, theta_val in enumerate(theta_grid):
        ll = 0.0
        for resp in team_resps:
            ll += team_loglike(theta_val, resp, items)
        log_post[g] = ll + st.norm(0, prior_sd).logpdf(theta_val)
    log_post -= log_post.max()
    probs = np.exp(log_post)
    probs /= probs.sum()
    return probs

theta_est_new = []
theta_ci_low  = []
theta_ci_high = []

for t in range(L):
    idx = np.where(team_id_new == t)[0]
    team_resps = Y_new[idx,:]
    probs = team_posterior(team_resps, calibrated_items, theta_grid, prior_sd=0.5)
    mean_est = np.sum(theta_grid * probs)
    cdf = np.cumsum(probs) / probs.sum()
    ci_low = np.interp(0.025, cdf, theta_grid)
    ci_high = np.interp(0.975, cdf, theta_grid)
    theta_est_new.append(mean_est)
    theta_ci_low.append(ci_low)
    theta_ci_high.append(ci_high)

theta_est_new = np.array(theta_est_new)
theta_ci_low  = np.array(theta_ci_low)
theta_ci_high = np.array(theta_ci_high)

# ============================================================
# 5. Compare true vs estimated new team θ (with CI + team sizes)
# ============================================================
r = np.corrcoef(theta_team_new, theta_est_new)[0,1]
rmse = np.sqrt(np.mean((theta_team_new - np.array(theta_est_new))**2))
print(f"Bayesian ability recovery: r={r:.3f}, RMSE={rmse:.3f}")

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8,8))
# marker size proportional to sqrt(team size) so area ~ team size
sizes = np.sqrt(team_sizes_new) * 20
plt.errorbar(theta_team_new, theta_est_new,
             yerr=[theta_est_new-theta_ci_low, theta_ci_high-theta_est_new],
             fmt='o', alpha=0.7, capsize=3, markersize=0)  # error bars only
plt.scatter(theta_team_new, theta_est_new,
            s=sizes, c=team_sizes_new, cmap="viridis", alpha=0.8, edgecolor="k")
plt.colorbar(label="Team size")
plt.plot([-3,3],[-3,3],'r--')
plt.xlabel("True team θ")
plt.ylabel("Posterior mean θ (fixed items)")
plt.title("Team θ recovery with Bayesian updating\n(marker size/color = team size)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("plots/team_theta_recovery_bayes_sizes.png", dpi=150)
plt.close()

# ============================================================
# 6. Posterior density curves per team (exact)
# ============================================================
plt.figure(figsize=(12,6))   # wider figure
for t in range(L):
    probs = team_posterior(Y_new[team_id_new==t], calibrated_items, theta_grid, prior_sd=0.5)
    plt.plot(theta_grid, probs, label=f"Team {t} (n={team_sizes_new[t]})")

plt.xlabel("θ")
plt.ylabel("Posterior density")
plt.title("Posterior of team θ (exact, fixed items)\n(n = team size)")

# put legend outside, on the right
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1, borderaxespad=0.)

plt.tight_layout()
plt.savefig("plots/team_theta_posteriors_exact_sizes.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 7. Correlation between team size and uncertainty
# ============================================================
ci_widths = theta_ci_high - theta_ci_low

cor_size_unc = np.corrcoef(team_sizes_new, ci_widths)[0,1]
print(f"Correlation between team size and CI width: {cor_size_unc:.3f}")

plt.figure(figsize=(7,5))
plt.scatter(team_sizes_new, ci_widths, alpha=0.7)
plt.xlabel("Team size (n)")
plt.ylabel("95% CI width for θ")
plt.title("Team size vs. posterior uncertainty")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("plots/team_size_vs_uncertainty.png", dpi=150)
plt.close()
