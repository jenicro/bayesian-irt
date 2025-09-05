import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmdstanpy
import os, glob
import scipy.stats as st
import seaborn as sns

# ============================================================
# Project configuration
# ============================================================
PROJECT = "extremely_skewed_items"   # <<< CHANGE THIS PER PROJECT

stan_file = "stan_files/laplace_grm.stan"
output_dir = os.path.join("stan_output", PROJECT)
plot_dir   = os.path.join("plots", PROJECT)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================
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

# ============================================================
# Item bank (example items)
# ============================================================
items = [
    dict(a=0.9, thresholds=[-4.37, -3.42, -2.56, -1.70, -0.86,  0.00]),
    dict(a=1.2, thresholds=[-3.28, -2.57, -1.92, -1.29, -0.67, -0.07]),
    dict(a=1.5, thresholds=[-2.62, -2.05, -1.51, -0.98, -0.47,  0.03]),
    dict(a=0.7, thresholds=[-5.63, -4.38, -3.28, -2.21, -1.16, -0.14]),
    dict(a=1.8, thresholds=[-2.19, -1.71, -1.25, -0.80, -0.36,  0.07]),
    dict(a=1.0, thresholds=[-3.93, -3.08, -2.27, -1.48, -0.70,  0.05]),
    dict(a=1.3, thresholds=[-2.90, -2.27, -1.65, -1.06, -0.48,  0.09]),
    dict(a=1.6, thresholds=[-2.44, -1.92, -1.42, -0.93, -0.45,  0.03]),
]



# ============================================================
# Simulation setup
# ============================================================
rng = np.random.default_rng(42)

L = 30           # number of teams
J = len(items)   # items
K = 7            # categories

# Team sizes uniformly between 5 and 20
team_sizes = rng.integers(low=5, high=20, size=L)

# Variance split
var_teams = 0.25
var_within = 0.75
sigma_between = np.sqrt(var_teams)
sigma_within  = np.sqrt(var_within)

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
N_ind     = len(theta_ind)

# Generate responses (long format)
rows = []
for i in range(N_ind):
    for j,it in enumerate(items):
        y = simulate_response(theta_ind[i], it["a"], np.array(it["thresholds"]), rng)
        rows.append((i+1, team_id[i], j+1, y))

df = pd.DataFrame(rows, columns=["ind","leader","item","y"])

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

# ============================================================
# Fit or reload Stan
# ============================================================
model = cmdstanpy.CmdStanModel(stan_file=stan_file,
                               model_name=f"laplace_grm_{PROJECT}")

csv_files = glob.glob(os.path.join(output_dir, f"laplace_grm_{PROJECT}-*.csv"))

if csv_files:
    print(f"🔄 Reloading existing Stan fit for {PROJECT}...")
    fit = cmdstanpy.from_csv(csv_files)
else:
    print(f"⚡ Running Stan fit for {PROJECT}...")
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
def grm_probs_norm(theta, a, thresholds):
    K = len(thresholds) + 1
    cum = logistic(np.outer(theta, a) - a * thresholds)
    p_ge = np.concatenate([np.ones((theta.size,1)), cum, np.zeros((theta.size,1))], axis=1)
    probs = p_ge[:,:K] - p_ge[:,1:K+1]
    return probs / probs.sum(axis=1, keepdims=True)

def simulate_item(theta, a, thresholds, rng):
    probs = grm_probs_norm(theta, a, thresholds)
    cum = np.cumsum(probs, axis=1)
    r = rng.random((theta.size,1))
    return (r > cum).sum(axis=1)  # returns 0..K-1

a_hats, b_hats = [], []
for j in range(1, J+1):
    a_hat = posterior[f"a[{j}]"].median()
    cols = [c for c in posterior.columns if c.startswith(f"kappa[{j},")]
    cols = sorted(cols, key=lambda x: int(x.split(",")[1][:-1]))
    kappas = [posterior[c].median() for c in cols]
    b_hat = [k / a_hat for k in kappas]
    a_hats.append(a_hat)
    b_hats.append(b_hat)

calibrated_items = [dict(a=a, thresholds=np.array(b)) for a,b in zip(a_hats,b_hats)]

# ============================================================
# 3. Simulate new teams and individuals with variable sizes
# ============================================================
team_sizes_new = rng.integers(low=5, high=20, size=L)
theta_team_new = rng.normal(0, sigma_between, L)

theta_ind_new = []
team_id_new   = []
for t, n_members in enumerate(team_sizes_new):
    ind_theta = rng.normal(theta_team_new[t], sigma_within, n_members)
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
        probs = grm_probs_norm(np.array([theta_val]), it["a"], it["thresholds"])
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
print(f"Bayesian ability recovery for {PROJECT}: r={r:.3f}, RMSE={rmse:.3f}")

plt.figure(figsize=(8,8))
sizes = np.sqrt(team_sizes_new) * 20
plt.errorbar(theta_team_new, theta_est_new,
             yerr=[theta_est_new-theta_ci_low, theta_ci_high-theta_est_new],
             fmt='o', alpha=0.7, capsize=3, markersize=0)
plt.scatter(theta_team_new, theta_est_new,
            s=sizes, c=team_sizes_new, cmap="viridis", alpha=0.8, edgecolor="k")
plt.colorbar(label="Team size")
plt.plot([-3,3],[-3,3],'r--')
plt.xlabel("True team θ")
plt.ylabel("Posterior mean θ (fixed items)")
plt.title(f"Team θ recovery with Bayesian updating — {PROJECT}")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(plot_dir, f"team_theta_recovery_bayes_sizes_{PROJECT}.png"), dpi=150)
plt.close()

# ============================================================
# 6. Posterior density curves per team (exact)
# ============================================================
plt.figure(figsize=(12,6))
for t in range(L):
    probs = team_posterior(Y_new[team_id_new==t], calibrated_items, theta_grid, prior_sd=0.5)
    plt.plot(theta_grid, probs, label=f"Team {t} (n={team_sizes_new[t]})")

plt.xlabel("θ")
plt.ylabel("Posterior density")
plt.title(f"Posterior of team θ (exact, fixed items) — {PROJECT}\n(n = team size)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1, borderaxespad=0.)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"team_theta_posteriors_exact_sizes_{PROJECT}.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 7. Correlation between team size and uncertainty
# ============================================================
ci_widths = theta_ci_high - theta_ci_low
cor_size_unc = np.corrcoef(team_sizes_new, ci_widths)[0,1]
print(f"Correlation between team size and CI width ({PROJECT}): {cor_size_unc:.3f}")

plt.figure(figsize=(7,5))
plt.scatter(team_sizes_new, ci_widths, alpha=0.7)
plt.xlabel("Team size (n)")
plt.ylabel("95% CI width for θ")
plt.title(f"Team size vs. posterior uncertainty — {PROJECT}")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(plot_dir, f"team_size_vs_uncertainty_{PROJECT}.png"), dpi=150)
plt.close()


import seaborn as sns

# Assume Y_new (N x J matrix) exists from your new simulation
# Reshape to long format for plotting
df_long = pd.DataFrame(Y_new, columns=[f"item{j+1}" for j in range(J)])
df_long = df_long.melt(var_name="item", value_name="response")

plt.figure(figsize=(12,6))
sns.countplot(data=df_long, x="response", hue="item", palette="tab10")
plt.xlabel("Response category")
plt.ylabel("Count")
plt.title(f"Distribution of observed categories across items — {PROJECT}")
plt.legend(title="Item", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"category_frequencies_{PROJECT}.png"), dpi=150)
plt.close()

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_team_caterpillar(theta_est, ci_low, ci_high, team_sizes, project, plot_dir,
                          sort_desc=True, fname_prefix="team_caterpillar"):
    """
    Caterpillar plot of team-level posterior means with 95% CIs against a zero line.

    Parameters
    ----------
    theta_est : array-like (L,)
        Posterior mean (or median) theta per team.
    ci_low, ci_high : array-like (L,)
        Lower and upper 95% CI per team (same scale as theta_est).
    team_sizes : array-like (L,)
        Team sizes (n per team), used for sorting and coloring.
    project : str
        Project name for title/filename.
    plot_dir : str
        Directory to save the figure.
    sort_desc : bool, default True
        If True, sort by team size descending (largest at top).
    fname_prefix : str
        Prefix for the output filename.
    """
    theta_est = np.asarray(theta_est)
    ci_low = np.asarray(ci_low)
    ci_high = np.asarray(ci_high)
    team_sizes = np.asarray(team_sizes)

    # sort by team size
    order = np.argsort(team_sizes)
    if sort_desc:
        order = order[::-1]

    theta_ord = theta_est[order]
    low_ord = ci_low[order]
    high_ord = ci_high[order]
    n_ord = team_sizes[order]
    idx = np.arange(len(theta_ord))  # 0 .. L-1

    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, max(6, len(theta_ord) * 0.2)))  # auto height if many teams

    # zero reference line
    plt.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.9)

    # horizontal error bars (CIs)
    plt.hlines(idx, low_ord, high_ord, color="gray", alpha=0.7, linewidth=2)

    # scatter points colored by team size
    sc = plt.scatter(theta_ord, idx, c=n_ord, cmap="viridis", s=np.sqrt(n_ord) * 18,
                     alpha=0.9, edgecolor="k")

    cbar = plt.colorbar(sc)
    cbar.set_label("Team size (n)")

    # labels / title
    plt.yticks(idx, [f"Team {i}" for i in order])  # show original team indices
    plt.xlabel("Posterior mean θ (team)")
    plt.ylabel("Teams (sorted by size)")
    plt.title(f"Team posterior θ vs 0 (95% CI) — {project}")

    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    outpath = os.path.join(plot_dir, f"{fname_prefix}_{project}.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")

plot_team_caterpillar(
    theta_est=theta_est_new,
    ci_low=theta_ci_low,
    ci_high=theta_ci_high,
    team_sizes=team_sizes_new,
    project=PROJECT,
    plot_dir=plot_dir,
    sort_desc=True,                         # largest teams at top
    fname_prefix="team_posterior_caterpillar"
)
