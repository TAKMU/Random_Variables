"""Appendix: simulation of the random variables in the multimodal / QML staging hypothesis,
for a pooled cohort of K multimodal datasets (TCGA-BRCA + 4 further datasets).

Only invasive cancers (stages I-IV) are included.
PLACEHOLDERS: dataset sizes, per-dataset stage laws and every error rate except the
MRI-only one (~0.45, from the prototype's 0.5-0.6 accuracy on TCGA/Duke).
Replace them with real counts and held-out predictions when available.
"""
import numpy as np
from scipy.stats import norm, multivariate_normal

rng = np.random.default_rng(2026)

# Label support: TNM stages I..IV. Healthy and in-situ (stage 0) patients are excluded.
labels = np.array([1, 2, 3, 4])

datasets = {            # name: (usable invasive patients, stage I..IV law)  -- placeholders except TCGA's law
    "TCGA-BRCA": (90,  np.array([0.17, 0.57, 0.23, 0.03])),   # approx. TCGA-BRCA mix
    "D2":        (120, np.array([0.30, 0.50, 0.17, 0.03])),   # earlier-stage cohort (placeholder)
    "D3":        (80,  np.array([0.17, 0.57, 0.23, 0.03])),
    "D4":        (60,  np.array([0.10, 0.45, 0.35, 0.10])),   # later-stage cohort (placeholder)
    "D5":        (50,  np.array([0.20, 0.55, 0.20, 0.05])),
}
names = list(datasets)
sizes = np.array([datasets[k][0] for k in names])
n = int(sizes.sum())
pi = sizes / n                                          # dataset shares (mixture weights), not a random variable here
p_label = sum(pi[j] * datasets[k][1] for j, k in enumerate(names))   # mixture law of Y

# Marginal error probabilities P(E_m = 1)
p_err = {"M": 0.45,   # MRI-only model (prototype, acc ~0.55)
         "L": 0.42,   # late fusion          (placeholder)
         "I": 0.40,   # intermediate fusion  (placeholder)
         "A": 0.38,   # early fusion / multimodal AE + classical head (placeholder)
         "Q": 0.45,   # VQC on the same AE latent, finite shots     (placeholder)
         "R": 0.43}   # RNA-seq-only model                           (placeholder)
rho = 0.6             # loading on shared patient difficulty -> errors are positively correlated

# ---- simulate one pooled cohort -------------------------------------------
src = np.repeat(np.arange(1, len(names) + 1), sizes)    # dataset of origin, kept only for the check below
Y = np.concatenate([rng.choice(labels, size=datasets[k][0], p=datasets[k][1]) for k in names])
Z = rng.standard_normal(n)                              # patient difficulty (part of omega)
E = {}
for m, p in p_err.items():
    eps = rng.standard_normal(n)                        # model-specific noise; for Q includes shot noise
    E[m] = (rho * Z + np.sqrt(1 - rho**2) * eps > norm.ppf(1 - p)).astype(int)
D = E["A"] - E["Q"]                                     # paired difference, positive = VQC better

# ---- model (closed-form) laws ---------------------------------------------
def p_joint_err(pa, pq, r=rho**2):
    """P(E_A=1, E_Q=1) under the Gaussian latent model."""
    ta, tq = norm.ppf(1 - pa), norm.ppf(1 - pq)
    return multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-ta, -tq])

p11 = p_joint_err(p_err["A"], p_err["Q"])
pD = {1: p_err["A"] - p11, -1: p_err["Q"] - p11}
pD[0] = 1 - pD[1] - pD[-1]

def ecdf_at(x, sample):
    return float(np.mean(sample <= x))

print("n =", n, " pi =", pi.round(3))
print("Y law:", p_label.round(3), " model F:", np.cumsum(p_label).round(3),
      " empirical F:", [round(ecdf_at(k, Y), 3) for k in labels])
for m in p_err:
    print(f"E_{m} model F(0)={1-p_err[m]:.3f} empirical F(0)={ecdf_at(0, E[m]):.3f}  mean={E[m].mean():.3f}")
print("D model P(-1,0,1)=", [round(pD[k], 3) for k in (-1, 0, 1)],
      " model F:", np.cumsum([pD[-1], pD[0], pD[1]]).round(3))
print("D empirical F:", [round(ecdf_at(k, D), 3) for k in (-1, 0, 1)],
      " mean=", round(D.mean(), 3), " discordant share=", round(np.mean(D != 0), 3))
print("most common class:", labels[p_label.argmax()], " baseline error =", round(1 - p_label.max(), 3))
print("E[D] model =", round(p_err["A"] - p_err["Q"], 3), " Var(D) model =",
      round(pD[1] + pD[-1] - (pD[1] - pD[-1])**2, 3))
print("Chebyshev bound D, eps=.07:", round((pD[1] + pD[-1] - (pD[1] - pD[-1])**2) / (n * 0.07**2), 3))
print("Chebyshev bound E, p=.4, eps=.1:", round(0.24 / (n * 0.01), 3))
# Pooling check: error rate and label mix per dataset of origin
print("per-dataset error rate of E_A (sample):",
      {k: round(E["A"][src == j + 1].mean(), 2) for j, k in enumerate(names)})
print("per-dataset share of stage III-IV:", {k: round(np.mean(Y[src == j + 1] >= 3), 2) for j, k in enumerate(names)})

# ---- plots: model F (solid) and empirical F (dashed), jumps marked ---------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def step_cdf(ax, support, F, lo, hi, **kw):
    """Right-continuous step CDF: filled dot at (x_k, F_k), open dot at (x_k, F_{k-1})."""
    xs = [lo] + list(support) + [hi]
    Fs = [0.0] + list(F)
    for k in range(len(Fs)):
        ax.hlines(Fs[k], xs[k], xs[k + 1], **kw)
    for k, x in enumerate(support):
        ax.plot(x, F[k], "o", color=kw.get("colors"), ms=5)
        ax.plot(x, Fs[k], "o", mfc="white", color=kw.get("colors"), ms=5)

def emp(sample, support):
    return [ecdf_at(x, sample) for x in support]

fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharey=True)
axs = axs.ravel()   # now axs[0]..axs[3] are individual Axes (row-major order)

step_cdf(axs[0], labels, np.cumsum(p_label), 0, 5, colors="k")
step_cdf(axs[0], labels, emp(Y, labels), 0, 5, colors="tab:blue", linestyles="--")
axs[0].set_title("Y (TNM stage I-IV)")

for m, c in zip("MRLIA", ["0.75", "0.55", "tab:orange", "tab:green", "tab:blue"]):
    step_cdf(axs[1], [0, 1], [1 - p_err[m], 1], -0.5, 1.5, colors=c)
    axs[1].plot([], [], color=c, label=f"E_{m}: F(0) = {1 - p_err[m]:.2f}")
axs[1].legend(loc="lower right", fontsize=8)
axs[1].set_title("E coordinates (model)")

step_cdf(axs[2], [0, 1], [1 - p_err["Q"], 1], -0.5, 1.5, colors="k")
step_cdf(axs[2], [0, 1], emp(E["Q"], [0, 1]), -0.5, 1.5, colors="tab:blue", linestyles="--")
axs[2].set_title("E_Q (VQC)")

step_cdf(axs[3], [-1, 0, 1], np.cumsum([pD[-1], pD[0], pD[1]]), -1.5, 1.5, colors="k")
step_cdf(axs[3], [-1, 0, 1], emp(D, [-1, 0, 1]), -1.5, 1.5, colors="tab:blue", linestyles="--")
axs[3].set_title("D = E_A - E_Q")

for ax in (axs[0], axs[2]):          # left column of the 2x2 grid
    ax.set_ylabel("F_X(x) = P(X <= x)")


fig.tight_layout()
fig.savefig("cdf_plots.png", dpi=150)

# To use real held-out predictions instead of the simulation, load a CSV with columns
# patient_id, dataset, y_true, pred_M, pred_R, pred_L, pred_I, pred_A, pred_Q  and set
#   E[m] = (df[f"pred_{m}"] != df.y_true).astype(int).to_numpy()