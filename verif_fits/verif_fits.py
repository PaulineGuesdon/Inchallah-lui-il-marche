"""Refit des spectres 87Sr normalisés (259, 276, 377 MHz) pour vérification.
259/276 : points extraits des PDF vectoriels composite_*.pdf ; 377 : resultats_normalises.csv."""
import pypdf, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
IMG = "/Users/pauline/Documents/GitHub/Inchallah-lui-il-marche/Manuscrit_these/Chapitres/Engineering highly entangled system of photoassociated 87Sr atoms/images/87Sr/"
CSV = "/Users/pauline/Documents/GitHub/Code-analyse/RE_ excel frequences 88/resultats_normalises.csv"
def lor(x, a, x0, g, off): return off + a / (1 + ((x - x0) / (g / 2))**2)

def from_pdf(fn, xticks, yticks):
    c = pypdf.PdfReader(IMG + fn).pages[0].get_contents().get_data().decode("latin1")
    m = re.search(r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re W n", c)
    X0, Y0, W, H = map(float, m.groups())
    xt = [float(a) for a, b in re.findall(r"([\d.]+) %s m \1 ([\d.]+) l B" % re.escape(m.group(2)), c)][:len(xticks)]
    yt = [float(a) for a in re.findall(r"%s ([\d.]+) m [\d.]+ \1 l B" % re.escape(m.group(1)), c)][:len(yticks)]
    ax, ay = np.polyfit(xt, xticks, 1), np.polyfit(yt, yticks, 1)
    seg = [(float(x), float(a), float(b)) for x, a, b in re.findall(r"([\d.]+) ([\d.]+) m \1 ([\d.]+) l S", c)]
    seg = [s for s in seg if X0 + .5 < s[0] < X0 + W - .5 and Y0 <= (s[1] + s[2]) / 2 <= Y0 + H]
    x = np.polyval(ax, [s[0] for s in seg]); y = np.polyval(ay, [(s[1] + s[2]) / 2 for s in seg])
    e = np.abs(np.polyval(ay, [s[2] for s in seg]) - np.polyval(ay, [s[1] for s in seg])) / 2
    # courbe verte du fit d'origine (premier chemin 2 J 1.6 w)
    i = c.find("2 J 1.6 w"); j = c.find(" S", i)
    pts = re.findall(r"([\d.]+) ([\d.]+) [ml]", c[i:j])
    cx = np.polyval(ax, [float(p[0]) for p in pts]); cy = np.polyval(ay, [float(p[1]) for p in pts])
    o = np.argsort(x); return x[o], y[o], e[o], cx, cy, None

def from_csv():
    d = pd.read_csv(CSV).dropna(subset=["Nb_atoms_norm"]).copy()
    d["det"] = 2 * d["index"] - 500
    d = d[d["Nb_atoms_norm"].between(0.3, 2) & (d["er_Nb_atoms_norm"] < 0.2 * d["Nb_atoms_norm"])].sort_values("det")
    x, y, e = d["det"].values, d["Nb_atoms_norm"].values, d["er_Nb_atoms_norm"].values
    keep = np.ones_like(x, bool); p = [-0.08, 376, 20, np.median(y)]
    for _ in range(10):  # même clipping 3 sigma que fit375.py
        p, _c = curve_fit(lor, x[keep], y[keep], p0=p)
        r = y - lor(x, *p); s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) < 3 * s
        if (new == keep).all(): break
        keep = new
    return x, y, e, None, None, keep

def fit(x, y, e, p0, x0_fixed=None):
    if x0_fixed is None:
        p, C = curve_fit(lor, x, y, p0=p0, sigma=e, absolute_sigma=True)
    else:
        f = lambda x, a, g, off: lor(x, a, x0_fixed, g, off)
        q, Cq = curve_fit(f, x, y, p0=[p0[0], p0[2], p0[3]], sigma=e, absolute_sigma=True)
        p = np.array([q[0], x0_fixed, q[1], q[2]]); C = np.zeros((4, 4))
        idx = [0, 2, 3]
        for a in range(3):
            for b in range(3): C[idx[a], idx[b]] = Cq[a, b]
    chi = np.sum(((y - lor(x, *p)) / e)**2) / (len(x) - (4 if x0_fixed is None else 3))
    s = max(1, np.sqrt(chi)); C = C * s**2
    a, x0, g, off = p
    J = np.array([-1 / off, 0, 0, a / off**2]); A, sA = -a / off, np.sqrt(J @ C @ J)
    return p, np.sqrt(np.diag(C)), A, sA, chi

lines = [
    ("-259 MHz", from_pdf("composite_259MHz.pdf", [250, 252.5, 255, 257.5, 260, 262.5, 265, 267.5, 270], [.5, .6, .7, .8, .9]), [-0.35, 259, 7, 0.9], 259.01, 0.31),
    ("-276 MHz", from_pdf("composite_276MHz.pdf", [272, 274, 276, 278, 280, 282], [.6, .65, .7, .75, .8, .85, .9, .95, 1.0]), [-0.3, 276.5, 5, 0.95], 276.83, 0.37),
    ("-377 MHz", from_csv(), [-0.08, 377, 20, 0.97], 377.2, 0.087),
]
fig, axs = plt.subplots(3, 1, figsize=(9, 17))
for ax, (name, (x, y, e, cx, cy, keep), p0, f_txt, A_txt) in zip(axs, lines):
    k = np.ones_like(x, bool) if keep is None else keep
    pf, ef, Af, sAf, chif = fit(x[k], y[k], e[k], p0)
    px, ex, Ax, sAx, chix = fit(x[k], y[k], e[k], p0, x0_fixed=f_txt)
    ax.errorbar(x[k], y[k], e[k], fmt="o", ms=3, color="purple", ecolor="0.6", lw=.8, label="points")
    if keep is not None: ax.plot(x[~k], y[~k], "o", mfc="none", color="purple", ms=3, label="exclus (3σ)")
    xx = np.linspace(x.min(), x.max(), 600)
    if cx is not None: ax.plot(cx, cy, color="limegreen", lw=4, alpha=.45, label="fit d'origine (figure)")
    ax.plot(xx, lor(xx, *pf), "k-", lw=1.3, label=f"refit centre libre: f0={pf[1]:.2f}±{ef[1]:.2f}, Γ={pf[2]:.2f}±{ef[2]:.2f} MHz, prof.={Af:.3f}±{sAf:.3f}, χ²r={chif:.2f}")
    ax.plot(xx, lor(xx, *px), "r--", lw=1.3, label=f"refit centre fixé à {f_txt}: Γ={px[2]:.2f}±{ex[2]:.2f} MHz, prof.={Ax:.3f}±{sAx:.3f}, χ²r={chix:.2f}")
    ax.axhline(pf[3], color="k", ls=":", lw=.8); ax.axhline(1, color="0.7", lw=.6)
    ax.axvline(f_txt, color="r", ls=":", lw=.8)
    ax.set_title(f"{name} — profondeur dans le texte avant correction : {A_txt}", fontsize=10)
    ax.set_xlabel("|Δ| (MHz)"); ax.set_ylabel("rapport N_PA / N_sans PA"); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False)
    if keep is not None: ax.set_ylim(0.75, 1.1)
    print(name, "libre:", np.round(pf, 3), np.round(ef, 3), "prof", round(Af, 3), round(sAf, 3), "chi2", round(chif, 2))
    print(name, "fixé :", np.round(px, 3), np.round(ex, 3), "prof", round(Ax, 3), round(sAx, 3), "chi2", round(chix, 2))
plt.tight_layout(); plt.savefig("/Users/pauline/Documents/GitHub/Inchallah-lui-il-marche/verif_fits/verif_fits_87Sr.png", dpi=130)
