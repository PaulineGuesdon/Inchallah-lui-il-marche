"""Graphes de comparaison des raies de PA pour la soutenance.

Valeurs reprises des tableaux tab:88Sr_results et tab:summary_fits_PA_final du
chapitre 3 (méthode 2 = déclin N(t)) et des références citées dans la thèse.
Lancer depuis n'importe où :  python3 graphes_PA.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

SORTIE = Path(__file__).resolve().parent.parent / "figures"

# Couleurs du thème beamer « strontium »
VIOLET = "#800080"
VERT = "#00AA44"
GRIS = "#8A8F98"
ENCRE = "#2A2232"

if any("Fira Sans" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Fira Sans"
plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRIS,
    "axes.labelcolor": ENCRE,
    "xtick.color": ENCRE,
    "ytick.color": ENCRE,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def graphe_lopt():
    """l_opt/I en fonction de l'énergie de liaison (échelle log-log)."""
    # (|f| en MHz, l_opt/I en a0/(W/cm²), incertitude, étiquette)
    sr88 = [(23.99, 9.2e3, 2.4e3, r"$\nu=-1$"), (222.70, 56, 10, r"$\nu=-2$")]
    sr87 = [(259.01, 3.4, 1.9, "259 MHz"), (276.83, 3.1, 0.5, "277 MHz"),
            (377.2, 0.49, 0.35, "377 MHz")]
    litt = [(23.932, 15.8e3, 0.8e3, ""), (222.161, 68, 3, ""),
            (228, 228, 42, r"$^{84}$Sr"), (1143, 0.95, 0.12, r"$^{84}$Sr")]

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.set_xscale("log")
    ax.set_yscale("log")

    x, y, e, _ = zip(*litt)
    ax.errorbar(x, y, yerr=e, fmt="s", ms=8, mfc="none", mec=GRIS, ecolor=GRIS,
                mew=1.5, label="Littérature", zorder=2)
    for xi, yi, _, t in litt:
        if t:
            ax.annotate(t, (xi, yi), xytext=(7, -3), textcoords="offset points",
                        color=GRIS, fontsize=9)

    x, y, e, _ = zip(*sr88)
    ax.errorbar(x, y, yerr=e, fmt="o", ms=7, color=VERT, label=r"$^{88}$Sr (cette thèse)",
                zorder=3)
    for xi, yi, _, t in sr88:
        ax.annotate(t, (xi, yi), xytext=(-8, 8), textcoords="offset points",
                    color=VERT, fontsize=9, ha="right")

    x, y, e, _ = zip(*sr87)
    # barres basses tronquées pour rester dans l'échelle log
    e_bas = [min(ei, 0.9 * yi) for yi, ei in zip(y, e)]
    ax.errorbar(x, y, yerr=[e_bas, e], fmt="^", ms=8, color=VIOLET,
                label=r"$^{87}$Sr (cette thèse)", zorder=4)

    ax.set_xlim(15, 2000)
    ax.set_ylim(0.1, 1e5)
    ax.set_xlabel("Énergie de liaison |f| (MHz)")
    ax.set_ylabel(r"$l_{\mathrm{opt}}/I$  ($a_0$/(W/cm$^2$))")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(SORTIE / "lopt_comparaison.pdf", transparent=True)
    plt.close(fig)


def graphe_elargissement():
    """Facteur d'élargissement eta = Gamma_fit / (Gamma_rad/2pi), échelle log."""
    # (étiquette, eta, incertitude, couleur)
    raies = [
        (r"$^{84}$Sr (litt.)", 1.5, 0.1, GRIS),
        (r"$^{88}$Sr", 22, 0, VERT),
        (r"$^{87}$Sr  277 MHz", 332, 17, VIOLET),
        (r"$^{87}$Sr  259 MHz", 493, 60, VIOLET),
        (r"$^{87}$Sr  377 MHz", 1349, 155, VIOLET),
    ]
    noms, eta, err, couleurs = zip(*raies)
    pos = range(len(raies))[::-1]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.barh(list(pos), eta, xerr=list(err), color=couleurs, height=0.6,
            error_kw={"ecolor": ENCRE, "capsize": 3, "lw": 1})
    for p, v in zip(pos, eta):
        ax.text(v * 1.35, p, f"{v:g}", va="center", color=ENCRE, fontsize=10)
    ax.set_yticks(list(pos))
    ax.set_yticklabels(noms)
    ax.set_xscale("log")
    ax.set_xlim(1, 5000)
    ax.set_xlabel(r"$\eta = \Gamma_{\mathrm{fit}} \,/\, (\Gamma_{\mathrm{rad}}/2\pi)$")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(SORTIE / "elargissement_eta.pdf", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    SORTIE.mkdir(exist_ok=True)
    graphe_lopt()
    graphe_elargissement()
    print("Figures écrites dans", SORTIE)
