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
    """l_opt/I en fonction de l'énergie de liaison (échelle log-log).

    Couleur = statistique (vert : bosons, violet : fermions) ;
    marqueur plein = cette thèse, vide = littérature.
    """
    # (|f| en MHz, l_opt/I en a0/(W/cm²), incertitude)
    sr88 = [(23.99, 9.2e3, 2.4e3), (222.70, 56, 10)]
    sr88_blatt = [(23.932, 15.8e3, 0.8e3), (222.161, 68, 3)]          # Blatt 2011
    sr84 = [(228, 228, 42), (1143, 0.95, 0.12)]                        # Reschovsky 2018
    sr87 = [(259.01, 3.4, 1.9), (276.83, 3.1, 0.5), (377.2, 0.49, 0.35)]
    # Han 2018 : « l_opt ~ 10 a0 at I_PA = 1 W/cm² » en supposant eta ~ 1,
    # raies à -791 et -796 MHz ; ordre de grandeur, sans barre d'erreur
    yb173 = (793, 10)

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.set_xscale("log")
    ax.set_yscale("log")

    def points(data, fmt, couleur, plein, label, ms=7):
        x, y, e = zip(*data)
        e_bas = [min(ei, 0.9 * yi) for yi, ei in zip(y, e)]   # reste dans l'échelle log
        ax.errorbar(x, y, yerr=[e_bas, e], fmt=fmt, ms=ms, color=couleur, ecolor=couleur,
                    mfc=couleur if plein else "none", mec=couleur, mew=1.5, label=label,
                    zorder=4 if plein else 3)

    points(sr88, "o", VERT, True, r"$^{88}$Sr, cette thèse")
    points(sr88_blatt, "o", VERT, False, r"$^{88}$Sr, Blatt 2011", ms=10)
    points(sr84, "s", VERT, False, r"$^{84}$Sr, Reschovsky 2018", ms=8)
    points(sr87, "^", VIOLET, True, r"$^{87}$Sr, cette thèse", ms=8)
    ax.plot(*yb173, "D", ms=8, mfc="none", mec=VIOLET, mew=1.5,
            label=r"$^{173}$Yb, Han 2018", zorder=3)

    # écart bosons / fermions à énergie de liaison comparable
    ax.annotate("", xy=(245, 4.5), xytext=(245, 170),
                arrowprops={"arrowstyle": "-|>", "color": ENCRE, "lw": 1.2})
    ax.text(200, 28, "÷ 10 à 100", color=ENCRE, fontsize=10, ha="right", va="center")

    ax.annotate("ordre de grandeur", yb173, xytext=(0, 9), textcoords="offset points",
                color=VIOLET, fontsize=8, ha="center")
    ax.annotate("nœud", (1143, 0.95), xytext=(10, 0), textcoords="offset points",
                color=GRIS, fontsize=8, ha="left", va="center")

    ax.set_xlim(15, 2000)
    ax.set_ylim(0.1, 1e5)
    ax.set_xlabel("Énergie de liaison |f| (MHz)")
    ax.set_ylabel(r"$l_{\mathrm{opt}}/I$  ($a_0$/(W/cm$^2$))")

    # deux légendes : bosons puis fermions
    poignees, noms = ax.get_legend_handles_labels()
    par_nom = dict(zip(noms, poignees))
    for titre, couleur, cles, ancre in [
            ("Bosons", VERT, noms[1:4], (0.6, 1.0)),
            ("Fermions", VIOLET, [noms[4], noms[0]], (0.6, 0.70))]:
        leg = ax.legend([par_nom[k] for k in cles], cles, frameon=False, fontsize=8.5,
                        loc="upper left", bbox_to_anchor=ancre, title=titre,
                        title_fontsize=9.5, alignment="left")
        leg.get_title().set_color(couleur)
        ax.add_artist(leg)
    fig.tight_layout()
    fig.savefig(SORTIE / "lopt_comparaison.pdf", transparent=True)
    fig.savefig(SORTIE / "lopt_comparaison.png", transparent=True, dpi=428)
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
