"""Figures de la présentation SFO, étape d'animation par étape d'animation.

L'export PDF de PowerPoint montre l'état final de chaque diapo, avec tous les
éléments animés superposés. Ce script fabrique une copie du pptx par étape k
(chaque diapo montrée après ses k premiers clics), l'exporte en PDF avec
PowerPoint, puis découpe les régions de figures utiles en petits PDF.

  python3 sfo_etapes.py etapes     # copies par étape + export PDF (lent)
  python3 sfo_etapes.py decoupes   # découpes dans figures/sfo/
"""
import subprocess
import sys
import zipfile
from pathlib import Path

from lxml import etree
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject

ICI = Path(__file__).resolve().parent.parent
PPTX = ICI / "SFO_VF_Guesdon.pptx"
ETAPES = ICI / "out" / "sfo_etapes"
FIGURES = ICI / "figures" / "sfo"
NB_ETAPES = 11  # la diapo la plus animée (10) a 10 clics : étapes 0 à 10

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}
FORMES = ("sp", "pic", "grpSp", "cxnSp", "graphicFrame")


def clics(arbre):
    """Liste des clics ; chaque clic = liste ordonnée de (classe, spid)."""
    seq = arbre.find(".//p:cTn[@nodeType='mainSeq']/p:childTnLst", NS)
    if seq is None:
        return []
    resultat = []
    for par in seq.findall("p:par", NS):
        effets = []
        for ctn in par.iter("{%s}cTn" % NS["p"]):
            classe = ctn.get("presetClass")
            if classe not in ("entr", "exit"):
                continue
            cible = ctn.find(".//p:spTgt", NS)
            if cible is not None:
                effets.append((classe, cible.get("spid")))
        resultat.append(effets)
    return resultat


def caches_a_l_etape(liste_clics, k):
    """Identifiants des formes invisibles après k clics."""
    premier = {}
    for effets in liste_clics:
        for classe, spid in effets:
            premier.setdefault(spid, classe)
    caches = {spid for spid, classe in premier.items() if classe == "entr"}
    for effets in liste_clics[:k]:
        for classe, spid in effets:
            if classe == "entr":
                caches.discard(spid)
            else:
                caches.add(spid)
    return caches


def retirer_formes(arbre, caches):
    for nom in FORMES:
        for el in arbre.findall(".//p:cSld//p:%s" % nom, NS):
            cnv = el.find("./*/p:cNvPr", NS)
            if cnv is None or cnv.get("id") not in caches:
                continue
            # les équations sont enveloppées dans mc:AlternateContent
            cible = el
            while cible.getparent() is not None and cible.getparent().tag.startswith("{%s}" % NS["mc"]):
                cible = cible.getparent()
            if cible.getparent() is not None:
                cible.getparent().remove(cible)


# Textes anglais retirés de toutes les étapes : ils sont réécrits en français dans le beamer
# (numéro de diapo -> identifiants p:cNvPr)
A_TRADUIRE = {
    4: {"67", "68", "69"},
    5: {"13", "19"},
    6: {"9", "27", "165", "11", "53", "166", "167", "45"},
    7: {"68", "55", "171", "174", "20"},
    8: {"2", "4", "5", "15", "16", "11", "12"},
    10: {"9", "10", "11", "12", "21", "22", "24", "26", "50", "60", "61", "68", "71"},
    11: {"5", "7", "9", "14", "21", "30"},
}


def version_etape(k, sortie):
    with zipfile.ZipFile(PPTX) as src, zipfile.ZipFile(sortie, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            donnees = src.read(info.filename)
            if info.filename.startswith("ppt/slides/slide") and info.filename.endswith(".xml"):
                diapo = int(info.filename[len("ppt/slides/slide"):-len(".xml")])
                arbre = etree.fromstring(donnees)
                caches = caches_a_l_etape(clics(arbre), k) | A_TRADUIRE.get(diapo, set())
                retirer_formes(arbre, caches)
                timing = arbre.find("p:timing", NS)
                if timing is not None:
                    arbre.remove(timing)
                donnees = etree.tostring(arbre, xml_declaration=True, encoding="UTF-8", standalone=True)
            dst.writestr(info, donnees)


def exporter_pdf(pptx, pdf):
    script = f'''
    tell application "Microsoft PowerPoint"
      open POSIX file "{pptx}"
      delay 3
      save active presentation in (POSIX file "{pdf}") as save as PDF
      close active presentation saving no
    end tell'''
    subprocess.run(["osascript", "-e", script], check=True)


def etapes():
    ETAPES.mkdir(parents=True, exist_ok=True)
    for k in range(NB_ETAPES):
        pptx = ETAPES / f"etape_{k:02d}.pptx"
        version_etape(k, pptx)
        exporter_pdf(pptx, ETAPES / f"etape_{k:02d}.pdf")
        print("étape", k, "exportée")


# (nom, diapo, étape, x0, y0, x1, y1) : région en fraction de la page, origine en haut à gauche
DECOUPES = [
    *[(f"zeeman_{k}", 4, k, 0.02, 0.16, 0.86, 0.89) for k in (0, 1, 2, 4, 6)],
    *[(f"tls_{k}", 5, k, 0.0, 0.13, 0.66, 1.0) for k in (0, 2)],
    *[(f"atomes_{k}", 6, k, 0.01, 0.125, 0.23, 0.42) for k in (0, 1, 2)],
    ("rabi", 6, 0, 0.25, 0.31, 0.80, 0.85),
    ("ramsey_schema", 7, 0, 0.04, 0.40, 0.80, 0.64),
    *[(f"ramsey_franges_{k}", 8, k, 0.02, 0.22, 0.70, 0.69) for k in (0, 1)],
    ("ramsey_contraste", 8, 1, 0.70, 0.55, 0.99, 0.95),
    ("detecteurs", 9, 0, 0.38, 0.21, 0.57, 0.47),
    ("noncomm_schema", 10, 0, 0.0, 0.21, 0.70, 0.62),
    *[(f"bloch_{k}", 10, k, 0.36, 0.58, 0.60, 0.98) for k in range(3, 10)],
    ("noncomm_resultats", 11, 0, 0.10, 0.34, 0.86, 0.82),
    ("ramsey4_schema", 15, 1, 0.03, 0.30, 0.99, 0.74),
    ("ramsey4_resultats", 16, 0, 0.0, 0.235, 0.93, 0.98),
]

# Les diapos masquées n'apparaissent pas dans le PDF
DIAPOS_MASQUEES = {3}


def page_de(diapo):
    return diapo - 1 - sum(1 for m in DIAPOS_MASQUEES if m < diapo)


def decoupes():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for nom, diapo, k, x0, y0, x1, y1 in DECOUPES:
        page = PdfReader(ETAPES / f"etape_{k:02d}.pdf").pages[page_de(diapo)]
        l, b, r, t = (float(v) for v in page.mediabox)
        w, h = r - l, t - b
        gauche, bas = l + x0 * w, t - y1 * h
        largeur, hauteur = (x1 - x0) * w, (y1 - y0) * h
        # on ramène le coin de la découpe en (0, 0) : certains lecteurs ignorent l'origine des boîtes
        page.add_transformation(Transformation().translate(-gauche, -bas))
        for attr in ("mediabox", "cropbox", "trimbox", "artbox", "bleedbox"):
            setattr(page, attr, RectangleObject((0, 0, largeur, hauteur)))
        ecrivain = PdfWriter()
        ecrivain.add_page(page)
        with open(FIGURES / f"{nom}.pdf", "wb") as f:
            ecrivain.write(f)
        print(nom)


if __name__ == "__main__":
    {"etapes": etapes, "decoupes": decoupes}[sys.argv[1]]()
