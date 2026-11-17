"""Transcription de la soutenance Beamer (soutenance.tex) en PowerPoint modifiable.

Textes, puces, tableaux et blocs sont des objets PowerPoint natifs ; les équations
sont des équations PowerPoint (OMML, via pandoc) ; les figures sont des images PNG
tirées des PDF du manuscrit ; les pas \\item<n-> / \\only<n> deviennent des
animations « apparaître » au clic.

  python soutenance_pptx.py        # écrit ../soutenance.pptx

Dépendances : python-pptx, lxml, Pillow, pandoc, ghostscript, lualatex.
"""
import copy
import re
import subprocess
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ICI = Path(__file__).resolve().parent.parent
PDF_BEAMER = ICI / "out" / "soutenance.pdf"
MEDIA = ICI / "out" / "pptx_media"
SORTIE = ICI / "soutenance.pptx"
CHEMINS_FIGURES = [
    ICI / "figures",
    ICI / "../Manuscrit_these/appendix/images",
    ICI / "../Manuscrit_these/Chapitres/Gas preparation and spin control/images",
    ICI / "../Manuscrit_these/Chapitres/Ramsey interferometers on qudit/article Coherent Manip",
    ICI / "../Manuscrit_these/Chapitres/Engineering highly entangled system of photoassociated 87Sr atoms/images",
]

# ---------------------------------------------------------------------
#  Géométrie : on garde les coordonnées Beamer (16 cm × 9 cm) et on agrandit
# ---------------------------------------------------------------------
ECHELLE = 13.333 * 2.54 / 16          # 16 cm Beamer -> 13,333 pouces PowerPoint
TOTAL = 40                            # nombre de diapos numérotées (hors annexes)


def cm(x):
    return Emu(int(round(x * ECHELLE * 360000)))


def pt(beamer_pt):
    return Pt(round(beamer_pt * 2.0, 1))


TAILLES = {"tiny": 6, "scriptsize": 8, "footnotesize": 9, "small": 10, "normalsize": 11,
           "large": 12, "Large": 14.4, "LARGE": 17.28, "huge": 20.74}

# ---------------------------------------------------------------------
#  Couleurs du thème strontium
# ---------------------------------------------------------------------
COULEURS = {
    "nuit": (0x3E, 0x2A, 0x57), "encre": (0x2A, 0x22, 0x32), "papier": (0xFA, 0xF7, 0xF1),
    "voile": (0xEF, 0xEA, 0xE0), "gris": (0x8A, 0x8F, 0x98), "violet": (0x80, 0x00, 0x80),
    "vert": (0x00, 0xAA, 0x44), "white": (0xFF, 0xFF, 0xFF), "black": (0, 0, 0),
    "blue": (0x00, 0x00, 0xFF), "red": (0xFF, 0x00, 0x00),
    "laserrouge": (0xFF, 0x4B, 0x4B), "laserbleu": (0x4D, 0x8D, 0xFF),
    "laserviolet": (0xC1, 0x7D, 0xFF), "laservert": (0x3D, 0xDC, 0x84),
}
COULEURS["accent"] = COULEURS["violet"]


def couleur(nom):
    """Couleur xcolor : 'violet', 'vert!70!black', 'gris!50', 'white!60'."""
    morceaux = nom.split("!")
    c = COULEURS[morceaux[0]]
    i = 1
    while i < len(morceaux):
        p = float(morceaux[i]) / 100
        autre = COULEURS[morceaux[i + 1]] if i + 1 < len(morceaux) else COULEURS["white"]
        c = tuple(p * a + (1 - p) * b for a, b in zip(c, autre))
        i += 2
    return RGBColor(*(int(round(v)) for v in c))


POLICE = "Avenir Next"
POLICE_FORTE = "Avenir Next Medium"
POLICE_GRASSE = "Avenir Next Demi Bold"
POLICE_FINE = "Avenir Next Ultra Light"

# ---------------------------------------------------------------------
#  Petit lecteur de LaTeX -> morceaux de texte enrichi
# ---------------------------------------------------------------------
EQUATIONS = []      # code LaTeX de chaque équation, repéré par un marqueur dans le texte
MARQUEUR = "\u27e6EQ{}\u27e7"

UNITES = {r"\micro": "µ", r"\percent": "%", r"\ohm": "Ω"}


def _num_fr(n):
    n = n.strip()
    if "e" in n:
        mant, exp = n.split("e")
        exp = exp.replace("+", "")
        mant = mant.replace(".", ",")
        if mant in ("", "1"):
            return [("10", {}), (_moins(exp), {"sup": True})]
        return [(mant + " × 10", {}), (_moins(exp), {"sup": True})]
    return [(_moins(n.replace(".", ",")), {})]


def _moins(s):
    return s.replace("-", "\u2212")


def _unite(u):
    """'cm^{-3}' -> morceaux avec exposants ; '\\micro K' -> 'µK'."""
    for k, v in UNITES.items():
        u = u.replace(k, v)
    u = u.replace(" ", "")
    morceaux = []
    for m in re.finditer(r"\^\{?([-0-9]+)\}?|([^\^]+)", u):
        if m.group(1):
            morceaux.append((_moins(m.group(1)), {"sup": True}))
        else:
            morceaux.append((m.group(2), {}))
    return morceaux


def _si(num, unite):
    return _num_fr(num) + [("\u202f", {})] + _unite(unite)


def math_pour_pandoc(code):
    """Adapte le LaTeX des diapos à ce que comprend texmath (pandoc)."""
    code = re.sub(r"\\SI\{([^}]*)\}\{([^}]*)\}", lambda m: _si_math(m.group(1), m.group(2)), code)
    code = re.sub(r"\\ket\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\\left|\1\\right\\rangle", code)
    code = code.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    return code


def _si_math(num, unite):
    num = num.strip()
    if "e" in num:
        mant, exp = num.split("e")
        num = (mant + r"\times" if mant not in ("", "1") else "") + "10^{" + exp + "}"
    unite = unite.replace(r"\micro", r"\mu ")
    return num + r"\,\mathrm{" + unite + "}"


class Lecteur:
    """Convertit un fragment LaTeX en liste de (texte, format)."""

    COMMANDES_A_IGNORER = {"centering", "raggedright", "hfill", "par", "noindent", "relax",
                           "selectfont", "displaystyle"}

    def __init__(self, s):
        self.s = s
        self.i = 0

    def lire(self, fmt):
        sortie = []
        fmt = dict(fmt)
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "}":
                self.i += 1
                return sortie
            if c == "{":
                self.i += 1
                sortie += self.lire(fmt)
            elif c == "$":
                fin = self.s.index("$", self.i + 1)
                EQUATIONS.append(self.s[self.i + 1:fin])
                sortie.append((MARQUEUR.format(len(EQUATIONS) - 1), dict(fmt, math=True)))
                self.i = fin + 1
            elif c == "~":
                sortie.append(("\u00a0", fmt))
                self.i += 1
            elif c == "-" and self.s.startswith("--", self.i):
                sortie.append(("\u2013", fmt))
                self.i += 2
            elif c == "%":
                fin = self.s.find("\n", self.i)
                self.i = len(self.s) if fin < 0 else fin + 1
            elif c == "\\":
                sortie += self.commande(fmt)
            elif c in "\n\t":
                sortie.append((" ", fmt))
                self.i += 1
            else:
                j = self.i
                while j < len(self.s) and self.s[j] not in "{}$~\\%\n\t" and not self.s.startswith("--", j):
                    j += 1
                sortie.append((self.s[self.i:j], fmt))
                self.i = j
        return sortie

    def argument(self):
        while self.s[self.i] == " ":
            self.i += 1
        assert self.s[self.i] == "{", self.s[self.i:self.i + 30]
        prof, j = 0, self.i
        while True:
            if self.s[j] == "{":
                prof += 1
            elif self.s[j] == "}":
                prof -= 1
                if prof == 0:
                    break
            j += 1
        arg = self.s[self.i + 1:j]
        self.i = j + 1
        return arg

    def option(self):
        if self.i < len(self.s) and self.s[self.i] == "[":
            j = self.s.index("]", self.i)
            self.i = j + 1

    def commande(self, fmt):
        m = re.match(r"\\([A-Za-z]+|.)", self.s[self.i:])
        nom = m.group(1)
        self.i += len(m.group(0))
        if nom.isalpha():       # comme TeX : les blancs après un mot de contrôle sont mangés
            while self.i < len(self.s) and self.s[self.i] == " ":
                self.i += 1
        sous = lambda arg, f: Lecteur(arg).lire(f)
        if nom == "\\":
            self.option()
            while self.i < len(self.s) and self.s[self.i] in " \n\t":
                self.i += 1
            return [("\n", fmt)]
        if nom in (",", ";", " "):
            return [("\u202f" if nom == "," else " ", fmt)]
        if nom in ("qquad", "quad"):
            return [("\u2003\u2003" if nom == "qquad" else "\u2003", fmt)]
        if nom in ("ldots", "dots"):
            return [("…", fmt)]
        if nom == "textperiodcentered":
            return [("·", fmt)]
        if nom in ("%", "&", "_", "#"):
            return [(nom, fmt)]
        if nom == "alert":
            return sous(self.argument(), dict(fmt, couleur="violet"))
        if nom == "textbf":
            return sous(self.argument(), dict(fmt, gras=True))
        if nom in ("textit", "emph"):
            return sous(self.argument(), dict(fmt, italique=True))
        if nom == "underline":
            return sous(self.argument(), dict(fmt, souligne=True))
        if nom == "textcolor":
            c = self.argument()
            return sous(self.argument(), dict(fmt, couleur=c))
        if nom == "textsuperscript":
            return sous(self.argument(), dict(fmt, sup=True))
        if nom == "Sr":
            return [(self.argument(), dict(fmt, sup=True)), ("Sr", fmt)]
        if nom == "SI":
            num, unite = self.argument(), self.argument()
            return [(t, dict(fmt, **f)) for t, f in _si(num, unite)]
        if nom == "si":
            return [(t, dict(fmt, **f)) for t, f in _unite(self.argument())]
        if nom == "SIrange":
            a, b, u = self.argument(), self.argument(), self.argument()
            return [(t, dict(fmt, **f)) for t, f in _si(a, u) + [(" à ", {})] + _si(b, u)]
        if nom == "color":
            fmt["couleur"] = self.argument()
            return []
        if nom == "fontforte":
            fmt["forte"] = True
            return []
        if nom in TAILLES:
            fmt["taille"] = TAILLES[nom]
            return []
        if nom in ("vspace", "hspace", "phantom"):
            self.argument()
            return []
        if nom in self.COMMANDES_A_IGNORER:
            return []
        raise ValueError("commande inconnue : \\" + nom)


def riche(s, **fmt):
    """Morceaux de texte enrichi à partir de LaTeX."""
    return Lecteur(s.strip()).lire(fmt)


# ---------------------------------------------------------------------
#  Écriture du texte dans PowerPoint
# ---------------------------------------------------------------------
def _run(paragraphe, texte, fmt, base):
    f = dict(base)
    f.update(fmt)
    r = paragraphe.add_run()
    r.text = texte
    police = r.font
    police.size = pt(f.get("taille", 11))
    police.name = POLICE_GRASSE if f.get("gras") else POLICE_FORTE if f.get("forte") else POLICE
    police.italic = bool(f.get("italique"))
    police.underline = bool(f.get("souligne"))
    police.color.rgb = couleur(f.get("couleur", "encre"))
    if f.get("sup"):
        r._r.get_or_add_rPr().set("baseline", "30000")
    if f.get("sub"):
        r._r.get_or_add_rPr().set("baseline", "-25000")
    r._r.get_or_add_rPr().set("lang", "fr-FR")
    return r


def ecrire(paragraphe, morceaux, base):
    """Ajoute des morceaux ; un '\\n' devient un retour à la ligne dans le paragraphe."""
    for texte, fmt in morceaux:
        parts = texte.split("\n")
        for k, part in enumerate(parts):
            if k:
                paragraphe.add_line_break()
            if part:
                _run(paragraphe, part, fmt, base)


def _puce(paragraphe, niveau, numero=None):
    pPr = paragraphe._p.get_or_add_pPr()
    marge = [0, 0.45, 0.9][niveau]
    pPr.set("marL", str(int(cm(marge + 0.45))))
    pPr.set("indent", str(int(-cm(0.45))))
    if numero is not None:
        etree.SubElement(pPr, qn("a:buClr")).append(_srgb("violet"))
        f = etree.SubElement(pPr, qn("a:buFont"))
        f.set("typeface", POLICE_FORTE)
        a = etree.SubElement(pPr, qn("a:buAutoNum"))
        a.set("type", "arabicPeriod")
        a.set("startAt", str(numero))
        return
    clr = etree.SubElement(pPr, qn("a:buClr"))
    clr.append(_srgb("violet" if niveau == 1 else "vert"))
    etree.SubElement(pPr, qn("a:buSzPct")).set("val", "70000" if niveau == 1 else "90000")
    f = etree.SubElement(pPr, qn("a:buFont"))
    f.set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", "●" if niveau == 1 else "○")


def _srgb(nom):
    e = etree.Element(qn("a:srgbClr"))
    e.set("val", str(couleur(nom)))
    return e


def _espace_avant(paragraphe, points):
    paragraphe.space_before = Pt(points)


def zone_texte(diapo, x, y, w, h, ancre="haut", marge=0.0):
    tb = diapo.shapes.add_textbox(cm(x), cm(y), cm(w), cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    for cote in ("left", "right", "top", "bottom"):
        setattr(tf, "margin_" + cote, cm(marge))
    tf.vertical_anchor = {"haut": MSO_ANCHOR.TOP, "milieu": MSO_ANCHOR.MIDDLE,
                          "bas": MSO_ANCHOR.BOTTOM}[ancre]
    return tb


# ---------------------------------------------------------------------
#  Animations : « apparaître » au clic
# ---------------------------------------------------------------------
class Animations:
    def __init__(self):
        self.effets = []       # (clic, spid, paragraphe ou None)

    def ajouter(self, pas, forme, paragraphe=None):
        """pas = numéro Beamer (1 = visible d'emblée)."""
        if pas and pas > 1:
            self.effets.append((pas - 1, forme.shape_id, paragraphe))

    def ecrire(self, diapo):
        if not self.effets:
            return
        P = "http://schemas.openxmlformats.org/presentationml/2006/main"
        ids = iter(range(1, 10000))
        clics = sorted({c for c, _, _ in self.effets})
        x = ['<p:timing xmlns:p="%s"><p:tnLst><p:par><p:cTn id="%d" dur="indefinite" '
             'restart="never" nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek">'
             '<p:cTn id="%d" dur="indefinite" nodeType="mainSeq"><p:childTnLst>' % (P, next(ids), next(ids))]
        groupes = {}
        for clic in clics:
            x.append('<p:par><p:cTn id="%d" fill="hold"><p:stCondLst><p:cond delay="indefinite"/>'
                     '</p:stCondLst><p:childTnLst><p:par><p:cTn id="%d" fill="hold"><p:stCondLst>'
                     '<p:cond delay="0"/></p:stCondLst><p:childTnLst>' % (next(ids), next(ids)))
            for k, (c, spid, par) in enumerate(e for e in self.effets if e[0] == clic):
                groupes.setdefault(spid, set()).add(par is not None)
                cible = ('<p:spTgt spid="%d"><p:txEl><p:pRg st="%d" end="%d"/></p:txEl></p:spTgt>'
                         % (spid, par, par)) if par is not None else '<p:spTgt spid="%d"/>' % spid
                x.append('<p:par><p:cTn id="%d" presetID="1" presetClass="entr" presetSubtype="0" '
                         'fill="hold" grpId="0" nodeType="%s"><p:stCondLst><p:cond delay="0"/>'
                         '</p:stCondLst><p:childTnLst><p:set><p:cBhvr><p:cTn id="%d" dur="1" '
                         'fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
                         '<p:tgtEl>%s</p:tgtEl><p:attrNameLst><p:attrName>style.visibility'
                         '</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/>'
                         '</p:to></p:set></p:childTnLst></p:cTn></p:par>'
                         % (next(ids), "clickEffect" if k == 0 else "withEffect", next(ids), cible))
            x.append('</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>')
        x.append('</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl>'
                 '<p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" '
                 'delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
                 '</p:childTnLst></p:cTn></p:par></p:tnLst><p:bldLst>')
        formes_texte = {f.shape_id for f in diapo.shapes if f.has_text_frame}
        for spid, genres in groupes.items():
            if spid in formes_texte:
                x.append('<p:bldP spid="%d" grpId="0" %s/>' % (spid, 'build="p"' if True in genres else 'animBg="1"'))
        x.append('</p:bldLst></p:timing>')
        diapo._element.append(etree.fromstring("".join(x)))


# ---------------------------------------------------------------------
#  Images
# ---------------------------------------------------------------------
def trouver(nom):
    for d in CHEMINS_FIGURES:
        p = d / nom
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(nom)


def png(nom, fond=None):
    """PNG (environ 2400 px de large) d'une figure PDF ; fond opaque si demandé."""
    src = trouver(nom)
    if src.suffix.lower() == ".png":
        return src
    MEDIA.mkdir(parents=True, exist_ok=True)
    dst = MEDIA / (nom.replace("/", "__").replace(".pdf", "") + ("_opaque" if fond else "") + ".png")
    if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
        return dst
    largeur_pt = _largeur_pdf(src)
    dpi = max(150, min(600, int(2400 / (largeur_pt / 72))))
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pngalpha",
                    "-dUseCropBox", f"-r{dpi}", "-dFirstPage=1", "-dLastPage=1",
                    f"-sOutputFile={dst}", str(src)], check=True)
    if fond:
        im = Image.open(dst).convert("RGBA")
        bg = Image.new("RGBA", im.size, tuple(couleur(fond)) + (255,))
        Image.alpha_composite(bg, im).convert("RGB").save(dst)
    return dst


def _largeur_pdf(src):
    sortie = subprocess.run(["gs", "-q", "-dNODISPLAY", "-dNOSAFER", "-dNOPAUSE", "-dBATCH",
                             f"-sFile={src}", "-c",
                             "File (r) file runpdfbegin 1 pdfgetpage /CropBox pget not "
                             "{1 pdfgetpage /MediaBox pget pop} if aload pop exch 4 -1 roll sub "
                             "== pop quit"],
                            capture_output=True, text=True).stdout.split()
    try:
        return abs(float(sortie[0]))
    except (IndexError, ValueError):
        return 400.0


def image(diapo, chemin, x, y, w, h, align="centre", valign="milieu"):
    """Image ajustée dans la boîte (x, y, w, h) en cm Beamer, sans déformation."""
    larg, haut = Image.open(chemin).size
    r = larg / haut
    iw, ih = (w, w / r) if w / r <= h else (h * r, h)
    ix = x + (w - iw) / 2 if align == "centre" else x if align == "gauche" else x + w - iw
    iy = y + (h - ih) / 2 if valign == "milieu" else y if valign == "haut" else y + h - ih
    return diapo.shapes.add_picture(str(chemin), cm(ix), cm(iy), cm(iw), cm(ih))


def tikz_standalone(nom, corps, pas=None):
    """Compile un dessin TikZ du dossier figures (ou un code) en PNG à fond transparent."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    base = MEDIA / (nom + (f"_{pas}" if pas else ""))
    tex = base.with_suffix(".tex")
    tex.write_text(r"""\documentclass[11pt, border=3pt]{standalone}
\usepackage{fontspec}\usepackage{unicode-math}
\setsansfont{Fira Sans}[UprightFont=*-Light, BoldFont=*-SemiBold]
\setmathfont{Fira Math}\renewcommand{\familydefault}{\sfdefault}
\usepackage{tikz}\usetikzlibrary{arrows.meta, calc}
\definecolor{nuit}{HTML}{3E2A57}\definecolor{gris}{HTML}{8A8F98}
\definecolor{violet}{HTML}{800080}\definecolor{vert}{HTML}{00AA44}
\def\pas{%s}
\def\only<#1->#2{\ifnum#1>\pas\relax\else#2\fi}
\begin{document}%s\end{document}
""" % (pas or 99, corps))
    subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                   cwd=MEDIA, check=True, capture_output=True)
    pdf = base.with_suffix(".pdf")
    sortie = base.with_suffix(".png")
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha", "-r400",
                    f"-sOutputFile={sortie}", str(pdf)], check=True)
    return sortie


def fond_beamer(page, nom):
    """Page du PDF Beamer rendue sans son texte : sert de fond aux pages de titre et de section."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    dst = MEDIA / f"{nom}.png"
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dFILTERTEXT", "-sDEVICE=png16m", "-r300",
                    f"-dFirstPage={page}", f"-dLastPage={page}", f"-sOutputFile={dst}",
                    str(PDF_BEAMER)], check=True)
    return dst


# ---------------------------------------------------------------------
#  Le gabarit d'une diapo
# ---------------------------------------------------------------------
class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = cm(16)
        self.prs.slide_height = cm(9)
        self.vide = self.prs.slide_layouts[6]
        self.numero = 0
        self.annexe = False

    def fond(self, diapo, nom):
        f = diapo.background.fill
        f.solid()
        f.fore_color.rgb = couleur(nom)

    def nouvelle(self, titre=None, sous_titre=None, numerotee=True):
        d = self.prs.slides.add_slide(self.vide)
        self.fond(d, "papier")
        d.anim = Animations()
        if titre is not None:
            tb = zone_texte(d, 1.1, 0.42, 13.8, 0.7, ancre="bas")
            p = tb.text_frame.paragraphs[0]
            ecrire(p, riche(titre), {"taille": 14.4, "forte": True, "couleur": "nuit"})
            bas = 1.2
            if sous_titre:
                tb = zone_texte(d, 1.1, 1.13, 13.8, 0.4)
                ecrire(tb.text_frame.paragraphs[0], riche(sous_titre), {"taille": 10, "couleur": "gris"})
                bas = 1.6
            trait_laser(d, 1.1, bas + 0.05, 2.2)
            d.haut = bas + 0.35
        if numerotee:
            if not self.annexe:
                self.numero += 1
            pied(d, self.numero, self.annexe)
        return d

    def finir(self, d):
        d.anim.ecrire(d)


def trait_laser(diapo, x, y, w, h=0.055):
    r = diapo.shapes.add_shape(MSO_SHAPE.RECTANGLE, cm(x), cm(y), cm(w), cm(h))
    r.line.fill.background()
    r.fill.gradient()
    r.fill.gradient_angle = 0
    stops = r.fill.gradient_stops
    stops[0].color.rgb = couleur("violet")
    stops[0].position = 0
    stops[1].color.rgb = couleur("vert")
    stops[1].position = 1.0
    r.name = "Trait laser"
    return r


def pied(diapo, numero, annexe):
    tb = zone_texte(diapo, 1.1, 8.33, 10, 0.3, ancre="milieu")
    p = tb.text_frame.paragraphs[0]
    ecrire(p, riche(r"Pauline Guesdon {\color{gris!50}|} Contrôle optique du \Sr{87} ultrafroid"),
           {"taille": 6, "couleur": "gris"})
    tb.name = "Pied de page"
    if not annexe:
        tb = zone_texte(diapo, 11.0, 8.33, 3.9, 0.3, ancre="milieu")
        p = tb.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.RIGHT
        ecrire(p, [(f"{numero}\u2009/\u2009{TOTAL}", {})], {"taille": 6, "couleur": "gris"})
        tb.name = "Numéro"
    fond = diapo.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, cm(9 - 0.07), cm(16), cm(0.07))
    fond.fill.solid()
    fond.fill.fore_color.rgb = couleur("gris!15")
    fond.line.fill.background()
    fond.name = "Barre de progression (fond)"
    if not annexe:
        barre = trait_laser(diapo, 0, 9 - 0.07, 16 * numero / TOTAL, 0.07)
        barre.name = "Barre de progression"


# ---------------------------------------------------------------------
#  Briques de contenu
# ---------------------------------------------------------------------
def puces(diapo, x, y, w, h, items, taille=11, ancre="milieu", espace=6, numerotees=False):
    """items : liste de (latex, pas) ou (latex, pas, niveau) ; niveau 1 = puce, 2 = sous-puce,
    0 = paragraphe sans puce."""
    tb = zone_texte(diapo, x, y, w, h, ancre=ancre)
    tf = tb.text_frame
    num = 0
    for k, item in enumerate(items):
        latex, pas = item[0], item[1]
        niveau = item[2] if len(item) > 2 else 1
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        if niveau:
            num += 1
            _puce(p, niveau, num if (numerotees and niveau == 1) else None)
        else:
            p._p.get_or_add_pPr().set("marL", "0")
        if k:
            _espace_avant(p, espace * 2)
        ecrire(p, riche(latex), {"taille": taille if niveau != 2 else taille})
        diapo.anim.ajouter(pas, tb, k)
    return tb


def paragraphes(diapo, x, y, w, h, lignes, ancre="milieu", align="gauche", espace=0):
    """lignes : liste de (latex, base, pas) ; un paragraphe par ligne, sans puce."""
    tb = zone_texte(diapo, x, y, w, h, ancre=ancre)
    tf = tb.text_frame
    for k, (latex, base, pas) in enumerate(lignes):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = {"gauche": PP_ALIGN.LEFT, "centre": PP_ALIGN.CENTER,
                       "droite": PP_ALIGN.RIGHT}[align]
        if k and espace:
            _espace_avant(p, espace * 2)
        ecrire(p, riche(latex), dict(base))
        diapo.anim.ajouter(pas, tb, k)
    return tb


def a_retenir(diapo, latex, pas=None, y=None, h=0.75):
    """Message clé encadré en bas de diapo (\\amretenir)."""
    y = 8.05 - h if y is None else y
    boite = diapo.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cm(1.1), cm(y), cm(13.8), cm(h))
    boite.adjustments[0] = 0.08
    boite.fill.solid()
    boite.fill.fore_color.rgb = couleur("voile")
    boite.line.fill.background()
    boite.shadow.inherit = False
    tf = boite.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left, tf.margin_right = cm(0.35), cm(0.25)
    tf.margin_top = tf.margin_bottom = cm(0.08)
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    ecrire(tf.paragraphs[0], riche(latex), {"taille": 10})
    barre = diapo.shapes.add_shape(MSO_SHAPE.RECTANGLE, cm(1.1), cm(y), cm(0.09), cm(h))
    barre.fill.solid()
    barre.fill.fore_color.rgb = couleur("violet")
    barre.line.fill.background()
    boite.name, barre.name = "À retenir", "À retenir (trait)"
    diapo.anim.ajouter(pas, boite)
    diapo.anim.ajouter(pas, barre)
    return boite


def bloc(diapo, x, y, w, titre, lignes, h=None, alerte=False, pas=None, taille=11):
    """Bloc Beamer : titre sur fond voile, puis contenu centré (équations)."""
    h = h or 0.55 + 0.75 * len(lignes)
    boite = diapo.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cm(x), cm(y), cm(w), cm(h))
    boite.adjustments[0] = 0.06
    boite.fill.solid()
    boite.fill.fore_color.rgb = couleur("voile")
    boite.line.fill.background()
    boite.shadow.inherit = False
    tf = boite.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = cm(0.2)
    tf.margin_top, tf.margin_bottom = cm(0.1), cm(0.1)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    ecrire(p, riche(titre), {"taille": taille, "forte": True,
                             "couleur": "violet" if alerte else "nuit"})
    for latex in lignes:
        p = tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        _espace_avant(p, 6)
        ecrire(p, riche(latex), {"taille": taille})
    boite.name = "Bloc : " + re.sub(r"[\\${}]", "", titre)[:30]
    diapo.anim.ajouter(pas, boite)
    return boite


def cadre_vide(diapo, x, y, w, h):
    r = diapo.shapes.add_shape(MSO_SHAPE.RECTANGLE, cm(x), cm(y), cm(w), cm(h))
    r.fill.solid()
    r.fill.fore_color.rgb = couleur("voile")
    r.line.color.rgb = couleur("gris!50")
    r.line.dash_style = MSO_LINE.DASH
    r.shadow.inherit = False
    tf = r.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    ecrire(p, [("figure", {})], {"taille": 10, "couleur": "gris"})
    r.name = "Figure à remplacer"
    return r


def tableau(diapo, x, y, w, lignes, largeurs, taille=10, h_ligne=0.55, pas=None, regles=(0, 1)):
    """lignes : liste de listes de LaTeX ; filets en haut, après la ligne d'en-tête et en bas."""
    n, m = len(lignes), len(lignes[0])
    gf = diapo.shapes.add_table(n, m, cm(x), cm(y), cm(w), cm(h_ligne * n))
    t = gf.table
    tblPr = t._tbl.tblPr
    for attr in ("firstRow", "bandRow"):
        tblPr.set(attr, "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # « aucun style, aucune grille »
    total = sum(largeurs)
    for j, lw in enumerate(largeurs):
        t.columns[j].width = cm(w * lw / total)
    for i, ligne in enumerate(lignes):
        t.rows[i].height = cm(h_ligne)
        for j, contenu in enumerate(ligne):
            c = t.cell(i, j)
            c.fill.background()
            c.margin_left = c.margin_right = cm(0.1)
            c.margin_top = c.margin_bottom = cm(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            ecrire(p, riche(contenu), {"taille": taille})
            _filets(c, haut=(i == 0), bas=(i in regles[1:] or i == n - 1), epais=(i == 0 or i == n - 1))
    gf.name = "Tableau"
    diapo.anim.ajouter(pas, gf)
    return gf


def _filets(cellule, haut, bas, epais):
    tcPr = cellule._tc.get_or_add_tcPr()
    for tag, actif, larg in (("a:lnL", False, 0), ("a:lnR", False, 0),
                             ("a:lnT", haut, 1.0), ("a:lnB", bas, 1.0 if epais else 0.5)):
        ln = etree.SubElement(tcPr, qn(tag))
        if actif:
            ln.set("w", str(int(Pt(larg))))
            sf = etree.SubElement(ln, qn("a:solidFill"))
            sf.append(_srgb("encre"))
        else:
            ln.set("w", "0")
            etree.SubElement(ln, qn("a:noFill"))
    # l'ordre du schéma : lnL, lnR, lnT, lnB doivent précéder le remplissage de la cellule
    for el in list(tcPr):
        if el.tag in (qn("a:noFill"), qn("a:solidFill")):
            tcPr.remove(el)
            tcPr.append(el)


def texte_sur(diapo, img, fx, fy, latex, base, pas=None, w=4.0, h=0.6, fond=None):
    """Texte centré au point (fx, fy) de l'image, en coordonnées de l'image (0,0 en bas à gauche)."""
    x = Emu(img.left + int(fx * img.width)) / 360000 / ECHELLE
    y = Emu(img.top + int((1 - fy) * img.height)) / 360000 / ECHELLE
    tb = zone_texte(diapo, x - w / 2, y - h / 2, w, h, ancre="milieu")
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    ecrire(p, riche(latex), base)
    if fond:
        tb.fill.solid()
        tb.fill.fore_color.rgb = couleur(fond)
    diapo.anim.ajouter(pas, tb)
    return tb


def colonnes(*largeurs):
    """Positions x (cm) des colonnes Beamer : l'espace libre est réparti autour des colonnes."""
    libre = 13.8 - sum(l * 13.8 for l in largeurs)
    g = libre / (len(largeurs) + 1)
    xs, x = [], 1.1 + g
    for l in largeurs:
        xs.append((x, l * 13.8))
        x += l * 13.8 + g
    return xs


def fleche(diapo, x1, y1, x2, y2, nom_couleur, epaisseur=1.3):
    c = diapo.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, cm(x1), cm(y1), cm(x2), cm(y2))
    c.line.color.rgb = couleur(nom_couleur)
    c.line.width = Pt(epaisseur * 2)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def disque(diapo, x, y, r, nom_couleur):
    d = diapo.shapes.add_shape(MSO_SHAPE.OVAL, cm(x - r), cm(y - r), cm(2 * r), cm(2 * r))
    d.fill.solid()
    d.fill.fore_color.rgb = couleur(nom_couleur)
    d.line.fill.background()
    d.shadow.inherit = False
    return d


# ---------------------------------------------------------------------
#  Les diapos
# ---------------------------------------------------------------------
def page_titre(deck):
    d = deck.nouvelle(numerotee=False)
    d.shapes.add_picture(str(fond_beamer(1, "fond_titre")), 0, 0, cm(16), cm(9))
    lignes = [
        (r"Contrôle optique du \Sr{87} ultrafroid", {"taille": 17.28, "forte": True, "couleur": "white"}),
        (r"Des manipulations de spin à un corps à la photoassociation de deux corps",
         {"taille": 10, "couleur": "white!70!nuit"}),
    ]
    tb = zone_texte(d, 1.1, 0.95, 8.9, 3.3, ancre="haut")
    for k, (latex, base) in enumerate(lignes):
        p = tb.text_frame.paragraphs[0] if k == 0 else tb.text_frame.add_paragraph()
        if k:
            _espace_avant(p, 20)
        ecrire(p, riche(latex), base)
    tb = zone_texte(d, 1.1, 4.75, 8.5, 1.3, ancre="haut")
    ecrire(tb.text_frame.paragraphs[0], riche("Pauline Guesdon"),
           {"taille": 11, "forte": True, "couleur": "white"})
    p = tb.text_frame.add_paragraph()
    _espace_avant(p, 4)
    ecrire(p, riche(r"Soutenance de thèse \textperiodcentered{} 14 décembre 2026"),
           {"taille": 10, "couleur": "white!60!nuit"})
    tb = zone_texte(d, 4.5, 8.1, 10.4, 0.5, ancre="milieu")
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    ecrire(p, riche(r"Laboratoire de Physique des Lasers \textperiodcentered{} Université Sorbonne Paris Nord"),
           {"taille": 8, "couleur": "gris"})
    deck.finir(d)


def page_section(deck, page_pdf, numero, titre):
    d = deck.nouvelle(numerotee=False)
    d.shapes.add_picture(str(fond_beamer(page_pdf, f"fond_section_{numero}")), 0, 0, cm(16), cm(9))
    laser = ["laserrouge", "laserbleu", "laserviolet", "laservert"][(numero - 1) % 4]
    tb = zone_texte(d, 1.1, 2.05, 12, 1.9, ancre="bas")
    ecrire(tb.text_frame.paragraphs[0], [(f"{numero:02d}", {})],
           {"taille": 30, "couleur": laser})
    for r in tb.text_frame.paragraphs[0].runs:
        r.font.name = POLICE_FINE
        r.font.size = Pt(130)
    tb = zone_texte(d, 1.1, 4.0, 12, 0.9, ancre="haut")
    ecrire(tb.text_frame.paragraphs[0], riche(titre), {"taille": 17.28, "forte": True, "couleur": "white"})
    deck.finir(d)


def page_forte(deck, lignes):
    d = deck.nouvelle(numerotee=False)
    deck.fond(d, "nuit")
    tb = zone_texte(d, 1.1, 1.0, 13.8, 7.0, ancre="milieu")
    for k, (latex, base) in enumerate(lignes):
        p = tb.text_frame.paragraphs[0] if k == 0 else tb.text_frame.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        if k:
            _espace_avant(p, 10)
        ecrire(p, riche(latex), base)
    return d


def diapos(deck, pages_section):
    N = "normalsize"
    page_titre(deck)

    # ======================= Introduction =======================
    page_section(deck, pages_section[1], 1, "Introduction")

    d = deck.nouvelle("Le sujet de ma thèse")
    (x1, w1), (x2, w2) = colonnes(0.55, 0.42)
    puces(d, x1, 1.9, w1, 6.0, [
        (r"\textbf{But : les pertes à deux corps du \Sr{87}}\\{\small un fermion, par photoassociation (PA)}", 1),
        (r"\textbf{Les paires qui collisionnent : \textcolor{violet}{singulets}}\\"
         r"{\small spectroscopie, les molécules elles-mêmes} {\scriptsize\color{gris} $\to$ partie 1}", 2),
        (r"\textbf{Les paires qui restent : \textcolor{vert!70!black}{triplets}}\\"
         r"{\small immunes (onde $p$) : elles aussi éclairent la PA}", 3),
        (r"\textbf{Une plateforme de contrôle de spin}\\"
         r"{\small pour préparer et sonder ces états} {\scriptsize\color{gris} $\to$ partie 2}", 4),
    ], espace=4)
    cx = x2 + 0.6
    f = paragraphes(d, x2, 2.45, w2, 1.0, [
        (r"$\Psi = \psi_{\text{spatial}}(\vec r)\otimes\chi_{\text{spin}}$", {}, None),
        (r"{\small fermions : $\Psi$ \alert{impaire} par échange}", {}, None),
    ], align="centre", ancre="haut")
    d.anim.ajouter(2, f)
    sing = [paragraphes(d, cx - 0.3, 3.5, 4.5, 0.45,
                        [(r"{\fontforte singulet} $\uparrow\downarrow-\downarrow\uparrow$", {"couleur": "violet", "taille": 10}, None)],
                        ancre="milieu"),
            disque(d, cx, 4.3, 0.18, "violet!70"), disque(d, cx + 0.42, 4.3, 0.18, "violet!70"),
            fleche(d, cx + 0.85, 4.3, cx + 1.95, 4.3, "laserrouge"),
            paragraphes(d, cx + 0.9, 3.93, 1.1, 0.35, [("PA", {"couleur": "laserrouge", "taille": 10}, None)],
                        align="centre"),
            paragraphes(d, cx + 2.05, 4.1, 1.6, 0.4, [("perdus", {"couleur": "violet", "taille": 10}, None)]),
            paragraphes(d, cx - 0.3, 4.55, 4.5, 0.35,
                        [(r"$\chi$ impaire, $\psi$ paire : onde $s$", {"couleur": "violet", "taille": 8}, None)])]
    trip = [paragraphes(d, cx - 0.3, 5.15, 4.5, 0.45,
                        [(r"{\fontforte triplet} $\uparrow\uparrow$", {"couleur": "vert!70!black", "taille": 10}, None)]),
            disque(d, cx, 5.85, 0.18, "vert!80"), disque(d, cx + 1.0, 5.85, 0.18, "vert!80"),
            paragraphes(d, cx + 2.05, 5.65, 1.6, 0.4, [("immunes", {"couleur": "vert!70!black", "taille": 10}, None)]),
            paragraphes(d, cx - 0.3, 6.1, 4.5, 0.35,
                        [(r"$\chi$ paire, $\psi$ impaire : onde $p$", {"couleur": "vert!70!black", "taille": 8}, None)])]
    barriere = d.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, cm(cx + 0.5), cm(5.65), cm(cx + 0.5), cm(6.05))
    barriere.line.color.rgb = couleur("gris")
    barriere.line.width = Pt(2.4)
    trip.append(barriere)
    for s in sing:
        d.anim.ajouter(2, s)
    for s in trip:
        d.anim.ajouter(3, s)
    deck.finir(d)

    d = deck.nouvelle(r"Le \Sr{87} : un atome à dix états de spin")
    (x1, w1), (x2, w2) = colonnes(0.46, 0.5)
    puces(d, x1, 1.6, w1, 5.2, [
        (r"Spin nucléaire $I=9/2$ : \alert{10 états} $m_F$ dans le fondamental", 1),
        (r"Fondamental $^1S_0$ ($J=0$) : \alert{faible sensibilité} au champ magnétique", 2),
        (r"Raie étroite $^1S_0\to{}^3P_1$ (\SI{7.5}{kHz}) pour adresser chaque état", 3),
    ])
    image(d, png("energy_levels_sr.pdf"), x2, 1.6, w2, 0.58 * 7.9)
    a_retenir(d, r"Un système à grand nombre d'états, robuste et adressable optiquement.")
    deck.finir(d)

    d = deck.nouvelle(r"Pourquoi ? Métrologie et information quantiques")
    (x1, w1), (x2, w2) = colonnes(0.5, 0.45)
    puces(d, x1, 1.6, w1, 6.4, [
        (r"Qudits : plus d'information par atome", 1),
        (r"Précision d'un interféromètre à $N$ atomes :", 2),
        (r"limite standard : $\Delta\phi = 1/\sqrt{N}$", 2, 2),
        (r"limite de Heisenberg : $\Delta\phi = 1/N$", 2, 2),
        (r"Deux leviers : \textcolor{vert}{dissipation} à deux corps, \alert{contrôle cohérent} à un corps", 3),
    ])
    cadre_vide(d, x2 + (w2 - 5.5) / 2, 4.8 - 2.25, 5.5, 4.5)
    deck.finir(d)

    d = deck.nouvelle(r"Le dispositif expérimental")
    (x1, w1), (x2, w2) = colonnes(0.4, 0.56)
    puces(d, x1, 1.6, w1, 6.4, [
        (r"Refroidissement laser (bleu, puis rouge)", 1),
        (r"Piège dipolaire croisé, évaporation", 1),
        (r"Gaz de quelques \si{\micro K}, $\langle n\rangle\sim 10^{13}\,\mathrm{cm^{-3}}$", 1),
        (r"Détection : imagerie d'absorption", 1),
    ])
    image(d, png("setup_exp.pdf"), x2, 1.6, w2, 0.7 * 7.9 + 0.8)
    deck.finir(d)

    d = deck.nouvelle("Sommaire")
    puces(d, 1.1, 1.6, 13.8, 6.4, [
        (r"\textbf{Photoassociation du \Sr{87}}\\{\small\color{gris} des pertes à deux corps sélectives en spin}", 1),
        (r"\textbf{Manipulations cohérentes de pseudo-spins}\\{\small\color{gris} préparer, contrôler et mesurer des compositions de spin}", 1),
        (r"\textbf{Conclusion et perspectives}\\{\small\color{gris} sonder les atomes photoassociés avec nos interféromètres}", 1),
    ], numerotees=True, espace=10)
    deck.finir(d)

    # ======================= Photoassociation =======================
    page_section(deck, pages_section[2], 2, r"Photoassociation du \Sr{87}")

    d = deck.nouvelle("La photoassociation")
    (x1, w1), (x2, w2) = colonnes(0.48, 0.48)
    puces(d, x1, 1.6, w1, 5.6, [
        (r"\textbf{Élastique} : énergie cinétique conservée, atomes piégés", 1),
        (r"\textbf{Inélastique} : énergie interne libérée, atomes éjectés", 2),
        (r"PA : un laser couple une paire à un \alert{état moléculaire excité}", 3),
        (r"Désexcitation : atomes chauds ou molécule, donc \alert{perdus}", 4),
    ])
    image(d, png("Figures_pour_theorie/Guttridge_PA_process.pdf"), x2, 2.0, w2, 0.5 * 7.9)
    paragraphes(d, x2, 6.0, w2, 0.35, [(r"Adapté de Guttridge \textit{et al.}, 2019", {"taille": 6, "couleur": "gris"}, None)],
                align="centre")
    a_retenir(d, r"Une raie moléculaire = un creux dans le nombre d'atomes.", pas=4)
    deck.finir(d)

    d = deck.nouvelle(r"Pourquoi la photoassociation du \Sr{87} ?")
    (x1, w1), (x2, w2) = colonnes(0.55, 0.4)
    puces(d, x1, 1.6, w1, 5.6, [
        (r"PA sur $^1S_0$--$^3P_1$ bien connue pour \Sr{88}, \Sr{84}, \Sr{86}", 1),
        (r"\Sr{87} : seulement sur $^1P_1$ ; spectre dense sur $^3P_1$ (structure hyperfine)", 2),
        (r"Idée (Foss-Feig 2012) : la dissipation à deux corps comme \alert{moteur}", 3),
        (r"État stationnaire de Dicke ($m=0$) : vers $\Delta\phi\propto 1/N$", 4),
    ])
    cadre_vide(d, x2 + (w2 - 5) / 2, 2.0, 5, 4.5)
    a_retenir(d, r"Première étape : identifier les raies moléculaires.", pas=4)
    deck.finir(d)

    d = deck.nouvelle("Le protocole")
    (x1, w1), (x2, w2) = colonnes(0.42, 0.54)
    puces(d, x1, 1.6, w1, 6.4, [
        (r"Repère : la transition atomique $^1S_0\to{}^3P_1$", 1),
        (r"On balaye le laser de PA dans le rouge", 2),
        (r"Illumination (\SI{1}{s}), puis on compte les atomes restants", 3),
        (r"Raie = creux : ajustement lorentzien (centre, largeur)", 4),
    ])
    corps = (ICI / "figures" / "schema_PA.tex").read_text()
    for pas in (1, 2, 3):
        im = image(d, tikz_standalone("schema_PA", corps, pas), x2, 1.75, w2, 0.72 * 7.9)
        im.name = f"Schéma PA, étape {pas}"
        d.anim.ajouter(pas, im)
    deck.finir(d)

    d = deck.nouvelle(r"Validation sur le \Sr{88} : les spectres")
    (x1, w1), (x2, w2) = colonnes(0.4, 0.56)
    puces(d, x1, 1.6, w1, 6.4, [
        (r"$\nu=-1$ et $\nu=-2$ retrouvées", 1),
        (r"Positions à moins de \SI{0.5}{MHz} de la littérature", 1),
        (r"Largeur $\approx 330\,\mathrm{kHz}$ ($\eta\approx22$)", 1),
    ])
    image(d, png("88Sr/resonance_88.pdf"), x2, 1.6, w2, 0.74 * 7.9 + 0.3)
    deck.finir(d)

    d = deck.nouvelle(r"Validation sur le \Sr{88} : le déclin $N(t)$")
    (x1, w1), (x2, w2) = colonnes(0.4, 0.56)
    puces(d, x1, 1.6, w1, 5.4, [
        (r"Temps courts : deux corps (non exponentiel)", 1),
        (r"Temps longs : un corps (droite en échelle log)", 2),
        (r"Ajustement : $K_2$, puis $l_{\text{opt}}/I$", 3),
        (r"$\nu=-2$ : $56\pm10$ contre $68(3)$ $a_0$/(W/cm$^2$) (Blatt 2011)", 4),
    ])
    image(d, png("88Sr/fig_149_sr88.pdf"), x2 + 0.075 * w2, 1.7, 0.85 * w2, 0.3 * 7.9)
    image(d, png("88Sr/fig_148_sr88.pdf"), x2 + 0.075 * w2, 1.7 + 0.3 * 7.9, 0.85 * w2, 0.3 * 7.9)
    a_retenir(d, r"Montage validé : accord avec la littérature à un facteur $\lesssim 2$.", pas=4)
    deck.finir(d)

    d = deck.nouvelle("Une perte à deux corps")
    bloc(d, 1.1, 1.75, 13.8, "Dynamique des pertes",
         [r"$\frac{dN}{dt} = -\beta N^2 - \Gamma N, \qquad \beta \propto K_2 / V_e(T)$"], h=1.5)
    puces(d, 1.1, 3.4, 13.8, 1.3, [
        (r"$K_2$ : la force de la raie (ne dépend pas de la densité)", 2),
        (r"$K_2 \propto l_{\text{opt}}$, et $l_{\text{opt}} \propto I$ : on compare $l_{\text{opt}}/I$", 3),
    ], ancre="haut")
    bloc(d, 1.1, 4.85, 13.8, "Raie élargie",
         [r"$K_2(\text{résonance}) = \frac{16h}{m_{\text{at}}}\,\frac{l_{\text{opt}}}{\eta}, \qquad "
          r"\eta = \frac{\Gamma_{\text{fit}}}{\Gamma_{\text{rad}}/2\pi}$"], h=1.6, alerte=True, pas=4)
    a_retenir(d, r"Une raie $\eta$ fois trop large a un $K_2$ $\eta$ fois plus faible.", pas=5)
    deck.finir(d)

    d = deck.nouvelle("Reconnaître une vraie raie moléculaire", "Une leçon tirée d'un faux signal")
    (x1, w1), (x2, w2) = colonnes(0.44, 0.52)
    puces(d, x1, 2.0, w1, 6.0, [
        (r"À \SI{-6}{MHz} : déclin non exponentiel\ldots", 1),
        (r"\ldots{} mais signal \alert{atomique} : bande latérale de bruit du laser", 2),
        (r"La forme de $N(t)$ seule ne suffit pas", 3),
        (r"Test : chercher la même perte à $+\delta$ et $-\delta$ de l'atome", 4),
    ])
    image(d, png("87Sr/data16_exp_bruit_6MHz.pdf"), x2, 2.0, w2, 0.7 * 7.9 - 0.1)
    deck.finir(d)

    d = deck.nouvelle(r"Ce que l'on attend pour le \Sr{87}")
    (x1, w1), (x2, w2) = colonnes(0.44, 0.52)
    puces(d, x1, 1.6, w1, 6.4, [
        (r"Largeur : $\Gamma_{\text{rad}}/2\pi\approx 15\,\mathrm{kHz}$", 1),
        (r"$\sim 3\times10^3$ raies dans les \SI{380}{MHz} explorés", 2),
        (r"Positions prédites par loi d'échelle en masse", 3),
    ])
    image(d, png("Figures_pour_theorie/mass_scaling.pdf"), x2, 1.6, w2, 0.68 * 7.9)
    paragraphes(d, x2, 7.05, w2, 0.35, [(r"Borkowski \textit{et al.}, 2014", {"taille": 6, "couleur": "gris"}, None)],
                align="centre")
    deck.finir(d)

    d = deck.nouvelle("Ce que l'on observe : une raie en détail")
    image(d, png("87Sr/composite_276MHz.pdf"), 1.1, 1.6, 13.8, 0.64 * 7.9 + 0.6)
    paragraphes(d, 1.1, 7.4, 13.8, 0.4, [(
        r"Raie à \SI{-276.83}{MHz} \textperiodcentered{} spectre : $I=5,2\,\mathrm{W/cm^2}$, "
        r"$\langle n\rangle=1,6\times10^{13}\,\mathrm{cm^{-3}}$ \textperiodcentered{} déclin : "
        r"$I=31\,\mathrm{W/cm^2}$, $\langle n\rangle=7,8\times10^{12}\,\mathrm{cm^{-3}}$ "
        r"\textperiodcentered{} illumination \SI{1}{s}", {"taille": 6, "couleur": "gris"}, None)], align="centre")
    deck.finir(d)

    d = deck.nouvelle("Seulement trois raies")
    tableau(d, 2.3, 2.1, 11.4, [
        [r"Raie (MHz)", r"$-259.01$", r"$-276.83$", r"$-377.2$"],
        [r"Largeur $\Gamma_{\text{fit}}$ (MHz)", r"$7.4\pm0.9$", r"$5.0\pm0.3$", r"$20.2\pm2.3$"],
        [r"Profondeur après \SI{1}{s}", r"40\,\%", r"35\,\%", r"9\,\%"],
        [r"Temps caractéristique $\tau$", r"\SI{286}{ms}", r"\SI{117}{ms}", r"\SI{5.3}{s}"],
        [r"$l_{\text{opt}}/I$ ($a_0$/(W/cm$^2$))", r"$3.4\pm1.9$", r"$3.1\pm0.5$", r"$0.49\pm0.35$"],
    ], [4.2, 2.4, 2.4, 2.4], taille=11, h_ligne=0.62)
    puces(d, 1.1, 5.6, 13.8, 1.3, [
        (r"Aucune raie à moins de \SI{259}{MHz} de l'asymptote", 2),
        (r"Largeurs de 5 à \SI{20}{MHz} au lieu de \SI{15}{kHz}", 3),
    ], ancre="haut")
    deck.finir(d)

    for titre, items, fig, larg in [
        ("Où se placent ces raies ?", [
            (r"\textcolor{vert}{\Sr{88}} : sur la littérature", 1),
            (r"\textcolor{violet}{\Sr{87}} : 1 à 2 ordres de grandeur plus bas", 2),
            (r"Au niveau de la raie la plus faible connue du \Sr{84}", 3)], "lopt_comparaison.pdf", 0.74),
        ("Le vrai message : la largeur", [
            (r"$\eta$ : $1.5 \to 22 \to 330$--$1350$", 1),
            (r"$K_2(\text{résonance}) \propto l_{\text{opt}}/\eta$", 2),
            (r"Raies peu profondes et raies larges : \alert{le même effet}", 3)], "elargissement_eta.pdf", 0.66)]:
        d = deck.nouvelle(titre)
        (x1, w1), (x2, w2) = colonnes(0.36, 0.6)
        puces(d, x1, 1.6, w1, 6.4, items)
        image(d, png(fig), x2, 1.6, w2, larg * 7.9 + 0.4)
        deck.finir(d)

    d = page_forte(deck, [(r"L'anomalie des raies du \Sr{87} est leur largeur",
                           {"taille": 14.4, "forte": True, "couleur": "white"}),
                          (r"et non leur force", {"taille": 11, "couleur": "white!60!nuit"})])
    deck.finir(d)

    d = deck.nouvelle("D'où vient cette largeur ?")
    (x1, w1), (x2, w2) = colonnes(0.48, 0.48)
    paragraphes(d, x1, 2.0, w1, 0.5, [(r"{\fontforte\color{violet} Causes techniques}", {}, None)], ancre="haut")
    puces(d, x1, 2.6, w1, 4.0, [
        (r"Pureté spectrale du laser de PA", 1),
        (r"Déplacement lumineux inhomogène du faisceau de PA", 1),
        (r"Lumière du piège : transitions multi-photons, déplacement lumineux", 1)], ancre="haut")
    paragraphes(d, x2, 2.0, w2, 0.5, [(r"{\fontforte\color{vert} Causes physiques}", {}, None)], ancre="haut")
    puces(d, x2, 2.6, w2, 4.0, [
        (r"\textbf{Prédissociation} vers $^3P_0$ : raie large, perte faible au pic", 1),
        (r"Croisements avec des potentiels plus hauts ($^1S_0+{}^1D_2$)", 1),
        (r"Statistique quantique et sélection de spin : abaissent $l_{\text{opt}}$, mais n'élargissent pas", 1)],
        ancre="haut")
    a_retenir(d, r"Hypothèses les plus probables : prédissociation et déplacements lumineux.")
    deck.finir(d)

    d = deck.nouvelle("La prédissociation", "Une sortie de plus pour la molécule")
    (x1, w1), (x2, w2) = colonnes(0.46, 0.5)
    image(d, png("Figures_pour_theorie/predissociation.pdf"), x1, 2.1, w1, 0.55 * 7.9)
    puces(d, x2, 2.0, w2, 5.0, [
        (r"\textbf{A} : deux atomes libres $^1S_0+{}^1S_0$", 1),
        (r"\textbf{B} : état lié $\nu_B$ de $^1S_0+{}^3P_1$, atteint par le laser de PA", 2),
        (r"\textbf{C} : un potentiel dont l'asymptote est \alert{sous} $\nu_B$ (pour le \Sr{87} : vers $^1S_0+{}^3P_0$)", 3),
        (r"C croise B : $\nu_B$ couplé au \alert{continuum} de C, la paire se dissocie", 4),
    ])
    a_retenir(d, r"La molécule a deux sorties : l'émission spontanée et la prédissociation.", pas=5)
    deck.finir(d)

    d = deck.nouvelle("Largeur effective de la raie")
    (x1, w1), (x2, w2) = colonnes(0.58, 0.38)
    bloc(d, x1, 1.75, w1, r"Taux de sortie de l'état $\nu_B$",
         [r"$\Gamma_{\text{mol}} = \Gamma_{\text{rad}} + \Gamma_{\text{pd}}, \qquad "
          r"\Gamma_{\text{pd}} = \frac{2\pi}{\hbar}\,|W_{BC}|^2\,\rho_C(E_{\nu_B})$"], h=1.55, taille=10)
    puces(d, x1, 3.45, w1, 3.5, [
        (r"Les taux s'additionnent : la raie s'élargit", 2),
        (r"$\Gamma_{\text{perte}}(\Delta) \propto l_{\text{opt}}\,\frac{\Gamma_{\text{rad}}\,\Gamma_{\text{mol}}}{\Delta^2 + \Gamma_{\text{mol}}^2/4}$", 3),
        (r"Pic abaissé de $1/\eta$, \alert{aire inchangée}", 4),
        (r"$\Gamma_{\text{pd}}$ indépendant de l'intensité de PA", 5),
    ], taille=10, ancre="haut")
    lorentz = r"""\begin{tikzpicture}[x=0.75cm, y=1.1cm]
        \draw[gris, ->] (-3.2,0) -- (3.3,0) node[below left, font=\scriptsize] {$\Delta$};
        \draw[gris!70, line width=1pt, domain=-3:3, samples=120] plot (\x, {2*0.25/(\x*\x + 0.0625)/3.1416});
        \draw[violet, line width=1.2pt, domain=-3:3, samples=120] plot (\x, {2*1/(\x*\x + 1)/3.1416});
        \node[gris, font=\scriptsize, anchor=west] at (0.35,2.3) {sans prédissociation};
        \node[violet, font=\scriptsize, anchor=west] at (1.1,0.75) {avec};
      \end{tikzpicture}"""
    im = image(d, tikz_standalone("lorentz", lorentz), x2, 2.3, w2, 3.4)
    d.anim.ajouter(4, im)
    t = paragraphes(d, x2, 5.8, w2, 0.4, [(r"Même aire sous les deux courbes", {"taille": 8, "couleur": "gris"}, None)],
                    align="centre")
    d.anim.ajouter(4, t)
    a_retenir(d, r"Raie large et peu profonde : la signature attendue d'une forte prédissociation.", pas=5)
    deck.finir(d)

    d = deck.nouvelle("Une observation similaire : RbYb",
                      r"Franzen \textit{et al.}, Phys. Rev. A \textbf{107}, 023114 (2023)")
    puces(d, 1.1, 2.0, 13.8, 2.6, [
        (r"PA \textsuperscript{87}Rb--\textsuperscript{170}Yb près de la raie d'intercombinaison de Yb", 1),
        (r"Recherche de \SIrange{0.1}{11}{GHz} : \alert{une seule paire} de raies, vers \SI{3.1}{GHz}", 2),
        (r"Raies larges, qui s'élargissent avec $B$ et ne s'affinent pas à plus faible $I$", 3),
        (r"Interprétation proposée : \alert{prédissociation} vers $^3P_0$", 4),
    ], ancre="haut", espace=3)
    tableau(d, 2.0, 4.75, 12.0, [
        ["", "RbYb", r"\Sr{87} (cette thèse)"],
        ["Raies trouvées", "1 paire", "3"],
        [r"Pertes (cm$^3$/s)", r"$5\times10^{-14}$ (\SI{40}{W/cm^2})",
         r"$3\times10^{-14}$ -- $10^{-12}$ (\SIrange{5}{31}{W/cm^2})"],
        ["Largeur / largeur radiative", r"$\approx 17$", r"$330$ à $1350$"],
    ], [3.0, 3.4, 5.6], taille=9, h_ligne=0.45, pas=5)
    a_retenir(d, r"Question ouverte : pourquoi le \Sr{87}, et pas le \textsuperscript{173}Yb, lui aussi hyperfin ?",
              pas=6, h=0.6, y=7.5)
    deck.finir(d)

    d = deck.nouvelle("Les pistes à explorer")
    (x1, w1), (x2, w2) = colonnes(0.5, 0.45)
    paragraphes(d, x1, 1.9, w1, 0.45, [(r"{\fontforte\color{gris} Déjà testé, sans effet}", {"taille": 10}, None)],
                ancre="haut")
    puces(d, x1, 2.35, w1, 1.4, [(r"Polarisation du laser de PA", 1), (r"Champ magnétique", 1)],
          taille=10, ancre="haut")
    paragraphes(d, x1, 3.9, w1, 0.45, [(r"{\fontforte\color{violet} À faire}", {"taille": 10}, None)], ancre="haut")
    puces(d, x1, 4.35, w1, 3.0, [
        (r"Largeur en fonction de $I$ : déplacement lumineux ou prédissociation ?", 1),
        (r"Séquence de \textit{chopping} piège / PA sur le \Sr{87}", 1),
        (r"Pompage optique : moins de potentiels, moins de prédissociation", 1)], taille=10, ancre="haut")
    image(d, png("Figures_pour_theorie/predissociation.pdf"), x2, 1.9, w2, 0.6 * 7.9)
    deck.finir(d)

    d = deck.nouvelle("Bilan de la photoassociation")
    paragraphes(d, 1.1, 2.6, 13.8, 0.7, [(r"\underline{Objectif} : une dissipation à deux corps sélective en spin",
                                          {"taille": 12, "couleur": "violet"}, None)], align="centre")
    puces(d, 3.1, 3.6, 9.8, 2.6, [
        (r"{\color{vert}Point fort} : une voie vers la limite de Heisenberg, encore inexplorée sur $^3P_1$", 1),
        (r"{\color{violet}Résultat surprenant} : seulement trois raies, $10^2$ à $10^3$ fois trop larges", 1),
    ], ancre="haut")
    deck.finir(d)

    # ======================= Manipulations cohérentes =======================
    page_section(deck, pages_section[3], 3, "Manipulations cohérentes")

    d = deck.nouvelle("Objectif et atouts de notre dispositif")
    paragraphes(d, 1.1, 2.3, 13.8, 1.6, [
        (r"\underline{Objectif} : un espace de Hilbert accessible de grande dimension", {"taille": 14.4, "couleur": "violet"}, None),
        (r"$\to$ un outil pour contrôler et mesurer des grandeurs physiques", {"taille": 12, "couleur": "violet"}, None),
    ], align="centre", espace=3)
    paragraphes(d, 1.1, 4.8, 13.8, 1.4, [
        (r"Gaz de \Sr{87} ultrafroid, piégé optiquement", {"taille": 12}, None),
        (r"Spin nucléaire $I = 9/2$ $\to$ du pseudo-spin $\frac{1}{2}$ au qudit", {"taille": 12}, None),
    ], align="centre", espace=3)
    deck.finir(d)

    d = deck.nouvelle("Contrôler l'état de spin", "Contrôler la taille de l'espace")
    (x1, w1), (x2, w2) = colonnes(0.52, 0.44)
    for k in range(5):
        im = image(d, png(f"sfo/pompage_{k}.pdf", fond="papier"), x1, 2.2, w1, 0.46 * 7.9)
        im.name = f"Pompage, étape {k + 1}"
        d.anim.ajouter(k + 1, im)
    t1 = paragraphes(d, x1, 2.2 + 0.46 * 7.9 + 0.1, w1, 0.5,
                     [(r"{\fontforte\color{violet}10 états de spin !}", {}, None)], align="centre")
    t2 = paragraphes(d, x1, 2.2 + 0.46 * 7.9 + 0.1, w1, 0.5,
                     [(r"{\fontforte\color{violet}1 seul état de spin !}", {}, None)], align="centre")
    t2.fill.solid()
    t2.fill.fore_color.rgb = couleur("papier")
    d.anim.ajouter(3, t2)
    puces(d, x2, 2.0, w2, 4.5, [
        (r"\textbf{Pompage optique} : tous les atomes dans le même état $m_F$", 2),
        (r"\textbf{Transition Raman} : on les transfère vers l'état choisi", 4),
    ])
    a_retenir(d, r"De dix états à un seul, puis à celui que l'on choisit.", pas=5)
    deck.finir(d)

    d = deck.nouvelle("Déplacement linéaire : non sélectif en spin")
    images = []
    for k, nom in enumerate(["zeeman_0", "zeeman_1", "zeeman_2", "zeeman_4", "zeeman_6"]):
        im = image(d, png(f"sfo/{nom}.pdf", fond="papier"), 1.1, 1.65, 13.8, 0.6 * 7.9)
        im.name = f"Zeeman, étape {k + 1}"
        d.anim.ajouter(k + 1, im)
        images.append(im)
    ref = images[0]
    texte_sur(d, ref, 0.79, 0.62, "Transition Raman", {"taille": 12, "couleur": "nuit"}, pas=2, w=4.5)
    texte_sur(d, ref, 0.8, 0.06, "Créer un état arbitraire", {"couleur": "nuit"}, w=4.5)
    texte_sur(d, ref, 0.8, 0.06, "Interaction Zeeman", {"couleur": "nuit"}, pas=3, w=4.5, fond="papier")
    a_retenir(d, r"Écarts égaux : la transition Raman couple tous les états voisins.", pas=5)
    deck.finir(d)

    d = deck.nouvelle("Comment isoler deux états parmi dix ?")
    (x1, w1), (x2, w2) = colonnes(0.6, 0.38)
    image(d, png("sfo/tls_0.pdf", fond="papier"), x1, 1.6, w1, 6.4)
    im = image(d, png("sfo/tls_2.pdf", fond="papier"), x1, 1.6, w1, 6.4)
    d.anim.ajouter(3, im)
    t = paragraphes(d, x2, 2.6, w2, 1.0, [(r"Hors résonance : comme si les états n'étaient pas couplés",
                                            {"couleur": "nuit"}, None)])
    d.anim.ajouter(3, t)
    bloc(d, x2, 4.1, w2, r"Déplacement lumineux tensoriel", [r"$\delta E \propto q\,m_F^2 + \beta\,m_F$"],
         h=1.45, pas=2)
    deck.finir(d)

    d = deck.nouvelle("Préparation d'un état cohérent du qubit")
    (x1, w1), (x2, w2) = colonnes(0.3, 0.68)
    for k, pas in ((0, 1), (1, 2), (2, 3)):
        im = image(d, png(f"sfo/atomes_{k}.pdf", fond="papier"), x1 + 0.1 * w1, 2.0, 0.8 * w1, 2.6, valign="haut")
        d.anim.ajouter(pas, im)
    texte_sur(d, im, 0.85, 1.05, "Superposition", {"taille": 10, "couleur": "nuit"}, pas=3, w=2.5)
    paragraphes(d, x1, 4.75, w1, 0.9, [
        (r"{\color{blue}●} population $-3/2$", {"taille": 10}, None),
        (r"{\color{violet}●} population $-7/2$ {\scriptsize(spectatrice)}", {"taille": 10}, None)], ancre="haut")
    t = paragraphes(d, x1, 5.8, w1 + 0.6, 0.6, [(r"$\left|\psi_0\right\rangle = N\left(a\ket{-\tfrac32} + b\ket{-\tfrac52}\right)$",
                                                 {"taille": 10}, None)], ancre="haut")
    d.anim.ajouter(3, t)
    t = paragraphes(d, x1, 6.5, w1, 0.5, [(r"$a = b = 0.5$", {"taille": 10}, None)], align="centre", ancre="haut")
    d.anim.ajouter(4, t)
    paragraphes(d, x2, 1.85, w2, 0.9, [
        (r"{\color{nuit}Temps de cohérence de \SI{300(20)}{ms} $\to$ 21 cycles de Rabi}", {"taille": 10}, None),
        (r"{\color{violet}Oscillations de Rabi} entre $m_F = -3/2$ et $m_F = -5/2$", {"taille": 10}, None),
    ], align="centre", ancre="haut")
    rabi = image(d, png("sfo/rabi.pdf"), x2, 2.85, w2, 0.44 * 7.9)
    x0 = rabi.left / 360000 / ECHELLE
    y0 = (rabi.top + rabi.height) / 360000 / ECHELLE
    lw, lh = rabi.width / 360000 / ECHELLE, rabi.height / 360000 / ECHELLE
    fl = fleche(d, x0, y0 + 0.12 * lh, x0 + 0.115 * lw, y0 - 0.45 * lh, "red!80!black", epaisseur=3)
    d.anim.ajouter(4, fl)
    t = paragraphes(d, x2, 6.6, w2, 0.5, [(r"Déterminer $t_{\pi/2}$ : moitié de la population dans chaque état",
                                           {"taille": 10}, None)], align="centre")
    d.anim.ajouter(4, t)
    deck.finir(d)

    d = deck.nouvelle("Contrôle de phase et cohérence du qubit", "Contrôle de la phase par un interféromètre de Ramsey")
    t = paragraphes(d, 1.1, 2.0, 13.8, 0.8, [(r"$\varphi = \left(E_{-5/2} - E_{-7/2} - \hbar\delta\right)\frac{T}{\hbar}, "
                                               r"\qquad \delta = \omega_{L1} - \omega_{L2}$", {}, None)], align="centre")
    d.anim.ajouter(2, t)
    image(d, png("sfo/ramsey_schema.pdf"), 1.1 + 0.075 * 13.8, 2.9, 0.85 * 13.8, 3.9)
    paragraphes(d, 1.1, 6.9, 6.0, 0.9, [(r"$\left|\psi_0\right\rangle = \frac{1}{\sqrt2}\left(\ket{-\tfrac52} + \ket{-\tfrac72}\right)$",
                                         {}, None)], align="centre")
    t = paragraphes(d, 8.2, 6.9, 6.7, 0.9, [(r"$\left|\psi_t\right\rangle = \frac{1}{\sqrt2}\left(\ket{-\tfrac52} + e^{-i\varphi}\ket{-\tfrac72}\right)$",
                                             {}, None)], align="centre")
    d.anim.ajouter(3, t)
    deck.finir(d)

    d = deck.nouvelle("Contrôle de phase et cohérence du qubit", "Mesure d'un long temps de cohérence")
    (x1, w1), (x2, w2) = colonnes(0.66, 0.32)
    image(d, png("sfo/ramsey_franges_0.pdf", fond="papier"), x1, 2.0, w1, 5.2)
    im = image(d, png("sfo/ramsey_franges_1.pdf", fond="papier"), x1, 2.0, w1, 5.2)
    d.anim.ajouter(2, im)
    paragraphes(d, x1, 7.25, w1, 0.4, [(r"{\color{blue}●} population $-5/2$", {"taille": 10}, None)])
    paragraphes(d, x2, 2.0, w2, 1.3, [
        (r"$\varphi = \left(E_{-5/2} - E_{-7/2} - \hbar\delta\right)\frac{T}{\hbar}$", {}, None),
        (r"La phase varie avec $T$", {"taille": 10}, None)], align="centre", ancre="haut")
    t = paragraphes(d, x2, 3.6, w2, 0.9, [(r"{\color{violet}\fontforte Même contraste pendant plus de \SI{3}{s} !}", {}, None)],
                    align="centre")
    d.anim.ajouter(2, t)
    im = image(d, png("sfo/ramsey_contraste.pdf"), x2, 4.6, w2, 0.32 * 7.9)
    d.anim.ajouter(2, im)
    deck.finir(d)

    d = deck.nouvelle("Exploiter le grand espace de Hilbert")
    image(d, png("sfo/detecteurs.pdf"), 1.1, 1.8, 13.8, 0.32 * 7.9)
    paragraphes(d, 1.1, 4.6, 13.8, 2.4, [
        (r"\underline{De nouvelles possibilités ?}", {"taille": 14.4, "couleur": "nuit"}, None),
        (r"Mesurer des observables non commutables en une seule opération", {"taille": 12}, None),
        (r"grâce au grand spin", {"couleur": "gris"}, None)], align="centre", ancre="haut", espace=3)
    deck.finir(d)

    d = deck.nouvelle("Au-delà du qubit", "Mesure simultanée d'observables non commutables")
    (x1, w1), (x2, w2) = colonnes(0.58, 0.4)
    sch = image(d, png("sfo/noncomm_schema.pdf"), x1, 1.95, w1, 3.6, valign="haut")
    texte_sur(d, sch, 0.43, 0.53, r"Préparation\\cohérente\\du pseudo-\\spin",
              {"taille": 6, "forte": True, "couleur": "white"}, w=1.6, h=1.0)
    rose = d.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                              Emu(sch.left + int(0.535 * sch.width)), Emu(sch.top + int((1 - 0.36) * sch.height)),
                              Emu(int(0.345 * sch.width)), Emu(int(0.13 * sch.height)))
    rose.fill.solid()
    rose.fill.fore_color.rgb = RGBColor(0xF3, 0xD7, 0xFD)
    rose.line.fill.background()
    rose.shadow.inherit = False
    texte_sur(d, sch, 0.705, 0.295, "Schéma de mesure", {"taille": 8}, w=3.0, h=0.45)
    paragraphes(d, x2, 2.0, w2, 3.4, [
        (r"{\color{violet}\scriptsize Non fermé $\to$ mesure de population}", {"taille": 10}, 2),
        (r"$\hat S^Z \to p_{-3/2} - p_{-9/2}$", {"taille": 10}, 2),
        (r"{\color{violet}\scriptsize Mesure de phase}", {"taille": 10}, 3),
        (r"$\hat S_\varphi = \hat S^Y\cos\varphi + \hat S^X\sin\varphi$", {"taille": 10}, 3),
        (r"$\hat S^X \to \ket{-\tfrac52} + e^{i\pi/2}\ket{-\tfrac72}$", {"taille": 10}, 3),
        (r"$\hat S^Y \to \ket{-\tfrac52} + \ket{-\tfrac72}$", {"taille": 10}, 3)], ancre="haut")
    (y1, v1), (y2, v2), (y3, v3) = colonnes(0.3, 0.28, 0.38)
    for pas, latex in ((5, r"$\varphi = 0$\\$\langle\hat S^Z\rangle = 0$, $\langle\hat S^Y\rangle = \hbar/2$,\\$\langle\hat S^X\rangle = 0$"),
                       (7, r"$\varphi = \pi/2$"),
                       (9, r"$\varphi = \pi/2$\\$\langle\hat S^Z\rangle = 0$, $\langle\hat S^Y\rangle = 0$,\\$\langle\hat S^X\rangle = \hbar/2$")):
        t = paragraphes(d, y1, 5.75, v1, 1.6, [(latex, {"taille": 10}, None)], align="centre")
        t.fill.solid()
        t.fill.fore_color.rgb = couleur("papier")
        d.anim.ajouter(pas, t)
    for k, pas in zip(range(3, 10), range(4, 11)):
        im = image(d, png(f"sfo/bloch_{k}.pdf", fond="papier"), y2, 5.55, v2, 0.31 * 7.9)
        d.anim.ajouter(pas, im)
    paragraphes(d, y3, 5.7, v3, 1.0, [
        (r"{\color{violet}$\hat S^Z, \hat S^X, \hat S^Y$ mais $[\hat S^{X,Y}, \hat S^Z] \neq 0$\\$\to$ agrandir l'espace de Hilbert}",
         {"taille": 10}, None)], ancre="haut")
    t = paragraphes(d, y3, 6.9, v3, 0.5, [(r"{\color{nuit}\underline{Limite} : bruit de projection}", {"taille": 10}, None)])
    d.anim.ajouter(11, t)
    deck.finir(d)

    d = deck.nouvelle("Résultats : deux observables non commutables", "Chaque tir mesure les deux observables")
    paragraphes(d, 1.1, 1.95, 13.8, 0.6, [(r"$\hat S_\varphi = \hat S^Y\cos\varphi + \hat S^X\sin\varphi$", {}, None)],
                align="centre")
    image(d, png("sfo/noncomm_resultats.pdf"), 1.1 + 0.11 * 13.8, 2.55, 0.78 * 13.8, 4.6)
    (x1, w1), (x2, w2) = colonnes(0.48, 0.48)
    paragraphes(d, x1, 7.2, w1, 0.5, [("Différence de population attendue nulle", {"taille": 10}, None)], align="centre")
    paragraphes(d, x2, 7.2, w2, 0.5, [("Fluctuations comprises, et compensables", {"taille": 10}, None)], align="centre")
    deck.finir(d)

    d = deck.nouvelle("Mesurer deux champs à la fois", "Deux interféromètres de Ramsey sur quatre états")
    (x1, w1), (x2, w2) = colonnes(0.68, 0.3)
    sch = image(d, png("sfo/ramsey4_schema.pdf", fond="papier"), x1, 2.0, w1, 5.0)
    cache = d.shapes.add_shape(MSO_SHAPE.RECTANGLE, sch.left, Emu(sch.top + int(0.88 * sch.height)),
                               Emu(int(0.04 * sch.width)), Emu(int(0.12 * sch.height)))
    cache.fill.solid()
    cache.fill.fore_color.rgb = couleur("white")
    cache.line.fill.background()
    cache.shadow.inherit = False
    cache.name = "Cache (reste d'un carré du pptx SFO)"
    paragraphes(d, x2, 2.6, w2, 1.6, [
        (r"$\Delta E^{-3/2}_{-5/2} = -4q + \beta$", {}, 2),
        (r"$\Delta E^{-9/2}_{-7/2} = -8q + \beta$", {}, 2)], espace=4)
    paragraphes(d, x2, 4.4, w2, 0.5, [(r"$\delta E \propto q\,m_F^2 + \beta\,m_F$", {"taille": 10, "couleur": "gris"}, None)])
    a_retenir(d, r"Deux paires d'états, deux énergies différentes : deux grandeurs mesurées en une seule réalisation.", pas=2)
    deck.finir(d)

    d = deck.nouvelle("Mesurer deux champs à la fois", "Résultats")
    (x1, w1), (x2, w2) = colonnes(0.68, 0.3)
    (x1, w1), (x2, w2) = colonnes(0.62, 0.36)
    image(d, png("sfo/ramsey4_resultats.pdf"), x1, 1.9, w1, 5.0)
    paragraphes(d, x2, 1.9, w2, 5.0, [
        (r"{\color{blue}●} $m_F=-3/2$", {}, None),
        (r"● $m_F=-7/2$", {}, None),
        (r"$\Delta E^{-3/2}_{-5/2} = -4q + \beta$", {}, None),
        (r"$\Delta E^{-9/2}_{-7/2} = -8q + \beta$", {}, None),
        (r"$\left(E_{-9/2} - E_{-7/2} - \hbar\omega_{RF}\right)\frac{T}{\hbar} = \phi$", {"taille": 10}, None),
        (r"$\omega_{RF} = \omega_{L1} - \omega_{L2}$", {"taille": 10}, None)], espace=3)
    a_retenir(d, r"Deux périodes différentes : les deux grandeurs sont bien distinguées.")
    deck.finir(d)

    d = deck.nouvelle("Bilan des manipulations cohérentes")
    paragraphes(d, 1.1, 1.9, 13.8, 0.7, [(r"\underline{Objectif} : une plateforme pour manipuler et caractériser des qudits",
                                          {"taille": 12, "couleur": "violet"}, None)], align="centre")
    puces(d, 3.1, 2.8, 9.8, 3.6, [
        (r"Isoler des paires d'états (qubits)", 1),
        (r"Les contrôler de façon cohérente\\$\to$ espèce à grand spin nucléaire, quasi insensible au champ magnétique", 1),
        (r"Mesurer des observables non commutables\\$=$ mesure à valeurs d'opérateurs positifs (POVM)", 1),
    ], ancre="haut")
    a_retenir(d, r"{\color{vert}Points forts} : au-delà de Stern-Gerlach, deux états isolés parmi dix ; "
                 r"deux observables non commutables mesurées en une seule réalisation.", h=0.95)
    deck.finir(d)

    # ======================= Conclusion =======================
    page_section(deck, pages_section[4], 4, "Conclusion")

    d = deck.nouvelle("Conclusion")
    (x1, w1), (x2, w2) = colonnes(0.48, 0.48)
    paragraphes(d, x1, 2.0, w1, 0.5, [(r"{\fontforte\color{violet} Photoassociation}", {}, None)], ancre="haut")
    puces(d, x1, 2.6, w1, 3.6, [
        (r"Protocole validé sur le \Sr{88}", 1),
        (r"Trois raies du \Sr{87} sur $^3P_1$", 1),
        (r"Raies $10^2$ à $10^3$ fois trop larges : une découverte inattendue", 1)], ancre="haut")
    paragraphes(d, x2, 2.0, w2, 0.5, [(r"{\fontforte\color{vert} Manipulations cohérentes}", {}, None)], ancre="haut")
    puces(d, x2, 2.6, w2, 3.6, [
        (r"Pseudo-spins isolés parmi dix états", 1),
        (r"Mesure simultanée de deux grandeurs", 1),
        (r"Deux observables non commutables en une réalisation", 1)], ancre="haut")
    a_retenir(d, r"Perspective : utiliser les outils de contrôle de spin pour sonder les atomes photoassociés.")
    deck.finir(d)

    d = deck.nouvelle("Perspectives : sonder la PA avec nos interféromètres")
    puces(d, 1.1, 1.75, 13.8, 5.3, [
        (r"\textbf{Interféromètres parallèles}\\{\small mesurer \textit{in situ} l'effet du laser de PA sur les atomes "
         r"(déplacements $q$ et $b$ : intensité, polarisation, inhomogénéité)}", 1),
        (r"\textbf{Mesure POVM}\\{\small caractériser en une seule réalisation la population ($\hat S^z$) \alert{et} "
         r"la phase de l'état après PA}", 2),
        (r"\textbf{Mesure de brisure de symétrie}\\{\small qubit $\{a,b\}$ + une troisième composante $c$ ; si la PA "
         r"n'a pas la même force entre $\{a,b\}$ et $\{b,c\}$, la symétrie du qubit est brisée}\\"
         r"{\small $\to$ $\langle \hat S^z\rangle$ nul sans PA vers $c$, non nul avec ; possible avec deux interféromètres parallèles}", 3),
    ], espace=4)
    a_retenir(d, r"Les outils de la partie 2 deviennent des sondes de la partie 1.", pas=3, h=0.6)
    deck.finir(d)

    d = deck.nouvelle("Perspectives : mesurer l'intrication de l'état photoassocié")
    (x1, w1), (x2, w2) = colonnes(0.55, 0.42)
    puces(d, x1, 1.75, w1, 5.0, [
        (r"La PA ne retire que les singulets : l'état final peut être un \alert{état de Dicke} intriqué", 1),
        (r"La PA change $N$ et $\hat S^{x}$, $\hat S^{y}$, $\hat S^{z}$, mais ne doit pas ajouter de fluctuations", 2),
        (r"Il faut un \alert{état initial précis} (le pompage optique ne suffit pas) :\\"
         r"{\small moitié des atomes polarisée de façon déterministe, ou réseau + TLS + renversement adiabatique du spin}", 3),
    ], espace=4)
    bloc(d, x2, 2.0, w2, "Témoin d'intrication",
         [r"$\xi_D^2 = \frac{N\,\mathrm{Var}(\hat S^z)}{\langle \hat S_x^2 + \hat S_y^2\rangle - N/2} < 1$"], h=1.9)
    t = paragraphes(d, x2, 4.05, w2, 1.6, [
        (r"$\langle \hat S_x^2 + \hat S_y^2\rangle$ : accessible avec la POVM", {"taille": 10}, None),
        (r"$\mathrm{Var}(\hat S^z)$ : il faut un bruit de détection $<$ bruit de projection $\sqrt{N}$", {"taille": 10}, None)],
        ancre="haut", espace=2)
    d.anim.ajouter(4, t)
    a_retenir(d, r"Une texture de spin déterministe ; la limite actuelle est la détection, pas le principe.", pas=4, h=0.6)
    deck.finir(d)

    d = deck.nouvelle("Perspectives : vers les horloges",
                      r"Foss-Feig, Daley, Thompson, Rey, PRL \textbf{109}, 230501 (2012)")
    (x1, w1), (x2, w2) = colonnes(0.5, 0.46)
    paragraphes(d, x1, 2.0, w1, 0.5, [(r"{\fontforte\color{vert} L'idée}", {}, None)], ancre="haut")
    puces(d, x1, 2.5, w1, 4.4, [
        (r"Préparer un état de Dicke, puis l'interroger sur la transition d'horloge", 1),
        (r"Pas de collision en onde $s$ $\to$ \alert{pas de déplacement collisionnel} (mesuré dans les horloges actuelles)", 2),
        (r"Interroger $\pm m_F$ : deux horloges à la fois", 3)], ancre="haut")
    paragraphes(d, x2, 2.0, w2, 0.5, [(r"{\fontforte\color{violet} Les limites}", {}, None)], ancre="haut")
    puces(d, x2, 2.5, w2, 4.4, [
        (r"Peu d'atomes : précision proche d'une horloge usuelle", 4),
        (r"Onde $p$ non supprimée", 4),
        (r"Laser d'horloge inhomogène : intrication perdue", 4)], ancre="haut")
    a_retenir(d, r"Notre PA passe par $^3P_1$, pas par l'horloge : pas le même objet, mais on peut tester la "
                 r"symétrie SU($N$) du fondamental.", pas=5, h=0.8)
    deck.finir(d)

    d = page_forte(deck, [("Merci de votre attention", {"taille": 14.4, "forte": True, "couleur": "white"})])
    trait_laser(d, 6.5, 5.0, 3.0)
    deck.finir(d)

    # ======================= Annexes =======================
    deck.annexe = True
    deck.numero = 0

    d = deck.nouvelle(r"Ajustement de $N(t)$ avec chauffage")
    bloc(d, 1.1, 1.75, 13.8, r"Fonction d'ajustement (éq. 3.79 de la thèse)",
         [r"$N(t) = \frac{N_0 e^{-\Gamma t}}{1 + \left[\frac{K_2}{8\pi^{3/2}}\,m_{\text{at}}^{3/2}\,\omega_x\omega_y\omega_z\right] "
          r"N_0 \int_0^t \left(\frac{1}{k_B T(t')}\right)^{3/2} e^{-\Gamma t'}\,dt'}$"], h=2.3)
    puces(d, 1.1, 4.4, 13.8, 2.5, [
        (r"$T(t)$ mesurée, interpolée linéairement, \alert{fixée} avant l'ajustement", 1),
        (r"Paramètres libres : $\Gamma$, $K_2$, $N_0$, offset (minimisation du $\chi^2$)", 1),
        (r"$K_2$ constant ; $\beta(t)$ varie seulement parce que $T(t)$ varie", 1)], ancre="haut")
    deck.finir(d)

    d = deck.nouvelle(r"Résultats complets sur le \Sr{87}")
    tableau(d, 2.0, 1.75, 12.0, [
        [r"Raie (MHz)", r"$-259.01 \pm 0.12$", r"$-276.83 \pm 0.03$", r"$-377.2 \pm 0.5$"],
        [r"\textit{Spectre}", "", "", ""],
        ["Profondeur", r"$0.40 \pm 0.02$", r"$0.35 \pm 0.01$", r"$0.09 \pm 0.01$"],
        [r"Largeur $\Gamma_{\text{fit}}$ (MHz)", r"$7.4 \pm 0.9$", r"$5.0 \pm 0.3$", r"$20.2 \pm 2.3$"],
        [r"$T$ (µK)", r"$2.38\pm0.08$", r"$1.25\pm0.03$", r"$2.38\pm0.04$"],
        [r"$\langle n\rangle$ ($10^{12}$ cm$^{-3}$)", r"$7.0\pm1.0$", r"$16\pm2$", r"$7.2\pm1.0$"],
        [r"$I$ (W/cm$^2$)", r"$19.0\pm1.1$", r"$5.2\pm0.3$", r"$5.6\pm0.4$"],
        [r"\textit{Déclin $N(t)$}", "", "", ""],
        [r"$I$ (W/cm$^2$)", r"$19.0\pm1.1$", r"$31.0\pm1.8$", r"$19.0\pm1.1$"],
        [r"$\langle n\rangle$ ($10^{12}$ cm$^{-3}$)", r"$7.0\pm1.0$", r"$7.8\pm1.1$", r"$7.0\pm1.0$"],
        [r"$K_2$ ($10^{-13}$ cm$^3$/s)", r"$5.0\pm2.8$", r"$11\pm2$", r"$0.27\pm0.19$"],
        [r"$\eta$", r"$493\pm60$", r"$332\pm17$", r"$1349\pm155$"],
        [r"$l_{\text{opt}}/I$, méthode 1 (spectre)", r"$0.63\pm0.13$", r"$0.54\pm0.08$", r"$0.84\pm0.17$"],
        [r"$l_{\text{opt}}/I$, méthode 2 (déclin)", r"$3.4\pm1.9$", r"$3.1\pm0.5$", r"$0.49\pm0.35$"],
    ], [4.2, 2.6, 2.6, 2.6], taille=7, h_ligne=0.43, regles=(0, 0, 6, 10))
    deck.finir(d)

    d = deck.nouvelle(r"Les autres raies du \Sr{87}")
    image(d, png("87Sr/composite_259MHz.pdf"), 1.1, 1.8, 6.8, 0.72 * 7.9)
    image(d, png("87Sr/composite_375MHz.pdf"), 8.1, 1.8, 6.8, 0.72 * 7.9)
    deck.finir(d)

    d = deck.nouvelle(r"Effet du laser sur la largeur (\Sr{88})")
    image(d, png("88Sr/comparaison_raie_24MHz_88_jolie.pdf"), 1.1, 1.7, 13.8, 0.66 * 7.9)
    paragraphes(d, 1.1, 7.15, 13.8, 0.4, [(r"MOGlabs : \SI{231}{kHz} \textperiodcentered{} laser injecté : \SI{122}{kHz}",
                                           {"taille": 6, "couleur": "gris"}, None)], align="centre")
    deck.finir(d)

    d = deck.nouvelle(r"Séquence de \textit{chopping} piège / PA")
    image(d, png("Figures_pour_theorie/sequence_chopping.pdf"), 1.1, 1.7, 13.8, 0.7 * 7.9)
    deck.finir(d)

    d = deck.nouvelle(r"Le faux signal à $\pm 6\,\mathrm{MHz}$")
    image(d, png("87Sr/data15_6MHz_noise.pdf"), 1.1, 1.8, 6.8, 0.66 * 7.9)
    image(d, png("87Sr/6MHz_blue_detuned.pdf"), 8.1, 1.8, 6.8, 0.66 * 7.9)
    deck.finir(d)

    d = deck.nouvelle("Limites de la mesure non commutable")
    puces(d, 1.1, 1.6, 13.8, 6.4, [
        (r"Étendre à six états pour $\hat S^Z$, $\hat S^X$, $\hat S^Y$ en une réalisation\\$\to$ plus de bruit de projection quantique", 1),
        (r"Plus d'états $m_F$ $\to$ les états positifs sont mal isolés", 1),
        (r"Réduire les fuites en adoucissant les rampes des impulsions", 1)])
    deck.finir(d)


# ---------------------------------------------------------------------
#  Équations : LaTeX -> OMML (pandoc), puis remplacement des marqueurs
# ---------------------------------------------------------------------
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
      "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
      "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006"}


def omml_des_equations():
    MEDIA.mkdir(parents=True, exist_ok=True)
    md = MEDIA / "equations.md"
    md.write_text("".join(f"# E{k}\n\n${math_pour_pandoc(e)}$\n\n" for k, e in enumerate(EQUATIONS)))
    pptx = MEDIA / "equations.pptx"
    subprocess.run(["pandoc", str(md), "-o", str(pptx)], check=True)
    z = zipfile.ZipFile(pptx)
    resultat = {}
    for k in range(len(EQUATIONS)):
        arbre = etree.fromstring(z.read(f"ppt/slides/slide{k + 1}.xml"))
        om = arbre.find(".//m:oMath", NS)
        resultat[k] = om
    return resultat


def poser_equations(chemin):
    """Remplace chaque marqueur ⟦EQn⟧ par l'équation native, au format du texte qui l'entoure."""
    omml = omml_des_equations()
    manquantes = []
    tmp = chemin.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(chemin) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename.startswith("ppt/slides/slide") and info.filename.endswith(".xml"):
                arbre = etree.fromstring(data)
                if _remplacer(arbre, omml, manquantes):
                    data = etree.tostring(arbre, xml_declaration=True, encoding="UTF-8", standalone=True)
            dst.writestr(info, data)
    tmp.replace(chemin)
    if manquantes:
        print("équations non converties (laissées en texte) :", *manquantes, sep="\n  ")


def _remplacer(arbre, omml, manquantes):
    modifie = False
    for r in arbre.iter("{%s}r" % NS["a"]):
        t = r.find("a:t", NS)
        m = re.fullmatch("\u27e6EQ(\\d+)\u27e7", t.text or "")
        if not m:
            continue
        k = int(m.group(1))
        rpr = r.find("a:rPr", NS)
        om = omml.get(k)
        if om is None:
            manquantes.append(EQUATIONS[k])
            t.text = EQUATIONS[k]
            continue
        om = copy.deepcopy(om)
        for mr in om.iter("{%s}r" % NS["m"]):
            arpr = copy.deepcopy(rpr)
            for attr in ("b", "baseline"):
                arpr.attrib.pop(attr, None)
            droit = mr.find("m:rPr/m:sty", NS) is not None
            arpr.set("i", "0" if droit else "1")
            latin = arpr.find("a:latin", NS)
            if latin is None:
                latin = etree.SubElement(arpr, "{%s}latin" % NS["a"])
            latin.set("typeface", "Cambria Math")
            for el in list(arpr):
                if el.tag in ("{%s}ea" % NS["a"], "{%s}cs" % NS["a"]):
                    arpr.remove(el)
            mt = mr.find("m:t", NS)
            mt.addprevious(arpr)
        a14m = etree.Element("{%s}m" % NS["a14"], nsmap={"a14": NS["a14"]})
        a14m.append(_seule_a_gauche(r, om))
        r.addprevious(a14m)
        r.getparent().remove(r)
        modifie = True
    if modifie:
        _envelopper(arbre)
    return modifie


def _seule_a_gauche(r, om):
    """Une équation seule dans un paragraphe aligné à gauche : PowerPoint la centrerait
    (mode « affichage ») ; on la met dans un m:oMathPara justifié à gauche."""
    p = r.getparent()
    autres = [t.text for t in p.findall("a:r/a:t", NS) if t.getparent() is not r and (t.text or "").strip()]
    ppr = p.find("a:pPr", NS)
    if autres or p.findall("a14:m", NS) or (ppr is not None and ppr.get("algn") in ("ctr", "r")):
        return om
    para = etree.Element("{%s}oMathPara" % NS["m"], nsmap={"m": NS["m"]})
    pr = etree.SubElement(para, "{%s}oMathParaPr" % NS["m"])
    etree.SubElement(pr, "{%s}jc" % NS["m"]).set("{%s}val" % NS["m"], "left")
    para.append(om)
    return para


def _envelopper(arbre):
    """Les formes qui contiennent des équations vont dans mc:AlternateContent, comme le fait PowerPoint."""
    sptree = arbre.find(".//p:spTree", NS)
    for forme in list(sptree):
        if forme.find(".//a14:m", NS) is None:
            continue
        repli = copy.deepcopy(forme)
        for m in repli.findall(".//a14:m", NS):
            texte = "".join(m.itertext())
            r = etree.Element("{%s}r" % NS["a"])
            etree.SubElement(r, "{%s}t" % NS["a"]).text = texte
            m.addprevious(r)
            m.getparent().remove(m)
        alt = etree.Element("{%s}AlternateContent" % NS["mc"], nsmap={"mc": NS["mc"]})
        choix = etree.SubElement(alt, "{%s}Choice" % NS["mc"], nsmap={"a14": NS["a14"]})
        choix.set("Requires", "a14")
        forme.addprevious(alt)
        choix.append(forme)
        etree.SubElement(alt, "{%s}Fallback" % NS["mc"]).append(repli)


# ---------------------------------------------------------------------
def pages_de_section():
    """Numéro de page du PDF Beamer de chaque page de section, lu dans le fichier .nav."""
    nav = (ICI / "out" / "soutenance.nav").read_text(encoding="utf-8")
    return {int(n): int(p) for n, p in re.findall(r"\\sectionentry \{(\d+)\}\{.*?\}\{(\d+)\}", nav)}


def main():
    deck = Deck()
    diapos(deck, pages_de_section())
    deck.prs.save(SORTIE)
    poser_equations(SORTIE)
    print(f"{SORTIE} : {len(deck.prs.slides)} diapos, {len(EQUATIONS)} équations")


if __name__ == "__main__":
    main()
