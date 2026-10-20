# -*- coding: utf-8 -*-
"""
RÉGLAGES DES VOIX CLONÉES — pour botqwenClaude.py
==================================================

Tout ce qui sert à choisir et à expliquer le réglage d'une voix clonée, sans dépendre des modèles
(on peut le tester seul) :

  - la MANIÈRE DE COPIER la voix de référence :
      « imitation » : la voix ET le texte de la référence guident Qwen → elle garde l'accent, l'âge,
                      le grain, le rythme ;
      « timbre »    : seule la couleur de la voix est gardée → un français neutre, sans accent ;
  - le CARACTÈRE (la « température » de Qwen), en 5 crans, de « Très posée » (0.4) à « Très vivante » (0.9) ;
  - le choix automatique après le test d'une voix (même règle que banc_voix/banc_bibliotheque.py) ;
  - la traduction des mesures en mots (« très fidèle », « 3 mots manqués sur 100 »…).
"""

import re
import time
import unicodedata
from statistics import mean

# ── La manière de copier ───────────────────────────────────────────────────
MODES = {
    "imitation": ("Imitation",
                  "Copie aussi la façon de parler de l'enregistrement : l'accent, l'âge, le grain, le rythme. "
                  "C'est le plus ressemblant."),
    "timbre": ("Timbre seul",
               "Garde seulement la couleur de la voix : elle parle un français neutre, sans accent ni "
               "particularités. Utile si l'imitation avale des mots."),
}

# ── Le caractère : 5 crans ────────────────────────────────────────────────
CRANS = [
    (0.4, "Très posée",
     "Toujours égale, presque lue. Bien pour de longues lectures, mais peut sembler monotone "
     "et lisser une voix rocailleuse."),
    (0.5, "Posée",
     "Régulière et fidèle : la voix ne dérive pas. Le choix le plus sûr."),
    (0.6, "Équilibrée",
     "Un peu plus d'intonation, toujours sûre."),
    (0.7, "Vivante",
     "Plus de souffle, de relief et de grain : bien pour un personnage. Dérive parfois légèrement."),
    (0.9, "Très vivante",
     "Le plus expressif : l'accent, la rocaille et les émotions ressortent. Risque de mots avalés "
     "ou d'une voix qui change en cours de phrase."),
]
TEMPERATURES_CRANS = [c[0] for c in CRANS]

# ── Le test d'une voix ────────────────────────────────────────────────────
TEST_TEMPERATURES = [0.5, 0.7, 0.9]
TEST_PHRASES = [
    "Bonjour ! Je suis ravie de vous accueillir ici aujourd'hui.",
    ("Imaginez une grande salle pleine de machines à écrire : chacune tape à son rythme, "
     "et pourtant, ensemble, elles finissent par composer une sorte de musique."),
]
IMITATION_MOTS_RATES_MAX = 0.30   # au-delà, l'imitation déforme trop la parole : on se rabat sur le timbre


def cran_le_plus_proche(temperature):
    try:
        t = float(temperature)
    except (TypeError, ValueError):
        t = 0.5
    return min(TEMPERATURES_CRANS, key=lambda c: abs(c - t))


def nom_cran(temperature):
    t = cran_le_plus_proche(temperature)
    return next(n for c, n, _ in CRANS if c == t)


def explication_cran(temperature):
    t = cran_le_plus_proche(temperature)
    return next(e for c, _, e in CRANS if c == t)


def nom_mode(mode):
    return MODES.get(mode, (mode or "?", ""))[0]


# ── Mots compris ──────────────────────────────────────────────────────────
def mots(texte):
    t = unicodedata.normalize("NFD", (texte or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", t)


def taux_erreur(reference, entendu):
    """Proportion de mots avalés, ajoutés ou déformés (0 = parfait)."""
    r, h = mots(reference), mots(entendu)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prec, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prec, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prec + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


# ── Le choix après le test ────────────────────────────────────────────────
def choisir(essais):
    """essais : [{temperature, mode, ressemblance (ou None), erreurs (ou None)}]
    → (réglage retenu, bilan de chaque réglage essayé)."""
    par_reglage = {}
    for e in essais:
        par_reglage.setdefault((e["temperature"], e["mode"]), []).append(e)
    bilans = []
    for (t, m), liste in par_reglage.items():
        ress = [x["ressemblance"] for x in liste if x.get("ressemblance") is not None]
        err = [x["erreurs"] for x in liste if x.get("erreurs") is not None]
        bilans.append({"temperature": t, "mode_clone": m,
                       "ressemblance_min": round(min(ress), 3) if ress else None,
                       "ressemblance": round(float(mean(ress)), 3) if ress else None,
                       "erreurs": round(float(mean(err)), 3) if err else None})
    return retenir(bilans), bilans


def retenir(bilans, teste_le=None):
    """Même règle que banc_bibliotheque.py :
    1. la MANIÈRE : « imitation » (elle garde le caractère de la voix) tant qu'elle reste compréhensible
       (au plus 30 % de mots ratés) ; sinon « timbre » ;
    2. le CARACTÈRE : le plus stable dans cette manière, parmi les réglages bien compris ; à égalité,
       la meilleure ressemblance, puis le plus posé."""
    def err(b):
        return b["erreurs"] if b["erreurs"] is not None else 0.0

    def rmin(b):
        return b["ressemblance_min"] if b["ressemblance_min"] is not None else 1.0

    def rmoy(b):
        return b["ressemblance"] if b["ressemblance"] is not None else 1.0

    imitation = [b for b in bilans if b["mode_clone"] == "imitation"]
    timbre = [b for b in bilans if b["mode_clone"] == "timbre"]
    note = ""
    if imitation and min(err(b) for b in imitation) <= IMITATION_MOTS_RATES_MAX:
        bilans = imitation
    elif imitation and timbre:
        note = (f" (en imitation, {min(err(b) for b in imitation):.0%} de mots ratés : le timbre seul est plus "
                "clair mais perd le caractère de la voix ; un enregistrement plus net, au texte exact, aiderait)")
        bilans = timbre
    meilleure = min(err(b) for b in bilans)
    candidats = [b for b in bilans if err(b) <= meilleure + 0.08]
    choix = sorted(candidats, key=lambda b: (-rmin(b), -rmoy(b), err(b), b["temperature"],
                                             b["mode_clone"] != "imitation"))[0]
    if rmin(choix) >= 0.97 and err(choix) <= 0.15:
        verdict = "stable"
    elif rmin(choix) >= 0.93 and err(choix) <= 0.25:
        verdict = "correcte"
    else:
        verdict = "fragile : enregistrement de référence à refaire (plus propre, plus long, texte exact)"
    return {"temperature": choix["temperature"], "mode_clone": choix["mode_clone"],
            "ressemblance_min": choix["ressemblance_min"], "mots_rates": choix["erreurs"],
            "verdict": verdict + note, "teste_le": teste_le or time.strftime("%Y-%m-%d")}


# ── Les mesures en mots ───────────────────────────────────────────────────
def fidelite_en_mots(ressemblance_min):
    if ressemblance_min is None:
        return "non mesurée"
    r = float(ressemblance_min)
    if r >= 0.97:
        return "très fidèle"
    if r >= 0.95:
        return "fidèle"
    if r >= 0.93:
        return "varie un peu"
    return "dérive"


def comprehension_en_mots(erreurs):
    if erreurs is None:
        return "non mesurée"
    n = round(float(erreurs) * 100)
    if n <= 3:
        return "tous les mots compris" if n == 0 else f"presque tout compris ({n} mots manqués sur 100)"
    if n <= 15:
        return f"{n} mots manqués sur 100"
    if n <= 30:
        return f"{n} mots manqués sur 100 : certains mots sont avalés"
    return f"{n} mots manqués sur 100 : difficile à comprendre"


def date_lisible(iso):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(iso or ""))
    return f"{m.group(3)}/{m.group(2)}/{m.group(1)}" if m else ""


def resume_reglage(reglages, defaut_temperature=0.5):
    """Une phrase : le réglage en vigueur et d'où il vient."""
    if not reglages:
        return (f"**{nom_mode('imitation')} · {nom_cran(defaut_temperature)}** "
                "— réglage par défaut : cette voix n'a pas encore été testée.")
    t = reglages.get("temperature", defaut_temperature)
    mode = reglages.get("mode_clone") or "imitation"
    tete = f"**{nom_mode(mode)} · {nom_cran(t)}**"
    quand = date_lisible(reglages.get("teste_le"))
    if reglages.get("fixe"):
        return f"{tete} — choisi à l'oreille" + (f" le {quand}" if quand else "") + "."
    verdict = str(reglages.get("verdict") or "")
    morceaux = []
    if reglages.get("ressemblance_min") is not None:
        morceaux.append(fidelite_en_mots(reglages["ressemblance_min"]))
    if reglages.get("mots_rates") is not None:
        morceaux.append(comprehension_en_mots(reglages["mots_rates"]))
    detail = ", ".join(morceaux)
    phrase = f"{tete} — choisi par le test" + (f" du {quand}" if quand else "")
    if detail:
        phrase += f" : {detail}"
    phrase += "."
    if verdict.startswith("fragile"):
        phrase += " ⚠️ Voix fragile : un enregistrement de référence plus propre, plus long, au texte exact, aiderait."
    return phrase


def tableau_bilans(bilans, choisi=None):
    """Les résultats du test, en mots (Markdown)."""
    if not bilans:
        return "_Pas encore de test pour cette voix._"
    lignes = ["| Réglage essayé | Ressemblance | Compréhension |", "|---|---|---|"]
    for b in sorted(bilans, key=lambda b: (b["mode_clone"] != "imitation", b["temperature"])):
        nom = f"{nom_mode(b['mode_clone'])} · {nom_cran(b['temperature'])}"
        if choisi and (b["mode_clone"], cran_le_plus_proche(b["temperature"])) == \
                (choisi.get("mode_clone"), cran_le_plus_proche(choisi.get("temperature"))):
            nom = f"**{nom}** ✓"
        lignes.append(f"| {nom} | {fidelite_en_mots(b.get('ressemblance_min'))} | "
                      f"{comprehension_en_mots(b.get('erreurs'))} |")
    return "\n".join(lignes)


def nom_fichier(texte):
    """Un nom de dossier sûr pour un profil (« Femme Théa / joie » → « Femme_Thea_joie »)."""
    t = unicodedata.normalize("NFD", texte or "voix")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", t).strip("_") or "voix"
