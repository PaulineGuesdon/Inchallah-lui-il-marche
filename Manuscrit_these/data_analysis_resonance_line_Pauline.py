# -*- coding: utf-8 -*-
"""
Analyse de données Atomes Froids - Pauline Guesdon
Fit 2D (Images) + Fit Lorentzien 1D (Spectre Final) avec calcul du Chi²
"""

"""A modifier soit même : 
    - P0_2D pour les guess initiaux du découpage de l'image (et les autres paramètres de config et constantes si besoin)
    - le type de fit TYPE_ANALYSE_FINALE DECAY ou SPECTRE en fonction de si fit de résonance ou de décroissance exp
    - le chemin dir_path
    - col_index qui sélectionne la bonne colonne du fichier excel pour les freq (ou le temps)
    - le centre de la courbe de fit p0_spec parce que il fait n'importe quoi de lui-même
    - err_systematique_rel qui représente le bruit rsm du nuage d'atomes sans laser de PA, qui corrige l'estimation
    de l'erreur par propagation d'erreur """


#%%
# -*- coding: utf-8 -*-
"""
Analyse de données Atomes Froids - Pauline Guesdon
Modèle Dual : Spectre (Lorentz) OU Cinétique de pertes (1+2 corps)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import glob
import os
import re

# =============================================================================
# --- 1. CONFIGURATION À MODIFIER ---
# =============================================================================

# Choix de l'analyse : 'SPECTRE' (Lorentz) ou 'DECAY' (Pertes 1+2 corps)
TYPE_ANALYSE_FINALE = 'DECAY' 

# Paramètres des images RAW et constantes physiques
MODE_FIT_IMAGE = 'gauss'
SIZE = 1024
PIXEL_SIZE_REAL = 6.2e-6 
lambd = 689.45e-9 
SIGMA_ABS = (3 * lambd**2) / (2 * np.pi)
sigma_x = sigma_y = sigma_z = 16*1e-6
V_e = (2*np.pi)**(3/2)*sigma_x**3
h = 6.64*1e-34
m_atome = 1.4*1e-25
a_0 = 5.291*1e-11

# Guess pour le fit 2D des images [Amplitude, xo, yo, sx, sy, offset]
P0_2D = [0.3, 100, 100, 40, 40, 0]

# =============================================================================
# --- 2. MODÈLES DE FIT ---
# =============================================================================

def gaussian_2d_no_rot(coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = coords
    return (offset + amplitude * np.exp(-(2*(x-xo)**2)/(sigma_x**2) - (2*(y-yo)**2)/(sigma_y**2))).ravel()

def lorentzian_2d(coords, amplitude, xo, yo, gx, gy, offset):
    x, y = coords
    return (offset + amplitude / (1 + ((x-xo)/(gx/2))**2 + ((y-yo)/(gy/2))**2)).ravel()

def lorentzian_1d(x, amplitude, x0, gamma, offset):
    """Modèle pour le spectre (pertes vs fréquence)"""
    return offset + amplitude / (1 + ((x - x0) / (gamma / 2))**2)

def decay_1_and_2_body(t, gamma, beta_prime, N0, offset):
    """Modèle cinétique : dN/dt = -gamma*N - beta_prime*N^2"""
    exponent = np.clip(-gamma * t, -100, 100)
    exp_t = np.exp(exponent)
    num = gamma * N0 * exp_t
    den = gamma + beta_prime * N0 * (1 - exp_t)
    return (num / (den + 1e-20)) + offset

# =============================================================================
# --- 3. FONCTIONS DE CHARGEMENT ---
# =============================================================================

def load_picture(path, size):
    try:
        data = np.fromfile(path, dtype=np.dtype('>u2'))[4:]
        return data[:size*size].reshape((size, size))
    except: return None

def load_data(path, size):
    m_at = load_picture(path + "Atoms.dat", size)
    m_no = load_picture(path + "NoAtoms.dat", size)
    m_dk = load_picture(path + "Dark.dat", size)
    if m_at is None or m_no is None or m_dk is None: return None
    diff_at = np.maximum(m_at.astype(float) - m_dk, 1)
    diff_no = np.maximum(m_no.astype(float) - m_dk, 1)
    return np.clip(np.log(diff_no) - np.log(diff_at), 0, 2)

# =============================================================================
# --- 4. TRAITEMENT DES IMAGES ---
# =============================================================================

if __name__ == "__main__":
    dir_data = 'C:/Users/pauline.guesdon/Pictures/Figures_thèses/datas_PA/data94_atomloss_f_time/data94/'
    excel_path = 'C:/Users/pauline.guesdon/Pictures/Figures_thèses/datas_PA/datasPA_waves.xlsx'
    
    df_excel = pd.read_excel(excel_path)
    # Colonne 2 pour Fréquence (Spectre), Colonne 3 pour Temps (Decay)
    col_index = 3 if TYPE_ANALYSE_FINALE == 'DECAY' else 2
    valeurs_physiques = df_excel.iloc[:, col_index].values.astype(float)
    
    files_atoms = sorted(glob.glob(os.path.join(dir_data, "*Atoms.dat")))
    indices_sets, nb_atomes_brut, erreurs_brut = [], [], []
    func_2d = gaussian_2d_no_rot if MODE_FIT_IMAGE == 'gauss' else lorentzian_2d

    print(f"--- Analyse de {len(files_atoms)} images ---")

    for f_path in files_atoms:
        prefix = f_path.replace("Atoms.dat", "")
        n_match = re.search(r'RAW_(\d+)_', os.path.basename(f_path))
        n_data = int(n_match.group(1)) if n_match else 0

        mat_do = load_data(prefix, SIZE)
        if mat_do is not None:
            crop = mat_do[400:600, 400:600]
            y_g, x_g = np.indices(crop.shape)
            try:
                popt, pcov = curve_fit(func_2d, (x_g, y_g), crop.ravel(), p0=P0_2D)
                perr = np.sqrt(np.diag(pcov))
                A, sx, sy = popt[0], np.abs(popt[3]), np.abs(popt[4])
                # Calcul du nombre d'atomes N
                N = (np.abs(A) * (2*np.pi) * sx * sy) * (PIXEL_SIZE_REAL**2 / SIGMA_ABS)
                rel_err = np.sqrt((perr[0]/A)**2 + (perr[3]/sx)**2 + (perr[4]/sy)**2)
                
                if n_data < len(valeurs_physiques):
                    indices_sets.append(valeurs_physiques[n_data])
                    nb_atomes_brut.append(N)
                    erreurs_brut.append(N * rel_err)
            except: continue

    # =========================================================================
    # --- 5. ANALYSE FINALE ---
    # =========================================================================

    if len(nb_atomes_brut) > 5:
        x_s = np.array(indices_sets)
        y_s = np.array(nb_atomes_brut)
        dy_tot = np.sqrt(np.array(erreurs_brut)**2 + (0.15 * y_s)**2)
        

        # Nettoyage et Tri
        mask = np.isfinite(x_s) & np.isfinite(y_s)
        x_s, y_s, dy_tot = x_s[mask], y_s[mask], dy_tot[mask]
        idx = np.argsort(x_s)
        x_s, y_s, dy_tot = x_s[idx], y_s[idx], dy_tot[idx]

        x_s, y_s, dy_tot = x_s[:-1], y_s[:-1], dy_tot[:-1]
        
        try:
            if TYPE_ANALYSE_FINALE == 'SPECTRE':
                # [Amplitude, Centre, Largeur, Offset]
                offset_guess = np.median(y_s)
                p0_final = [np.min(y_s) - offset_guess, 130.2, 0.05, offset_guess]
                func_final = lorentzian_1d
                # Bornes : Amplitude doit être négative (creux)
                lower_b = [-1e9, x_s.min(), 0.001, 0]
                upper_b = [0, x_s.max(), 2.0, 1e9]
                bounds = (lower_b, upper_b)
            else:
                # [gamma, beta_prime, N0, offset]
                n0_guess = np.max(y_s)
                # Tes estimations : offset peut être négatif (-80000)
                p0_final = [3e-4, 3e-9, 1e6, -80000]
                func_final = decay_1_and_2_body
                # Bornes : On autorise l'offset négatif et N0 jusqu'à 1e8
                lower_b = [0, 0, 0, -500000]
                upper_b = [1000, 1e-5, 1e8, 1e8]
                bounds = (lower_b, upper_b)

            # Sécurité : On s'assure que p0 est bien DANS les bornes avant de lancer
            for i in range(len(p0_final)):
                p0_final[i] = np.clip(p0_final[i], lower_b[i], upper_b[i])

            popt_s, pcov_s = curve_fit(func_final, x_s, y_s, p0=p0_final, 
                                       absolute_sigma=True, 
                                       maxfev=20000, bounds=bounds)
            
            # 1. Calcul des incertitudes
            perr_s = np.sqrt(np.diag(pcov_s))

            # 2. Extraction propre des paramètres
            gamma_opt, beta_opt, n0_opt, off_opt = popt_s
            gamma_err, beta_err, n0_err, off_err = perr_s

            # 3. Affichage textuel dans la console
            print("\n" + "="*40)
            print(f"{'PARAMÈTRE':<15} | {'VALEUR':<12} | {'ERREUR':<12}")
            print("-" * 40)
            print(f"{'gamma (1-body)':<15} | {gamma_opt:<12.2e} | {gamma_err:<12.2e} s⁻¹")
            print(f"{'beta (2-body)':<15} | {beta_opt:<12.2e} | {beta_err:<12.2e} at⁻¹·s⁻¹")
            print(f"{'N0 (Initial)':<15} | {n0_opt:<12.2e} | {n0_err:<12.2e} atomes")
            print(f"{'Offset':<15} | {off_opt:<12.2e} | {off_err:<12.2e}")
            print("="*40 + "\n")

            # 4. Ajout des résultats sur le graphique (Légende enrichie)
            texte_resultats = (
                f"N0 = {n0_opt:.2e} ± {n0_err:.1e}\n"
                f"$\gamma$ = {gamma_opt:.2e} ± {gamma_err:.1e} s$^{{-1}}$\n"
                f"$\\beta'$ = {beta_opt:.2e} ± {beta_err:.1e}"
            )
            
            # On place le texte dans un coin du graph
            plt.gca().text(0.95, 0.95, texte_resultats, transform=plt.gca().transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            l_opt = (m_atome * beta_opt * V_e) / (16 * h)
            l_opt = l_opt/a_0
            
            # Calcul de l'erreur sur l_opt (par propagation de l'erreur sur beta)
            l_opt_err = l_opt * (beta_err / beta_opt)

            # --- AFFICHAGE MIS À JOUR ---
            print(f"{'l_opt':<15} | {l_opt:<12.2e}a0 | {l_opt_err:<12.2e}")
            
            # Ajout sur le graphique
            texte_physique = f"$l_{{opt}}$ = {l_opt:.2e} ± {l_opt_err:.1e}a0"
            
            # On peut l'ajouter à la suite du texte précédent
            plt.gca().text(0.95, 0.75, texte_physique, transform=plt.gca().transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # --- AFFICHAGE ---
            plt.figure(figsize=(10, 6))
            plt.errorbar(x_s, y_s, yerr=dy_tot, fmt='o', label='Données', alpha=0.6)
            x_fine = np.linspace(x_s.min(), x_s.max(), 1000)
            plt.plot(x_fine, func_final(x_fine, *popt_s), 'r-', lw=2, label=f'Fit {TYPE_ANALYSE_FINALE}')
            plt.ylabel("Nombre d'atomes")
            plt.xlabel("Temps [ms]")
            plt.title(f"Analyse Finale : {TYPE_ANALYSE_FINALE}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('exponential_decay_data88.pdf')
            plt.show()
        
            
            
        except Exception as e:
            print(f"❌ Erreur lors du fit final : {e}")