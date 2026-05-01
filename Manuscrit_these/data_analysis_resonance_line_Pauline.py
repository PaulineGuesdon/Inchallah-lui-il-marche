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
    de l'erreur par propagation d'erreur 
    - nom fichier à save """


#%%
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import re
from pathlib import Path

# =============================================================================
# --- 1. CONFIGURATION À MODIFIER ---
# =============================================================================

# Choix de l'analyse : 'SPECTRE' (Lorentz) ou 'DECAY' (Pertes 1+2 corps)
TYPE_ANALYSE_FINALE = 'SPECTRE' 

# Paramètres des images RAW et constantes physiques
MODE_FIT_IMAGE = 'gauss'
SIZE = 1024
PIXEL_SIZE_REAL = 6.5e-6 
lambd = 461e-9 
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
    return (offset + amplitude * np.exp(-((x-xo)**2)/(2*sigma_x**2) - ((y-yo)**2)/(2*sigma_y**2))).ravel()

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
    path = Path(path)
    m_at = load_picture(path.parent / f"{path.name}Atoms.dat", size)
    m_no = load_picture(path.parent / f"{path.name}NoAtoms.dat", size)
    m_dk = load_picture(path.parent / f"{path.name}Dark.dat", size)
    if m_at is None or m_no is None or m_dk is None: return None
    diff_at = np.maximum(m_at.astype(float) - m_dk, 1)
    diff_no = np.maximum(m_no.astype(float) - m_dk, 1)
    return np.clip(np.log(diff_no) - np.log(diff_at), 0, 2)

# =============================================================================
# --- 4. TRAITEMENT DES IMAGES ---
# =============================================================================

if __name__ == "__main__":
    # Override these with environment variables when moving between machines:
    #   THESIS_DATA_DIR=/path/to/data
    #   THESIS_EXCEL_PATH=/path/to/datasPA_waves.xlsx
    dir_data = Path(os.environ.get(
        "THESIS_DATA_DIR",
        "/Volumes/Elements/datas_PA/data146_spectro_5µW_10ms_MOG/data146",
    )).expanduser()
    excel_path = Path(os.environ.get(
        "THESIS_EXCEL_PATH",
        "/Volumes/Elements/datas_PA/datasPA_waves.xlsx",
    )).expanduser()

    if not dir_data.exists():
        raise FileNotFoundError(
            f"Data directory not found: {dir_data}. "
            "Set THESIS_DATA_DIR to the folder containing the *Atoms.dat files."
        )
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {excel_path}. "
            "Set THESIS_EXCEL_PATH to the workbook with the frequency/time column."
        )
    
    df_excel = pd.read_excel(excel_path)
    # Colonne 2 pour Fréquence (Spectre), Colonne 3 pour Temps (Decay)
    col_index = 3 if TYPE_ANALYSE_FINALE == 'DECAY' else 4
    valeurs_physiques = df_excel.iloc[:, col_index].values.astype(float)
    
    files_atoms = sorted(dir_data.glob("*Atoms.dat"))
    indices_sets, nb_atomes_brut, erreurs_brut = [], [], []
    func_2d = gaussian_2d_no_rot if MODE_FIT_IMAGE == 'gauss' else lorentzian_2d

    print(f"--- Analyse de {len(files_atoms)} images ---")

    for f_path in files_atoms:
        prefix = f_path.with_name(f_path.name.replace("Atoms.dat", ""))
        n_match = re.search(r'RAW_(\d+)_', f_path.name)
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
    # --- 5. ANALYSE FINALE (Calcul Brut / Plot Normalisé) ---
    # =========================================================================

    if len(nb_atomes_brut) > 5:
        x_s = np.array(indices_sets)
        y_s = np.array(nb_atomes_brut) # On garde les vraies valeurs pour le fit
        dy_tot = np.sqrt(np.array(erreurs_brut)**2 + (0.15 * y_s)**2)

        # Nettoyage et Tri
        mask = np.isfinite(x_s) & np.isfinite(y_s)
        x_s, y_s, dy_tot = x_s[mask], y_s[mask], dy_tot[mask]
        idx = np.argsort(x_s)
        x_s, y_s, dy_tot = x_s[idx], y_s[idx], dy_tot[idx]

        try:
            # --- 1. CONFIGURATION DU FIT (VALEURS BRUTES) ---
            if TYPE_ANALYSE_FINALE == 'SPECTRE':
                offset_guess = np.median(y_s)
                # [Amplitude, Centre, Largeur, Offset]
                p0_final = [np.min(y_s) - offset_guess, 232.2, 0.05, offset_guess]
                func_final = lorentzian_1d
                bounds = ([-1e9, x_s.min(), 0.0001, 0], [0, x_s.max(), 5.0, 1e9])
            else:
                # [gamma, beta_prime, N0, offset]
                n0_guess = np.max(y_s)
                p0_final = [3e-4, 3e-9, n0_guess, 0.0] 
                func_final = decay_1_and_2_body
                bounds = ([0, 0, 0, -1e6], [1000, 1e-5, 1e9, 1e9])

            # --- 2. RÉALISATION DU FIT ---
            popt_s, pcov_s = curve_fit(func_final, x_s, y_s, p0=p0_final, 
                                       sigma=dy_tot, absolute_sigma=True, 
                                       maxfev=20000, bounds=bounds)
            perr_s = np.sqrt(np.diag(pcov_s))

            # --- 3. GESTION DE L'AFFICHAGE SELON LE MODE ---
            print("\n" + "="*50)
            print(f"RÉSULTATS DU FIT : {TYPE_ANALYSE_FINALE}")
            print("-" * 50)
            print(f"{'PARAMÈTRE':<18} | {'VALEUR':<12} | {'ERREUR':<12}")
            print("-" * 50)

            if TYPE_ANALYSE_FINALE == 'SPECTRE':
                amp, center, width, off = popt_s
                amp_e, cent_e, wid_e, off_e = perr_s
                
                print(f"{'Amplitude (Pertes)':<18} | {amp:<12.2e} | {amp_e:<12.2e}")
                print(f"{'Centre (MHz)':<18} | {center:<12.4f} | {cent_e:<12.4f}")
                print(f"{'Largeur FWHM':<18} | {width:<12.4f} | {wid_e:<12.4f}")
                print(f"{'Offset (Fond)':<18} | {off:<12.2e} | {off_e:<12.2e}")
                
                res_text = (f"Centre: {center:.3f} MHz\n"
                            f"Largeur: {width:.3f} MHz\n"
                            f"Pertes: {amp:.1e}")

            else: # MODE DECAY
                gamma_opt, beta_opt, n0_opt, off_opt = popt_s
                gamma_err, beta_err, n0_err, off_err = perr_s
                
                l_opt = (m_atome * beta_opt * V_e) / (16 * h * a_0)
                l_opt_err = l_opt * (beta_err / beta_opt) if beta_opt != 0 else 0

                print(f"{'gamma (1-body)':<18} | {gamma_opt:<12.2e} | {gamma_err:<12.2e} s⁻¹")
                print(f"{'beta (2-body)':<18} | {beta_opt:<12.2e} | {beta_err:<12.2e}")
                print(f"{'N0 (Initial)':<18} | {n0_opt:<12.2e} | {n0_err:<12.2e}")
                print(f"{'Offset':<18} | {off_opt:<12.2e} | {off_err:<12.2e}")
                print("-" * 50)
                print(f"{'l_opt':<18} | {l_opt:<12.2e} a0 | {l_opt_err:<12.2e}")
                
                res_text = (f"N0 = {n0_opt:.2e}\n$\gamma$ = {gamma_opt:.2e}\n"
                            f"$\\beta'$ = {beta_opt:.2e}\n$l_{{opt}}$ = {l_opt:.1f} $a_0$")

            print("="*50 + "\n")

            # --- 4. GRAPHIQUE NORMALISÉ ---
            norm = np.median(y_s)
            cm=2.54
            plt.figure(figsize=(20/cm, 12/cm))
            plt.errorbar(x_s, y_s / norm, yerr=dy_tot / norm, fmt='ko', label='Données (Norm)', alpha=0.6)
            
            x_fine = np.linspace(x_s.min(), x_s.max(), 500)
            plt.plot(x_fine, func_final(x_fine, *popt_s) / norm, 'r-', lw=2, label=f'Fit {TYPE_ANALYSE_FINALE}')
            
            plt.gca().text(0.95, 0.95, res_text, transform=plt.gca().transAxes, 
                           va='top', ha='right', fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            
            plt.ylabel("Population relative ($N / N_{max}$)")
            plt.xlabel("Fréquence [MHz]" if TYPE_ANALYSE_FINALE == 'SPECTRE' else "Temps [ms]")
            plt.title(f"Analyse {TYPE_ANALYSE_FINALE} - {os.path.basename(dir_data)}")
            plt.grid(True, alpha=0.2)
            plt.legend()
            plt.tight_layout()
            plt.savefig('fit_SPECTRE_data.pdf')
            plt.show()

        except Exception as e:
            print(f"❌ Erreur de fit : {e}")
            import traceback
            traceback.print_exc()
