import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. CONFIGURATION ET CONSTANTES --- FILE_PATH = 'D:/DATA/2026_03_06/Labview/RAW_0_Atoms.dat'
SIZE = 1024
SIZE_CROP = 200  # Taille de la zone à analyser (ex: 200x200 pixels)

# Paramètres initiaux [amplitude, xo, yo, sigma_x, sigma_y, offset]
P0 = [0.3, 80, 175, 20, 20, 0]

# --- 2. FONCTIONS ---
def gaussian_2d_no_rot(coords, amplitude, xo, yo, sigma_x, sigma_y,
offset):
     """Modèle gaussien 2D sans rotation pour le fit."""
     x, y = coords
     g = offset + amplitude * np.exp(-(2*(x - xo)**2) / (sigma_x**2) - (2*(y - yo)**2) / (sigma_y**2))
     return g.ravel()


def load_picture(path, size):
     """Charge et redimensionne les données binaires."""
     try:
         # Lecture en Big-Endian unsigned 16-bit ('>u2')
         data = np.fromfile(path, dtype=np.dtype('>u2'))

         # On ignore les 4 premiers éléments (header potentiel)
         data = data[4:]
         return data[:size*size].reshape((size, size))
     except Exception as e:
         print(f"Erreur de chargement : {e}")
         return None
     
def plout_picture(matrice):
    if matrice is not None:
        # Préparation des coordonnées
        matrice_crop = matrice[400:600, 400:600] 
        y_grid, x_grid = np.indices(matrice_crop.shape)
        
        # Création de la figure
        fig, axs = plt.subplots(1, 1, figsize=(18, 5))
        vmin, vmax = np.min(matrice_crop), np.max(matrice_crop)

        # Affichage
        im0 = axs.imshow(matrice_crop, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axs.set_title('Données Originales')
        
        # On transforme axs en [axs] pour que le zip fonctionne
        for ax, im in zip([axs], [im0]):
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        plt.show() # Optionnel selon ton environnement
           
def load_data(path, size):
        """Charge les données ede trois fichiers et renvoie la matrice finale après soustraction."""
        path_atoms=path+"Atoms.dat"
        path_noatoms=path+"NoAtoms.dat"
        path_dark=path+"Dark.dat"
        matrice_atoms = load_picture(path_atoms, size)
        matrice_noatoms = load_picture(path_noatoms, size)
        matrice_dark = load_picture(path_dark, size)
        if matrice_atoms is None or matrice_noatoms is None or matrice_dark is None:
            print("Erreur : Impossible de charger toutes les matrices nécessaires.")
            return None

       # On calcule les différences d'abord
        diff_atoms = matrice_atoms - matrice_dark
        diff_noatoms = matrice_noatoms - matrice_dark

        # On applique le log seulement si la valeur est strictement positive (> 0)
        matrice = np.where(diff_atoms > 0, np.log(diff_atoms), 0) - np.where(diff_noatoms > 0, np.log(diff_noatoms), 0)
        matrice_filter = np.clip(-matrice, 0, 1)
        return matrice_filter
# --- 3. TRAITEMENT ---

def fit_matrice(matrice, P0, plot_fit=False):
     """Effectue le fit gaussien 2D et affiche les résultats avec incertitudes."""
     if matrice is not None:
        """Préparation des coordonnées"""
        matrice_crop = matrice[300:500, 400:600] # On se concentre sur une zone de 200x200 pixels autour du centre
        y_grid, x_grid = np.indices(matrice_crop.shape)
        coords = (x_grid, y_grid)

        try:
            # Fit avec récupération de la matrice de covariance (pcov)
            popt, pcov = curve_fit(gaussian_2d_no_rot, coords, matrice_crop.ravel(), p0=P0)

            # Calcul des incertitudes (1 cart-type)
            perr = np.sqrt(np.diag(pcov))
            if plot_fit:
                names = ["Amplitude", "Centre X", "Centre Y", "Sigma X", "Sigma Y", "Offset"]
                print("--- RÉSULTATS DU FIT (avec erreurs) ---")
                for i, name in enumerate(names):
                    print(f"{name:10}: {popt[i]:.3f} ± {perr[i]:.3f}")
            P0 = popt  # Mise à jour de P0 pour les itérations suivantes
            
            if plot_fit:
                # Génération du modèle et des résidus
                image_model = gaussian_2d_no_rot(coords, *popt).reshape(SIZE_CROP,SIZE_CROP)
                residus = matrice_crop - image_model

                # --- 4. VISUALISATION 2D ---
                fig, axs = plt.subplots(1, 3, figsize=(18, 5))
                vmin, vmax = np.min(matrice_crop), np.max(matrice_crop)

                # Original
                im0 = axs[0].imshow(matrice_crop, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
                axs[0].set_title('Données Originales')

                # Modèle
                im1 = axs[1].imshow(image_model, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
                axs[1].set_title(f'Modèle\nX0={popt[1]:.1f}±{perr[1]:.1f}')

                # Résidus
                abs_max_res = np.max(np.abs(residus))
                im2 = axs[2].imshow(residus, cmap='seismic', vmin=-abs_max_res, vmax=abs_max_res, origin='lower')
                axs[2].set_title('Résidus (Data - Fit)')

                for ax, im in zip(axs, [im0, im1, im2]):
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()

                # --- 5. COUPES AVEC BARRES D'ERREUR ---
                ix, iy = int(round(popt[1])), int(round(popt[2]))

                # Estimation du bruit local pour les barres d'erreur (écart-type des résidus)
                bruit_sigma = np.std(residus)

                fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(15, 5))
                # Coupe Horizontale
                # On affiche un point sur 10 pour ne pas surcharger le graphique avec les barres d'erreur
                step = 1
                ax_h.errorbar(x_grid[0, ::step], matrice_crop[iy, ::step], yerr=bruit_sigma,
                            fmt='b.', alpha=0.4, label='Données + Bruit')
                ax_h.plot(x_grid[0, :], image_model[iy, :], 'r-', lw=2, label='Fit Gaussien')
                ax_h.set_title("Coupe Horizontale (Axe X)")
                ax_h.legend()

                # Coupe Verticale
                ax_v.errorbar(y_grid[::step, 0], matrice_crop[::step, ix], yerr=bruit_sigma,
                            fmt='b.', alpha=0.4, label='Données + Bruit')
                ax_v.plot(y_grid[:, 0], image_model[:, ix], 'r-', lw=2, label='Fit Gaussien')
                ax_v.set_title("Coupe Verticale (Axe Y)")
                ax_v.legend()
                plt.tight_layout()
                plt.show()
            return popt, perr
        except RuntimeError:
            print("Échec du fit : Vérifiez vos paramètres initiaux (P0).")
            return None, None
     else:
        print("Erreur : Matrice non chargée.")
        return None, None


if __name__ == "__main__":
    # --- 6. EXÉCUTION PRINCIPALE ---
    path = 'C:/Users/pauline.guesdon/Pictures/Figures  thèses/datas PA/data141_l_opt2.1µW/data141/RAW_2_'
    # path = 'C:/Users/alban.meyroneinc/Desktop/data/data160/RAW_1_'  # Assurez-vous que ce chemin est correct
    matrice = load_data(path, SIZE)
    plout_picture(matrice)
    #fit_matrice(matrice, P0, plot_fit=True)
    
    
    
    