"""Plot the left panels of composite_375MHz from fit375.py output (needs matplotlib >= 3.2)."""
import sys
import numpy as np, matplotlib, matplotlib.patches
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager

NPZ, OUT = sys.argv[1], sys.argv[2]
COURIER = "/System/Library/Fonts/Supplemental/Courier New.ttf"
r = np.load(NPZ)
x, y, ey, keep, popt = r["x"], r["y"], r["ey"], r["keep"], r["popt"]
# total error bar: 2D-fit uncertainty (covariance diagonal) + shot-to-shot RMS fluctuation,
# estimated on the first 100 points of the scan (off resonance, outliers excluded), in quadrature
edge = np.argsort(x)[:100]
edge = edge[keep[edge]]
rms_rel = np.std(y[edge]) / np.mean(y[edge])
print("RMS on the first 100 points:", rms_rel, "(", len(edge), "points used )")
ey = y * np.sqrt((ey / y)**2 + rms_rel**2)

def lorentzian_1d(x, amplitude, x0, gamma, offset):
    return offset + amplitude / (1 + ((x - x0) / (gamma / 2))**2)

font_manager.fontManager.addfont(COURIER)
plt.rcParams.update({"font.family": "Courier New", "font.size": 10, "pdf.fonttype": 42})
W, H = 862.5260620117188, 638.0543823242188
fig = plt.figure(figsize=(W / 72, H / 72)); fig.patch.set_alpha(0)
fig.patches.append(matplotlib.patches.Rectangle((0, 0), 440.0 / W, 1, transform=fig.transFigure, color="white", zorder=-10))
def ax_at(x0, y0, x1, y1):
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H])
ax1 = ax_at(38.2625, 7.2000, 378.8923, 261.9069)
ax2 = ax_at(38.2625, 343.4131, 378.8923, 598.1200)

eb = dict(fmt="o", ms=3.2, mfc="purple", mec="black", mew=0.3, ecolor=(0.53, 0.81, 0.92, 0.6), elinewidth=0.8, capsize=1.5, lw=0.6)
ax1.errorbar(x[keep], y[keep], yerr=ey[keep], **eb)
# points exclus par le clipping 3 sigma : retirés de la figure
xp = np.linspace(x.min(), x.max(), 600)
ax1.plot(xp, lorentzian_1d(xp, *popt), color="#00aa44", lw=1.8, zorder=5)

ax2.errorbar(x[keep], r["T"][keep], yerr=r["eT"][keep], **eb)
x0, hw = popt[1], abs(popt[2]) / 2   # half width at half maximum
ax2.axvspan(x0 - hw, x0 + hw, color="#00aa44", alpha=0.08, lw=0)
ax2.axvline(x0, color="#00aa44", ls="--", lw=1.2)
for a, lab in ((ax1, "Atom number ratio"), (ax2, "Temperature [µK]")):
    a.set_xlabel("Laser detuning |Δ| [MHz]")
    a.set_xlim(340, 410)
    # y-axis title on the right, reading top to bottom, starting at the top of the frame
    a.text(1.015, 1.0, lab, transform=a.transAxes, rotation=270, rotation_mode="anchor",
           ha="left", va="bottom", fontsize=11)
fig.savefig(OUT)
