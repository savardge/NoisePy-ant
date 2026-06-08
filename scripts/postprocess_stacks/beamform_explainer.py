"""
Simple graphic to explain beamforming slowness interpretation.
Two panels: physical space (left) and slowness space (right).
"""
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# =============================================================================
# LEFT PANEL: Physical space (km)
# =============================================================================
ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)
ax1.set_aspect('equal')
ax1.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax1.axvline(0, color='gray', lw=0.5, alpha=0.5)

# Direction labels
ax1.text(0, 47, 'N', fontsize=14, ha='center', fontweight='bold')
ax1.text(0, -49, 'S', fontsize=14, ha='center', fontweight='bold')
ax1.text(47, 0, 'E', fontsize=14, ha='center', va='center', fontweight='bold')
ax1.text(-49, 0, 'W', fontsize=14, ha='center', va='center', fontweight='bold')

# Draw stations
stations = [(-20, 10), (15, -10), (-5, -20), (10, 15), (-15, -5)]
for i, sta in enumerate(stations):
    ax1.plot(*sta, 'k^', markersize=12)
ax1.text(0, -35, 'Station array', fontsize=11, ha='center')

# Source in NE (outside the array)
source_pos = (70, 70)  # conceptually outside plot
ax1.annotate('', xy=(40, 40), xytext=(15, 15),
            arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax1.text(35, 30, 'Source\n(in NE)', fontsize=12, color='green', fontweight='bold', ha='center')

# Wavefronts coming from NE, traveling to SW
for offset in [-10, 5, 20, 35]:
    x_wf = np.array([-30, 30]) + offset
    y_wf = np.array([30, -30]) + offset
    ax1.plot(x_wf, y_wf, 'b-', lw=1.5, alpha=0.5)

# Arrow showing wave propagation direction
ax1.annotate('', xy=(-25, -25), xytext=(5, 5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=3))
ax1.text(-20, 0, 'Waves traveling\ntowards SW', fontsize=11, color='blue', ha='center')

ax1.set_xlabel('East-West [km]', fontsize=12)
ax1.set_ylabel('North-South [km]', fontsize=12)
ax1.set_title('Physical Space\n(stations and wave propagation)', fontsize=13)

# =============================================================================
# RIGHT PANEL: Slowness space (s/km)
# =============================================================================
ax2.set_xlim(-0.8, 0.8)
ax2.set_ylim(-0.8, 0.8)
ax2.set_aspect('equal')
ax2.axhline(0, color='gray', lw=1)
ax2.axvline(0, color='gray', lw=1)

# Direction labels
ax2.text(0, 0.72, 'N', fontsize=14, ha='center', fontweight='bold')
ax2.text(0, -0.76, 'S', fontsize=14, ha='center', fontweight='bold')
ax2.text(0.72, 0, 'E', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(-0.76, 0, 'W', fontsize=14, ha='center', va='center', fontweight='bold')

# Draw slowness circles
for radius in [0.2, 0.4, 0.6]:
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(radius*np.cos(theta), radius*np.sin(theta), 'k--', lw=0.5, alpha=0.3)
    ax2.text(radius+0.02, 0.02, f'{radius}', fontsize=8, alpha=0.5)

# Beam maximum in SW quadrant
beam_center = (-0.35, -0.35)
circle = plt.Circle(beam_center, 0.12, color='red', alpha=0.7)
ax2.add_patch(circle)
ax2.text(beam_center[0], beam_center[1], 'Beam\nmax', ha='center', va='center',
         fontsize=10, fontweight='bold', color='white')

# Arrow from origin to beam max (slowness vector direction)
ax2.annotate('', xy=beam_center, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax2.text(-0.1, -0.2, '(ux, uy)', fontsize=11, color='blue')

ax2.set_xlabel('Slowness E-W (ux) [s/km]', fontsize=12)
ax2.set_ylabel('Slowness N-S (uy) [s/km]', fontsize=12)
ax2.set_title('Slowness Space\n(beamforming result)', fontsize=13)

# Summary box
textstr = 'Interpretation:\n• Beam max at (ux, uy) in SW\n• Waves travel TOWARDS SW\n• Sources are in the NE'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax2.text(0.05, -0.55, textstr, fontsize=10, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/Users/genevievesavard/Codes/NoisePy-ant/scripts/postprocess_stacks/beamform_explainer.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to beamform_explainer.png")
