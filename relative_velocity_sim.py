"""
Relative line-of-sight (LOS) velocity between SDO and a point on the Sun's
surface at heliographic latitude phi, as a function of time, for several
values of phi.

Replicates the vector model from the Desmos 3D graph
(https://www.desmos.com/3d/yn1xguuseh), with the damping factor `g` removed
(it was only needed to keep coordinates inside Desmos's render threshold)
and theta0 = theta1 = 0.

Geometry (Earth-centered frame, meters, seconds, radians):
    p_sdo(t)      = R_sdo * (cos(w_sdo t), sin(w_sdo t), 0)
    p_sun(phi, t) = R_sun * (-cos(phi) cos(w_sun(phi) t),
                              -cos(phi) sin(w_sun(phi) t),
                               sin(phi))
    r_sdo(t)      = p_sdo(t)
    r_sun(phi, t) = (d, 0, 0) + p_sun(phi, t)
    u(phi, t)     = r_sdo(t) - r_sun(phi, t)

Relative velocity is the analytic derivative du/dt (differentiated by hand,
since every term is a simple sinusoid) evaluated at each phi/t, then its
Euclidean norm is plotted. u_z is constant in t, so it drops out of du/dt.
"""
import numpy as np
import matplotlib.pyplot as plt

# --- constants (from the Desmos graph, g dropped / set to 1) ---
w_sdo = 2 * np.pi / (1436 * 60)                    # SDO orbital angular rate [rad/s]
R_sdo = 6571 * 1000 + 35756 * 1000                 # SDO orbital radius [m]
R_sun = 695508 * 1000                              # solar radius [m]
d = 1.495979e11                                    # Earth-Sun distance [m]
T_hours = 324                                      # simulated duration [hours]


def w_sun(phi_rad):
    """Differential rotation rate at latitude phi, converted deg/day -> rad/s."""
    deg_per_day = 14.713 - 2.396 * np.sin(phi_rad) ** 2 - 1.787 * np.sin(phi_rad) ** 4
    return (np.pi / 180 / 86400) * deg_per_day


def relative_speed(phi_deg, t):
    """|du/dt| at latitude phi_deg (degrees) over time array t (seconds)."""
    phi = np.radians(phi_deg)
    ws = w_sun(phi)
    cphi = np.cos(phi)
    alpha_deg = 26
    cos_alpha = np.cos(np.radians(alpha_deg))
    sin_alpha = np.sin(np.radians(alpha_deg))

    dux_dt = -R_sdo * w_sdo * np.sin(w_sdo * t) * cos_alpha - R_sun * cphi * ws * np.sin(ws * t)
    duy_dt = R_sdo * w_sdo * np.cos(w_sdo * t) + R_sun * cphi * ws * np.cos(ws * t)
    duz_dt = -R_sdo * w_sdo * np.sin(w_sdo * t) * sin_alpha

    return np.sqrt(dux_dt ** 2 + duy_dt ** 2 + duz_dt ** 2)


def plot_relative_velocity():
    t = np.linspace(0, T_hours * 3600, 5000)  # seconds
    t_hours = t / 3600

    phi_values_deg = [0, 15, 30, 45, 60, 75]

    fig, ax = plt.subplots(figsize=(10, 6))
    for phi_deg in phi_values_deg:
        v = relative_speed(phi_deg, t)
        ax.plot(t_hours, v, label=f"phi = {phi_deg} deg")

    ax.set_xlabel("t (hours)")
    ax.set_ylabel("|du/dt|  (m/s)")
    ax.set_title("Relative LOS velocity between SDO and a solar surface point vs time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = "/home/thomas/Projects/sun_rotacion_sim/relative_velocity_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_depth_comparison(phi=45):
    """Plot relative velocity for a given phi, comparing different depths."""
    t = np.linspace(0, T_hours * 3600, 5000)  # seconds
    t_hours = t / 3600

    depths = [0, 600*1000,1200*1000]  # depths in meters
    fig, ax = plt.subplots(figsize=(10, 6))
    for depth in depths:
        global R_sun
        R_sun_original = R_sun
        R_sun -= depth  # adjust Earth-Sun distance for depth
        v = relative_speed(phi, t)
        R_sun = R_sun_original  # reset R_sun
        v_original = relative_speed(phi, t)  # compute original for comparison
        ax.plot(t_hours, v-v_original, label=f"depth = {depth} m")

    ax.set_xlabel("t (hours)")
    ax.set_ylabel("|du/dt|  (m/s)")
    ax.set_title(f"Relative LOS velocity at phi={phi} deg vs time for different depths")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = f"/home/thomas/Projects/sun_rotacion_sim/relative_velocity_depth_comparison_phi{phi}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved depth comparison plot to {out_path}")

if __name__ == "__main__":
    plot_relative_velocity()
    plot_depth_comparison(phi=45)
