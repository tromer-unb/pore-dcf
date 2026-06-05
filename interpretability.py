# ============================================================
# PORE-DCF INTERPRETABILITY
# Several particles per radius, with editable parameters at the top
# ============================================================
#
# Input:
#   - green/black segmented image
#   - pore_dcf_descriptor.csv
#
# Output:
#   - interpretability figures
#   - trajectories by radius
#   - all trajectories combined
#   - text summary
#
# Green = pore
# Black = matrix
#
# ============================================================


# ============================================================
# 0. LIST OF PARAMETERS TO EDIT
# ============================================================

# ----------------------------
# Input files
# ----------------------------

IMAGE_PATH = "../poros/poros/patch_y7600_x53200_c0_mask.png"
CSV_PATH = "pore_dcf_descriptor.csv"

# ----------------------------
# Output directory
# ----------------------------

OUTPUT_DIR = "pore_dcf_interpretability_outputs"

# ----------------------------
# Physical scale
# ----------------------------
# From your metadata:
# 4.4e-7 m = 0.44 micrometer per pixel

PIXEL_SIZE_UM = 0.44

# ----------------------------
# Radii to be used
# ----------------------------
# These names must exist in the CSV:
# r5_ball_radius_px
# r25_ball_radius_px
# r50_ball_radius_px
# r75_ball_radius_px
# r90_ball_radius_px

RADIUS_PREFIXES = ["r1", "r5", "r10", "r25", "r50", "r75", "r90", "r95"]
# You can also use, for example:
# RADIUS_PREFIXES = ["r25", "r50", "r75"]
# or:
# RADIUS_PREFIXES = ["r50"]

# ----------------------------
# Number of particles per radius
# ----------------------------
# This is the main parameter for making the figure denser.
# Teste:
#   20  = fast
#   100 = good
#   300 = dense
#   500 = very dense and slower

TRAJECTORIES_PER_RADIUS = 100

# ----------------------------
# Number of steps for each particle
# ----------------------------
# Larger values produce longer trajectories.
# Teste:
#   500  = short
#   1500 = good
#   3000 = long
#   5000 = very long

NSTEPS_VISUAL = 1500

# ----------------------------
# Particle step size
# ----------------------------
# None uses automatic mode:
# step = max(1 px, 0.25 * particle_radius_px)
#
# To control it manually, use for example:
# FIXED_STEP_PX = 1.0
# FIXED_STEP_PX = 2.0

FIXED_STEP_PX = None

# ----------------------------
# Trajectory visualization
# ----------------------------

TRAJECTORY_ALPHA = 0.22
TRAJECTORY_LINEWIDTH = 0.35

# ----------------------------
# Collision-point visualization
# ----------------------------

PLOT_COLLISION_POINTS = True
COLLISION_POINT_SIZE = 1.0
COLLISION_ALPHA = 0.20

# ----------------------------
# Figures to generate
# ----------------------------

SAVE_BASIC_INTERPRETABILITY = True
SAVE_TRAJECTORIES_BY_RADIUS = True
SAVE_ALL_TRAJECTORIES_TOGETHER = True
SAVE_ALL_TRAJECTORIES_LINES_ONLY = True
SAVE_TEXT_SUMMARY = True

# ----------------------------
# FFT modes shown in the plot
# ----------------------------

FFT_MODES = list(range(1, 13))

# ----------------------------
# Random seed
# ----------------------------

SEED_BASE = 12345

# ----------------------------
# Figure DPI
# ----------------------------

FIGURE_DPI = 300


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates


# ============================================================
# 2. PREPARATION
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. MASK READING
# ============================================================

def load_green_mask(image_path):
    """
    Reads the RGB image and returns:
        mask = True in green pores
        rgb  = original image
    """

    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)

    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    # Criterion for strong green.
    # Adjust this if your segmentation uses another shade of green.
    mask = (g > 80) & (g > 1.5 * r) & (g > 1.5 * b)

    return mask, rgb


# ============================================================
# 4. INTERPOLATION AND ACCESSIBLE REGION
# ============================================================

def interp_field(field, pos):
    """
    Interpolates a 2D field at a subpixel position.
    pos = [y, x]
    """

    y, x = pos

    return map_coordinates(
        field,
        [[y], [x]],
        order=1,
        mode="nearest"
    )[0]


def is_inside_accessible_region(distance_map, pos, ball_radius_px):
    """
    The particle can only be where:
        distance_map >= ball_radius_px

    In other words, the particle center can only occupy pixels whose distance
    to the pore wall is greater than or equal to the particle radius.
    """

    y, x = pos
    height, width = distance_map.shape

    if x < 0 or x >= width or y < 0 or y >= height:
        return False

    d = interp_field(distance_map, pos)

    return d >= ball_radius_px


# ============================================================
# 5. SIMULATION OF ONE TRAJECTORY
# ============================================================

def simulate_trajectory_for_visualization(
    distance_map,
    ball_radius_px,
    nsteps=1500,
    step_px=None,
    seed=1
):
    """
    Simulates a particle inside the pores for visualization.

    Returns:
        trajectory_points: trajectory points
        collision_points: collision points
    """

    rng = np.random.default_rng(seed)

    accessible = distance_map >= ball_radius_px
    ys, xs = np.where(accessible)

    if len(xs) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    if step_px is None:
        step_px = max(1.0, 0.25 * ball_radius_px)

    # Random initial position inside the accessible region
    idx = rng.integers(0, len(xs))
    pos = np.array([ys[idx], xs[idx]], dtype=float)

    # Random initial direction
    angle = rng.uniform(0, 2 * np.pi)
    vel = np.array([np.sin(angle), np.cos(angle)], dtype=float)
    vel = vel / np.linalg.norm(vel)

    # Gradient of the distance map.
    # It points toward more internal pore regions.
    grad_y, grad_x = np.gradient(distance_map.astype(float))

    trajectory_points = [pos.copy()]
    collision_points = []

    for _ in range(nsteps):

        new_pos = pos + vel * step_px

        # Free motion
        if is_inside_accessible_region(distance_map, new_pos, ball_radius_px):
            pos = new_pos
            trajectory_points.append(pos.copy())
            continue

        # Collision: find impact point by bisection
        low = pos.copy()
        high = new_pos.copy()

        for _ in range(15):
            mid = 0.5 * (low + high)

            if is_inside_accessible_region(distance_map, mid, ball_radius_px):
                low = mid
            else:
                high = mid

        hit = low.copy()
        collision_points.append(hit.copy())

        # Local wall normal
        ny = interp_field(grad_y, hit)
        nx = interp_field(grad_x, hit)

        normal = np.array([ny, nx], dtype=float)
        norm = np.linalg.norm(normal)

        if norm < 1e-12:
            random_angle = rng.uniform(0, 2 * np.pi)
            normal = np.array([np.sin(random_angle), np.cos(random_angle)])
        else:
            normal = normal / norm

        # Elastic reflection:
        # v' = v - 2(v.n)n
        vel = vel - 2.0 * np.dot(vel, normal) * normal
        vel = vel / np.linalg.norm(vel)

        # Small displacement to avoid getting stuck at the boundary
        pos_candidate = hit + vel * 0.5

        if is_inside_accessible_region(distance_map, pos_candidate, ball_radius_px):
            pos = pos_candidate
        else:
            pos_candidate = hit + normal * 0.5

            if is_inside_accessible_region(distance_map, pos_candidate, ball_radius_px):
                pos = pos_candidate
            else:
                # If stuck, restart at another accessible point
                idx = rng.integers(0, len(xs))
                pos = np.array([ys[idx], xs[idx]], dtype=float)

                angle = rng.uniform(0, 2 * np.pi)
                vel = np.array([np.sin(angle), np.cos(angle)], dtype=float)
                vel = vel / np.linalg.norm(vel)

        trajectory_points.append(pos.copy())

    return np.array(trajectory_points), np.array(collision_points)


# ============================================================
# 6. BASIC INTERPRETABILITY FIGURES
# ============================================================

def save_mask_plot(mask):
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="gray")
    plt.title("Máscara binária: poro = verde / matriz = preto")
    plt.axis("off")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "interpretability_mask.png")
    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


def save_distance_map_plot(distance_map):
    distance_um = distance_map * PIXEL_SIZE_UM

    plt.figure(figsize=(8, 8))
    plt.imshow(distance_um)
    plt.colorbar(label="Distância até a parede do poro (µm)")
    plt.title("Mapa de distância interna dos poros")
    plt.axis("off")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "interpretability_distance_map.png")
    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


def save_radius_distribution_plot(distance_map, descriptor):
    valid = distance_map[distance_map > 0] * PIXEL_SIZE_UM

    plt.figure(figsize=(8, 5))
    plt.hist(valid, bins=100)
    plt.xlabel("Raio máximo local permitido (µm)")
    plt.ylabel("Frequência de pixels")
    plt.title("Distribuição local de tamanhos possíveis de bolinha")

    ymax = plt.ylim()[1]

    for prefix in RADIUS_PREFIXES:
        col = f"{prefix}_ball_radius_um"

        if col in descriptor:
            value = descriptor[col]
            plt.axvline(value, linestyle="--", linewidth=1.2)
            plt.text(value, ymax * 0.85, prefix, rotation=90)

    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "interpretability_radius_distribution.png")
    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


def save_accessible_fraction_plot(descriptor):
    xs = []
    ys = []

    for prefix in RADIUS_PREFIXES:
        rcol = f"{prefix}_ball_radius_um"
        acol = f"{prefix}_accessible_fraction"

        if rcol in descriptor and acol in descriptor:
            xs.append(descriptor[rcol])
            ys.append(descriptor[acol])

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Raio da bolinha (µm)")
    plt.ylabel("Fração da imagem acessível")
    plt.title("Acessibilidade dos poros em função do tamanho da bolinha")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "interpretability_accessible_fraction.png")
    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


def save_mean_free_path_plot(descriptor):
    xs = []
    ys = []
    yerr = []

    for prefix in RADIUS_PREFIXES:
        rcol = f"{prefix}_ball_radius_um"
        mcol = f"{prefix}_mean_free_path_um"
        scol = f"{prefix}_std_free_path_um"

        if rcol in descriptor and mcol in descriptor:
            xs.append(descriptor[rcol])
            ys.append(descriptor[mcol])
            yerr.append(descriptor.get(scol, 0.0))

    plt.figure(figsize=(7, 5))
    plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4)
    plt.xlabel("Raio da bolinha (µm)")
    plt.ylabel("Caminho livre médio (µm)")
    plt.title("Caminho livre médio por tamanho de bolinha")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "interpretability_mean_free_path.png")
    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


def save_entropy_plot(descriptor):
    xs = []
    ys = []

    for prefix in RADIUS_PREFIXES:
        rcol = f"{prefix}_ball_radius_um"
        ecol = f"{prefix}_angular_entropy_norm"

        if rcol in descriptor and ecol in descriptor:
            xs.append(descriptor[rcol])
            ys.append(descriptor[ecol])

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Raio da bolinha (µm)")
    plt.ylabel("Entropia angular normalizada")
    plt.title("Desordem angular das colisões")
    plt.ylim(0, 1.05)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "interpretability_entropy.png")
    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


def save_fft_modes_plot(descriptor):
    plt.figure(figsize=(10, 6))

    for prefix in RADIUS_PREFIXES:
        values = []

        for mode in FFT_MODES:
            col = f"{prefix}_fft_mode_{mode}"
            values.append(descriptor.get(col, 0.0))

        plt.plot(FFT_MODES, values, marker="o", label=prefix)

    plt.xlabel("Modo angular FFT")
    plt.ylabel("Intensidade normalizada")
    plt.title("Assinatura angular das colisões por raio")
    plt.xticks(FFT_MODES)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "interpretability_fft_modes.png")
    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


# ============================================================
# 7. TRAJECTORIES: SEVERAL PARTICLES PER RADIUS
# ============================================================

def save_trajectories_one_radius(mask, distance_map, descriptor, prefix):
    """
    Generates a separate figure for one radius.
    Example:
        r5  -> several small particles
        r90 -> several large particles
    """

    rcol_px = f"{prefix}_ball_radius_px"
    rcol_um = f"{prefix}_ball_radius_um"

    if rcol_px not in descriptor:
        print(f"Aviso: coluna {rcol_px} não encontrada no CSV.")
        return

    radius_px = float(descriptor[rcol_px])
    radius_um = float(descriptor.get(rcol_um, radius_px * PIXEL_SIZE_UM))

    plt.figure(figsize=(12, 12))
    plt.imshow(mask, cmap="gray")

    total_collisions = 0
    total_points = 0

    for j in range(TRAJECTORIES_PER_RADIUS):

        traj, collisions = simulate_trajectory_for_visualization(
            distance_map=distance_map,
            ball_radius_px=radius_px,
            nsteps=NSTEPS_VISUAL,
            step_px=FIXED_STEP_PX,
            seed=SEED_BASE + 100000 + j
        )

        if len(traj) > 0:
            plt.plot(
                traj[:, 1],
                traj[:, 0],
                linewidth=TRAJECTORY_LINEWIDTH,
                alpha=TRAJECTORY_ALPHA
            )
            total_points += len(traj)

        if PLOT_COLLISION_POINTS and len(collisions) > 0:
            plt.scatter(
                collisions[:, 1],
                collisions[:, 0],
                s=COLLISION_POINT_SIZE,
                alpha=COLLISION_ALPHA
            )
            total_collisions += len(collisions)

    title = (
        f"Trajetórias Pore-DCF - {prefix}\n"
        f"raio = {radius_um:.3f} µm | "
        f"bolinhas = {TRAJECTORIES_PER_RADIUS} | "
        f"passos/bolinha = {NSTEPS_VISUAL} | "
        f"colisões visuais = {total_collisions}"
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    out = os.path.join(
        OUTPUT_DIR,
        f"interpretability_trajectories_{prefix}.png"
    )

    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


def save_trajectories_all_radii(mask, distance_map, descriptor):
    """
    Generates a single figure containing several particles for all radii.
    """

    plt.figure(figsize=(14, 14))
    plt.imshow(mask, cmap="gray")

    total_collisions = 0

    for i, prefix in enumerate(RADIUS_PREFIXES):

        rcol_px = f"{prefix}_ball_radius_px"
        rcol_um = f"{prefix}_ball_radius_um"

        if rcol_px not in descriptor:
            print(f"Aviso: coluna {rcol_px} não encontrada no CSV.")
            continue

        radius_px = float(descriptor[rcol_px])
        radius_um = float(descriptor.get(rcol_um, radius_px * PIXEL_SIZE_UM))

        for j in range(TRAJECTORIES_PER_RADIUS):

            traj, collisions = simulate_trajectory_for_visualization(
                distance_map=distance_map,
                ball_radius_px=radius_px,
                nsteps=NSTEPS_VISUAL,
                step_px=FIXED_STEP_PX,
                seed=SEED_BASE + 200000 + i * 10000 + j
            )

            if len(traj) > 0:
                plt.plot(
                    traj[:, 1],
                    traj[:, 0],
                    linewidth=TRAJECTORY_LINEWIDTH,
                    alpha=TRAJECTORY_ALPHA,
                    label=f"{prefix} ({radius_um:.2f} µm)" if j == 0 else None
                )

            if PLOT_COLLISION_POINTS and len(collisions) > 0:
                plt.scatter(
                    collisions[:, 1],
                    collisions[:, 0],
                    s=COLLISION_POINT_SIZE,
                    alpha=COLLISION_ALPHA
                )
                total_collisions += len(collisions)

    title = (
        "Todas as trajetórias Pore-DCF\n"
        f"{TRAJECTORIES_PER_RADIUS} bolinhas por raio | "
        f"passos/bolinha = {NSTEPS_VISUAL} | "
        f"colisões visuais = {total_collisions}"
    )

    plt.title(title)
    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()

    out = os.path.join(
        OUTPUT_DIR,
        "interpretability_trajectories_all_radii.png"
    )

    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


def save_trajectories_all_radii_lines_only(mask, distance_map, descriptor):
    """
    Clean version:
    plots only the trajectory lines, without collision points.
    """

    plt.figure(figsize=(14, 14))
    plt.imshow(mask, cmap="gray")

    for i, prefix in enumerate(RADIUS_PREFIXES):

        rcol_px = f"{prefix}_ball_radius_px"
        rcol_um = f"{prefix}_ball_radius_um"

        if rcol_px not in descriptor:
            print(f"Aviso: coluna {rcol_px} não encontrada no CSV.")
            continue

        radius_px = float(descriptor[rcol_px])
        radius_um = float(descriptor.get(rcol_um, radius_px * PIXEL_SIZE_UM))

        for j in range(TRAJECTORIES_PER_RADIUS):

            traj, collisions = simulate_trajectory_for_visualization(
                distance_map=distance_map,
                ball_radius_px=radius_px,
                nsteps=NSTEPS_VISUAL,
                step_px=FIXED_STEP_PX,
                seed=SEED_BASE + 300000 + i * 10000 + j
            )

            if len(traj) > 0:
                plt.plot(
                    traj[:, 1],
                    traj[:, 0],
                    linewidth=TRAJECTORY_LINEWIDTH,
                    alpha=TRAJECTORY_ALPHA,
                    label=f"{prefix} ({radius_um:.2f} µm)" if j == 0 else None
                )

    title = (
        "Todas as trajetórias Pore-DCF - linhas apenas\n"
        f"{TRAJECTORIES_PER_RADIUS} bolinhas por raio | "
        f"passos/bolinha = {NSTEPS_VISUAL}"
    )

    plt.title(title)
    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()

    out = os.path.join(
        OUTPUT_DIR,
        "interpretability_trajectories_all_radii_lines_only.png"
    )

    plt.savefig(out, dpi=FIGURE_DPI)
    plt.close()

    print(f"Salvo: {out}")


# ============================================================
# 8. AUTOMATIC TEXT SUMMARY
# ============================================================

def generate_text_summary(descriptor):
    lines = []

    lines.append("Resumo interpretativo do Pore-DCF")
    lines.append("=" * 45)
    lines.append("")

    porosity = descriptor.get("porosity_2d", np.nan)
    n_pores = descriptor.get("n_pores", np.nan)
    max_r = descriptor.get("max_inscribed_radius_um", np.nan)
    mean_area = descriptor.get("mean_pore_area_um2", np.nan)
    median_area = descriptor.get("median_pore_area_um2", np.nan)

    lines.append(f"Imagem: {IMAGE_PATH}")
    lines.append(f"CSV: {CSV_PATH}")
    lines.append("")
    lines.append(f"Pixel size: {PIXEL_SIZE_UM:.6f} µm/pixel")
    lines.append(f"Porosidade 2D: {porosity:.6f}")
    lines.append(f"Número de poros conectados: {int(n_pores)}")
    lines.append(f"Área média dos poros: {mean_area:.6f} µm²")
    lines.append(f"Área mediana dos poros: {median_area:.6f} µm²")
    lines.append(f"Maior raio inscrito observado: {max_r:.6f} µm")
    lines.append("")

    lines.append("Parâmetros usados na interpretabilidade:")
    lines.append("")
    lines.append(f"Raios analisados: {RADIUS_PREFIXES}")
    lines.append(f"Bolinhas por raio: {TRAJECTORIES_PER_RADIUS}")
    lines.append(f"Passos por bolinha: {NSTEPS_VISUAL}")
    lines.append(f"Passo fixo em px: {FIXED_STEP_PX}")
    lines.append(f"Alpha das trajetórias: {TRAJECTORY_ALPHA}")
    lines.append(f"Espessura das trajetórias: {TRAJECTORY_LINEWIDTH}")
    lines.append(f"Plotar pontos de colisão: {PLOT_COLLISION_POINTS}")
    lines.append("")

    lines.append("Interpretação por raio:")
    lines.append("")

    for prefix in RADIUS_PREFIXES:
        r = descriptor.get(f"{prefix}_ball_radius_um", np.nan)
        rpx = descriptor.get(f"{prefix}_ball_radius_px", np.nan)
        acc = descriptor.get(f"{prefix}_accessible_fraction", np.nan)
        mfp = descriptor.get(f"{prefix}_mean_free_path_um", np.nan)
        std = descriptor.get(f"{prefix}_std_free_path_um", np.nan)
        ent = descriptor.get(f"{prefix}_angular_entropy_norm", np.nan)
        mode = descriptor.get(f"{prefix}_dominant_fft_mode", np.nan)
        intensity = descriptor.get(f"{prefix}_dominant_fft_intensity", np.nan)
        ncoll = descriptor.get(f"{prefix}_n_collisions", np.nan)

        lines.append(f"{prefix}:")
        lines.append(f"  raio da bolinha: {r:.6f} µm")
        lines.append(f"  raio da bolinha: {rpx:.6f} px")
        lines.append(f"  fração acessível: {acc:.6f}")
        lines.append(f"  colisões no descritor original: {int(ncoll)}")
        lines.append(f"  caminho livre médio: {mfp:.6f} µm")
        lines.append(f"  desvio do caminho livre: {std:.6f} µm")
        lines.append(f"  entropia angular: {ent:.6f}")
        lines.append(f"  modo FFT dominante: {int(mode)}")
        lines.append(f"  intensidade FFT dominante: {intensity:.6f}")

        if acc < 0.02:
            lines.append("  leitura: esse raio acessa apenas cavidades/canais grandes.")
        elif acc < 0.06:
            lines.append("  leitura: esse raio acessa uma sub-rede intermediária dos poros.")
        else:
            lines.append("  leitura: esse raio acessa boa parte da rede porosa verde.")

        if ent > 0.9:
            lines.append("  leitura angular: colisões muito distribuídas, geometria heterogênea ou pouco direcional.")
        elif ent > 0.7:
            lines.append("  leitura angular: há alguma direção preferencial, mas com dispersão.")
        else:
            lines.append("  leitura angular: colisões mais organizadas em ângulos específicos.")

        if intensity > 0.20:
            lines.append("  leitura FFT: orientação geométrica forte.")
        elif intensity > 0.08:
            lines.append("  leitura FFT: orientação geométrica detectável.")
        else:
            lines.append("  leitura FFT: orientação angular fraca.")

        lines.append("")

    r5_mfp = descriptor.get("r5_mean_free_path_um", np.nan)
    r90_mfp = descriptor.get("r90_mean_free_path_um", np.nan)
    r5_acc = descriptor.get("r5_accessible_fraction", np.nan)
    r90_acc = descriptor.get("r90_accessible_fraction", np.nan)

    lines.append("Tendência global:")
    lines.append("")

    if np.isfinite(r5_mfp) and np.isfinite(r90_mfp):
        if r90_mfp > r5_mfp:
            lines.append(
                "O caminho livre médio aumenta para bolinhas maiores. "
                "Isso indica que as bolinhas grandes ficam restritas aos poros mais largos, "
                "onde percorrem distâncias maiores entre colisões."
            )
        else:
            lines.append(
                "O caminho livre médio não aumenta para bolinhas maiores. "
                "Isso sugere maior confinamento ou fragmentação da rede porosa."
            )

    if np.isfinite(r5_acc) and np.isfinite(r90_acc) and r5_acc > 0:
        ratio = r90_acc / r5_acc
        lines.append(
            f"A razão de acessibilidade r90/r5 é {ratio:.6f}. "
            "Quanto menor esse valor, mais seletiva é a rede porosa para partículas grandes."
        )

    output_txt = os.path.join(OUTPUT_DIR, "pore_dcf_interpretation_summary.txt")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Salvo: {output_txt}")


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():
    print("")
    print("==============================================")
    print("PORE-DCF INTERPRETABILITY")
    print("==============================================")
    print("")
    print("Parâmetros principais:")
    print(f"  IMAGE_PATH = {IMAGE_PATH}")
    print(f"  CSV_PATH = {CSV_PATH}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"  PIXEL_SIZE_UM = {PIXEL_SIZE_UM}")
    print(f"  RADIUS_PREFIXES = {RADIUS_PREFIXES}")
    print(f"  TRAJECTORIES_PER_RADIUS = {TRAJECTORIES_PER_RADIUS}")
    print(f"  NSTEPS_VISUAL = {NSTEPS_VISUAL}")
    print(f"  FIXED_STEP_PX = {FIXED_STEP_PX}")
    print("")

    print("Lendo imagem...")
    mask, rgb = load_green_mask(IMAGE_PATH)

    print("Calculando mapa de distância...")
    distance_map = ndi.distance_transform_edt(mask)

    print("Lendo CSV do descritor...")
    df = pd.read_csv(CSV_PATH)

    if len(df) == 0:
        raise ValueError("CSV vazio.")

    descriptor = df.iloc[0].to_dict()

    if SAVE_BASIC_INTERPRETABILITY:
        print("")
        print("Gerando figuras básicas...")
        save_mask_plot(mask)
        save_distance_map_plot(distance_map)
        save_radius_distribution_plot(distance_map, descriptor)
        save_accessible_fraction_plot(descriptor)
        save_mean_free_path_plot(descriptor)
        save_entropy_plot(descriptor)
        save_fft_modes_plot(descriptor)

    if SAVE_TRAJECTORIES_BY_RADIUS:
        print("")
        print("Gerando trajetórias separadas por raio...")
        for prefix in RADIUS_PREFIXES:
            save_trajectories_one_radius(mask, distance_map, descriptor, prefix)

    if SAVE_ALL_TRAJECTORIES_TOGETHER:
        print("")
        print("Gerando todas as trajetórias juntas...")
        save_trajectories_all_radii(mask, distance_map, descriptor)

    if SAVE_ALL_TRAJECTORIES_LINES_ONLY:
        print("")
        print("Gerando todas as trajetórias juntas, sem pontos de colisão...")
        save_trajectories_all_radii_lines_only(mask, distance_map, descriptor)

    if SAVE_TEXT_SUMMARY:
        print("")
        print("Gerando resumo textual...")
        generate_text_summary(descriptor)

    print("")
    print("==============================================")
    print("FINALIZADO")
    print("==============================================")
    print(f"Arquivos salvos em: {OUTPUT_DIR}")
    print("")
    print("Arquivos principais:")
    print("  interpretability_trajectories_r5.png")
    print("  interpretability_trajectories_r25.png")
    print("  interpretability_trajectories_r50.png")
    print("  interpretability_trajectories_r75.png")
    print("  interpretability_trajectories_r90.png")
    print("  interpretability_trajectories_all_radii.png")
    print("  interpretability_trajectories_all_radii_lines_only.png")
    print("  pore_dcf_interpretation_summary.txt")
    print("")


if __name__ == "__main__":
    main()
