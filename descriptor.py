# ============================================================
# PORE-DCF COMPLETE
# Dynamic collision descriptor for 2D pores
# ============================================================
#
# Green = pore
# Black = matrix
#
# The code:
#   1. reads the segmented image
#   2. computes morphological descriptors
#   3. computes the internal pore distance map
#   4. selects probe-particle radii
#   5. launches several particles per radius
#   6. computes free paths, collisions, angular entropy, and FFT
#   7. saves a CSV file with the final descriptor
#
# ============================================================


# The analysis uses multiple probe-particle radii because each radius samples a
# different scale of the porous network. Small particles enter fine pores,
# narrow throats, and rough regions, capturing local anomalies and
# wall microtexture. Intermediate particles evaluate partial connectivity
# of the network. Large particles access only wide cavities and channels,
# representing the dominant porosity that may be more relevant for
# flow. Thus, the multi-radius descriptor provides a multiscale signature of the
# rock, combining information from microtexture, connectivity, and macropores.

# ============================================================
# 0. LIST OF PARAMETERS TO EDIT
# ============================================================

# ----------------------------
# Input file
# ----------------------------

IMAGE_PATH = "../poros/poros/patch_y7600_x53200_c0_mask.png"

# ----------------------------
# Output file
# ----------------------------

OUTPUT_CSV = "pore_dcf_descriptor.csv"

# ----------------------------
# Physical scale
# ----------------------------
# From the metadata:
# pixel_size = 4.4e-7 m = 0.44 micrometer

PIXEL_SIZE_UM = 0.44

# ----------------------------
# Probe-particle radii
# ----------------------------
# Radii are selected as percentiles of the distance map.
#
# Example 1: very small particles
# RADIUS_PERCENTILES = (1, 2, 3, 4, 5)
#
# Example 2: small, medium, and large particles
# RADIUS_PERCENTILES = (5, 25, 50, 75, 90)
#
# Example 3: more detailed
# RADIUS_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95)

RADIUS_PERCENTILES = (1, 5, 10 , 25, 50, 75, 90, 95)

# ----------------------------
# Number of launches per radius
# ----------------------------
# Each launch = one particle with a random initial position and direction.
#
# Quick test:
# NLAUNCHES = 50
#
# Good:
# NLAUNCHES = 100
#
# More robust:
# NLAUNCHES = 1000

NLAUNCHES = 1000

# ----------------------------
# Number of steps per particle
# ----------------------------
# Larger values produce longer trajectories and more collisions.
#
# Quick test:
# NSTEPS = 1000
#
# Good:
# NSTEPS = 3000
#
# More robust:
# NSTEPS = 6000 or 10000

NSTEPS = 6000

# ----------------------------
# Particle step size
# ----------------------------
# None uses automatic mode:
# step_px = max(1.0, STEP_FACTOR * particle_radius_px)
#
# To set it manually:
# FIXED_STEP_PX = 1.0
# FIXED_STEP_PX = 2.0

FIXED_STEP_PX = None

# Factor used when FIXED_STEP_PX = None
STEP_FACTOR = 0.25

# ----------------------------
# Morphological noise filter
# ----------------------------
# Pores with area smaller than this value are ignored
# in the morphological descriptors.
#
# Note: they remain in the distance map if they are present in the mask.
# To remove them completely, use REMOVE_SMALL_OBJECTS = True.

MIN_PORE_AREA_PX = 4
REMOVE_SMALL_OBJECTS = False

# ----------------------------
# Green detection
# ----------------------------
# Adjust this if your mask uses a different shade of green.

GREEN_MIN = 80
GREEN_RATIO = 1.5

# ----------------------------
# Angular descriptor
# ----------------------------

ANGULAR_NBINS = 180
MAX_FFT_MODE = 12

# ----------------------------
# Collision / numerical stability
# ----------------------------

MAX_BACKTRACK_ITER = 15
WALL_EPSILON_PX = 0.5

# ----------------------------
# Random seed
# ----------------------------

SEED_BASE = 1000

# ----------------------------
# Printing
# ----------------------------

PRINT_DESCRIPTOR_TRANSPOSED = True


# ============================================================
# 1. IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from PIL import Image

from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.stats import skew, kurtosis

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


# ============================================================
# 2. READING THE GREEN/BLACK MASK
# ============================================================

def load_green_mask(image_path):
    """
    Reads an RGB image and returns a boolean mask.

    True  = green pore
    False = black matrix
    """

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)

    mask = (
        (g > GREEN_MIN) &
        (g > GREEN_RATIO * r) &
        (g > GREEN_RATIO * b)
    )

    if REMOVE_SMALL_OBJECTS:
        mask = remove_small_objects(mask, min_size=MIN_PORE_AREA_PX)

    return mask


# ============================================================
# 3. MORPHOLOGICAL DESCRIPTORS OF THE PORES
# ============================================================

def morphology_descriptors(mask, pixel_size_um=0.44, min_pore_area_px=4):
    """
    Computes direct pore descriptors:
    porosity, areas, equivalent radii, and maximum inscribed radius.
    """

    labeled = label(mask)
    props = regionprops(labeled)

    areas_px = []
    max_inscribed_radii_px = []

    distance_map = ndi.distance_transform_edt(mask)

    for p in props:
        if p.area < min_pore_area_px:
            continue

        coords = p.coords
        areas_px.append(p.area)

        local_distances = distance_map[coords[:, 0], coords[:, 1]]
        max_inscribed_radii_px.append(np.max(local_distances))

    areas_px = np.array(areas_px, dtype=float)
    max_inscribed_radii_px = np.array(max_inscribed_radii_px, dtype=float)

    desc = {}
    desc["porosity_2d"] = float(mask.mean())
    desc["n_pores"] = int(len(areas_px))

    if len(areas_px) == 0:
        desc["mean_pore_area_um2"] = 0.0
        desc["median_pore_area_um2"] = 0.0
        desc["std_pore_area_um2"] = 0.0
        desc["min_pore_area_um2"] = 0.0
        desc["max_pore_area_um2"] = 0.0

        desc["mean_equivalent_radius_um"] = 0.0
        desc["median_equivalent_radius_um"] = 0.0
        desc["max_equivalent_radius_um"] = 0.0

        desc["mean_max_inscribed_radius_um"] = 0.0
        desc["median_max_inscribed_radius_um"] = 0.0
        desc["max_inscribed_radius_um"] = 0.0

        desc["pore_area_skewness"] = 0.0
        desc["pore_area_kurtosis"] = 0.0

        return desc, distance_map, labeled

    areas_um2 = areas_px * pixel_size_um**2
    equivalent_radii_um = np.sqrt(areas_um2 / np.pi)
    max_inscribed_radii_um = max_inscribed_radii_px * pixel_size_um

    desc["mean_pore_area_um2"] = float(np.mean(areas_um2))
    desc["median_pore_area_um2"] = float(np.median(areas_um2))
    desc["std_pore_area_um2"] = float(np.std(areas_um2))
    desc["min_pore_area_um2"] = float(np.min(areas_um2))
    desc["max_pore_area_um2"] = float(np.max(areas_um2))

    desc["mean_equivalent_radius_um"] = float(np.mean(equivalent_radii_um))
    desc["median_equivalent_radius_um"] = float(np.median(equivalent_radii_um))
    desc["max_equivalent_radius_um"] = float(np.max(equivalent_radii_um))

    desc["mean_max_inscribed_radius_um"] = float(np.mean(max_inscribed_radii_um))
    desc["median_max_inscribed_radius_um"] = float(np.median(max_inscribed_radii_um))
    desc["max_inscribed_radius_um"] = float(np.max(max_inscribed_radii_um))

    if len(areas_um2) > 2:
        desc["pore_area_skewness"] = float(skew(areas_um2))
        desc["pore_area_kurtosis"] = float(kurtosis(areas_um2))
    else:
        desc["pore_area_skewness"] = 0.0
        desc["pore_area_kurtosis"] = 0.0

    return desc, distance_map, labeled


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
    Checks whether the particle center can be located at pos.

    The particle can only be where:
        distance to wall >= radius.
    """

    y, x = pos
    height, width = distance_map.shape

    if x < 0 or x >= width or y < 0 or y >= height:
        return False

    d = interp_field(distance_map, pos)

    return d >= ball_radius_px


# ============================================================
# 5. SIMULATION OF ONE PARTICLE INSIDE THE PORES
# ============================================================

def simulate_one_particle(
    distance_map,
    ball_radius_px,
    nsteps=3000,
    step_px=None,
    seed=None,
    max_backtrack_iter=15
):
    """
    Simulates a circular particle moving inside the pore.

    Allowed region:
        distance_map >= ball_radius_px

    When the particle attempts to leave this region,
    an elastic collision with the pore wall occurs.
    """

    rng = np.random.default_rng(seed)

    accessible = distance_map >= ball_radius_px
    ys, xs = np.where(accessible)

    if len(xs) == 0:
        return np.array([]), np.array([]), np.empty((0, 2))

    if step_px is None:
        step_px = max(1.0, STEP_FACTOR * ball_radius_px)

    idx = rng.integers(0, len(xs))
    pos = np.array([ys[idx], xs[idx]], dtype=float)

    angle = rng.uniform(0, 2 * np.pi)
    vel = np.array([np.sin(angle), np.cos(angle)], dtype=float)
    vel = vel / np.linalg.norm(vel)

    grad_y, grad_x = np.gradient(distance_map.astype(float))

    collision_points = []
    collision_angles = []
    free_paths = []

    last_collision_pos = pos.copy()

    for _ in range(nsteps):

        new_pos = pos + vel * step_px

        if is_inside_accessible_region(distance_map, new_pos, ball_radius_px):
            pos = new_pos
            continue

        low = pos.copy()
        high = new_pos.copy()

        for _ in range(max_backtrack_iter):
            mid = 0.5 * (low + high)

            if is_inside_accessible_region(distance_map, mid, ball_radius_px):
                low = mid
            else:
                high = mid

        hit = low.copy()

        free_path_px = np.linalg.norm(hit - last_collision_pos)

        if free_path_px > 1e-9:
            free_paths.append(free_path_px)

        last_collision_pos = hit.copy()
        collision_points.append(hit.copy())

        ny = interp_field(grad_y, hit)
        nx = interp_field(grad_x, hit)

        normal = np.array([ny, nx], dtype=float)
        norm = np.linalg.norm(normal)

        if norm < 1e-12:
            random_angle = rng.uniform(0, 2 * np.pi)
            normal = np.array([np.sin(random_angle), np.cos(random_angle)])
        else:
            normal = normal / norm

        vel = vel - 2.0 * np.dot(vel, normal) * normal
        vel = vel / np.linalg.norm(vel)

        theta = np.arctan2(normal[0], normal[1])
        collision_angles.append(theta)

        pos_candidate = hit + vel * WALL_EPSILON_PX

        if is_inside_accessible_region(distance_map, pos_candidate, ball_radius_px):
            pos = pos_candidate
        else:
            pos = hit + normal * WALL_EPSILON_PX

            if not is_inside_accessible_region(distance_map, pos, ball_radius_px):
                idx = rng.integers(0, len(xs))
                pos = np.array([ys[idx], xs[idx]], dtype=float)

                angle = rng.uniform(0, 2 * np.pi)
                vel = np.array([np.sin(angle), np.cos(angle)], dtype=float)
                vel = vel / np.linalg.norm(vel)

                last_collision_pos = pos.copy()

    return (
        np.array(free_paths),
        np.array(collision_angles),
        np.array(collision_points)
    )


# ============================================================
# 6. ANGULAR DESCRIPTORS
# ============================================================

def angular_descriptor(angles, nbins=180, max_mode=12):
    """
    Computes angular entropy and Fourier modes.
    """

    desc = {}

    if len(angles) == 0:
        desc["angular_entropy_norm"] = 0.0

        for mode in range(1, max_mode + 1):
            desc[f"fft_mode_{mode}"] = 0.0

        desc["dominant_fft_mode"] = 0
        desc["dominant_fft_intensity"] = 0.0

        return desc

    hist, _ = np.histogram(
        angles,
        bins=nbins,
        range=(-np.pi, np.pi),
        density=False
    )

    total = hist.sum()

    if total == 0:
        desc["angular_entropy_norm"] = 0.0

        for mode in range(1, max_mode + 1):
            desc[f"fft_mode_{mode}"] = 0.0

        desc["dominant_fft_mode"] = 0
        desc["dominant_fft_intensity"] = 0.0

        return desc

    p = hist / total
    p_nonzero = p[p > 0]

    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    entropy_norm = entropy / np.log(nbins)

    fft_values = np.abs(np.fft.rfft(hist))

    if fft_values[0] > 0:
        fft_values = fft_values / fft_values[0]

    desc["angular_entropy_norm"] = float(entropy_norm)

    mode_values = []

    for mode in range(1, max_mode + 1):
        value = float(fft_values[mode]) if mode < len(fft_values) else 0.0
        desc[f"fft_mode_{mode}"] = value
        mode_values.append(value)

    dominant_mode = int(np.argmax(mode_values) + 1)
    dominant_intensity = float(np.max(mode_values))

    desc["dominant_fft_mode"] = dominant_mode
    desc["dominant_fft_intensity"] = dominant_intensity

    return desc


# ============================================================
# 7. DYNAMIC DESCRIPTOR FOR ONE RADIUS
# ============================================================

def dynamic_descriptor_for_radius(
    distance_map,
    ball_radius_px,
    pixel_size_um=0.44,
    nlaunches=100,
    nsteps=3000,
    seed_base=1234,
    max_fft_mode=12
):
    """
    Runs several particles with the same radius and returns the descriptor.
    """

    all_paths_px = []
    all_angles = []
    total_collisions = 0

    for i in range(nlaunches):
        paths_px, angles, _ = simulate_one_particle(
            distance_map=distance_map,
            ball_radius_px=ball_radius_px,
            nsteps=nsteps,
            step_px=FIXED_STEP_PX,
            seed=seed_base + i,
            max_backtrack_iter=MAX_BACKTRACK_ITER
        )

        if len(paths_px) > 0:
            all_paths_px.extend(paths_px)

        if len(angles) > 0:
            all_angles.extend(angles)

        total_collisions += len(paths_px)

    all_paths_px = np.array(all_paths_px, dtype=float)
    all_angles = np.array(all_angles, dtype=float)

    desc = {}

    desc["ball_radius_px"] = float(ball_radius_px)
    desc["ball_radius_um"] = float(ball_radius_px * pixel_size_um)
    desc["accessible_fraction"] = float(np.mean(distance_map >= ball_radius_px))
    desc["n_collisions"] = int(total_collisions)

    if len(all_paths_px) > 2:
        paths_um = all_paths_px * pixel_size_um

        desc["mean_free_path_um"] = float(np.mean(paths_um))
        desc["median_free_path_um"] = float(np.median(paths_um))
        desc["std_free_path_um"] = float(np.std(paths_um))
        desc["min_free_path_um"] = float(np.min(paths_um))
        desc["max_free_path_um"] = float(np.max(paths_um))

        desc["free_path_skewness"] = float(skew(paths_um))
        desc["free_path_kurtosis"] = float(kurtosis(paths_um))

        desc["geometric_diffusivity_um2"] = float(np.mean(paths_um**2) / 4.0)

    else:
        desc["mean_free_path_um"] = 0.0
        desc["median_free_path_um"] = 0.0
        desc["std_free_path_um"] = 0.0
        desc["min_free_path_um"] = 0.0
        desc["max_free_path_um"] = 0.0
        desc["free_path_skewness"] = 0.0
        desc["free_path_kurtosis"] = 0.0
        desc["geometric_diffusivity_um2"] = 0.0

    ang_desc = angular_descriptor(
        all_angles,
        nbins=ANGULAR_NBINS,
        max_mode=max_fft_mode
    )

    desc.update(ang_desc)

    return desc


# ============================================================
# 8. COMPLETE MULTI-RADIUS PORE-DCF
# ============================================================

def pore_dcf_descriptor(
    image_path,
    pixel_size_um=0.44,
    radius_percentiles=(5, 25, 50, 75, 90),
    nlaunches=100,
    nsteps=3000,
    min_pore_area_px=4,
    max_fft_mode=12
):
    """
    Reads the image, computes morphology, selects particle radii,
    and computes dynamic descriptors for each radius.
    """

    mask = load_green_mask(image_path)

    morph_desc, distance_map, labeled = morphology_descriptors(
        mask,
        pixel_size_um=pixel_size_um,
        min_pore_area_px=min_pore_area_px
    )

    desc = {}
    desc["image_path"] = image_path
    desc["pixel_size_um"] = pixel_size_um
    desc["image_height_px"] = int(mask.shape[0])
    desc["image_width_px"] = int(mask.shape[1])
    desc["radius_percentiles"] = str(radius_percentiles)
    desc["nlaunches"] = int(nlaunches)
    desc["nsteps"] = int(nsteps)
    desc["fixed_step_px"] = str(FIXED_STEP_PX)
    desc["step_factor"] = float(STEP_FACTOR)

    desc.update(morph_desc)

    valid_distances = distance_map[distance_map > 0]

    if len(valid_distances) == 0:
        return desc

    selected_radii_px = np.percentile(valid_distances, radius_percentiles)

    for percentile, radius_px in zip(radius_percentiles, selected_radii_px):

        dyn_desc = dynamic_descriptor_for_radius(
            distance_map=distance_map,
            ball_radius_px=radius_px,
            pixel_size_um=pixel_size_um,
            nlaunches=nlaunches,
            nsteps=nsteps,
            seed_base=SEED_BASE + int(percentile) * 100,
            max_fft_mode=max_fft_mode
        )

        prefix = f"r{percentile}"

        for key, value in dyn_desc.items():
            desc[f"{prefix}_{key}"] = value

    return desc


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":

    print("")
    print("==============================================")
    print("PORE-DCF COMPLETE")
    print("==============================================")
    print("")
    print("Parâmetros:")
    print(f"  IMAGE_PATH = {IMAGE_PATH}")
    print(f"  OUTPUT_CSV = {OUTPUT_CSV}")
    print(f"  PIXEL_SIZE_UM = {PIXEL_SIZE_UM}")
    print(f"  RADIUS_PERCENTILES = {RADIUS_PERCENTILES}")
    print(f"  NLAUNCHES = {NLAUNCHES}")
    print(f"  NSTEPS = {NSTEPS}")
    print(f"  FIXED_STEP_PX = {FIXED_STEP_PX}")
    print(f"  STEP_FACTOR = {STEP_FACTOR}")
    print(f"  MIN_PORE_AREA_PX = {MIN_PORE_AREA_PX}")
    print(f"  REMOVE_SMALL_OBJECTS = {REMOVE_SMALL_OBJECTS}")
    print(f"  ANGULAR_NBINS = {ANGULAR_NBINS}")
    print(f"  MAX_FFT_MODE = {MAX_FFT_MODE}")
    print("")

    descriptor = pore_dcf_descriptor(
        image_path=IMAGE_PATH,
        pixel_size_um=PIXEL_SIZE_UM,
        radius_percentiles=RADIUS_PERCENTILES,
        nlaunches=NLAUNCHES,
        nsteps=NSTEPS,
        min_pore_area_px=MIN_PORE_AREA_PX,
        max_fft_mode=MAX_FFT_MODE
    )

    df = pd.DataFrame([descriptor])
    df.to_csv(OUTPUT_CSV, index=False)

    print("")
    print("Descritor Pore-DCF gerado com sucesso.")
    print(f"Arquivo salvo em: {OUTPUT_CSV}")
    print("")

    if PRINT_DESCRIPTOR_TRANSPOSED:
        print(df.T)
