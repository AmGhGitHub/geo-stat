"""
===========================================================================
GeostatsPy Reservoir Geomodel Generator for Miscible Flood Studies
===========================================================================

This script generates a geostatistically heterogeneous 2D reservoir model
suitable for investigating the effect of heterogeneity and conformance on
miscible gas injection performance (e.g., using the Todd-Longstaff model).

It demonstrates:
  1. Setting up a 2D grid
  2. Generating correlated porosity fields via Sequential Gaussian Simulation
  3. Deriving permeability from porosity using a transform
  4. Computing heterogeneity metrics (Dykstra-Parsons, Lorenz coefficient)
  5. Generating multiple realizations to study uncertainty
  6. Exporting grids in formats usable by OPM Flow / Eclipse

Requirements (install via pip):
    pip install geostatspy numpy matplotlib pandas scipy

Author: Generated for reservoir engineering study
Date:   March 2026
===========================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, truncnorm

# ── Try importing GeostatsPy ──
try:
    import geostatspy.geostats as geostats
    import geostatspy.GSLIB as GSLIB

    GEOSTATSPY_AVAILABLE = True
    print("✓ GeostatsPy successfully imported.")
except ImportError:
    GEOSTATSPY_AVAILABLE = False
    print("✗ GeostatsPy not found. Install it with: pip install geostatspy")
    print("  Falling back to built-in Sequential Gaussian Simulation.\n")


# =========================================================================
# SECTION 1: USER PARAMETERS — MODIFY THESE FOR YOUR STUDY
# =========================================================================

# ── Grid dimensions ──
nx = 150  # number of cells in x-direction
ny = 150  # number of cells in y-direction
dx = 5.0  # cell size in x (meters)
dy = 5.0  # cell size in y (meters)

# ── Porosity distribution ──
por_mean = 0.10  # mean porosity (fraction)
por_std = 0.04  # standard deviation of porosity (MAXIMIZED for extreme heterogeneity)
por_min = 0.05  # minimum porosity (truncation) - very tight rock
por_max = 0.20  # maximum porosity (truncation) - high perm channels

# ── Variogram parameters for porosity ──
# These control the spatial correlation structure
variogram_type = 1  # 1 = spherical, 2 = exponential, 3 = Gaussian
range_major = 100.0  # correlation range in major direction (m)
range_minor = 60.0  # correlation range in minor direction (m) - SHORT for high contrast
azimuth = 1.0  # azimuth of major direction (degrees from N)
nugget = 0.0  # nugget effect - ZERO for smooth transitions
sill = 1.0  # sill (normalized; actual variance = por_std^2)

# ── Porosity-to-Permeability transform ──
# log10(k) = a * porosity + b   (common empirical relationship)
perm_a = 40.0  # slope of poro-perm transform (MODERATE for realistic contrast)
perm_b = -1.5  # intercept of poro-perm transform
# Realistic ranges: por=0.01 → k~0.003 mD, por=0.15 → k~100 mD, por=0.35 → k~3,000,000 mD

# ── Number of realizations ──
n_realizations = 5

# ── Output directory ──
output_dir = "geomodel_output"

# Clear previous output files before running
if os.path.exists(output_dir):
    import shutil

    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    print(f"  Cleared previous output from: {output_dir}/")

os.makedirs(output_dir, exist_ok=True)

# ── Random seed (for reproducibility) ──
base_seed = 42


# =========================================================================
# SECTION 2: BUILT-IN SGS (fallback if GeostatsPy unavailable)
# =========================================================================


def variogram_model(h, vtype, a, nugget, sill):
    """Compute variogram value for lag distance h."""
    if h == 0:
        return 0.0
    hr = h / a
    if vtype == 1:  # Spherical
        if hr >= 1.0:
            return nugget + sill
        return nugget + sill * (1.5 * hr - 0.5 * hr**3)
    elif vtype == 2:  # Exponential
        return nugget + sill * (1.0 - np.exp(-3.0 * hr))
    elif vtype == 3:  # Gaussian
        return nugget + sill * (1.0 - np.exp(-3.0 * hr**2))
    return nugget + sill


def covariance_model(h, vtype, a, nugget, sill):
    """Covariance = sill - variogram (for stationary model)."""
    return (nugget + sill) - variogram_model(h, vtype, a, nugget, sill)


def sgs_2d_builtin(
    nx,
    ny,
    dx,
    dy,
    mean,
    std,
    vtype,
    range_maj,
    range_min,
    azimuth_deg,
    nugget,
    sill,
    seed,
    n_cond_max=24,
):
    """
    Simple 2D Sequential Gaussian Simulation (built-in fallback).

    Uses a simplified approach with nearest-neighbor kriging for
    demonstration purposes. For production work, use GeostatsPy.
    """
    rng = np.random.RandomState(seed)
    az_rad = np.radians(azimuth_deg)

    # Create grid coordinates
    xcoords = np.arange(0.5 * dx, nx * dx, dx)
    ycoords = np.arange(0.5 * dy, ny * dy, dy)

    # Initialize output grid
    grid = np.full((ny, nx), np.nan)

    # Random path through all cells
    indices = [(iy, ix) for iy in range(ny) for ix in range(nx)]
    rng.shuffle(indices)

    # Store simulated values with coordinates
    sim_x = []
    sim_y = []
    sim_val = []

    for count, (iy, ix) in enumerate(indices):
        cx = xcoords[ix]
        cy = ycoords[iy]

        if len(sim_val) == 0:
            # First point: draw from marginal distribution
            grid[iy, ix] = rng.normal(0, 1)
        else:
            # Find nearest already-simulated points
            sx = np.array(sim_x)
            sy = np.array(sim_y)
            sv = np.array(sim_val)

            dists_raw = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)
            n_use = min(n_cond_max, len(sv))
            nearest_idx = np.argsort(dists_raw)[:n_use]

            # Compute anisotropic distances
            ddx = sx[nearest_idx] - cx
            ddy = sy[nearest_idx] - cy

            # Rotate to variogram axes
            ddx_rot = ddx * np.cos(az_rad) + ddy * np.sin(az_rad)
            ddy_rot = -ddx * np.sin(az_rad) + ddy * np.cos(az_rad)

            # Anisotropic distance
            h_vals = (
                np.sqrt((ddx_rot / range_maj) ** 2 + (ddy_rot / range_min) ** 2)
                * range_maj
            )

            # Simple kriging
            n = len(nearest_idx)
            C_dd = np.zeros((n, n))
            C_d0 = np.zeros(n)

            for i in range(n):
                C_d0[i] = covariance_model(h_vals[i], vtype, range_maj, nugget, sill)
                for j in range(n):
                    if i == j:
                        C_dd[i, j] = covariance_model(0, vtype, range_maj, nugget, sill)
                    else:
                        di = nearest_idx[i]
                        dj = nearest_idx[j]
                        ddx_ij = sx[di] - sx[dj]
                        ddy_ij = sy[di] - sy[dj]
                        ddx_r = ddx_ij * np.cos(az_rad) + ddy_ij * np.sin(az_rad)
                        ddy_r = -ddx_ij * np.sin(az_rad) + ddy_ij * np.cos(az_rad)
                        h_ij = (
                            np.sqrt((ddx_r / range_maj) ** 2 + (ddy_r / range_min) ** 2)
                            * range_maj
                        )
                        C_dd[i, j] = covariance_model(
                            h_ij, vtype, range_maj, nugget, sill
                        )

            # Add small diagonal for numerical stability
            C_dd += np.eye(n) * 1e-8

            try:
                weights = np.linalg.solve(C_dd, C_d0)
                sk_mean = np.dot(weights, sv[nearest_idx])
                sk_var = (nugget + sill) - np.dot(weights, C_d0)
                sk_var = max(sk_var, 0.001)
            except np.linalg.LinAlgError:
                sk_mean = np.mean(sv[nearest_idx])
                sk_var = 0.5

            grid[iy, ix] = sk_mean + rng.normal(0, 1) * np.sqrt(sk_var)

        sim_x.append(cx)
        sim_y.append(cy)
        sim_val.append(grid[iy, ix])

        # Progress update
        if (count + 1) % 2000 == 0:
            print(
                f"  SGS progress: {count+1}/{nx*ny} cells ({100*(count+1)/(nx*ny):.0f}%)"
            )

    # Back-transform from N(0,1) to target distribution
    grid = norm.cdf(grid)  # uniform [0,1]
    grid = norm.ppf(grid, loc=mean, scale=std)  # target N(mean, std)
    grid = np.clip(grid, por_min, por_max)

    return grid, xcoords, ycoords


# =========================================================================
# SECTION 3: GENERATE POROSITY REALIZATIONS
# =========================================================================


def generate_porosity_geostatspy(seed):
    """Generate porosity field using GeostatsPy SGS."""
    # Create grid coordinates for GeostatsPy
    xmin, xmax = 0.5 * dx, (nx - 0.5) * dx
    ymin, ymax = 0.5 * dy, (ny - 0.5) * dy

    # GeostatsPy uses GSLIB-style variogram parameters
    # [type, range_major, range_minor, azimuth, nugget, sill]
    vario = GSLIB.make_variogram(
        nug=nugget,
        nst=1,
        it1=variogram_type,
        cc1=sill,
        azi1=azimuth,
        hmaj1=range_major,
        hmin1=range_minor,
    )

    # Run Sequential Gaussian Simulation
    sim = geostats.sgsim(
        1,
        nx,
        xmin,
        dx,
        ny,
        ymin,
        dy,
        pd.DataFrame({"X": [nx * dx / 2], "Y": [ny * dy / 2], "Porosity": [por_mean]}),
        "X",
        "Y",
        "Porosity",
        0,
        0,  # no secondary variable
        por_min,
        por_max,
        1,
        por_mean,
        por_std**2,
        vario,
        seed,
    )

    por_grid = sim.reshape((ny, nx))
    xcoords = np.linspace(xmin, xmax, nx)
    ycoords = np.linspace(ymin, ymax, ny)

    return por_grid, xcoords, ycoords


def generate_porosity_builtin(seed):
    """Generate porosity field using built-in SGS."""
    return sgs_2d_builtin(
        nx,
        ny,
        dx,
        dy,
        por_mean,
        por_std,
        variogram_type,
        range_major,
        range_minor,
        azimuth,
        nugget,
        sill,
        seed,
    )


def porosity_to_permeability(porosity):
    """Convert porosity to permeability using log-linear transform."""
    log_k = perm_a * porosity + perm_b
    return 10.0**log_k


def preview_expected_properties():
    """
    Preview expected porosity and permeability ranges based on current parameters.
    Shows what values to expect before running the full simulation.
    """
    print("\n" + "=" * 70)
    print("  EXPECTED PROPERTY RANGES (Parameter Preview)")
    print("=" * 70)

    # Porosity expectations
    print(f"\n  POROSITY DISTRIBUTION:")
    print(f"    Target: Normal distribution N(μ={por_mean}, σ={por_std})")
    print(f"    Truncated to: [{por_min}, {por_max}]")

    # Expected percentiles for truncated normal
    a_trunc = (por_min - por_mean) / por_std
    b_trunc = (por_max - por_mean) / por_std
    tn = truncnorm(a_trunc, b_trunc, loc=por_mean, scale=por_std)

    print(f"    Expected mean (truncated):   {tn.mean():.4f}")
    print(f"    Expected std (truncated):    {tn.std():.4f}")
    print(f"    Expected P10:                {tn.ppf(0.10):.4f}")
    print(f"    Expected P50 (median):       {tn.ppf(0.50):.4f}")
    print(f"    Expected P90:                {tn.ppf(0.90):.4f}")

    # Permeability expectations
    print(f"\n  PERMEABILITY DISTRIBUTION:")
    print(f"    Transform: log10(k) = {perm_a} * φ + {perm_b}")
    print(f"             k = 10^({perm_a} * φ + {perm_b})")

    # Calculate permeability at key porosity values
    por_vals = np.linspace(por_min, por_max, 100)
    perm_vals = porosity_to_permeability(por_vals)

    # Expected permeability statistics (approximate)
    por_samples = tn.rvs(size=10000, random_state=42)
    perm_samples = porosity_to_permeability(por_samples)

    print(f"    At φ = {por_min:.2f}:  k ≈ {porosity_to_permeability(por_min):.1f} mD")
    print(f"    At φ = {por_mean:.3f}: k ≈ {porosity_to_permeability(por_mean):.1f} mD")
    print(f"    At φ = {por_max:.2f}:  k ≈ {porosity_to_permeability(por_max):.1f} mD")
    print(f"\n    Expected permeability range:")
    print(f"      P10: {np.percentile(perm_samples, 10):.1f} mD")
    print(f"      P50: {np.percentile(perm_samples, 50):.1f} mD")
    print(f"      P90: {np.percentile(perm_samples, 90):.1f} mD")
    print(f"      Min: {np.min(perm_samples):.1f} mD")
    print(f"      Max: {np.max(perm_samples):.1f} mD")

    # Heterogeneity expectations
    print(f"\n  EXPECTED HETEROGENEITY:")
    print(f"    Based on porosity std = {por_std} and perm slope = {perm_a}:")

    # Approximate V_DP from log-perm variance
    log_perm = np.log10(perm_samples)
    log_var = np.var(log_perm)
    vdp_approx = 1 - np.exp(-np.sqrt(log_var))
    print(f"    Approx. Dykstra-Parsons:     {vdp_approx:.3f}")

    # Lorenz coefficient hint
    if por_std < 0.02:
        lorenz_hint = "~0.1-0.2 (mild heterogeneity)"
    elif por_std < 0.04:
        lorenz_hint = "~0.2-0.4 (moderate heterogeneity)"
    else:
        lorenz_hint = "~0.4-0.6 (strong heterogeneity)"
    print(f"    Expected Lorenz coeff range: {lorenz_hint}")

    print("\n" + "=" * 70)
    print("  Run the simulation to generate actual realizations.")
    print("=" * 70 + "\n")


# =========================================================================
# SECTION 4: HETEROGENEITY METRICS
# =========================================================================


def dykstra_parsons(perm_array):
    """
    Compute Dykstra-Parsons coefficient of permeability variation.
    V_DP = (k_50 - k_84.1) / k_50
    where k_50 = median, k_84.1 = 84.1th percentile of log-normal dist.
    Values range from 0 (homogeneous) to 1 (extremely heterogeneous).
    Typical reservoirs: 0.5 to 0.9
    """
    k_flat = perm_array.flatten()
    k_50 = np.percentile(k_flat, 50)
    k_84 = np.percentile(k_flat, 100 - 84.1)  # 15.9th percentile
    vdp = (k_50 - k_84) / k_50
    return vdp


def lorenz_coefficient(perm_array, por_array):
    """
    Compute Lorenz coefficient from flow capacity (kh) vs storage capacity (φh).
    L = 0: perfectly homogeneous
    L = 1: perfectly heterogeneous (all flow in one layer)
    """
    k_flat = perm_array.flatten()
    phi_flat = por_array.flatten()

    # Sort by k/phi ratio (decreasing)
    kphi_ratio = k_flat / phi_flat
    sort_idx = np.argsort(kphi_ratio)[::-1]

    k_sorted = k_flat[sort_idx]
    phi_sorted = phi_flat[sort_idx]

    # Cumulative flow capacity and storage capacity
    cum_k = np.cumsum(k_sorted) / np.sum(k_sorted)
    cum_phi = np.cumsum(phi_sorted) / np.sum(phi_sorted)

    # Lorenz = 2 * area between curve and diagonal
    area_under_curve = np.trapezoid(cum_k, cum_phi)
    lorenz = 2.0 * (area_under_curve - 0.5)

    return lorenz, cum_phi, cum_k


# =========================================================================
# SECTION 5: EXPORT TO OPM FLOW / ECLIPSE FORMAT
# =========================================================================


def export_eclipse_include(porosity, permeability, filename):
    """
    Export porosity and permeability grids as Eclipse INCLUDE files.
    These can be included in an OPM Flow / Eclipse data deck.
    """
    nz = 1  # single layer for 2D model

    with open(filename, "w") as f:
        f.write(f"-- Geostatistically generated reservoir properties\n")
        f.write(f"-- Grid: {nx} x {ny} x {nz}\n")
        f.write(f"-- Generated by GeostatsPy reservoir model script\n\n")

        # Porosity
        f.write("PORO\n")
        por_flat = porosity.flatten()
        for i, val in enumerate(por_flat):
            f.write(f"  {val:.6f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        f.write(" /\n\n")

        # Permeability X (horizontal)
        f.write("PERMX\n")
        k_flat = permeability.flatten()
        for i, val in enumerate(k_flat):
            f.write(f"  {val:.4f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        f.write(" /\n\n")

        # Permeability Y = Permeability X (isotropic in horizontal)
        f.write("PERMY\n")
        for i, val in enumerate(k_flat):
            f.write(f"  {val:.4f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        f.write(" /\n\n")

        # Permeability Z (vertical, typically reduced)
        kv_kh_ratio = 0.1  # common ratio
        f.write(f"-- PERMZ = {kv_kh_ratio} * PERMX\n")
        f.write("PERMZ\n")
        for i, val in enumerate(k_flat):
            f.write(f"  {val * kv_kh_ratio:.4f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        f.write(" /\n\n")

    print(f"  Exported Eclipse include file: {filename}")


def export_eclipse_data_deck(porosity, permeability, filename):
    """
    Export a COMPLETE minimal Eclipse / OPM Flow data deck
    for a 2D miscible flood using the Todd-Longstaff model.

    This is a working template — adjust fluid properties,
    well locations, and schedule for your specific case.
    """
    nz = 1
    dz = 10.0  # layer thickness (m)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            """-- ================================================================
-- MISCIBLE FLOOD MODEL WITH TODD-LONGSTAFF
-- Generated by GeostatsPy Reservoir Model Script
-- ================================================================

RUNSPEC
-- ----------------------------------------------------------------

TITLE
MISCIBLE_FLOOD_TODD_LONGSTAFF

DIMENS
-- NX   NY   NZ
"""
        )
        f.write(f"   {nx}   {ny}   {nz}  /\n")
        f.write(
            """
-- Phases present
OIL
WATER
GAS

-- Enable miscible flood (Todd-Longstaff solvent model)
MISCIBLE
-- This activates the 4-phase solvent model

METRIC

START
  1 'JAN' 2026 /

WELLDIMS
  4  100  4  4  /

TABDIMS
  1  1  40  40  /

-- ----------------------------------------------------------------
GRID
-- ----------------------------------------------------------------

-- Cartesian grid
DXV
"""
        )
        f.write(f"  {nx}*{dx:.1f}  /\n")
        f.write("DYV\n")
        f.write(f"  {ny}*{dy:.1f}  /\n")
        f.write("DZV\n")
        f.write(f"  {nz}*{dz:.1f}  /\n")
        f.write(
            """
TOPS
"""
        )
        f.write(f"  {nx*ny}*2000.0  /\n")
        f.write(
            """
-- Include geostatistical properties
-- (porosity and permeability from geomodel)

PORO
"""
        )
        por_flat = porosity.flatten()
        for i, val in enumerate(por_flat):
            f.write(f"  {val:.6f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        f.write(" /\n\n")

        f.write("PERMX\n")
        k_flat = permeability.flatten()
        for i, val in enumerate(k_flat):
            f.write(f"  {val:.4f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        f.write(" /\n\n")

        f.write("COPY\n")
        f.write("  PERMX  PERMY /\n")
        f.write("/\n\n")

        f.write("PERMZ\n")
        for i, val in enumerate(k_flat):
            f.write(f"  {val * 0.1:.4f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        f.write(" /\n\n")

        f.write(
            """
-- ----------------------------------------------------------------
PROPS
-- ----------------------------------------------------------------

-- Water PVT (dead water)
PVTW
-- Pref    Bw      Cw         Visc   Viscosibility
  300.0   1.01   4.0E-05    0.50    0.0  /

-- Oil PVT (dead oil, no dissolved gas)
PVDO
-- Press   Bo      Visco
  100.0   1.10    2.00
  200.0   1.08    2.10
  300.0   1.06    2.20
  400.0   1.05    2.30
  500.0   1.04    2.40  /

-- Gas (Solvent) PVT
PVDG
-- Press   Bg         Visc
  100.0   0.01000    0.020
  200.0   0.00500    0.025
  300.0   0.00340    0.035
  400.0   0.00260    0.045
  500.0   0.00210    0.050  /

-- Rock compressibility
ROCK
  300.0   4.0E-05  /

-- Density at surface conditions
DENSITY
-- Oil      Water    Gas
  800.0   1025.0   1.2  /

-- Water-oil relative permeability
SWOF
-- Sw       Krw      Kro      Pcow
  0.20    0.0000   1.0000    0.0
  0.30    0.0200   0.6500    0.0
  0.40    0.0600   0.4000    0.0
  0.50    0.1200   0.2200    0.0
  0.60    0.2000   0.1000    0.0
  0.70    0.3200   0.0300    0.0
  0.80    0.5000   0.0000    0.0  /

-- Gas-oil relative permeability (used for solvent)
-- NOTE: Pcgo = 0 for miscible flood
SGOF
-- Sg       Krg      Kro      Pcgo
  0.00    0.0000   1.0000    0.0
  0.10    0.0100   0.7000    0.0
  0.20    0.0400   0.4500    0.0
  0.30    0.1000   0.2500    0.0
  0.40    0.2000   0.1200    0.0
  0.50    0.3500   0.0400    0.0
  0.60    0.5500   0.0000    0.0  /

-- ================================================================
-- TODD-LONGSTAFF MIXING PARAMETER
-- ================================================================
-- This is the key parameter!
-- ω = 0.0 : no mixing (pure component properties, pessimistic)
-- ω = 1/3 : recommended for field-scale simulation
-- ω = 2/3 : matches lab-scale experiments  
-- ω = 1.0 : complete mixing (optimistic)

TLMIXPAR
  0.3333  /

-- Solvent PVT (same as gas for first-contact miscible)
SSFN
-- Sg     Krg_misc   Krg_immisc
  0.00   0.0000     0.0000
  0.10   0.0100     0.0100
  0.20   0.0400     0.0400
  0.30   0.1000     0.1000
  0.40   0.2000     0.2000
  0.50   0.3500     0.3500
  0.60   0.5500     0.5500  /

-- Miscibility function (= 1 everywhere for FCM)
MISC
  0.0  0.0
  0.1  1.0
  1.0  1.0  /

-- Solvent density and viscosity at reference pressure
SDENSITY
  160.0  /

SOLVNUM
"""
        )
        f.write(f"  {nx*ny}*1  /\n")

        f.write(
            """
-- ----------------------------------------------------------------
SOLUTION
-- ----------------------------------------------------------------

-- Initial conditions: oil zone with connate water
EQUIL
-- Datum   Pdat    WOC     Pcow   GOC    Pcgo
  2000.0  300.0  2100.0   0.0   1900.0  0.0  /

-- ----------------------------------------------------------------
SUMMARY
-- ----------------------------------------------------------------

-- Field-level output
FOPR
FOPT
FWPR
FWPT
FGPR
FGPT
FGOR
FWCT
FOE

-- Well-level
WOPR
/
WWPR
/
WGOR
/

-- ----------------------------------------------------------------
SCHEDULE
-- ----------------------------------------------------------------

-- Define wells
-- Injector: top-left corner (for line drive) or center
WELSPECS
  'INJ1'  'G1'  1    1    2000.0  'GAS'  /
  'PRD1'  'G1'  """
        )
        f.write(f"{nx}  {ny}")
        f.write(
            """  2000.0  'OIL'  /
/

COMPDAT
  'INJ1'  1    1    1  1  'OPEN'  0  0.0  0.15  /
"""
        )
        f.write(f"  'PRD1'  {nx}  {ny}  1  1  'OPEN'  0  0.0  0.15  /\n")
        f.write(
            """/

-- Injection control: inject solvent at constant rate
WCONINJE
  'INJ1'  'GAS'  'OPEN'  'RATE'  5000.0  1*  500.0  /
/

-- Producer control: produce at constant BHP
WCONPROD
  'PRD1'  'OPEN'  'BHP'  5*  200.0  /
/

-- IMPORTANT: Specify that injector injects SOLVENT (not regular gas)
WSOLVENT
  'INJ1'  1.0  /
/

-- Run for 3000 days with reporting every 100 days
TSTEP
  30*100.0  /

END
"""
        )

    print(f"  Exported complete OPM Flow data deck: {filename}")


# =========================================================================
# SECTION 6: VISUALIZATION
# =========================================================================


def plot_realization(por, perm, xcoords, ycoords, real_num, vdp, lorenz_val):
    """Create a comprehensive plot of a single realization."""

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    extent = [0, nx * dx, 0, ny * dy]

    # ── Porosity map ──
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        por, origin="lower", extent=extent, cmap="viridis", vmin=por_min, vmax=por_max
    )
    ax1.set_title(f"Porosity — Realization {real_num}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    plt.colorbar(im1, ax=ax1, label="Porosity (fraction)", shrink=0.8)

    # ── Permeability map (log scale) ──
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        np.log10(perm), origin="lower", extent=extent, cmap="jet", vmin=0, vmax=4
    )
    ax2.set_title(
        f"log₁₀(Permeability) — Realization {real_num}", fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    plt.colorbar(im2, ax=ax2, label="log₁₀(k) [mD]", shrink=0.8)

    # ── Permeability histogram ──
    ax3 = fig.add_subplot(gs[0, 2])
    k_flat = perm.flatten()
    ax3.hist(np.log10(k_flat), bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax3.set_xlabel("log₁₀(Permeability) [mD]")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Permeability Distribution", fontsize=11, fontweight="bold")
    ax3.axvline(
        np.log10(np.median(k_flat)),
        color="red",
        linestyle="--",
        label=f"Median = {np.median(k_flat):.0f} mD",
    )
    ax3.legend(fontsize=9)

    # ── Porosity histogram ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(por.flatten(), bins=50, color="green", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("Porosity (fraction)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Porosity Distribution", fontsize=11, fontweight="bold")
    ax4.axvline(por_mean, color="red", linestyle="--", label=f"Mean = {por_mean:.3f}")
    ax4.legend(fontsize=9)

    # ── Poro-Perm crossplot ──
    ax5 = fig.add_subplot(gs[1, 1])
    subsample = np.random.choice(
        len(por.flatten()), size=min(2000, len(por.flatten())), replace=False
    )
    ax5.scatter(
        por.flatten()[subsample],
        perm.flatten()[subsample],
        alpha=0.3,
        s=5,
        c="steelblue",
    )
    ax5.set_yscale("log")
    ax5.set_xlabel("Porosity (fraction)")
    ax5.set_ylabel("Permeability (mD)")
    ax5.set_title("Porosity–Permeability Crossplot", fontsize=11, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # ── Lorenz plot ──
    _, cum_phi, cum_k = lorenz_coefficient(perm, por)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(cum_phi, cum_k, "b-", linewidth=2, label=f"L = {lorenz_val:.3f}")
    ax6.plot([0, 1], [0, 1], "k--", linewidth=1, label="Homogeneous (L=0)")
    ax6.fill_between(cum_phi, cum_k, cum_phi, alpha=0.15, color="blue")
    ax6.set_xlabel("Cumulative Storage Capacity (Φh)")
    ax6.set_ylabel("Cumulative Flow Capacity (kh)")
    ax6.set_title("Lorenz Plot", fontsize=11, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_aspect("equal")
    ax6.grid(True, alpha=0.3)

    # ── Summary statistics text ──
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    stats_text = (
        f"═══════════════════════════════════════════════════════════════════\n"
        f"  REALIZATION {real_num} — HETEROGENEITY SUMMARY\n"
        f"═══════════════════════════════════════════════════════════════════\n"
        f"  Grid:  {nx} × {ny} cells  |  Cell size: {dx:.0f} × {dy:.0f} m  |  "
        f"Domain: {nx*dx:.0f} × {ny*dy:.0f} m\n"
        f"  Variogram: {'Spherical' if variogram_type==1 else 'Exponential' if variogram_type==2 else 'Gaussian'}  |  "
        f"Range: {range_major:.0f} × {range_minor:.0f} m  |  Azimuth: {azimuth:.0f}°\n"
        f"───────────────────────────────────────────────────────────────────\n"
        f"  Porosity:      mean = {np.mean(por):.4f}    std = {np.std(por):.4f}    "
        f"min = {np.min(por):.4f}    max = {np.max(por):.4f}\n"
        f"  Permeability:  P10 = {np.percentile(k_flat, 10):.1f} mD    "
        f"P50 = {np.percentile(k_flat, 50):.1f} mD    "
        f"P90 = {np.percentile(k_flat, 90):.1f} mD\n"
        f"───────────────────────────────────────────────────────────────────\n"
        f"  Dykstra-Parsons coefficient:  V_DP = {vdp:.3f}   "
        f"({'Low' if vdp < 0.5 else 'Moderate' if vdp < 0.7 else 'High' if vdp < 0.9 else 'Very High'} heterogeneity)\n"
        f"  Lorenz coefficient:           L    = {lorenz_val:.3f}   "
        f"({'Homogeneous' if lorenz_val < 0.1 else 'Mildly heterogeneous' if lorenz_val < 0.3 else 'Heterogeneous' if lorenz_val < 0.5 else 'Highly heterogeneous'})\n"
        f"═══════════════════════════════════════════════════════════════════"
    )
    ax7.text(
        0.02,
        0.5,
        stats_text,
        transform=ax7.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        f"Geostatistical Reservoir Model — Realization {real_num}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    figpath = os.path.join(output_dir, f"realization_{real_num}.png")
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {figpath}")


def plot_comparison(all_por, all_perm, all_vdp, all_lorenz):
    """Compare all realizations side by side."""
    n = len(all_por)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    extent = [0, nx * dx, 0, ny * dy]

    for i in range(n):
        # Porosity
        im1 = axes[0, i].imshow(
            all_por[i],
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=por_min,
            vmax=por_max,
        )
        axes[0, i].set_title(f"Realization {i+1}\nPorosity", fontsize=10)
        axes[0, i].set_xlabel("X (m)")
        if i == 0:
            axes[0, i].set_ylabel("Y (m)")

        # Permeability
        im2 = axes[1, i].imshow(
            np.log10(all_perm[i]),
            origin="lower",
            extent=extent,
            cmap="jet",
            vmin=0,
            vmax=4,
        )
        axes[1, i].set_title(
            f"V_DP={all_vdp[i]:.3f}  L={all_lorenz[i]:.3f}", fontsize=10
        )
        axes[1, i].set_xlabel("X (m)")
        if i == 0:
            axes[1, i].set_ylabel("Y (m)")

    fig.suptitle("Comparison of Multiple Realizations", fontsize=14, fontweight="bold")
    plt.tight_layout()

    figpath = os.path.join(output_dir, "comparison_all_realizations.png")
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison plot: {figpath}")


# =========================================================================
# SECTION 7: MAIN EXECUTION
# =========================================================================


def main():
    print("=" * 70)
    print("  GEOSTATISTICAL RESERVOIR MODEL GENERATOR")
    print("  For Miscible Flood Heterogeneity Studies")
    print("=" * 70)

    # Show expected property ranges before running
    preview_expected_properties()
    print(f"\n  Grid:       {nx} × {ny} cells ({nx*dx:.0f} × {ny*dy:.0f} m)")
    print(f"  Porosity:   mean={por_mean}, std={por_std}")
    print(f"  Variogram:  range={range_major}×{range_minor} m, azimuth={azimuth}°")
    print(f"  Realizations: {n_realizations}")
    print(f"  Output dir: {output_dir}/")
    print()

    # Choose SGS method
    if GEOSTATSPY_AVAILABLE:
        generate_func = generate_porosity_geostatspy
        print("  Using GeostatsPy for Sequential Gaussian Simulation.\n")
    else:
        generate_func = generate_porosity_builtin
        print("  Using built-in SGS (install geostatspy for better results).\n")

    all_por = []
    all_perm = []
    all_vdp = []
    all_lorenz = []

    for r in range(n_realizations):
        seed = base_seed + r * 1000
        print(f"── Realization {r+1}/{n_realizations} (seed={seed}) ──")

        # Generate porosity
        print(f"  Generating porosity field...")
        por, xcoords, ycoords = generate_func(seed)

        # Derive permeability
        perm = porosity_to_permeability(por)
        print(f"  Porosity:  mean={np.mean(por):.4f}, std={np.std(por):.4f}")
        print(
            f"  Perm (mD): P10={np.percentile(perm, 10):.1f}, "
            f"P50={np.percentile(perm, 50):.1f}, P90={np.percentile(perm, 90):.1f}"
        )

        # Compute heterogeneity metrics
        vdp = dykstra_parsons(perm)
        lorenz_val, _, _ = lorenz_coefficient(perm, por)
        print(f"  Dykstra-Parsons: {vdp:.3f}")
        print(f"  Lorenz coeff:    {lorenz_val:.3f}")

        # Store
        all_por.append(por)
        all_perm.append(perm)
        all_vdp.append(vdp)
        all_lorenz.append(lorenz_val)

        # Plot individual realization
        plot_realization(por, perm, xcoords, ycoords, r + 1, vdp, lorenz_val)

        # Export Eclipse include file
        include_file = os.path.join(output_dir, f"PROPS_REAL_{r+1}.INC")
        export_eclipse_include(por, perm, include_file)

        print()

    # Export a complete OPM Flow data deck for realization 1
    print("── Exporting complete OPM Flow / Eclipse data deck (Realization 1) ──")
    deck_file = os.path.join(output_dir, "MISCIBLE_FLOOD.DATA")
    export_eclipse_data_deck(all_por[0], all_perm[0], deck_file)

    # Comparison plot
    print("\n── Creating comparison plot ──")
    plot_comparison(all_por, all_perm, all_vdp, all_lorenz)

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY OF ALL REALIZATIONS")
    print("=" * 70)
    print(
        f"  {'Real':>5s}  {'Por_mean':>9s}  {'Por_std':>9s}  {'k_P50':>9s}  {'V_DP':>7s}  {'Lorenz':>7s}"
    )
    print(f"  {'':>5s}  {'':>9s}  {'':>9s}  {'(mD)':>9s}  {'':>7s}  {'':>7s}")
    print(
        f"  {'-'*5:>5s}  {'-'*9:>9s}  {'-'*9:>9s}  {'-'*9:>9s}  {'-'*7:>7s}  {'-'*7:>7s}"
    )
    for r in range(n_realizations):
        print(
            f"  {r+1:>5d}  {np.mean(all_por[r]):>9.4f}  {np.std(all_por[r]):>9.4f}  "
            f"{np.percentile(all_perm[r], 50):>9.1f}  {all_vdp[r]:>7.3f}  {all_lorenz[r]:>7.3f}"
        )

    print(f"\n  All output files saved to: {os.path.abspath(output_dir)}/")
    print(f"\n  To run the miscible flood simulation:")
    print(f"    1. Install OPM Flow: https://opm-project.org/")
    print(f"    2. Run:  flow {deck_file}")
    print(f"    3. Visualize with ResInsight: https://resinsight.org/")
    print("=" * 70)


if __name__ == "__main__":
    main()
