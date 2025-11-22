# THz Beam Profiling and Signal Analysis

This repository contains Python scripts and experimental data for two distinct phases of Terahertz (THz) research:

1. **Beam Analysis:** Characterizing the spatial profile, stability, and shape of the laser/THz beam.
2. **Signal Analysis:** Time-Domain Spectroscopy (TDS) analysis to determine the refractive index and absorption coefficients of waveguide samples.

## 📦 Dependencies

To run the analysis scripts, you need a standard scientific Python environment. You can install the required libraries via pip:

```bash
pip install numpy pandas matplotlib scipy
```

---

## Part 1: Beam Analysis

This module focuses on analyzing the beam spot captured by the profiling hardware. It performs Gaussian fitting to determine the beam center, width ($1/e^2$), and validates the profile using derivative analysis.

### 📂 Files

- **`beam_analysis_updated.py`**: The main script for this section. It loads the CSV data, cleans it, fits a Gaussian function to the beam intensity profile, and plots the results (including a 3D visualization).
- **`spot.cursor.csv`**: Contains the spatial intensity distribution data (the "image" of the beam spot).
- **`spot.results.csv`**: Time-series data recording beam parameters (Centroid X/Y, Total counts, Peak counts) over time, used to check stability.
- **`spot.sums.csv`**: Contains row/column summation data used for background analysis or quick profiling.

### 🚀 Usage

Run the script to visualize the beam profile and calculate fit parameters:

```bash
python beam_analysis_updated.py
```

**Key Outputs:**
- A 3D plot of the Gaussian Beam Profile.
- 2D cross-section fits (Horizontal and Vertical).
- Derivative checks to validate the analytical Gaussian fit against numerical gradients.

---

## Part 2: THz Signal Analysis

This module processes Time-Domain Spectroscopy (TDS) data. It compares a reference signal (Air) against a sample signal (Waveguide) to calculate the complex refractive index and absorption coefficient in the frequency domain via Fast Fourier Transform (FFT).

There are two datasets corresponding to different waveguide lengths (30mm and 85mm).

### 📂 Files

#### Scripts

- **`Analysis30_updated.py`**: Analysis logic specifically calibrated for the **30mm** waveguide data.
- **`Analysis85_updated.py`**: Analysis logic specifically calibrated for the **85mm** waveguide data.

#### Data (30mm Set)

- **`air_wg30_delay_2.txt`**: The reference time-domain signal (Air) for the 30mm setup.
- **`sam_wg30_delay_2.txt`**: The sample time-domain signal (Waveguide) for the 30mm setup.

#### Data (85mm Set)

- **`air_wg85_delay_2.txt`**: The reference time-domain signal (Air) for the 85mm setup.
- **Note:** The corresponding `sam_wg85` file is processed by `Analysis85_updated.py`.

#### Reference Material

- **`ge.txt`**: Reference data for Germanium (Ge), likely used for comparison or calibration of optical properties (Refractive Index vs. Frequency).

### 🚀 Usage

Run the specific script for the waveguide length you are analyzing:

```bash
# For the 30mm sample
python Analysis30_updated.py

# For the 85mm sample
python Analysis85_updated.py
```

**Key Outputs:**
- **Time Domain Plot:** Visualizes the delay between the Reference (Air) and Sample pulses.
- **Refractive Index Plot:** The calculated refractive index ($n$) over the valid THz frequency band (excluding noise at $\omega=0$).
- **Absorption Coefficient:** The calculated absorption ($\alpha$) plotted against the maximum dynamic range/detectable limit.
