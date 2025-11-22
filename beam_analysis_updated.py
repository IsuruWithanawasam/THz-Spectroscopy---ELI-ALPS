# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 18:41:17 2025

@author: Asus
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Helper Functions ---

def clean_row(data_row_series):
 
    numeric_data = pd.to_numeric(data_row_series, errors='coerce')
    clean_data = numeric_data.dropna().values
    return clean_data

def gaussian(x, I0, x0, w):
    """
    Gaussian function for fitting.
    I0 = amplitude, x0 = center, w = 1/e^2 radius
    """
    return I0 * np.exp(-2 * ((x - x0)**2) / w**2)

def gaussian_derivative_w(x, I0, x0, w):
    """
    Analytical derivative of the fitting function:
    I(x) = I0 * exp(-2 * (x - x0)**2 / w**2)
    """
    exponent = -2 * ((x - x0)**2) / w**2
    prefactor = (-4 * (x - x0)) / w**2
    return I0 * np.exp(exponent) * prefactor
# --- 2. Main Analysis Functions ---

def load_calibration_data(filepath):
    """
    Loads calibration data from the 'spot.results.csv' file.
    Prints a summary and returns the key values from the first row.
    """
    print("--- 1. Loading Calibration Data (spot.results.csv) ---")
    d_profile = pd.read_csv(filepath, encoding='utf-8-sig')
    
    selected_columns = d_profile[['Peak cnts', 'Centroid X mm', 'Centroid Y mm', 'D4σX mm', 'D4σY mm']]
    print(selected_columns.head().to_markdown(index=False, floatfmt=".3f"))
    
    # Get data from the first row
    data = selected_columns.values[0]
    
    calibration_values = {
        'centroid_x': data[1],
        'd4sigma_x': data[3],
        'centroid_y': data[2],
        'd4sigma_y': data[4]
    }
    
    print(f"\nLoaded X Cal: Centroid={calibration_values['centroid_x']:.3f} mm, D4σ={calibration_values['d4sigma_x']:.3f} mm")
    print(f"Loaded Y Cal: Centroid={calibration_values['centroid_y']:.3f} mm, D4σ={calibration_values['d4sigma_y']:.3f} mm")
    
    return calibration_values

def load_cursor_data(filepath):
    """
    Loads and cleans the horizontal and vertical cursor data.
    """
    print(f"\n--- 2. Loading Cursor Data ({filepath}) ---")
    cursor_data_raw = pd.read_csv(filepath, header=None)
    
    even_rows = cursor_data_raw[0::2]   # rows 0, 2, 4, 6, 8
    odd_rows  = cursor_data_raw[1::2]   # rows 1, 3, 5, 7, 9

    even_avg = np.mean(even_rows, axis=0)
    odd_avg  = np.mean(odd_rows, axis=0)
    
    h_array = clean_row(even_avg)
    v_array = clean_row(odd_avg)
   # h_array = clean_row(cursor_data_raw.iloc[0])
   # v_array = clean_row(cursor_data_raw.iloc[1])
    
    print(f"Cleaned h_array (cursor) has {len(h_array)} data points.")
    print(f"Cleaned v_array (cursor) has {len(v_array)} data points.")
    
    return h_array, v_array

def plot_raw_data(h_array, v_array):
    """
    Generates a plot of the raw cursor data.
    """
    print("\n--- 3. Generating Raw Data Plot ---")
    plt.figure("Raw Cursor Profiles", figsize=(12, 5))
    
    plt.figure()
    plt.plot(h_array,'o')
    plt.title('Horizontal Cursor Data (Raw)')
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity (cnts)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(v_array,'o')
    plt.title('Vertical Cursor Data (Raw)')
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity (cnts)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def fit_and_plot_profile(data_array, cal_centroid, cal_d4sigma, axis_name):
    """
    Performs a Gaussian fit on a data array and plots the
    calibrated result.
    """
    print(f"\n--- 4. Fitting {axis_name} Profile ---")
    
    # --- Fit in Pixels ---
    y_data_to_fit = data_array
    x_data_to_fit = np.arange(len(y_data_to_fit))
    
    I0_guess = np.nanmax(y_data_to_fit)
    x0_guess = np.nanargmax(y_data_to_fit)
    w_guess = 15.0
    
    print(f"Initial guesses: I0={I0_guess:.2f}, x0={x0_guess:.2f}, w={w_guess:.2f}")
    
    popt, pcov = curve_fit(gaussian, x_data_to_fit, y_data_to_fit, p0=[I0_guess, x0_guess, w_guess])
    I0_fit, x0_fit_pixels, w_fit_pixels = popt
    
    print(f"Fitted in pixels: I0={I0_fit:.2f}, x0={x0_fit_pixels:.2f}, w={w_fit_pixels:.2f} pixels")

    # --- Calibrate to mm ---
    mm_per_pixel = cal_d4sigma / (2.0 * w_fit_pixels)
    x_origin_mm = cal_centroid - (x0_fit_pixels * mm_per_pixel)
    
    x_axis_to_plot = x_data_to_fit * mm_per_pixel + x_origin_mm
    
    # Convert fitted parameters to mm for plotting
    x0_fit_mm = (x0_fit_pixels * mm_per_pixel) + x_origin_mm
    w_fit_mm = w_fit_pixels * mm_per_pixel
    
    print(f"Fitted in mm: x0={x0_fit_mm:.3f} mm, w={w_fit_mm:.3f} mm")

    # --- Plot Calibrated Fit ---
    x_new = np.linspace(min(x_axis_to_plot), max(x_axis_to_plot), 1000)
    y_fit = gaussian(x_new, I0_fit, x0_fit_mm, w_fit_mm)
    
    plt.figure(f"Calibrated Fit: {axis_name}", figsize=(10, 6))
    plt.plot(x_axis_to_plot, y_data_to_fit, 'bo', label='Raw Data', markersize=4)
    plt.plot(x_new, y_fit, 'r-', label='Gaussian Fit', linewidth=2)
    
    plt.title(f'Calibrated Gaussian Fit of {axis_name} Cursor Data')
    plt.xlabel(f'{axis_name} Position (mm)')
    plt.ylabel('Intensity (cnts)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    metad=I0_fit, x0_fit_mm, w_fit_mm
    
    return x_new,y_fit,metad



    

# --- 3. Main Script Execution ---

# Define file names
beam_prof_file = 'spot.results.csv'
cursor_file = 'spot.cursor.csv'

# Load all data
calib = load_calibration_data(beam_prof_file)
h_array, v_array = load_cursor_data(cursor_file)

# Plot raw data
plot_raw_data(h_array, v_array)

# Fit and plot Horizontal data
x_h,h_fit,h_met=fit_and_plot_profile(h_array, calib['centroid_x'], calib['d4sigma_x'], "Horizontal")

# Fit and plot Vertical data
x_v,v_fit,v_met=fit_and_plot_profile(v_array, calib['centroid_y'], calib['d4sigma_y'], "Vertical")

I0_fit_h, x0_fit_mm_h, w_fit_mm_h=h_met
I0_fit_v, x0_fit_mm_v, w_fit_mm_v=v_met

dHx_dx = -np.gradient(h_fit, x_h)
dHy_dy = -np.gradient(v_fit, x_v)

# Call using the new function
dGx_dx = gaussian_derivative_w(x_h, *h_met)
dGy_dy = gaussian_derivative_w(x_v, *v_met)

X, Y = np.meshgrid(x_h, x_v)
# Note the '2*' in the exponent, matching your 1D fit function
Z = np.exp(-2 * ((X-x0_fit_mm_h)**2 / w_fit_mm_h**2) + \
           -2 * ((Y-x0_fit_mm_v)**2 / w_fit_mm_v**2))
    
plt.figure(figsize=(6,5))
plt.imshow(Z, extent=[x_h.min(), x_h.max(), x_v.min(), x_v.max()], origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label="Intensity")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian Beam Profile")
plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='inferno', edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Intensity")
ax.set_title("3D Gaussian Beam Profile")
plt.show()


"""
# --- 4. Validation: Plotting Derivatives ---
# This confirms our analytical derivative matches the numerical derivative of the fit

print("\n--- 5. Plotting Derivative Validation ---")

# Calculate the numerical derivative of the fitted curve (no negative sign needed)
num_grad_h = np.gradient(h_fit, x_h) 
num_grad_v = np.gradient(v_fit, x_v)

# Plotting
fig_deriv, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Horizontal Plot ---
ax1.set_title('Horizontal Derivative Check')
ax1.plot(x_h, dGx_dx, 'r-', label='Analytical Derivative (dGx_dx)', linewidth=2)
ax1.plot(x_h, num_grad_h, 'b--', label='Numerical Derivative (np.gradient)', linewidth=2, alpha=0.7)
ax1.set_xlabel('Horizontal Position (mm)')
ax1.set_ylabel('d(Intensity)/dx')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Vertical Plot ---
ax2.set_title('Vertical Derivative Check')
ax2.plot(x_v, dGy_dy, 'r-', label='Analytical Derivative (dGy_dy)', linewidth=2)
ax2.plot(x_v, num_grad_v, 'b--', label='Numerical Derivative (np.gradient)', linewidth=2, alpha=0.7)
ax2.set_xlabel('Vertical Position (mm)')
ax2.set_ylabel('d(Intensity)/dy')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("\nAll analysis complete. Displaying plots...")
# Show all figures at the end
plt.show()
"""