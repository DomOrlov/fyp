import os
import ChiantiPy.core as ch  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from matplotlib import gridspec

error_log = []

title = {
    "CaAr": "Ca XIV 193.87 Å / Ar XIV 194.40 Å",
    "SiS": "Si X 258.37 Å / S X 264.23 Å",
}

pair_to_element = {
    ("ca_14", "ar_14"): "CaAr",
    ("si_10", "s_10"):  "SiS"
}

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.labelspacing": 0.4,
    "axes.titlesize": 16
})

unique_pairs = [
    ("ca_14", 193.87, "ar_14", 194.40),
    ("si_10", 258.37, "s_10", 264.23)
]
    
# Process matched pairs and calculate emissivities

T_range = np.logspace(5.8, 7.0, num=120)  # Extend to log(T) = 7 
logT = np.log10(T_range)
electron_densities = [1e8, 1e9, 1e10]
density_labels = {1e8: r"1e8 cm$^{-3}$", 1e9: r"1e9 cm$^{-3}$", 1e10: r"1e10 cm$^{-3}$"}
emissivity_data = {}

def find_line_index(ion_obj, target_wavelength):
    """Find the index of the spectral line closest to the target wavelength."""
    if not hasattr(ion_obj, "Wgfa") or "wvl" not in ion_obj.Wgfa:  # Check if the ion has wavelength data
        print(f"Skipping {ion_obj.Spectroscopic} - No 'wvl' data available")
        error_log.append(f"Skipping {ion_obj.Spectroscopic} - No 'wvl' data available")
        return None  # Skip this ion if no wavelength data exists
    wavelengths = np.asarray(ion_obj.Wgfa.get('wvl', []))  # Retrieve the wavelength array
    if wavelengths.size == 0:  # Check if the array is empty
        print(f"Skipping {ion_obj.Spectroscopic} - Empty wavelength data")
        error_log.append(f"Skipping {ion_obj.Spectroscopic} - Empty wavelength data")
        return None
    return np.argmin(np.abs(wavelengths - target_wavelength))  # Return the index of the closest wavelength

print(f"T_range min: {np.log10(min(T_range)):.2f}, max: {np.log10(max(T_range)):.2f}")
error_log.append(f"T_range min: {np.log10(min(T_range)):.2f}, max: {np.log10(max(T_range)):.2f}")

for low_ion, low_wvl, high_ion, high_wvl in unique_pairs:
    try:
        for ne in electron_densities:
            ne = float(ne) 
            ne_array = np.array([ne], dtype=float)  # Convert to a 1D NumPy array
            try:
                low_ion_obj = ch.ion(low_ion, temperature=np.array([T_range[0]], dtype=float), eDensity=ne_array)
                high_ion_obj = ch.ion(high_ion, temperature=np.array([T_range[0]], dtype=float), eDensity=ne_array)
                # Initialize 'low_index' and 'high_index' before use
                low_index, high_index = None, None
                try:
                    low_index = find_line_index(low_ion_obj, low_wvl)
                    high_index = find_line_index(high_ion_obj, high_wvl)
                except Exception as e:
                    print(f"Error: Exception occurred in find_line_index - {e}")
                    error_log.append(f"Error: Exception occurred in find_line_index - {e}")
                    continue  
                if low_index is None:
                    print(f"Warning: No valid index found for {low_ion} at {low_wvl}Å, skipping...")
                    error_log.append(f"Warning: No valid index found for {low_ion} at {low_wvl}Å, skipping...")
                    continue
                if high_index is None:
                    print(f"Warning: No valid index found for {high_ion} at {high_wvl}Å, skipping...")
                    error_log.append(f"Warning: No valid index found for {high_ion} at {high_wvl}Å, skipping...")
                    continue
                low_emissivities = []
                high_emissivities = []
                for T in T_range:
                    try:
                        low_ion_obj = ch.ion(low_ion, temperature=np.array([T], dtype=float), eDensity=ne_array)
                        high_ion_obj = ch.ion(high_ion, temperature=np.array([T], dtype=float), eDensity=ne_array)
                        low_ion_obj.intensity()
                        high_ion_obj.intensity()
                        try:
                            low_emiss = low_ion_obj.Intensity['intensity'][0, low_index]
                            high_emiss = high_ion_obj.Intensity['intensity'][0, high_index]
                            low_emissivities.append(low_emiss)
                            high_emissivities.append(high_emiss)
                        except IndexError as e:
                            print(f"Error: IndexError when accessing intensity data - {e}")
                            continue
                    except Exception as e:
                        print(f"Skipping temperature {T} for {low_ion} / {high_ion} at ne={ne} - CHIANTI error: {e}")
                        continue
                if len(low_emissivities) > 0 and len(high_emissivities) > 0:
                    emissivity_data[(low_ion, low_wvl, high_ion, high_wvl, ne)] = (np.array(low_emissivities),np.array(high_emissivities))
            except Exception as e:
                print(f"Skipping density {ne} for {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å - CHIANTI error: {e}")
                error_log.append(f"Skipping density {ne} for {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å - CHIANTI error: {e}")
                continue
    except Exception as e:
        print(f"Skipping {low_ion} {low_wvl} Å / {high_ion} {high_wvl} Å due to error: {e}")
        error_log.append(f"Skipping {low_ion} {low_wvl} Å / {high_ion} {high_wvl} Å due to error: {e}")

#--------------------------------------------------------------------------------
#now we finally plot the data



colors_ratio = ['purple', 'green', 'yellow']  # Colors for the ratio curves

def plot_emissivity_ratios(emissivity_data, logT, unique_pairs):
    fig = plt.figure(figsize=(10,12))
    fig.suptitle("Emissivities and ratios vs log T", fontsize=22, y=0.975)
    outer_grid = gridspec.GridSpec(2, 1, wspace=0.125, hspace=0.175)
    for idx, (low_ion, low_wvl, high_ion, high_wvl) in enumerate(unique_pairs):
        ax = fig.add_subplot(outer_grid[idx])
        #plt.title(f"{low_ion.replace('_', ' ')} {low_wvl} & {high_ion.replace('_', ' ')} {high_wvl} emissivities and ratio vs. log T")
        element_key = pair_to_element.get((low_ion, high_ion), f"{low_ion}_{high_ion}")
        low_label, high_label = title[element_key].split(" / ")
        # Strip wavelength (keep only ion name, e.g., "S XI")
        low_label = " ".join(low_label.split()[:2])
        high_label = " ".join(high_label.split()[:2])
        #fig, ax = plt.subplots(figsize=(8, 6))  # Single figure per ion pair
        ax.set_yscale('log')
        if idx == 1:
            ax.set_xlabel('Log T (K)', fontsize=16)
        else:
            ax.set_xlabel('')
        ax.set_ylabel('Emissivities and ratios', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        #ax.set_xlim(6.0, 7.2)  # **Updated to match the good plot**
        ax.set_xlim(5.8, 7.2)
        ax.set_ylim(0.1, 10)
        ax.set_yticks([1e-2, 1e-1, 1e0, 1e1])
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(ticker.NullFormatter())
        # Normalize emissivities at ne = 1e9
        ne_ref = 1e9
        key_ref = (low_ion, low_wvl, high_ion, high_wvl, ne_ref)
        if key_ref in emissivity_data:
            low_emiss, high_emiss = emissivity_data[key_ref]
            valid_indices = np.where((low_emiss > 0) & (high_emiss > 0))  # Only valid points
            low_emiss_valid = low_emiss[valid_indices]
            high_emiss_valid = high_emiss[valid_indices]
            if len(low_emiss_valid) > 0 and len(high_emiss_valid) > 0:
                low_emiss_norm = low_emiss / np.max(low_emiss_valid)
                high_emiss_norm = high_emiss / np.max(high_emiss_valid)
            else:
                print(f"Warning: No valid emissivity data for {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å at ne=1e9")
                error_log.append(f"Warning: No valid emissivity data for {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å at ne=1e9")            
                #ax.plot(logT, low_emiss_norm, 'k-', label=f'{low_ion.upper()} {low_wvl}Å at 1e9')
            #ax.plot(logT, high_emiss_norm, 'k--', label=f'{high_ion.upper()} {high_wvl}Å at 1e9')
            ax.plot(logT, low_emiss_norm, 'k-', label=f'{low_label} at {density_labels[ne_ref]}')
            ax.plot(logT, high_emiss_norm, 'k--', label=f'{high_label} at {density_labels[ne_ref]}')
        # Plot ratios at different densities
        for ne, color, linestyle in zip([1e8, 1e9, 1e10], colors_ratio, [':', '--', '-.']):
            key = (low_ion, low_wvl, high_ion, high_wvl, ne)
            if key in emissivity_data:
                low_emiss, high_emiss = emissivity_data[key]
                low_emiss = np.where(low_emiss > 1e-30, low_emiss, np.nan)
                high_emiss = np.where(high_emiss > 1e-30, high_emiss, np.nan)
                ratio = np.full_like(low_emiss, np.nan)  # Initialize with NaN values
                valid_ratio_indices = (high_emiss > 1e-30)  # Only compute ratio where high_emiss is significant
                ratio[valid_ratio_indices] = low_emiss[valid_ratio_indices] / high_emiss[valid_ratio_indices]
                #ax.plot(logT, ratio, color=color, linestyle=linestyle, label=f'{low_ion.upper()} / {high_ion.upper()} at {int(ne):.0e}')
                ax.plot(logT, ratio, color=color, linestyle=linestyle, label=f'{low_label} / {high_label} at {density_labels[ne]}')
        ax.axhline(1.0, color='gray', linewidth=1)
        ax.legend(loc='best')
        #plt.title(f"{title[element_key]} emissivities and ratio vs log T")
        ax.set_title(f"{title[element_key]}", fontsize=16, fontweight='bold')
        #filename = f"{low_ion.replace('_', ' ')} {low_wvl} & {high_ion.replace('_', ' ')} {high_wvl} emissivities and ratio vs. log T".title()
        #filename = filename.replace(" ", "_").replace(".", "_") + ".png"        
        #plt.savefig(filename, dpi=300)
        #plt.show(block=True)  
        #plt.close(fig)
        #print(f"Saved: {filename}")

    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    plt.savefig("Emissivities_Ratios.png", dpi=150, bbox_inches="tight")
    plt.show()

error_log_path = os.path.join("/home/ug/orlovsd2/fyp", "error_log.txt")

# Write the error log to a file
with open(error_log_path, "w", encoding="utf-8") as log_file:
    log_file.write(f"Log start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 50 + "\n")
    for err in error_log:
        log_file.write(err + "\n")
    log_file.write("\nLog end.\n")

# Run the function
plot_emissivity_ratios(emissivity_data, logT, unique_pairs)

