# scatter plot of intensity ratio vs loop lenght (Feature Detection)
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from sunpy.map import Map
import pandas as pd


title = {"CaAr":"Ca XIV 193.87 Å / Ar XIV 194.40 Å","FeS":"Fe XVI 262.98 Å / S XIII 256.69 Å","sis":"Si X 258.37 Å / S X 264.23 Å","sar":"S XI 188.68 Å / Ar XI 188.81 Å"}
PAIR_FOR = {"CaAr": "Ca_Ar", "FeS": "Fe_S", "sis": "Si_S", "sar": "S_Ar"}

elements = ["CaAr", "FeS", "sis", "sar"]
test_mode = False
test_target_ar = "11967"
catalogue = "AR_Catalogue.xlsx"
catalogue_sheet = "AR_Catalogue"
output_dir = "png"
diagnostics_path = os.path.join(output_dir, "scatter_diagnostics.txt")

all_closed_loop_files = sorted(glob.glob("/mnt/scratch/data/orlovsd2/sunpy/data/loop_length/loop_length_map_closed_*.fits"))
df_cat = pd.read_excel(catalogue, sheet_name=catalogue_sheet)
df_cat = df_cat[["ar_id", "date"]].dropna()
df_cat["ar_id"] = df_cat["ar_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

if test_mode:
    ar_list = [str(test_target_ar)]
else:
    ar_list = sorted(df_cat["ar_id"].unique().tolist())

print(f"Total ARs to process: {len(ar_list)}")
with open(diagnostics_path, "w") as fdiag:
    fdiag.write("scatter diagnostics\n\n")



# element_data = {e: {"abund": [], "loop": [], "open_mask": []} for e in elements}
element_data_all = {e: {"abund": [], "loop": [], "open_mask": []} for e in elements}
for ar_id in ar_list:
    df_ar = df_cat[df_cat["ar_id"] == str(ar_id)]
    dates = pd.to_datetime(df_ar["date"], errors="coerce")
    allowed_dates = set(dates.dt.strftime("%Y_%m_%d").dropna().unique())

    closed_loop_files = []
    for p in all_closed_loop_files:
        base = os.path.basename(p)
        datetime_str = "_".join(base.split("_")[4:])[:-5]
        date_part = datetime_str.split("__")[0]
        if date_part in allowed_dates:
            closed_loop_files.append(p)

    print(f"\nAR {ar_id}: loop-length files selected = {len(closed_loop_files)}")
    if len(closed_loop_files) == 0:
        print(f"AR {ar_id}: no loop-length files, skipping.")
        continue

    element_data = {e: {"abund": [], "loop": [], "open_mask": []} for e in elements}


    for closed_path in closed_loop_files:
        # Infer datetime string and open loop path
        basename = os.path.basename(closed_path)
        datetime_str = "_".join(basename.split("_")[4:])[:-5] 
        open_path = closed_path.replace("closed", "open")
        dt_parts = datetime_str.split("__")

        loop_map_closed = Map(closed_path, silence_warnings=True)
        loop_map_open = Map(open_path, silence_warnings=True)

        loop_lengths = np.where(np.isfinite(loop_map_open.data), loop_map_open.data, loop_map_closed.data)

        for element in elements:
            # Build the cleaned ratio filename sitting in intensity_ratio/
            pair_token = PAIR_FOR[element] 
            cleaned_path = f"/mnt/scratch/data/orlovsd2/sunpy/data/intensity_ratio/cleaned_relerr_size_intensity_map_ratio_{datetime_str}_{pair_token}.fits"
        
            if not os.path.exists(cleaned_path):
                print(f"Missing file: {cleaned_path}")
                continue
        
            abundance_map = Map(cleaned_path, silence_warnings=True)
            abundance = abundance_map.data.copy()
        
            if element == "sar":
                abundance = np.clip(abundance, 0, 1.5)
            else:
                abundance = np.clip(abundance, 0, 4)
        
            valid_mask = np.isfinite(abundance) & np.isfinite(loop_lengths)
        
            abund_vals = abundance[valid_mask]
            loop_vals = loop_lengths[valid_mask]
        
            open_flat = np.isfinite(loop_map_open.data).flatten()
            valid_flat = valid_mask.flatten()
            # open_mask_flat = open_flat[valid_flat]
            open_mask_flat = np.isfinite(loop_map_open.data[valid_mask])
        
            # Append values to the dictionary
            element_data[element]["abund"].append(abund_vals)
            element_data[element]["loop"].append(loop_vals)
            element_data[element]["open_mask"].append(open_mask_flat)
            element_data_all[element]["abund"].append(abund_vals)
            element_data_all[element]["loop"].append(loop_vals)
            element_data_all[element]["open_mask"].append(open_mask_flat)


    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(f"AR {ar_id}: FIP bias vs loop length", fontsize=28, y=0.92)
    with open(diagnostics_path, "a") as fdiag:
        fdiag.write(f"AR {ar_id}: FIP bias vs loop length\n\n")

    outer_grid = gridspec.GridSpec(2, 2, wspace=0.125, hspace=0.2)
    order = ["CaAr", "FeS", "sis", "sar"]

    for idx, element in enumerate(order):
        ax = fig.add_subplot(outer_grid[idx])
        
        # If nothing was appended for this element, skip
        if len(element_data[element]["abund"]) == 0:
            print(f"{element}: no data appended. Skipping this panel.")
            ax.set_title(f"{title[element]}\n(no data)")
            ax.axis("off")
            with open(diagnostics_path, "a") as fdiag:
                fdiag.write(f"=== Final Summary for {element} ===\n")
                fdiag.write("No data appended for this AR\n\n")
            continue
        abund_vals = np.concatenate(element_data[element]["abund"])
        loop_vals = np.concatenate(element_data[element]["loop"])
        open_mask_flat = np.concatenate(element_data[element]["open_mask"])
        closed_mask_flat = ~open_mask_flat

        # Need at least 2 points for regression
        if abund_vals.size < 2 or loop_vals.size < 2:
            print(f"{element}: not enough valid points for regression (N={abund_vals.size}). Skipping fits.")
            ax.set_title(f"{title[element]}\n(no valid pixels)")
            ax.axis("off")
            with open(diagnostics_path, "a") as fdiag:
                fdiag.write(f"=== Final Summary for {element} ===\n")
                fdiag.write(f"Not enough valid points (N={abund_vals.size})\n\n")
            continue

        print(f"\n=== Final Summary for {element} ===")
        print("Total abundance NaNs:", np.isnan(abund_vals).sum())
        print("Total loop length NaNs:", np.isnan(loop_vals).sum())
        print("Valid pixels (not NaN in both):", len(abund_vals))
        print("Open fieldline pixels:", np.sum(open_mask_flat))
        print("Closed fieldline pixels:", np.sum(closed_mask_flat))

        with open(diagnostics_path, "a") as fdiag:
            fdiag.write(f"=== Final Summary for {element} ===\n")
            fdiag.write(f"Total abundance NaNs: {np.isnan(abund_vals).sum()}\n")
            fdiag.write(f"Total loop length NaNs: {np.isnan(loop_vals).sum()}\n")
            fdiag.write(f"Valid pixels (not NaN in both): {len(abund_vals)}\n")
            fdiag.write(f"Open fieldline pixels: {np.sum(open_mask_flat)}\n")
            fdiag.write(f"Closed fieldline pixels: {np.sum(closed_mask_flat)}\n\n")

        open_abund = abund_vals[open_mask_flat]
        open_loop = loop_vals[open_mask_flat]
        
        # print(f"{element} open fieldline sample values (abund vs loop):")
        # for i in range(min(10, len(open_abund))):
        #     print(f"  {i}: Abundance = {open_abund[i]:.2f}, Loop Length = {open_loop[i]:.2f}")
        # print(f"  Open points out of Y range (>1.5 or >4): {(open_abund > (1.5 if element == 'sar' else 4)).sum()}")
        # print(f"  Open points out of X range (<1e3 or >3e5): {(open_loop < 1e3).sum()} below, {(open_loop > 3e5).sum()} above")

        # Linear regression
        slope_lin, intercept_lin, r_lin, p_lin, err_lin = linregress(loop_vals, abund_vals)
        x_fit_lin = np.linspace(min(loop_vals), max(loop_vals), 100)
        y_fit_lin = slope_lin * x_fit_lin + intercept_lin
        
        # Log-linear regression
        log_loop = np.log10(loop_vals)
        slope_log, intercept_log, r_log, p_log, err_log = linregress(log_loop, abund_vals)
        x_fit_log = np.linspace(min(log_loop), max(log_loop), 100)
        y_fit_log = slope_log * x_fit_log + intercept_log
        x_fit_log10 = 10 ** x_fit_log

        # Binning
        bin_width = 5000
        max_length = 250000
        bins = np.arange(0, max_length + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_vals = []
        median_vals = []
        p25_vals = []
        p75_vals = []
        center_vals = []

        for i in range(len(bins) - 1):
            bin_mask = (loop_vals >= bins[i]) & (loop_vals < bins[i + 1])
            bin_abund = abund_vals[bin_mask]
            if len(bin_abund) > 2:
                mean_vals.append(np.mean(bin_abund))
                median_vals.append(np.median(bin_abund))
                p25_vals.append(np.percentile(bin_abund, 25))
                p75_vals.append(np.percentile(bin_abund, 75))
                center_vals.append(bin_centers[i])

        center_vals = np.array(center_vals)
        mean_vals = np.array(mean_vals)
        median_vals = np.array(median_vals)
        p25_vals = np.array(p25_vals)
        p75_vals = np.array(p75_vals)

        # Plot
        ax.fill_between(center_vals, p25_vals, p75_vals, color="gray", alpha=0.3, label="25–75th percentile")
        ax.plot(center_vals, mean_vals, color="blue", label="Mean", linewidth=2, alpha=0.5)
        ax.plot(center_vals, median_vals, color="green", linestyle="--", label="Median", linewidth=2, alpha=0.5)
        ax.scatter(loop_vals[closed_mask_flat], abund_vals[closed_mask_flat], s=5, alpha=0.7, color="lightskyblue", label="Closed fieldlines")
        ax.scatter(loop_vals[open_mask_flat], abund_vals[open_mask_flat], s=10, alpha=0.9, color="green", label="Open fieldlines", zorder=5)
        # ax.plot(x_fit_log10, y_fit_log, color='red', label=f'Log Fit: y = {slope_log:.2e}·log₁₀(x) + {intercept_log:.2e}', alpha=1, linewidth=3)
        logfit_label = f'Log Fit: y = {slope_log:.2e}·log₁₀(x) + {intercept_log:.2e}'
        logfit_line, = ax.plot(x_fit_log10, y_fit_log, color='red', alpha=1, linewidth=5)
        ax.set_xlabel("Loop length (km)")
        ax.set_ylabel("Intensity ratio (num/den)")
        ax.set_title(f"{title[element]}", fontsize=20, fontweight="bold")
        ax.grid(True)
        ax.set_xscale("log")
        # plt.ylim(0, 1.5 if element == "sar" else 4)
        ax.set_ylim(0, 4)
        ax.set_xlim(1e3, 3e5)
        # plt.xlim(1e3, 1.5e6) # to see open fieldlines
        # plt.plot(x_fit, y_fit, color='red', linewidth=1, label=f'Best Fit: y = {slope:.2e}·log₁₀(x) + {intercept:.2e}', alpha=0.4)
        # ax.legend()

        # if element == "sar":
        #     main_legend = ax.legend(loc="upper right", fontsize=12.4)
        #     ax.add_artist(main_legend)
        
        # legend_loc = "upper left" if element == "sar" else "lower left"
        # ax.legend([logfit_line], [logfit_label], loc=legend_loc, fontsize=13, frameon=True)

        if element == "sar":
            # Combine log fit + main legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                [logfit_line] + handles[0:5],
                [logfit_label] + labels[0:5],
                loc="upper left",
                fontsize=20,
                frameon=True
            )
        else:
            # Leave all other elements unchanged
            ax.legend([logfit_line], [logfit_label], loc="lower left", fontsize=20, frameon=True, fancybox=True)

    plt.tight_layout()
    outname = os.path.join(output_dir, f"AR{ar_id}_Abundance_length.png")
    plt.savefig(outname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outname}")

# fig = plt.figure(figsize=(24, 18))
# fig.suptitle("All ARs: FIP bias vs loop length", fontsize=28, y=0.92)
# outer_grid = gridspec.GridSpec(2, 2, wspace=0.125, hspace=0.2)
# order = ["CaAr", "FeS", "sis", "sar"]

# for idx, element in enumerate(order):
with open(diagnostics_path, "a") as fdiag:
    fdiag.write("All ARs: FIP bias vs loop length diagnostics\n\n")


    fig = plt.figure(figsize=(24, 18))
    fig.suptitle("All ARs: FIP bias vs loop length", fontsize=28, y=0.92)
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.125, hspace=0.2)
    order = ["CaAr", "FeS", "sis", "sar"]

    for idx, element in enumerate(order):
        ax = fig.add_subplot(outer_grid[idx])
        
        if len(element_data_all[element]["abund"]) == 0:
            print(f"{element}: no ALL-AR data appended. Skipping this panel.")
            ax.set_title(f"{title[element]}\n(no data)")
            ax.axis("off")
            continue

        abund_vals = np.concatenate(element_data_all[element]["abund"])
        loop_vals = np.concatenate(element_data_all[element]["loop"])
        open_mask_flat = np.concatenate(element_data_all[element]["open_mask"])
        closed_mask_flat = ~open_mask_flat

        if abund_vals.size < 2 or loop_vals.size < 2:
            print(f"{element}: not enough ALL-AR points for regression (N={abund_vals.size}). Skipping fits.")
            ax.set_title(f"{title[element]}\n(no valid pixels)")
            ax.axis("off")
            continue

        print(f"\n=== Final Summary for {element} ===")
        print("Total abundance NaNs:", np.isnan(abund_vals).sum())
        print("Total loop length NaNs:", np.isnan(loop_vals).sum())
        print("Valid pixels (not NaN in both):", len(abund_vals))
        print("Open fieldline pixels:", np.sum(open_mask_flat))
        print("Closed fieldline pixels:", np.sum(closed_mask_flat))
        fdiag.write(f"=== Final Summary for {element} ===\n")
        fdiag.write(f"Total abundance NaNs: {np.isnan(abund_vals).sum()}\n")
        fdiag.write(f"Total loop length NaNs: {np.isnan(loop_vals).sum()}\n")
        fdiag.write(f"Valid pixels (not NaN in both): {len(abund_vals)}\n")
        fdiag.write(f"Open fieldline pixels: {np.sum(open_mask_flat)}\n")
        fdiag.write(f"Closed fieldline pixels: {np.sum(closed_mask_flat)}\n\n")
        
        open_abund = abund_vals[open_mask_flat]
        open_loop = loop_vals[open_mask_flat]
        
        # print(f"{element} open fieldline sample values (abund vs loop):")
        # for i in range(min(10, len(open_abund))):
        #     print(f"  {i}: Abundance = {open_abund[i]:.2f}, Loop Length = {open_loop[i]:.2f}")
        # print(f"  Open points out of Y range (>1.5 or >4): {(open_abund > (1.5 if element == 'sar' else 4)).sum()}")
        # print(f"  Open points out of X range (<1e3 or >3e5): {(open_loop < 1e3).sum()} below, {(open_loop > 3e5).sum()} above")

        # Linear regression
        slope_lin, intercept_lin, r_lin, p_lin, err_lin = linregress(loop_vals, abund_vals)
        x_fit_lin = np.linspace(min(loop_vals), max(loop_vals), 100)
        y_fit_lin = slope_lin * x_fit_lin + intercept_lin
        
        # Log-linear regression
        log_loop = np.log10(loop_vals)
        slope_log, intercept_log, r_log, p_log, err_log = linregress(log_loop, abund_vals)
        x_fit_log = np.linspace(min(log_loop), max(log_loop), 100)
        y_fit_log = slope_log * x_fit_log + intercept_log
        x_fit_log10 = 10 ** x_fit_log

        # Binning
        bin_width = 5000
        max_length = 250000
        bins = np.arange(0, max_length + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_vals = []
        median_vals = []
        p25_vals = []
        p75_vals = []
        center_vals = []

        for i in range(len(bins) - 1):
            bin_mask = (loop_vals >= bins[i]) & (loop_vals < bins[i + 1])
            bin_abund = abund_vals[bin_mask]
            if len(bin_abund) > 2:
                mean_vals.append(np.mean(bin_abund))
                median_vals.append(np.median(bin_abund))
                p25_vals.append(np.percentile(bin_abund, 25))
                p75_vals.append(np.percentile(bin_abund, 75))
                center_vals.append(bin_centers[i])

        center_vals = np.array(center_vals)
        mean_vals = np.array(mean_vals)
        median_vals = np.array(median_vals)
        p25_vals = np.array(p25_vals)
        p75_vals = np.array(p75_vals)

        # Plot
        ax.fill_between(center_vals, p25_vals, p75_vals, color="gray", alpha=0.3, label="25–75th percentile")
        ax.plot(center_vals, mean_vals, color="blue", label="Mean", linewidth=2, alpha=0.5)
        ax.plot(center_vals, median_vals, color="green", linestyle="--", label="Median", linewidth=2, alpha=0.5)
        ax.scatter(loop_vals[closed_mask_flat], abund_vals[closed_mask_flat], s=5, alpha=0.7, color="lightskyblue", label="Closed fieldlines")
        ax.scatter(loop_vals[open_mask_flat], abund_vals[open_mask_flat], s=10, alpha=0.9, color="green", label="Open fieldlines", zorder=5)
        # ax.plot(x_fit_log10, y_fit_log, color='red', label=f'Log Fit: y = {slope_log:.2e}·log₁₀(x) + {intercept_log:.2e}', alpha=1, linewidth=3)
        logfit_label = f'Log Fit: y = {slope_log:.2e}·log₁₀(x) + {intercept_log:.2e}'
        logfit_line, = ax.plot(x_fit_log10, y_fit_log, color='red', alpha=1, linewidth=5)
        ax.set_xlabel("Loop length (km)")
        ax.set_ylabel("Intensity ratio (num/den)")
        ax.set_title(f"{title[element]}", fontsize=20, fontweight="bold")
        ax.grid(True)
        ax.set_xscale("log")
        # plt.ylim(0, 1.5 if element == "sar" else 4)
        ax.set_ylim(0, 4)
        ax.set_xlim(1e3, 3e5)
        # plt.xlim(1e3, 1.5e6) # to see open fieldlines
        # plt.plot(x_fit, y_fit, color='red', linewidth=1, label=f'Best Fit: y = {slope:.2e}·log₁₀(x) + {intercept:.2e}', alpha=0.4)
        # ax.legend()

        # if element == "sar":
        #     main_legend = ax.legend(loc="upper right", fontsize=12.4)
        #     ax.add_artist(main_legend)
        
        # legend_loc = "upper left" if element == "sar" else "lower left"
        # ax.legend([logfit_line], [logfit_label], loc=legend_loc, fontsize=13, frameon=True)

        if element == "sar":
            # Combine log fit + main legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                [logfit_line] + handles[0:5],
                [logfit_label] + labels[0:5],
                loc="upper left",
                fontsize=20,
                frameon=True
            )
        else:
            # Leave all other elements unchanged
            ax.legend([logfit_line], [logfit_label], loc="lower left", fontsize=20, frameon=True, fancybox=True)

    plt.tight_layout()
    outname = os.path.join(output_dir, "ARall_Abundance_length.png")
    plt.savefig(outname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ALL-AR plot: {outname}")

