# scatter plot of intensity ratio vs loop lenght (Feature Detection)
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from sunpy.map import Map
import pandas as pd
from collections import defaultdict
import matplotlib.colors as mcolors
import eispac

title = {"CaAr":"Ca XIV 193.87 Å / Ar XIV 194.40 Å","FeS":"Fe XVI 262.98 Å / S XIII 256.69 Å","sis":"Si X 258.37 Å / S X 264.23 Å","sar":"S XI 188.68 Å / Ar XI 188.81 Å"}
PAIR_FOR = {"CaAr": "Ca_Ar", "FeS": "Fe_S", "sis": "Si_S", "sar": "S_Ar"}

elements = ["CaAr", "FeS", "sis", "sar"]
test_mode = True 
test_target_ar = "11967"
catalogue = "AR_Catalogue.xlsx"
catalogue_sheet = "AR_Catalogue"
output_dir = "png"
diagnostics_path = os.path.join(output_dir, "scatter_diagnostics.txt")

all_closed_loop_files = sorted(glob.glob("/mnt/scratch/data/orlovsd2/sunpy/data/loop_length/loop_length_map_closed_*.fits"))
all_mean_B_files = sorted(glob.glob("/mnt/scratch/data/orlovsd2/sunpy/data/mean_B/mean_B_map_combined_*.fits"))
all_mean_B_mask_files = sorted(glob.glob("/mnt/scratch/data/orlovsd2/sunpy/data/mean_B/mean_B_open_mask_map_*.fits"))
df_cat = pd.read_excel(catalogue, sheet_name=catalogue_sheet)
df_cat = df_cat[["ar_id", "date"]].dropna()
df_cat["ar_id"] = df_cat["ar_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
binned_stats = {e: {} for e in elements}

if test_mode:
    ar_list = [str(test_target_ar)]
else:
    ar_list = sorted(df_cat["ar_id"].unique().tolist())

print(f"Total ARs to process: {len(ar_list)}")
with open(diagnostics_path, "w") as fdiag:
    fdiag.write("scatter diagnostics\n\n")

show_colorbar_for = {"FeS"}
# element_data = {e: {"abund": [], "loop": [], "open_mask": []} for e in elements}
element_data_all = {e: {"abund": [], "loop": [], "open_mask": []} for e in elements}
element_data_all_B = {e: {"abund": [], "B": [], "open_mask": [], "loop": [], "length": [], "above_mask": [], "below_mask": [], "start_pix": [], 
                        "start_pix_above": [], "start_pix_below": [], "datetime": []} for e in elements}


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

    mean_B_files = []
    for p in all_mean_B_files:
        base = os.path.basename(p)
        datetime_str = "_".join(base.split("_")[4:])[:-5]
        date_part = datetime_str.split("__")[0]
        if date_part in allowed_dates:
            mean_B_files.append(p)

    print(f"\nAR {ar_id}: loop-length files selected = {len(closed_loop_files)}")
    if len(closed_loop_files) == 0:
        print(f"AR {ar_id}: no loop-length files, skipping.")
        continue

    element_data = {e: {"abund": [], "loop": [], "open_mask": []} for e in elements}
    element_data_B = {e: {"abund": [], "B": [], "open_mask": [], "loop": [], "length": [], "above_mask": [], "below_mask": [], "start_pix": [],
                        "start_pix_above": [], "start_pix_below": [], "datetime": []} for e in elements}
    display_map = {e: None for e in elements}
    display_bbox = {e: None for e in elements} 
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
            # Saves the first available cleaned map for this element
            if display_map[element] is None:
                display_map[element] = abundance_map

                # finite = np.isfinite(abundance_map.data)
                # if np.any(finite):
                #     yy, xx = np.where(finite)
                #     pad = 2
                #     x0 = max(int(xx.min()) - pad, 0)
                #     x1 = min(int(xx.max()) + pad, abundance_map.data.shape[1] - 1)
                #     y0 = max(int(yy.min()) - pad, 0)
                #     y1 = min(int(yy.max()) + pad, abundance_map.data.shape[0] - 1)
                #     display_bbox[element] = (x0, x1, y0, y1)

                uncleaned_path = f"/mnt/scratch/data/orlovsd2/sunpy/data/intensity_ratio/intensity_map_ratio_{datetime_str}_{pair_token}.fits"
                uncleaned_map = Map(uncleaned_path, silence_warnings=True)
                finite = np.isfinite(uncleaned_map.data)

                if np.any(finite):
                    yy, xx = np.where(finite)
                    pad = 2
                    ny, nx = uncleaned_map.data.shape
                    x0 = max(int(xx.min()) - pad, 0)
                    x1 = min(int(xx.max()) + pad, nx - 1)
                    y0 = max(int(yy.min()) - pad, 0)
                    y1 = min(int(yy.max()) + pad, ny - 1)
                    display_bbox[element] = (x0, x1, y0, y1)

        
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

    outer_grid = gridspec.GridSpec(2, 4, wspace=0.125, hspace=0.2)
    order = ["CaAr", "FeS", "sis", "sar"]

    # for idx, element in enumerate(order):
    #     ax = fig.add_subplot(outer_grid[idx])
    for idx, element in enumerate(order):
        row = idx // 2
        pair = idx % 2
        map_col = pair * 2
        scat_col = pair * 2 + 1

        # Left panel: cleaned ratio map
        ax_map = None
        if display_map[element] is not None:
            ax_map = fig.add_subplot(outer_grid[row, map_col], projection=display_map[element])
        else:
            ax_map = fig.add_subplot(outer_grid[row, map_col])
            ax_map.axis("off")

        # Right panel: scatter 
        ax = fig.add_subplot(outer_grid[row, scat_col])
        
        # If nothing was appended for this element, skip
        if len(element_data[element]["abund"]) == 0:
            print(f"{element}: no data appended. Skipping this panel.")
            ax.set_title(f"{title[element]}\n(no data)")
            ax.axis("off")
            ax_map.axis("off")
            with open(diagnostics_path, "a") as fdiag:
                fdiag.write(f"=== Final Summary for {element} ===\n")
                fdiag.write("No data appended for this AR\n\n")
            continue
        abund_vals = np.concatenate(element_data[element]["abund"])
        loop_vals = np.concatenate(element_data[element]["loop"])
        open_mask_flat = np.concatenate(element_data[element]["open_mask"])
        closed_mask_flat = ~open_mask_flat
        # Plot the representative cleaned ratio map on the left
        if display_map[element] is not None:
            vmax_map = 1.5 if element == "sar" else 4.0
            display_map[element].plot(axes=ax_map, vmin=0, vmax=vmax_map, cmap="viridis")
            if display_bbox[element] is not None:
                x0, x1, y0, y1 = display_bbox[element]
                ax_map.set_xlim(x0, x1)
                ax_map.set_ylim(y0, y1)
                ax_map.set_autoscale_on(False)
            ax_map.set_title("Cleaned ratio map", fontsize=14)
            ax_map.coords[1].set_ticklabel_visible(False)

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

    # plt.tight_layout()
    outname = os.path.join(output_dir, f"AR{ar_id}_Abundance_length_with_fip.png")
    # plt.savefig(outname, dpi=150, bbox_inches="tight")
    plt.savefig(outname, dpi=150)
    plt.close(fig)
    print(f"Saved: {outname}")

    for B_path in mean_B_files:
        # # cleaned_intensity_map_ratio_<YYYY>_<MM>_<DD>__<HH>_<MM>_<SS>_Ca_Ar.fits
        # base = os.path.basename(ab_path)
        # parts = base.split("_")
        # # join the datetime pieces back together: 2014_02_01__10_50_35
        # datetime_str = "_".join(parts[4:11])   # -> 2014_02_01__10_50_35
        # datetime_str = os.path.basename(ab_path).split("cleaned_relerr_intensity_map_ratio_")[1].split("_Ca_Ar.fits")[0]
        basename = os.path.basename(B_path)
        datetime_str = "_".join(basename.split("_")[4:])[:-5]

        # B_path = f"/mnt/scratch/data/orlovsd2/sunpy/data/mean_B/mean_B_map_combined_{datetime_str}.fits"
        mask_path = f"/mnt/scratch/data/orlovsd2/sunpy/data/mean_B/mean_B_open_mask_map_{datetime_str}.fits"
        loop_path_closed = f"/mnt/scratch/data/orlovsd2/sunpy/data/loop_length/loop_length_map_closed_{datetime_str}.fits"


        B_map = Map(B_path, silence_warnings=True) # Extracts actual vaues to get a Numpy array.
        mask_map = Map(mask_path, silence_warnings=True)
        loop_map_closed = Map(loop_path_closed, silence_warnings=True)
        B_vals = B_map.data.copy()
        # open_mask = mask_map.data.copy()
        open_mask = mask_map.data.copy().astype(bool)

        loop_length = loop_map_closed.data.copy()

        for element in elements:
            pair_token    = PAIR_FOR[element] 
            abundance_path = f"/mnt/scratch/data/orlovsd2/sunpy/data/intensity_ratio/cleaned_relerr_size_intensity_map_ratio_{datetime_str}_{pair_token}.fits"

            if not os.path.exists(abundance_path):
                print(f"Missing file: {abundance_path}")
                continue

            abundance_map = Map(abundance_path, silence_warnings=True)
            abundance = abundance_map.data.copy()

            if element == "sar":
                abundance = np.clip(abundance, 0, 1.5)
                # abundance = np.clip(abundance, 0, 4)
            else:
                abundance = np.clip(abundance, 0, 4)
            valid_mask = np.isfinite(abundance) & np.isfinite(B_vals)
            if valid_mask.sum() < 2:
                print(f"Skipping dt={datetime_str} elem={element}: finite(A&B)={valid_mask.sum()}")
                continue

            # if element in ["CaAr", "FeS"]:
            # maps_per_raster[datetime_str][element] = abundance_map
            
            # Original masking
            abund_vals = abundance[valid_mask] # This makes it 1D
            B_strength = B_vals[valid_mask]
            loop_length_valid = loop_length[valid_mask]
            # open_flat = open_mask.flatten()[valid_mask.flatten()]  # open = 1, closed = 0
            open_flat = open_mask[valid_mask]

            # Aditional Masking filter
            valid_range_mask = B_strength > 1
            if valid_range_mask.sum() < 2:
                print(f"Skipping dt={datetime_str} elem={element}: after(B>1)={valid_range_mask.sum()}")
                continue
            abund_vals = abund_vals[valid_range_mask]
            B_strength = B_strength[valid_range_mask]
            loop_length_valid = loop_length_valid[valid_range_mask]
            open_flat = open_flat[valid_range_mask]
            
            # closed_loop_vals = loop_length_valid[~open_flat.astype(bool)] # Extracts loop lengths for closed fieldlines only. open_flat.astype(bool) : [1, 0, 1] into [True, False, True]
            closed_loop_vals = loop_length_valid[~open_flat]

            # element_data[element]["abund"].append(abund_vals) # Dictionary of all valid abundances
            # element_data[element]["B"].append(B_strength)
            # element_data[element]["open_mask"].append(open_flat)
            # element_data[element]["length"].append(closed_loop_vals)
            element_data_B[element]["abund"].append(abund_vals)
            element_data_B[element]["B"].append(B_strength)
            element_data_B[element]["open_mask"].append(open_flat)
            element_data_B[element]["loop"].append(closed_loop_vals)

            element_data_all_B[element]["abund"].append(abund_vals)
            element_data_all_B[element]["B"].append(B_strength)
            element_data_all_B[element]["open_mask"].append(open_flat)
            element_data_all_B[element]["loop"].append(closed_loop_vals)

            # if element in ["CaAr", "FeS"]:
            # Extract fieldline starting pixel positions from map coordinates
            # Create a grid of (x, y) pixel coordinates for the whole map
            ny, nx = abundance.shape
            y_grid, x_grid = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
            pix_coords = np.stack((x_grid, y_grid), axis=-1).reshape(-1, 2)  # (N, 2) in (x, y) format
        
            # Only keep valid pixels based on previous mask
            start_pix = pix_coords[valid_mask.flatten()][valid_range_mask]
        
            # Save to element_data
            element_data_B[element]["start_pix"].append(start_pix)
            
            element_data_B[element]["datetime"].append(datetime_str)
            # Here we seperate data sets based on std below and above the log-linear regression (original single) (PER-RASTER)
            # Full-range log-linear regression (needed for residuals)
            log_B = np.log10(B_strength)
            slope_log, intercept_log, *_ = linregress(log_B, abund_vals)
        
            fit_line = slope_log * log_B + intercept_log  # The fitted abundance values
            residuals = abund_vals - fit_line  # Vertical difference from the fit line
            std_resid = np.std(residuals)  # Spread of the residuals
            
            # Masks for classification
            above_mask = residuals > std_resid   # Points > 1σ above fit line
            below_mask = residuals < -std_resid  # Points < 1σ below fit line
        
            # Save masks for later use
            element_data_B[element]["above_mask"].append(above_mask)
            element_data_B[element]["below_mask"].append(below_mask)
            element_data_B[element]["start_pix_above"].append(start_pix[above_mask])
            element_data_B[element]["start_pix_below"].append(start_pix[below_mask])

            element_data_all_B[element]["start_pix"].append(start_pix)
            element_data_all_B[element]["datetime"].append(datetime_str)
            element_data_all_B[element]["above_mask"].append(above_mask)
            element_data_all_B[element]["below_mask"].append(below_mask)
            element_data_all_B[element]["start_pix_above"].append(start_pix[above_mask])
            element_data_all_B[element]["start_pix_below"].append(start_pix[below_mask])

            # element_data[element]["datetime_above"].append([datetime_str] * np.sum(above_mask))
            # element_data[element]["datetime_below"].append([datetime_str] * np.sum(below_mask))


            # print(f"{element} Raster {datetime_str}")
            # print(f"  Valid points: {len(abund_vals)}")
            # print(f"  +1σ count: {np.sum(above_mask)}")
            # print(f"  –1σ count: {np.sum(below_mask)}")
        
    # Plotting
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(f"AR {ar_id}: FIP bias vs mean magnetic field strength", fontsize=28, y=0.92)
    with open(diagnostics_path, "a") as fdiag:
        fdiag.write(f"AR {ar_id}: FIP bias vs mean magnetic field strength\n\n")
    outer_grid = gridspec.GridSpec(2, 4, wspace=0.125, hspace=0.2)
    order = ["CaAr", "FeS", "sis", "sar"]

    # for idx, element in enumerate(order):
    #     inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[24, 1], wspace=0.05,
    #                                                 subplot_spec=outer_grid[idx])
    #     ax = fig.add_subplot(inner_grid[0])
    #     cax = fig.add_subplot(inner_grid[1]) if element in show_colorbar_for else None
    for idx, element in enumerate(order):
        row = idx // 2
        pair = idx % 2
        map_col = pair * 2
        scat_col = pair * 2 + 1

        # Left panel: cleaned ratio map
        ax_map = None
        if display_map[element] is not None:
            ax_map = fig.add_subplot(outer_grid[row, map_col], projection=display_map[element])
        else:
            ax_map = fig.add_subplot(outer_grid[row, map_col])
            ax_map.axis("off")

        # Right panel: scatter + optional colorbar
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            1, 2, width_ratios=[24, 1], wspace=0.05,
            subplot_spec=outer_grid[row, scat_col]
        )
        ax = fig.add_subplot(inner_grid[0])
        cax = fig.add_subplot(inner_grid[1]) if element in show_colorbar_for else None


        if len(element_data_B[element]["abund"]) == 0:
            ax_map.axis("off")
            ax.set_title(f"{title[element]}\n(no data)")
            ax.axis("off")
            continue
        abund_vals = np.concatenate(element_data_B[element]["abund"])
        B_vals = np.concatenate(element_data_B[element]["B"])
        open_mask_flat = np.concatenate(element_data_B[element]["open_mask"]).astype(bool)
        loop_length_vals = np.concatenate(element_data_B[element]["loop"])
        closed_mask_flat = ~open_mask_flat # Tells Boolean NOT to identify closed fieldlines.
        # start_pix = np.concatenate(element_data[element]["start_pix"])
        # datetimes = np.concatenate([
        #     [dt] * len(abund)  # Repeat this datetime for every pixel in this raster
        #     for dt, abund in zip(element_data[element]["datetime"], element_data[element]["abund"])
        # ])
        # Plot the representative cleaned ratio map on the left
        if display_map[element] is not None:
            # For the B-figure, keep the same visual scale across elements
            display_map[element].plot(axes=ax_map, vmin=0, vmax=4.0, cmap="viridis")
            if display_bbox[element] is not None:
                x0, x1, y0, y1 = display_bbox[element]
                ax_map.set_xlim(x0, x1)
                ax_map.set_ylim(y0, y1)
                ax_map.set_autoscale_on(False)
            ax_map.set_title("Cleaned ratio map", fontsize=14)
            ax_map.coords[1].set_ticklabel_visible(False)
        
        print(f"\n=== Final Summary for {element} ===")
        print("Total abundance NaNs:", np.isnan(abund_vals).sum())
        print("Total magnetic field NaNs:", np.isnan(B_vals).sum())
        print("Valid pixels (not NaN in both):", len(abund_vals))
        print("Open fieldline pixels:", np.sum(open_mask_flat))
        print("Closed fieldline pixels:", np.sum(closed_mask_flat))
        print(f"B-field range: {B_vals.min():.2f} to {B_vals.max():.2f}")
        print(f"B-field mean: {B_vals.mean():.2f}, median: {np.median(B_vals):.2f}")

        # Split data at {split_gauss} Gauss
        split_gauss = 150
        mask_low = (B_vals < split_gauss)
        mask_high = (B_vals >= split_gauss)

        n_low = np.sum(mask_low)
        n_high = np.sum(mask_high)

        have_low = n_low >= 2
        have_high = n_high >= 2
        
        # # Linear Regression (2 segments) 
        # slope_lin_low, intercept_lin_low, *_ = linregress(B_vals[mask_low], abund_vals[mask_low])
        # slope_lin_high, intercept_lin_high, *_ = linregress(B_vals[mask_high], abund_vals[mask_high])
        
        # x_fit_lin_low = np.linspace(B_vals[mask_low].min(), split_gauss, 100)
        # x_fit_lin_high = np.linspace(split_gauss, B_vals[mask_high].max(), 100)
        
        # y_fit_lin_low = slope_lin_low * x_fit_lin_low + intercept_lin_low
        # y_fit_lin_high = slope_lin_high * x_fit_lin_high + intercept_lin_high
        
        # # Log-Linear Regression (2 segments) 
        # log_B_low = np.log10(B_vals[mask_low])
        # log_B_high = np.log10(B_vals[mask_high])
        
        # slope_log_low, intercept_log_low, *_ = linregress(log_B_low, abund_vals[mask_low])
        # slope_log_high, intercept_log_high, *_ = linregress(log_B_high, abund_vals[mask_high])
        
        # x_fit_log10_low = np.logspace(np.log10(B_vals[mask_low].min()), np.log10(split_gauss), 100)
        # x_fit_log10_high = np.logspace(np.log10(split_gauss), np.log10(B_vals[mask_high].max()), 100)
        
        # y_fit_log_low = slope_log_low * np.log10(x_fit_log10_low) + intercept_log_low
        # y_fit_log_high = slope_log_high * np.log10(x_fit_log10_high) + intercept_log_high

        slope_lin_low = intercept_lin_low = None
        slope_lin_high = intercept_lin_high = None
        slope_log_low = intercept_log_low = None
        slope_log_high = intercept_log_high = None

        x_fit_log10_low = y_fit_log_low = None
        x_fit_log10_high = y_fit_log_high = None

        # Low side (< split_gauss)
        if have_low:
            # Linear (not plotted, but safe if you want it later)
            slope_lin_low, intercept_lin_low, *_ = linregress(B_vals[mask_low], abund_vals[mask_low])

            # Log-linear
            log_B_low = np.log10(B_vals[mask_low])
            slope_log_low, intercept_log_low, *_ = linregress(log_B_low, abund_vals[mask_low])

            x_fit_log10_low = np.logspace(np.log10(B_vals[mask_low].min()), np.log10(split_gauss), 100)
            y_fit_log_low = slope_log_low * np.log10(x_fit_log10_low) + intercept_log_low

        # High side (>= split_gauss)
        if have_high:
            slope_lin_high, intercept_lin_high, *_ = linregress(B_vals[mask_high], abund_vals[mask_high])

            log_B_high = np.log10(B_vals[mask_high])
            slope_log_high, intercept_log_high, *_ = linregress(log_B_high, abund_vals[mask_high])

            x_fit_log10_high = np.logspace(np.log10(split_gauss), np.log10(B_vals[mask_high].max()), 100)
            y_fit_log_high = slope_log_high * np.log10(x_fit_log10_high) + intercept_log_high

        # Binning
        bin_width = 20
        max_B = 2500
        bins = np.arange(0, max_B + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_vals, median_vals, p25_vals, p75_vals, center_vals = [], [], [], [], []

        for i in range(len(bins) - 1):
            bin_mask = (B_vals >= bins[i]) & (B_vals < bins[i + 1])
            bin_abund = abund_vals[bin_mask]
            if len(bin_abund) > 5:
                # print(f"Bin {bins[i]:.1f}–{bins[i+1]:.1f} G: {len(bin_abund)} points")
                mean_vals.append(np.mean(bin_abund))
                median_vals.append(np.median(bin_abund))
                p25_vals.append(np.percentile(bin_abund, 25))
                p75_vals.append(np.percentile(bin_abund, 75))
                center_vals.append(bin_centers[i])

        # Store binned results
        binned_stats[element]["bin_centers"] = np.array(center_vals)
        binned_stats[element]["mean"] = np.array(mean_vals)
        binned_stats[element]["median"] = np.array(median_vals)
        binned_stats[element]["p25"] = np.array(p25_vals)
        binned_stats[element]["p75"] = np.array(p75_vals)


        def truncate_colormap(cmap, minval=0, maxval=0.9, n=256):
            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                cmap(np.linspace(minval, maxval, n))
            )
            return new_cmap
        
        # Get truncated colormap
        original_cmap = plt.cm.cividis
        cropped_cmap = truncate_colormap(original_cmap, 0, 0.9)

        # plt.figure(figsize=(8, 6))

        # fig = plt.figure(figsize=(8, 6))

        ax.fill_between(center_vals, p25_vals, p75_vals, color="gray", alpha=0.3, label="25–75th percentile")
        ax.plot(center_vals, mean_vals, color="blue", label="Mean", linewidth=3, alpha=1, zorder=4, linestyle="--")
        ax.plot(center_vals, median_vals, color="#006400", label="Median", linewidth=3, alpha=1, zorder=4, linestyle="--")
        # plt.scatter(B_vals[closed_mask_flat], abund_vals[closed_mask_flat], s=3, alpha=0.5, color="lightskyblue", label="Closed")
        sc = ax.scatter(B_vals[closed_mask_flat], abund_vals[closed_mask_flat], s=4, alpha=0.4, c=loop_length_vals, cmap=cropped_cmap, label="Closed Fieldlines", zorder=2, vmin = 0, vmax = 0.1e6)
        if cax is not None:
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label("Loop length (km)")

            
        # ax.scatter(B_vals[open_mask_flat], abund_vals[open_mask_flat], s=15, alpha=1, color="#006400", label="Open")

        
        # plt.plot(x_fit_log10, y_fit_log, color='red', linewidth=1, label=f'Log Fit: y = {slope_log:.2e}·log₁₀(x) + {intercept_log:.2e}', alpha=0.8, zorder=4)
        # ax.plot(x_fit_log10_low, y_fit_log_low, color='red', linestyle='--',
        #          label=f'Fit < {split_gauss}G: y={slope_log_low:.2e}·log₁₀(x)+{intercept_log_low:.2f}', alpha=1, zorder=4, linewidth=2.3)
        # ax.plot(x_fit_log10_high, y_fit_log_high, color='red', linestyle='-',
        #          label=f'Fit ≥ {split_gauss}G: y={slope_log_high:.2e}·log₁₀(x)+{intercept_log_high:.2f}', alpha=1, zorder=4, linewidth=3)
        # ax.plot(x_fit_log10_low, y_fit_log_low, color='red', linestyle='--',
        #          alpha=1, zorder=4, linewidth=2.3)
        # ax.plot(x_fit_log10_high, y_fit_log_high, color='red', linestyle='-',
        #          alpha=1, zorder=4, linewidth=3)
        ax.set_xscale("log")
        ax.set_xlabel("Mean magnetic field strength (G)")
        ax.set_ylabel("Intensity ratio (num/den)")
        # ax.set_title(f"{title[element]} : abundance vs mean magnetic field strength", fontsize = 11)
        ax.set_title(f"{title[element]}", fontsize=20, fontweight="bold")
        ax.grid(True)
        # plt.ylim(0, 1.5 if element == "sar" else 4)
        ax.set_ylim(0, 4)
        ax.set_xlim(5, 3000)
        # ax.legend()
        # Only show full legend once
        # if element == "sar":
        #     main_legend = ax.legend(loc="upper right", fontsize=20)
        #     ax.add_artist(main_legend)
        
        # # Always show log fit legend
        # logfit_label_low = f'Fit < {split_gauss}G: y={slope_log_low:.2e}·log_10(x)+{intercept_log_low:.2f}'
        # logfit_label_high = f'Fit => {split_gauss}G: y={slope_log_high:.2e}·log_10(x)+{intercept_log_high:.2f}'
        # logfit_line_low = ax.plot(x_fit_log10_low, y_fit_log_low, color='red', linestyle='--',
        #                         alpha=1, zorder=4, linewidth=5)[0]
        # logfit_line_high = ax.plot(x_fit_log10_high, y_fit_log_high, color='red', linestyle='-',
        #                         alpha=1, zorder=4, linewidth=5)[0]
        
        # # legend_loc = "upper left" if element == "sar" else "lower left"
        # # ax.legend([logfit_line_low, logfit_line_high],
        # #           [logfit_label_low, logfit_label_high],
        # #           loc=legend_loc, fontsize=20, frameon=True)
        # if element == "sar":
        #     # Combine log fit + main legend in one box
        #     handles, labels = ax.get_legend_handles_labels()
        #     ax.legend(
        #         [logfit_line_low, logfit_line_high] + handles[0:5],
        #         [logfit_label_low, logfit_label_high] + labels[0:5],
        #         loc="upper left",
        #         fontsize=20,
        #         frameon=True,
        #         fancybox=True
        #     )
        # else:
        #     # Only show log fit lines for other elements
        #     ax.legend([logfit_line_low, logfit_line_high],
        #             [logfit_label_low, logfit_label_high],
        #             loc="lower left",
        #             fontsize=20,
        #             frameon=True)
        fit_lines = []
        fit_labels = []

        if have_low:
            logfit_label_low = f'Fit < {split_gauss}G: y={slope_log_low:.2e}·log_10(x)+{intercept_log_low:.2f}'
            logfit_line_low = ax.plot(
                x_fit_log10_low, y_fit_log_low,
                color='red', linestyle='--', alpha=1, zorder=4, linewidth=5
            )[0]
            fit_lines.append(logfit_line_low)
            fit_labels.append(logfit_label_low)

        if have_high:
            logfit_label_high = f'Fit => {split_gauss}G: y={slope_log_high:.2e}·log_10(x)+{intercept_log_high:.2f}'
            logfit_line_high = ax.plot(
                x_fit_log10_high, y_fit_log_high,
                color='red', linestyle='-', alpha=1, zorder=4, linewidth=5
            )[0]
            fit_lines.append(logfit_line_high)
            fit_labels.append(logfit_label_high)
        if element == "sar":
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                fit_lines + handles[0:5],
                fit_labels + labels[0:5],
                loc="upper left",
                fontsize=20,
                frameon=True,
                fancybox=True
            )
        else:
            # Only show fit legend if at least one fit exists
            if len(fit_lines) > 0:
                ax.legend(
                    fit_lines, fit_labels,
                    loc="lower left",
                    fontsize=20,
                    frameon=True
                )

            
        valid_pixels_low = np.sum(mask_low & closed_mask_flat)
        valid_pixels_high = np.sum(mask_high & closed_mask_flat)
        print(f"[{element}] Valid closed pixels: < {split_gauss} G = {valid_pixels_low}, >= {split_gauss} G = {valid_pixels_high}")


    outname = os.path.join(output_dir, f"AR{ar_id}_Abundance_B_with_fip.png")
    # plt.savefig(outname, dpi=150, bbox_inches="tight")
    plt.savefig(outname, dpi=150)
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

    # plt.tight_layout()
    outname = os.path.join(output_dir, "ARall_Abundance_length_with_fip.png")
    # plt.savefig(outname, dpi=150, bbox_inches="tight")
    # plt.savefig(outname, dpi=150)
    plt.close(fig)
    print(f"Saved ALL-AR plot: {outname}")


with open(diagnostics_path, "a") as fdiag:
    fdiag.write("All ARs: FIP bias vs mean magnetic field strength diagnostics\n\n")

    # Plotting
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle("All ARs: FIP bias vs mean magnetic field strength", fontsize=28, y=0.92)
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.08, hspace=0.2)
    order = ["CaAr", "FeS", "sis", "sar"]

    for idx, element in enumerate(order):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[24, 1], wspace=0.05,
                                                    subplot_spec=outer_grid[idx])
        ax = fig.add_subplot(inner_grid[0])
        cax = fig.add_subplot(inner_grid[1]) if element in show_colorbar_for else None


        if len(element_data_all_B[element]["abund"]) == 0:
            ax.set_title(f"{title[element]}\n(no data)")
            ax.axis("off")
            continue
        abund_vals = np.concatenate(element_data_all_B[element]["abund"])
        B_vals = np.concatenate(element_data_all_B[element]["B"])
        open_mask_flat = np.concatenate(element_data_all_B[element]["open_mask"]).astype(bool)
        loop_length_vals = np.concatenate(element_data_all_B[element]["loop"])
        closed_mask_flat = ~open_mask_flat # Tells Boolean NOT to identify closed fieldlines.
        # start_pix = np.concatenate(element_data[element]["start_pix"])
        # datetimes = np.concatenate([
        #     [dt] * len(abund)  # Repeat this datetime for every pixel in this raster
        #     for dt, abund in zip(element_data[element]["datetime"], element_data[element]["abund"])
        # ])
        
        print(f"\n=== Final Summary for {element} ===")
        print("Total abundance NaNs:", np.isnan(abund_vals).sum())
        print("Total magnetic field NaNs:", np.isnan(B_vals).sum())
        print("Valid pixels (not NaN in both):", len(abund_vals))
        print("Open fieldline pixels:", np.sum(open_mask_flat))
        print("Closed fieldline pixels:", np.sum(closed_mask_flat))
        print(f"B-field range: {B_vals.min():.2f} to {B_vals.max():.2f}")
        print(f"B-field mean: {B_vals.mean():.2f}, median: {np.median(B_vals):.2f}")

        fdiag.write(f"=== Final Summary for {element} ===\n")
        fdiag.write(f"Total abundance NaNs: {np.isnan(abund_vals).sum()}\n")
        fdiag.write(f"Total magnetic field NaNs: {np.isnan(B_vals).sum()}\n")
        fdiag.write(f"Valid pixels (not NaN in both): {len(abund_vals)}\n")
        fdiag.write(f"Open fieldline pixels: {np.sum(open_mask_flat)}\n")
        fdiag.write(f"Closed fieldline pixels: {np.sum(closed_mask_flat)}\n")
        fdiag.write(f"B-field range: {B_vals.min():.2f} to {B_vals.max():.2f}\n")
        fdiag.write(f"B-field mean: {B_vals.mean():.2f}, median: {np.median(B_vals):.2f}\n")
        fdiag.write("\n")


        # Split data at {split_gauss} Gauss
        split_gauss = 150
        mask_low = (B_vals < split_gauss)
        mask_high = (B_vals >= split_gauss)

        n_low = np.sum(mask_low)
        n_high = np.sum(mask_high)

        have_low = n_low >= 2
        have_high = n_high >= 2
        
        # # Linear Regression (2 segments) 
        # slope_lin_low, intercept_lin_low, *_ = linregress(B_vals[mask_low], abund_vals[mask_low])
        # slope_lin_high, intercept_lin_high, *_ = linregress(B_vals[mask_high], abund_vals[mask_high])
        
        # x_fit_lin_low = np.linspace(B_vals[mask_low].min(), split_gauss, 100)
        # x_fit_lin_high = np.linspace(split_gauss, B_vals[mask_high].max(), 100)
        
        # y_fit_lin_low = slope_lin_low * x_fit_lin_low + intercept_lin_low
        # y_fit_lin_high = slope_lin_high * x_fit_lin_high + intercept_lin_high
        
        # # Log-Linear Regression (2 segments) 
        # log_B_low = np.log10(B_vals[mask_low])
        # log_B_high = np.log10(B_vals[mask_high])
        
        # slope_log_low, intercept_log_low, *_ = linregress(log_B_low, abund_vals[mask_low])
        # slope_log_high, intercept_log_high, *_ = linregress(log_B_high, abund_vals[mask_high])
        
        # x_fit_log10_low = np.logspace(np.log10(B_vals[mask_low].min()), np.log10(split_gauss), 100)
        # x_fit_log10_high = np.logspace(np.log10(split_gauss), np.log10(B_vals[mask_high].max()), 100)
        
        # y_fit_log_low = slope_log_low * np.log10(x_fit_log10_low) + intercept_log_low
        # y_fit_log_high = slope_log_high * np.log10(x_fit_log10_high) + intercept_log_high

        slope_lin_low = intercept_lin_low = None
        slope_lin_high = intercept_lin_high = None
        slope_log_low = intercept_log_low = None
        slope_log_high = intercept_log_high = None

        x_fit_log10_low = y_fit_log_low = None
        x_fit_log10_high = y_fit_log_high = None

        # Low side (< split_gauss)
        if have_low:
            # Linear (not plotted, but safe if you want it later)
            slope_lin_low, intercept_lin_low, *_ = linregress(B_vals[mask_low], abund_vals[mask_low])

            # Log-linear
            log_B_low = np.log10(B_vals[mask_low])
            slope_log_low, intercept_log_low, *_ = linregress(log_B_low, abund_vals[mask_low])

            x_fit_log10_low = np.logspace(np.log10(B_vals[mask_low].min()), np.log10(split_gauss), 100)
            y_fit_log_low = slope_log_low * np.log10(x_fit_log10_low) + intercept_log_low

        # High side (>= split_gauss)
        if have_high:
            slope_lin_high, intercept_lin_high, *_ = linregress(B_vals[mask_high], abund_vals[mask_high])

            log_B_high = np.log10(B_vals[mask_high])
            slope_log_high, intercept_log_high, *_ = linregress(log_B_high, abund_vals[mask_high])

            x_fit_log10_high = np.logspace(np.log10(split_gauss), np.log10(B_vals[mask_high].max()), 100)
            y_fit_log_high = slope_log_high * np.log10(x_fit_log10_high) + intercept_log_high


        # Binning
        bin_width = 20
        max_B = 2500
        bins = np.arange(0, max_B + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_vals, median_vals, p25_vals, p75_vals, center_vals = [], [], [], [], []

        for i in range(len(bins) - 1):
            bin_mask = (B_vals >= bins[i]) & (B_vals < bins[i + 1])
            bin_abund = abund_vals[bin_mask]
            if len(bin_abund) > 5:
                # print(f"Bin {bins[i]:.1f}–{bins[i+1]:.1f} G: {len(bin_abund)} points")
                mean_vals.append(np.mean(bin_abund))
                median_vals.append(np.median(bin_abund))
                p25_vals.append(np.percentile(bin_abund, 25))
                p75_vals.append(np.percentile(bin_abund, 75))
                center_vals.append(bin_centers[i])

        # Store binned results
        binned_stats[element]["bin_centers"] = np.array(center_vals)
        binned_stats[element]["mean"] = np.array(mean_vals)
        binned_stats[element]["median"] = np.array(median_vals)
        binned_stats[element]["p25"] = np.array(p25_vals)
        binned_stats[element]["p75"] = np.array(p75_vals)


        def truncate_colormap(cmap, minval=0, maxval=0.9, n=256):
            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                cmap(np.linspace(minval, maxval, n))
            )
            return new_cmap
        
        # Get truncated colormap
        original_cmap = plt.cm.cividis
        cropped_cmap = truncate_colormap(original_cmap, 0, 0.9)

        # plt.figure(figsize=(8, 6))

        # fig = plt.figure(figsize=(8, 6))

        ax.fill_between(center_vals, p25_vals, p75_vals, color="gray", alpha=0.3, label="25–75th percentile")
        ax.plot(center_vals, mean_vals, color="blue", label="Mean", linewidth=3, alpha=1, zorder=4, linestyle="--")
        ax.plot(center_vals, median_vals, color="#006400", label="Median", linewidth=3, alpha=1, zorder=4, linestyle="--")
        # plt.scatter(B_vals[closed_mask_flat], abund_vals[closed_mask_flat], s=3, alpha=0.5, color="lightskyblue", label="Closed")
        sc = ax.scatter(B_vals[closed_mask_flat], abund_vals[closed_mask_flat], s=4, alpha=0.4, c=loop_length_vals, cmap=cropped_cmap, label="Closed Fieldlines", zorder=2, vmin = 0, vmax = 0.1e6)
        if cax is not None:
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label("Loop length (km)")

            
        # ax.scatter(B_vals[open_mask_flat], abund_vals[open_mask_flat], s=15, alpha=1, color="#006400", label="Open")

        
        # plt.plot(x_fit_log10, y_fit_log, color='red', linewidth=1, label=f'Log Fit: y = {slope_log:.2e}·log₁₀(x) + {intercept_log:.2e}', alpha=0.8, zorder=4)
        # ax.plot(x_fit_log10_low, y_fit_log_low, color='red', linestyle='--',
        #          label=f'Fit < {split_gauss}G: y={slope_log_low:.2e}·log₁₀(x)+{intercept_log_low:.2f}', alpha=1, zorder=4, linewidth=2.3)
        # ax.plot(x_fit_log10_high, y_fit_log_high, color='red', linestyle='-',
        #          label=f'Fit ≥ {split_gauss}G: y={slope_log_high:.2e}·log₁₀(x)+{intercept_log_high:.2f}', alpha=1, zorder=4, linewidth=3)
        # ax.plot(x_fit_log10_low, y_fit_log_low, color='red', linestyle='--',
        #          alpha=1, zorder=4, linewidth=2.3)
        # ax.plot(x_fit_log10_high, y_fit_log_high, color='red', linestyle='-',
        #          alpha=1, zorder=4, linewidth=3)
        ax.set_xscale("log")
        ax.set_xlabel("Mean magnetic field strength (G)")
        ax.set_ylabel("Intensity ratio (num/den)")
        # ax.set_title(f"{title[element]} : abundance vs mean magnetic field strength", fontsize = 11)
        ax.set_title(f"{title[element]}", fontsize=20, fontweight="bold")
        ax.grid(True)
        # plt.ylim(0, 1.5 if element == "sar" else 4)
        ax.set_ylim(0, 4)
        ax.set_xlim(5, 3000)
        # ax.legend()
        # Only show full legend once
        # if element == "sar":
        #     main_legend = ax.legend(loc="upper right", fontsize=20)
        #     ax.add_artist(main_legend)
        
        # # Always show log fit legend
        # logfit_label_low = f'Fit < {split_gauss}G: y={slope_log_low:.2e}·log_10(x)+{intercept_log_low:.2f}'
        # logfit_label_high = f'Fit => {split_gauss}G: y={slope_log_high:.2e}·log_10(x)+{intercept_log_high:.2f}'
        # logfit_line_low = ax.plot(x_fit_log10_low, y_fit_log_low, color='red', linestyle='--',
        #                         alpha=1, zorder=4, linewidth=5)[0]
        # logfit_line_high = ax.plot(x_fit_log10_high, y_fit_log_high, color='red', linestyle='-',
        #                         alpha=1, zorder=4, linewidth=5)[0]
        
        # # legend_loc = "upper left" if element == "sar" else "lower left"
        # # ax.legend([logfit_line_low, logfit_line_high],
        # #           [logfit_label_low, logfit_label_high],
        # #           loc=legend_loc, fontsize=20, frameon=True)
        # if element == "sar":
        #     # Combine log fit + main legend in one box
        #     handles, labels = ax.get_legend_handles_labels()
        #     ax.legend(
        #         [logfit_line_low, logfit_line_high] + handles[0:5],
        #         [logfit_label_low, logfit_label_high] + labels[0:5],
        #         loc="upper left",
        #         fontsize=20,
        #         frameon=True,
        #         fancybox=True
        #     )
        # else:
        #     # Only show log fit lines for other elements
        #     ax.legend([logfit_line_low, logfit_line_high],
        #             [logfit_label_low, logfit_label_high],
        #             loc="lower left",
        #             fontsize=20,
        #             frameon=True)
        fit_lines = []
        fit_labels = []

        if have_low:
            logfit_label_low = f'Fit < {split_gauss}G: y={slope_log_low:.2e}·log_10(x)+{intercept_log_low:.2f}'
            logfit_line_low = ax.plot(
                x_fit_log10_low, y_fit_log_low,
                color='red', linestyle='--', alpha=1, zorder=4, linewidth=5
            )[0]
            fit_lines.append(logfit_line_low)
            fit_labels.append(logfit_label_low)

        if have_high:
            logfit_label_high = f'Fit => {split_gauss}G: y={slope_log_high:.2e}·log_10(x)+{intercept_log_high:.2f}'
            logfit_line_high = ax.plot(
                x_fit_log10_high, y_fit_log_high,
                color='red', linestyle='-', alpha=1, zorder=4, linewidth=5
            )[0]
            fit_lines.append(logfit_line_high)
            fit_labels.append(logfit_label_high)

        # Legend handling 
        if element == "sar":
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                fit_lines + handles[0:5],
                fit_labels + labels[0:5],
                loc="upper left",
                fontsize=20,
                frameon=True,
                fancybox=True
            )
        else:
            # Only show fit legend if at least one fit exists
            if len(fit_lines) > 0:
                ax.legend(
                    fit_lines, fit_labels,
                    loc="lower left",
                    fontsize=20,
                    frameon=True
                )
            
        valid_pixels_low = np.sum(mask_low & closed_mask_flat)
        valid_pixels_high = np.sum(mask_high & closed_mask_flat)
        print(f"[{element}] Valid closed pixels: < {split_gauss} G = {valid_pixels_low}, >= {split_gauss} G = {valid_pixels_high}")
        fdiag.write(f"[{element}] Valid closed pixels: < {split_gauss} G = {valid_pixels_low}, >= {split_gauss} G = {valid_pixels_high}\n\n")


    outname = os.path.join(output_dir, "ARall_Abundance_B_with_fip.png")
    # plt.savefig(outname, dpi=150, bbox_inches="tight")
    # plt.savefig(outname, dpi=150)
    plt.close(fig)
    print(f"Saved: {outname}")
