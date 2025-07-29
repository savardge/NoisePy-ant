import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import numpy as np

# flist = glob.glob(os.path.join("/home/users/s/savardg/HV","output_files_AA", "*.csv"))
# flist = glob.glob(os.path.join("/home/users/s/savardg/HV","output_files_RI", "*.csv"))
# flist = glob.glob(os.path.join("output_files_AA", "*.csv"))
flist = glob.glob(os.path.join("output_files_AA_normZ", "*HV_pws.csv"))
# fname = flist[0]

for fname in flist:
    # if "300" in fname: continue
    station = os.path.split(fname)[1].split("_")[0]
    # if station[0:3] in ["BAS", "MUT", "RIE", "RHE", "LOR", "WEI", "BET", "GRE", "HAR", "INZ"]:
    #     continue

    print(station)
    df = pd.read_csv(fname)
    df = df.drop(columns=["Unnamed: 0"])
    print(f"{df.shape[0]} initial measurements")

    # Remove rows where SNR is 0
    snr_threshold = 8
    n_before = df.shape[0]
    df = df.loc[(df["snr_ZZ"] > snr_threshold) & (df["snr_RZ"] > snr_threshold) & (df["snr_ZR"] > snr_threshold) & (
                df["snr_RR"] > snr_threshold), :]
    print(f"{n_before - df.shape[0]} measurements removed after SNR criterion")

    #     # Keep consistent timings (within T/4)
    # df["time_std"] = df.loc[:, ["time_ZZ", "time_ZR", "time_RZ", "time_RR"]].std(axis=1)
    # df["time_max-min"] = df.loc[:, ["time_ZZ", "time_ZR", "time_RZ", "time_RR"]].max(axis=1) - df.loc[:, ["time_ZZ", "time_ZR", "time_RZ", "time_RR"]].min(axis=1)
    # n_before = df.shape[0]
    # df = df.loc[df["time_max-min"] < df["period"]/1, :]
    # print(f"{n_before - df.shape[0]} measurements removed after arrival time consistency criterion")

    # Keep measurements in far field approximation (3 wavelengths)
    dist_thresh_mult = 3
    n_before = df.shape[0]
    df = df.loc[df["dist_over_lambda"] > dist_thresh_mult, :]
    print(f"{n_before - df.shape[0]} measurements removed after d/lambda criterion")

    # Keep values where H/V smaller than 5
    df = df.loc[(df["HV_vertical"] < 5) & (df["HV_radial"] < 5), :]

    # summary
    N = df.shape[0]
    print(f"{N} measurements left")
    if N < 50:
        continue

    # Plot 1
    # fig, axs = plt.subplots(3,2,figsize=(15,15), sharex=False, sharey=True)
    # bins = np.arange(0,5,0.1)
    # # positive lag, HV vertical
    # df.loc[df["lag_type"] == "pos", "HV_vertical"].hist(bins=bins, ax=axs[0][0], grid=True)
    # axs[0][0].set_title("positive lag, vertical force")
    # # positive lag, HV radial
    # df.loc[df["lag_type"] == "pos", "HV_radial"].hist(bins=bins, ax=axs[0][1], grid=True)
    # axs[0][1].set_title("positive lag, radial force")
    # # sym lag, HV vertical
    # df.loc[df["lag_type"] == "sym", "HV_vertical"].hist(bins=bins, ax=axs[1][0], grid=True)
    # axs[1][0].set_title("symmetric lag, vertical force")
    # # sym lag, HV radial
    # df.loc[df["lag_type"] == "sym", "HV_radial"].hist(bins=bins, ax=axs[1][1], grid=True)
    # axs[1][1].set_title("symmetric lag, radial force")
    # # neg lag, HV vertical
    # df.loc[df["lag_type"] == "neg", "HV_vertical"].hist(bins=bins, ax=axs[2][0], grid=True)
    # axs[2][0].set_title("negative lag, vertical force")
    # # neg lag, HV radial
    # df.loc[df["lag_type"] == "neg", "HV_radial"].hist(bins=bins, ax=axs[2][1], grid=True)
    # axs[2][1].set_title("negative lag, radial force")
    # axs[0][0].set_xlabel("H/V")
    # axs[0][0].set_ylabel("Count") 
    # plt.suptitle(f"Station {station}")
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # Plot 2
    # fig, axs = plt.subplots(3,2,figsize=(15,15), sharex=False, sharey=True)
    # ccol = "dist_over_lambda" #"percent_diff" #"dist_over_lambda"
    # lag_types = ["pos", "pos", "sym", "sym", "neg", "neg"]
    # hv_types = ["HV_vertical", "HV_radial", "HV_vertical", "HV_radial", "HV_vertical", "HV_radial"]
    # for lag_type, hv_type, ax in zip(lag_types, hv_types, axs.ravel()):
    #     df.loc[df["lag_type"] == lag_type, :].plot.scatter(x="period", y=hv_type, c=ccol, ax=ax, grid=True, colormap='jet', title=f"{lag_type} lag, {hv_type}", ylabel="H/V", xlabel="period")
    #     tmp = df.loc[df["lag_type"] == lag_type, ["period",hv_type]].groupby(["period"])
    #     x = tmp.median().index
    #     y = tmp.median().iloc[:,0].values
    #     yerr = tmp.mad().iloc[:,0].values
    #     ax.errorbar(x, y, yerr=yerr, c="r", ecolor="r", elinewidth=2, marker="d", ms=8)
    # axs[0][0].set_ylim((0,5))
    # plt.suptitle(f"Station {station}")
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # Plot 3
    # fig, ax = plt.subplots(1,1,figsize=(12,7))
    # lag_types = ["pos", "pos", "sym", "sym", "neg", "neg"]
    # hv_types = ["HV_vertical", "HV_radial", "HV_vertical", "HV_radial", "HV_vertical", "HV_radial"]
    # for lag_type, hv_type in zip(lag_types, hv_types):
    #     # df.loc[df["lag_type"] == lag_type, :].plot.scatter(x="period", y=hv_type, c=ccol, ax=ax, grid=True, colormap='jet', title=f"{lag_type} lag, {hv_type}", ylabel="H/V", xlabel="period")
    #     tmp = df.loc[df["lag_type"] == lag_type, ["period",hv_type]].groupby(["period"])
    #     x = tmp.median().index
    #     y = tmp.median().iloc[:,0].values
    #     yerr = tmp.mad().iloc[:,0].values
    #     ax.errorbar(x, y, yerr=yerr, elinewidth=2, marker="d", ms=8,  capsize=10, label=f"{lag_type} lag, {hv_type}")            
    # all measurements
    Tall = []
    HVall = []
    for lag_type, hv_type in zip(["pos", "pos", "neg", "neg"],
                                 ["HV_vertical", "HV_radial", "HV_vertical", "HV_radial"]):
        tmp1 = df.loc[df["lag_type"] == lag_type, ["period", hv_type]]
        Tall += list(tmp1["period"].values)
        HVall += list(tmp1[hv_type].values)
    dfall = pd.DataFrame({"period": np.array(Tall), "HV": np.array(HVall)}).groupby(["period"])
    x = dfall.median().index
    y = dfall.median().iloc[:, 0].values
    yerr = dfall.mad().iloc[:, 0].values
    # ax.errorbar(x, y, yerr=yerr, elinewidth=4, c="k", ecolor="k", marker="d", ms=8,  capsize=10, label=f"{lag_type} lag, {hv_type}")  
    # Save to file
    dfsum = pd.DataFrame({"period": x,
                          "HV_median": dfall.median().iloc[:, 0].values,
                          "HV_mad": dfall.mad().iloc[:, 0].values,
                          "HV_mean": dfall.mean().iloc[:, 0].values,
                          "HV_std": dfall.std().iloc[:, 0].values,
                          "HV_count": dfall.count().iloc[:, 0].values})
    dfsum.to_csv(fname.replace("HV_pws.csv", "HV_pws_combined.csv"))
#    # plot setup
#     ax.set_ylim((0,5))
#     plt.legend()
#     plt.suptitle(f"Station {station}")
#     plt.tight_layout()
#     plt.show()
#     plt.close()

    # Alternative way: iteratively eliminate measurements outside 2 standard deviation
    dfa = pd.DataFrame({"period": np.array(Tall), "HV": np.array(HVall)})
    periods = x
    HV_periods = []
    HV_median = []
    HV_mean = []
    HV_std = []
    HV_count = []
    min_measurements = 50
    for per in x:
        hv = dfa.loc[dfa.period == per, "HV"].values
        irem = ([0],)
        while len(irem) == 0:
            hvmean = np.mean(hv)
            hvstd = np.std(hv)
            irem = np.where((hv < hvmean - 2 * hvstd) | (hv > hvmean + 2 * hvstd))
            hv = np.delete(hv, irem)
        if len(hv) < min_measurements:
            continue
        HV_periods.append(per)
        HV_median.append(np.median(hv))
        HV_mean.append(np.mean(hv))
        HV_std.append(np.std(hv))
        HV_count.append(len(hv))
    dfout = pd.DataFrame({
        "period": HV_periods,
        "HV_median": HV_median,
        "HV_mean": HV_mean,
        "HV_std": HV_std,
        "HV_count": HV_count
    })
    dfout.to_csv(fname.replace("HV_pws.csv", "HV_pws_combined_2sigma.csv"))
