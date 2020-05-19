import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
import calendar, argparse, pdb, os, shutil, requests, io, zipfile


def aggregate_df(df, args):
    # get all county level data points and add to get state level data points
    aggregation_functions = {
        args.new_cases_name: "sum",
        args.new_deaths_name: "sum",
        args.date_name: "first",
    }
    # add up all entries with same date
    df_new = df.groupby(df[args.date_name]).aggregate(aggregation_functions)
    # Get the number of new cases by taking the difference of cumulative cases given by NYT
    df_new["new_cases"] = df_new[args.new_cases_name].diff().fillna(0)
    df_new["new_deaths"] = df_new[args.new_deaths_name].diff().fillna(0)
    return df_new


def get_state(df, state, args):
    df_new = df.loc[df[args.state_name] == state]
    return df_new


def get_rmse(df1, df2, var):
    rmse = np.sqrt(mean_squared_error(df1[[var]], df2[[var]]))
    return round(rmse, 2)


def get_equal_len_df(df1, df2):
    df2_length = len(df2.index)
    truncated_df1 = df1.head(df2_length)
    return truncated_df1


def get_p_diff_z_score(var1, var2):
    # print(f'var1: {var1} var2: {var2}')
    if var1 == 0 and var2 == 0:
        p_diff = 0
        z_score = 0
    elif var1 == 0 and var2 != 0:  # may want to deal with small variations differently
        p_diff = (var2 - var1) / var2
        z_score = (var2 - var1) / 1  # I am hardcoding the error on zero to be 1
    elif var2 == 0 and var1 != 0:
        p_diff = (var2 - var1) / var1
        z_score = (var2 - var1) / 1  # again hardcoding error on zero to be 1
    else:
        p_diff = (abs(var2) - abs(var1)) / var1
        err = np.sqrt(abs(var1))
        z_score = (var2 - var1) / err  # to be added
    # print(f'pdiff: {p_diff} zscore: {z_score}')
    return p_diff, z_score


def get_compare_metrics(df1, df2, var):
    diff = []
    z_scores = []
    for i in range(len(df1.index)):
        var1 = df1[var].iloc[i]
        var2 = df2[var].iloc[i]
        p_diff, z_score = get_p_diff_z_score(var1, var2)

        z_scores.append(z_score)
        diff.append(p_diff)
    number_of_days_percent_threshold = sum(i > args.percent_threshold for i in diff)
    average_Z = round(np.mean(z_scores), 2)
    latest_Z = round(np.mean(z_scores[: -args.n_days_z_score_mean]), 2)
    return diff, z_scores, average_Z, latest_Z, number_of_days_percent_threshold


def average(list):
    if len(list) > 0:
        return sum(list) / len(list)
    else:
        return 0


def compare_data(var, df1, df2, df3, df1_name, df2_name, df3_name, args, state):
    # compare old prod data to latest nyt
    df_new_data, new_p_diffs, new_z_scores = check_new_data(df1, df3, args, var)

    new_days_over_thres = sum(i > args.new_day_percent_thres for i in new_p_diffs)
    new_data_days_abnormal = (
        new_days_over_thres > 0
    )  # there is at least one new day added that exceeds abnormal threshold
    # truncate df2 and df3 to df1 (this assumes df1 is the shortest)
    truncated_df2 = get_equal_len_df(df2, df1)
    truncated_df3 = get_equal_len_df(df3, df1)
    # pdb.set_trace()
    # Get comparison metrics
    rmse2 = get_rmse(df1, truncated_df2, var)
    rmse3 = get_rmse(df1, truncated_df3, var)
    diff_df2, z_df2, avg_z2, latest_avg_z2, days_over_z2 = get_compare_metrics(
        df1, truncated_df2, var
    )
    diff_df3, z_df3, avg_z3, latest_avg_z3, days_over_z3 = get_compare_metrics(
        df1, truncated_df3, var
    )

    # determine if data is different enough to plot for further investigation
    historical_data_disagree = days_over_z3 > args.n_days_over_threshold

    if historical_data_disagree or new_data_days_abnormal:
        fig, ax = plt.subplots(3, 1, sharex=True)
        fig.subplots_adjust(hspace=0)
        markersize1 = 8
        markersize2 = 10
        markersize3 = 4
        markersize4 = 4
        markerstyle1 = "."
        markerstyle2 = "."
        markerstyle3 = "."
        markerstyle4 = "."
        color1 = "blue"
        color2 = "orange"
        color3 = "purple"
        color4 = "purple"  # because we are only comparing changes in additional days from the latest NYT dataset
        fig.suptitle(var)
        ax[0].plot(
            df1.index.values,
            df1[var],
            color=color1,
            label=df1_name + " (RMSE: " + str(rmse2) + " $Z_{avg}$: " + str(avg_z2) + ")",
            markersize=markersize1,
            marker=markerstyle1,
            alpha=0.5,
        )
        ax[0].plot(
            df3.index.values,
            df3[var],
            color=color3,
            label=df3_name + " (RMSE: " + str(rmse3) + " $Z_{avg}$: " + str(avg_z3) + ")",
            markersize=markersize3,
            marker=markerstyle3,
            alpha=0.5,
        )
        ax[0].plot(
            df2.index.values,
            df2[var],
            color=color2,
            label=df2_name,
            markersize=markersize2,
            marker=markerstyle2,
            alpha=0.5,
        )
        ax[0].set(ylabel=var)
        ax[0].grid(True)
        ax[0].set_yscale("log")

        ax[1].plot(
            df1.index.values,
            diff_df2,
            color=color2,
            markersize=markersize2,
            marker=markerstyle2,
            alpha=0.5,
        )
        ax[1].plot(
            df1.index.values,
            diff_df3,
            color=color3,
            markersize=markersize3,
            marker=markerstyle3,
            alpha=0.5,
        )
        ax[1].plot(
            df_new_data.index.values,
            new_p_diffs,
            color=color4,
            markersize=markersize4,
            marker=markerstyle4,
            alpha=0.5,
            label="Additional New Data",
        )
        ax[1].set(ylabel="Percent Difference")
        ax[1].grid(True)

        ax[2].plot(
            df1.index.values,
            z_df2,
            color=color2,
            markersize=markersize2,
            marker=markerstyle2,
            alpha=0.5,
        )
        ax[2].plot(
            df1.index.values,
            z_df3,
            color=color3,
            markersize=markersize3,
            marker=markerstyle3,
            alpha=0.5,
        )
        ax[2].plot(
            df_new_data.index.values,
            new_z_scores,
            color=color4,
            markersize=markersize4,
            marker=markerstyle4,
            alpha=0.5,
            label="Additional New Data",
        )
        ax[2].set(ylabel="Z Score")
        ax[2].grid(True)

        plt.xticks(rotation=30)
        ax[0].legend(loc="upper left")
        ax[0].legend(loc=2, prop={"size": 4})
        if historical_data_disagree and not new_data_days_abnormal:
            output_path = args.old_data_abnormal_folder
        elif new_data_days_abnormal and not historical_data_disagree:
            output_path = args.new_data_abnormal_folder
        elif new_data_days_abnormal and historical_data_disagree:
            output_path = args.new_and_old_data_abnormal

        plt.savefig(
            args.output_dir + "/" + output_path + "/" + state + "_" + var + "_compare.pdf",
            bbox_inches="tight",
        )
        plt.close("all")
        return avg_z2, latest_avg_z2, days_over_z2, rmse2, rmse3, historical_data_disagree
    else:
        return 0, 0, 0, 0, 0, False


def compare_county_state_plot(var, df1, df2, df1_name, df2_name, args, state):
    rmse2 = get_rmse(df1, df2, var)
    plt.title(state)
    plt.xlabel(var)
    plt.ylabel(args.updated_date_name)
    plt.plot(
        df1[args.updated_date_name],
        df1[var],
        color="blue",
        label=df1_name + " NYT, RMSE: " + str(rmse2),
        markersize=8,
        marker=".",
        alpha=0.5,
    )
    plt.plot(
        df2[args.updated_date_name],
        df2[var],
        color="orange",
        label=df2_name + " NYT",
        markersize=8,
        marker=".",
        alpha=0.5,
    )
    plt.xticks(rotation=30)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig(state + "_raw_county_state" + var + "_compare.png", bbox_inches="tight")
    plt.close("all")


def get_current_day():
    current_month = calendar.month_name[datetime.now().month]
    current_day = datetime.now().day
    return current_month + " " + str(current_day)


def make_meta_comparison_plot(
    states_list, array1, name1, array2, name2, array3, name3, array4, name4, var
):
    x_values = np.arange(len(states_list))
    fig, ax = plt.subplots(2, 2)
    width = 1
    fig.suptitle(var)
    ax[0, 0].bar(x_values, array1, width=width)
    ax[0, 0].set_ylabel(name1)
    ax[0, 0].set_xticks(x_values)
    ax[0, 0].set_xticklabels(states_list)
    ax[0, 1].bar(x_values, array2, width=width)
    ax[0, 1].set_ylabel(name2)
    ax[0, 1].set_xticks(x_values)
    ax[0, 1].set_xticklabels(states_list)
    ax[1, 1].bar(x_values, array3, width=width)
    ax[1, 1].set_ylabel(name3)
    ax[1, 1].set_xticks(x_values)
    ax[1, 1].set_xticklabels(states_list)
    ax[1, 0].bar(x_values, array4, width=width)
    ax[1, 0].set_ylabel(name4)
    ax[1, 0].set_xticks(x_values)
    ax[1, 0].set_xticklabels(states_list)
    plt.savefig(args.output_dir + "/meta_compare_" + var + ".pdf")
    plt.close("all")
    return x_values


def check_new_data(df1, df2, args, var):
    number_of_new_days = len(df2.index) - len(df1.index)
    new_data = df2.tail(
        number_of_new_days + 1
    )  # the new data are the additional entries not in df1

    # get percent change of values relative to last data point

    p_diffs, z_scores = [], []
    for i in range(1, len(new_data.index)):
        ref_point = new_data[var].iloc[i - 1]
        var2 = new_data[var].iloc[i]
        p_diff, z_score = get_p_diff_z_score(ref_point, var2)
        p_diffs.append(p_diff)
        z_scores.append(z_score)
    return new_data.tail(len(new_data.index) - 1), p_diffs, z_scores


def get_production_hash():
    prod_snapshot_version = (
        requests.get(
            "https://raw.githubusercontent.com/covid-projections/covid-projections/master/src/assets/data/data_url.json"
        )
        .json()["data_url"]
        .split("/")[-2]
    )
    master_hash = requests.get(
        f"https://data.covidactnow.org/snapshot/{prod_snapshot_version}/version.json"
    ).json()["covid-data-public"]["hash"]
    return master_hash


def get_df_from_url_hash(thishash, basepath, filepath, args, name):
    this_file = requests.get(f"{BASE_PATH}/{thishash}/{filepath}").content
    this_df = pd.read_csv(io.StringIO(this_file.decode("utf-8")), parse_dates=[args.date_name])
    this_df.to_csv(f"{args.output_dir}/{args.output_data_dir}/{name}.csv")
    return this_df


def get_df_from_url(url, args, name):
    this_file = requests.get(url).content
    this_df = pd.read_csv(io.StringIO(this_file.decode("utf-8")), parse_dates=[args.date_name])
    this_df.to_csv(f"{args.output_dir}/{args.output_data_dir}/{name}.csv")
    return this_df


def make_outputdirs(args):
    # remove output directory if it exists
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    os.makedirs(args.output_dir + "/" + args.new_data_abnormal_folder)
    os.makedirs(args.output_dir + "/" + args.old_data_abnormal_folder)
    os.makedirs(args.output_dir + "/" + args.new_and_old_data_abnormal_folder)
    os.makedirs(args.output_dir + "/" + args.output_data_dir)
    return


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser for process.py")
    parser.add_argument(
        "-output_dir",
        "--output_dir",
        type=str,
        dest="output_dir",
        default="output",
        help="output_dir",
    )
    parser.add_argument(
        "-states",
        "--states",
        nargs="+",
        dest="states",
        default=["Wyoming"],
        help="name of state to process",
    )
    parser.add_argument(
        "-state_name",
        "--state_name",
        type=str,
        dest="state_name",
        default="state",
        help="name of state variable in input csv",
    )
    parser.add_argument(
        "-new_cases_name",
        "--new_cases_name",
        type=str,
        dest="new_cases_name",
        default="cases",
        help="name of new cases column in input csv",
    )
    parser.add_argument(
        "-new_deaths_name",
        "--new_deaths_name",
        type=str,
        dest="new_deaths_name",
        default="deaths",
        help="name of new deaths name in input csv",
    )
    parser.add_argument(
        "-date_name",
        "--date_name",
        type=str,
        dest="date_name",
        default="date",
        help="name of datesin input csv",
    )
    parser.add_argument(
        "-updated_date_name",
        "--updated_date_name",
        type=str,
        dest="updated_date_name",
        default="Date",
        help="name of date variable that is computed in this script, really just renaming",
    )
    parser.add_argument(
        "-rmse_threshold",
        "--rmse_threshold",
        type=float,
        dest="rmse_threshold",
        default=1,
        help="rmse threshold to trigger on",
    )
    parser.add_argument(
        "-percent_threshold",
        "--percent_threshold",
        type=float,
        dest="percent_threshold",
        default=0.1,
        help="percent difference threshold to trigger on",
    )
    parser.add_argument(
        "-z_threshold",
        "--z_threshold",
        type=float,
        dest="z_threshold",
        default=2,
        help="Z score threshold to trigger on",
    )
    parser.add_argument(
        "-n_days_over_threshold",
        "--n_days_over_threshold",
        dest="n_days_over_threshold",
        type=int,
        default=5,
        help="number of days to be over threshold to trigger on",
    )
    parser.add_argument(
        "-use_latest",
        "--use_latest",
        type=bool,
        default=False,
        help="set to true to use latest data * and * new data to determine unexpected changes",
    )
    parser.add_argument(
        "-n_days_z_score_mean",
        "--n_days_z_score_mean",
        type=int,
        default=14,
        help="how many days in the past to average the z score to look for unexpcted changes to the data",
    )
    parser.add_argument(
        "-new_day_percent_thres",
        "--new_day_percent_thres",
        type=float,
        dest="new_day_percent_thres",
        default=1.2,
        help="percent change threshold for new days to trigger on",
    )
    parser.add_argument(
        "-covid_data_public_dir",
        "--covid_data_public_dir",
        type=str,
        dest="covid_data_public_dir",
        default="../../covid-data-public",
        help="directory of covid-data-public",
    )
    args = parser.parse_args()

    args.new_data_abnormal_folder = "new_data_abnormal"
    args.old_data_abnormal_folder = "old_data_abnormal"
    args.new_and_old_data_abnormal_folder = "new_and_old_data_abnormal"
    args.output_data_dir = "data"

    # Make output dirs
    print("covid data public dir")
    print(args.covid_data_public_dir)
    make_outputdirs(args)

    # Get Data
    BASE_PATH = "https://raw.githubusercontent.com/covid-projections/covid-data-public/"
    COUNTY_CSV_PATH = "data/cases-nytimes/us-counties.csv"
    NYT_PATH = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"

    # Get Current Prod Data
    master_hash = get_production_hash()
    prod_df = get_df_from_url_hash(master_hash, BASE_PATH, COUNTY_CSV_PATH, args, "prod")

    # Get Current local cache being used
    current_cached_file = args.covid_data_public_dir + "/data/cases-nytimes/us-counties.csv"
    current_df = pd.read_csv(current_cached_file, parse_dates=[args.date_name])
    current_df.to_csv(f"{args.output_dir}/{args.output_data_dir}/current_cache.csv")

    # Get Latest NYT
    latest_nyt_df = get_df_from_url(NYT_PATH, args, "nyt_latest")

    # Variables to Compare
    variables = ["cases", "deaths", "new_cases", "new_deaths"]

    # Get all states in input dataset if user asks for all states
    if "All" in args.states:
        # could add start and end date here Natasha
        args.states = latest_nyt_df["state"].unique()

    # Iterate thru states
    for var in variables:
        (
            z_avg_list,
            z_latest_avg_list,
            days_over_thres_list,
            states_list,
            rmse_new_list,
            rmse_latest_list,
        ) = ([], [], [], [], [], [])
        for state in args.states:
            # Grab Data To Compare from CAN Data Caches
            this_prod_df = get_state(prod_df, state, args)
            this_local_df = get_state(current_df, state, args)
            this_latest_nyt_df = get_state(latest_nyt_df, state, args)
            prod_name = "CAN PROD"
            local_name = "CAN LOCAL CACHE"
            latest_nyt_name = "NYT LATEST"

            # Aggregate Datasets (i.e. combine counties to state level and calculate new cases and deaths)
            prod_ag = aggregate_df(this_prod_df, args)
            local_ag = aggregate_df(this_local_df, args)
            latest_ag = aggregate_df(this_latest_nyt_df, args)
            avg_z, latest_avg_z, days_over_z, rmse_new, rmse_latest, abnormal = compare_data(
                var,
                local_ag,
                prod_ag,
                latest_ag,
                local_name,
                prod_name,
                latest_nyt_name,
                args,
                state,
            )
            """
            avg_z, latest_avg_z, days_over_z, rmse_new, rmse_latest, abnormal = compare_data(
                var,
                prod_ag,
                local_ag,
                latest_ag,
                prod_name,
                local_name,
                latest_nyt_name,
                args,
                state,
            )
            """
            if abnormal:
                z_avg_list.append(avg_z)
                z_latest_avg_list.append(latest_avg_z)
                days_over_thres_list.append(days_over_z)
                states_list.append(state)
                rmse_new_list.append(rmse_new)
                rmse_latest_list.append(rmse_latest)

        # Create meta-compare charts for all abnormal areas
        states_list = list(dict.fromkeys(states_list))
        make_meta_comparison_plot(
            states_list,
            days_over_thres_list,
            "days_over_thres",
            rmse_new_list,
            "rmse_new",
            z_avg_list,
            "z_avg",
            z_latest_avg_list,
            "z_latest",
            var,
        )

        # zipf = zipfile.ZipFile(args.output_dir + "raw_data_QA.zip", "w", zipfile.ZIP_DEFLATED)
        # zipdir("./output", zipf)
        # zipf.close()

        # compare_county_state_plot('new_cases', df_state_ag, df_county_ag, 'County Sum', 'State', args, state)
