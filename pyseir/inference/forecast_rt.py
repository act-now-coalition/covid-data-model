import math
import us
from datetime import datetime, timedelta
import numpy as np
import logging
import pandas as pd
import os, sys, glob
from matplotlib import pyplot as plt
import us
import structlog

# from pyseir.utils import AggregationLevel, TimeseriesType
from pyseir.utils import get_run_artifact_path, RunArtifact
from structlog.threadlocal import bind_threadlocal, clear_threadlocal, merge_threadlocal
from structlog import configure
from enum import Enum

from tensorflow import keras
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping

configure(processors=[merge_threadlocal, structlog.processors.KeyValueRenderer()])
log = structlog.get_logger(__name__)


class ForecastRt:
    """
    Write doc string
    """

    def __init__(self, df_all=None):
        self.save_csv_output = False
        self.csv_output_folder = "./csv_files/"
        self.df_all = df_all
        self.states = "All"  # All to use All
        self.csv_path = "./pyseir_data/merged_results_delphi.csv"
        # self.csv_path = "./pyseir_data/delphi_merged.csv"

        self.merged_df = True  # set to true if input dataframe merges all areas
        self.states_only = True  # set to true if you only want to train on state level data (county level training not implemented...yet)
        self.ref_date = datetime(year=2020, month=1, day=1)
        self.debug_plots = True

        # Variable Names
        self.aggregate_level_name = "aggregate_level"
        self.state_aggregate_level_name = "state"
        self.state_var_name = "state"
        self.fips_var_name = "fips"  # name of fips var in input csv
        self.fips_var_name_int = (
            "fips_int"  # name of fips used in forecast (cast from input string to int)
        )
        self.sim_date_name = "sim_day"
        self.index_col_name_csv = "date"
        self.predict_variable = "Rt_MAP__new_cases"
        self.cases_cumulative = True
        self.deaths_cumulative = True
        self.case_var = "cases"
        self.death_var = "deaths"
        self.daily_var_prefix = "new_"
        self.daily_case_var = self.daily_var_prefix + self.case_var
        self.daily_death_var = self.daily_var_prefix + self.death_var
        self.d_predict_variable = f"d_{self.predict_variable}"
        self.forecast_variables = [
            self.sim_date_name,  # DO NOT MOVE THIS!!!!! EVA!!!!!
            "raw_search",
            "smoothed_search",
            self.daily_case_var,
            self.daily_death_var,
            self.d_predict_variable,
            self.predict_variable,
            self.fips_var_name_int,
            "positive_tests",
            "negative_tests",
        ]
        self.scaled_variable_suffix = "_scaled"

        # Seq2Seq Parameters
        self.max_scaling = 2  # multiply max feature values by this number for scaling set
        self.min_scaling = 0.5  # multiply min feature values by this number of scaling set
        self.days_between_samples = 7
        self.mask_value = -10
        self.min_number_of_days = 31
        self.sequence_length = 30
        self.sample_train_length = 30  # Set to -1 to use all historical data
        self.predict_days = 7
        self.percent_train = False
        self.train_size = 0.8
        self.n_test_days = 10
        self.n_batch = 1
        self.n_epochs = 1000
        self.n_hidden_layer_dimensions = 100
        self.dropout = 0
        self.patience = 50
        self.validation_split = 0.1

    @classmethod
    def run_forecast(cls, df_all=None):
        try:
            engine = cls(df_all)
            return engine.forecast_rt()
        except Exception:
            logging.exception("forecast failed : ( something unintended occured")
            return None

    def get_forecast_dfs(self):
        if self.merged_df:
            log.info("retrieving input csv")

            df_merge = pd.read_csv(
                self.csv_path,
                parse_dates=True,
                index_col=self.index_col_name_csv,
                converters={self.fips_var_name: str},
            )
            log.info("retrieved input csv")

            if self.states_only:
                if self.save_csv_output:
                    df_merge.to_csv(self.csv_output_folder + "MERGED_CSV.csv")
                # only store state information
                df_states_merge = df_merge[
                    df_merge[self.aggregate_level_name] == self.state_aggregate_level_name
                ]
                # create separate dataframe for each state
                log.info(df_states_merge)
                state_df_dictionary = dict(iter(df_states_merge.groupby(self.fips_var_name)))

                # process dataframe
                state_names, df_list = [], []
                for state in state_df_dictionary:
                    df = state_df_dictionary[state]
                    state_name = df[self.fips_var_name][0]

                    # Only keep data points where predict variable exists
                    first_valid_index = df[self.predict_variable].first_valid_index()
                    df = df[first_valid_index:].copy()

                    df[self.sim_date_name] = (df.index - self.ref_date).days + 1
                    # Calculate Rt derivative, exclude first row since-- zero derivative
                    df[self.d_predict_variable] = df[self.predict_variable].diff()
                    df = df[1:]
                    df[self.fips_var_name_int] = df[self.fips_var_name].astype(int)

                    if self.deaths_cumulative:
                        df[self.daily_case_var] = df[self.case_var].diff()
                    if self.cases_cumulative:
                        df[self.daily_death_var] = df[self.death_var].diff()

                    df_forecast = df[self.forecast_variables].copy()
                    # Fill empty values with mask value
                    df_forecast = df_forecast.fillna(self.mask_value)
                    # ignore last entry = NaN #TODO find a better way to do this!!!
                    # Is this necessary? dunno why some states have 0 for last Rt
                    df_forecast = df_forecast.iloc[:-1]

                    state_names.append(state_name)
                    df_list.append(df_forecast)
                    if self.save_csv_output:
                        df_forecast.to_csv(
                            self.csv_output_folder + df["state"][0] + "_forecast.csv"
                        )

            else:
                log.info("County level training not implemented yet :(")
                exit()

            return state_names, df_list

    def get_train_test_samples(self, df_forecast):
        # create list of dataframe samples
        df_samples = self.create_samples(df_forecast)

        # Determine size of train set to split sample list into training and testing
        if self.percent_train:
            train_set_length = int(len(df_samples) * self.train_size)
        else:
            train_set_length = int(len(df_samples)) - self.n_test_days

        # TODO could create scaling set more cleaning
        # Split sample list into training and testing
        train_samples_not_spaced = df_samples[:train_set_length]
        first_test_index = (
            self.days_between_samples * ((train_set_length // self.days_between_samples) + 1) - 1
        )
        # test_samples = df_samples[train_set_length:]
        test_samples = df_samples[first_test_index:]

        if 1 == 0:
            for i in range(len(train_samples_not_spaced)):
                df = train_samples_not_spaced[i]
                if self.save_csv_output:
                    df.to_csv(self.csv_output_folder + "df" + str(i) + "_train-notspaced.csv")

            for i in range(len(test_samples)):
                df = test_samples[i]
                if self.save_csv_output:
                    df.to_csv(self.csv_output_folder + "df" + str(i) + "_test-notspaced.csv")

        # For training only keep samples that are days_between_samples apart (avoid forecast learning meaningless correlations between labels)
        train_samples = train_samples_not_spaced[0 :: self.days_between_samples]

        # Scaling set is the concatenated train_samples
        scaling_set = pd.concat(train_samples)

        return train_samples, test_samples, scaling_set

    def plot_variables(self, df_list, state_fips, scalers_dict):
        for df, state in zip(df_list, state_fips):
            fig, ax = plt.subplots(figsize=(18, 12))
            for var in self.forecast_variables:
                ax.plot(df[var], label=var)
            ax.legend()
            plt.xticks(rotation=30, fontsize=14)
            plt.grid(which="both", alpha=0.5)
            output_path = get_run_artifact_path(state, RunArtifact.FORECAST_VAR_UNSCALED)
            plt.title(us.states.lookup(state).name)
            plt.savefig(output_path, bbox_inches="tight")

            fig2, ax2 = plt.subplots(figsize=(18, 12))
            for var in self.forecast_variables:
                reshaped_data = df[var].values.reshape(-1, 1)
                scaled_values = scalers_dict[var].transform(reshaped_data)
                ax2.plot(scaled_values, label=var)
            ax2.legend()
            plt.xticks(rotation=30, fontsize=14)
            plt.grid(which="both", alpha=0.5)
            plt.title(us.states.lookup(state).name)
            output_path = get_run_artifact_path(state, RunArtifact.FORECAST_VAR_SCALED)
            plt.savefig(output_path, bbox_inches="tight")

            plt.close("all")

        return

    def forecast_rt(self):
        """
        predict r_t for 14 days into the future
        Parameters
        df_all: dataframe with dates, new_cases, new_deaths, and r_t values
        Potential todo: add more features #ALWAYS
        Returns
        dates and forecast r_t values
        """
        # split merged dataframe into state level dataframes (this includes adding variables and masking nan values)
        state_fips, df_list = self.get_forecast_dfs()

        # get train, test, and scaling samples per state and append to list
        scaling_samples, train_samples, test_samples = [], [], []
        for df, fips in zip(df_list, state_fips):
            train, test, scaling = self.get_train_test_samples(df)
            scaling_samples.append(scaling)
            train_samples.append(train)
            test_samples.append(test)
            state_name = us.states.lookup(fips).name
            log.info(f"{state_name}: train_samples: {len(train)} test_samples: {len(test)}")
        # Get scaling dictionary
        # TODO add max min rows to avoid domain adaption issues
        train_scaling_set = pd.concat(scaling_samples)
        scalers_dict = self.get_scaling_dictionary(train_scaling_set)
        log.info("about to make debug plots")
        if self.debug_plots:
            self.plot_variables(df_list, state_fips, scalers_dict)
        log.info("made debug plots")
        # Create scaled train samples
        list_train_X, list_train_Y, list_test_X, list_test_Y = [], [], [], []
        # iterate over train/test_samples = list[state_dfs_samples]
        for train, test in zip(train_samples, test_samples):
            train_X, train_Y, train_df_list = self.get_scaled_X_Y(train, scalers_dict, "train")
            log.info(
                f"input train length: {len(train_samples)} train_X: {len(train_X)} train_Y: {len(train_Y)}"
            )
            test_X, test_Y, test_df_list = self.get_scaled_X_Y(test, scalers_dict, "test")
            list_train_X.append(train_X)
            list_train_Y.append(train_Y)
            list_test_X.append(test_X)
            list_test_Y.append(test_Y)

        final_list_train_X = np.concatenate(list_train_X)
        final_list_train_Y = np.concatenate(list_train_Y)
        model, history = self.build_model(final_list_train_X, final_list_train_Y)

        # Plot predictions for test and train sets

        for train_X, train_Y, test_X, test_Y, df_forecast, fips in zip(
            list_train_X, list_train_Y, list_test_X, list_test_Y, df_list, state_fips
        ):
            state_name = us.states.lookup(fips).name
            forecasts_train, dates_train = self.get_forecasts(train_X, train_Y, scalers_dict, model)

            for i in range(len(train_X)):
                df = pd.DataFrame(data=train_X[i])
                if self.save_csv_output:
                    df.to_csv(self.csv_output_folder + state_name + "_" + str(i) + ".csv")

            forecasts_test, dates_test = self.get_forecasts(test_X, test_Y, scalers_dict, model)
            DATA_LINEWIDTH = 1
            MODEL_LINEWIDTH = 2
            # plot training predictions
            plt.figure(figsize=(18, 12))
            for n in range(len(dates_train)):
                i = dates_train[n]
                newdates = dates_train[n]
                # newdates = convert_to_2020_date(i,args)
                j = np.squeeze(forecasts_train[n])
                if n == 0:
                    plt.plot(
                        newdates,
                        j,
                        color="green",
                        label="Train Set",
                        linewidth=MODEL_LINEWIDTH,
                        markersize=0,
                    )
                else:
                    plt.plot(newdates, j, color="green", linewidth=MODEL_LINEWIDTH, markersize=0)

            for n in range(len(dates_test)):
                i = dates_test[n]
                newdates = dates_test[n]
                # newdates = convert_to_2020_date(i,args)
                j = np.squeeze(forecasts_test[n])

                if n == 0:
                    plt.plot(
                        newdates,
                        j,
                        color="orange",
                        label="Test Set",
                        linewidth=MODEL_LINEWIDTH,
                        markersize=0,
                    )
                else:
                    plt.plot(newdates, j, color="orange", linewidth=MODEL_LINEWIDTH, markersize=0)

            plt.plot(
                df_forecast[self.sim_date_name],
                df_forecast[self.predict_variable],
                linewidth=DATA_LINEWIDTH,
                markersize=3,
                label="Data",
                marker="o",
            )
            plt.xlabel(self.sim_date_name)
            plt.ylabel(self.predict_variable)
            plt.legend()
            plt.grid(which="both", alpha=0.5)
            # Seq2Seq Parameters
            seq_params_dict = {
                "days_between_samples": self.days_between_samples,
                "min_number_days": self.min_number_of_days,
                "sequence_length": self.sequence_length,
                "train_length": self.sample_train_length,
                "% train": self.train_size,
                "batch size": self.n_batch,
                "epochs": self.n_epochs,
                "hidden layer dimensions": self.n_hidden_layer_dimensions,
                "dropout": self.dropout,
                "patience": self.patience,
                "validation split": self.validation_split,
                "mask value": self.mask_value,
            }
            for i, (k, v) in enumerate(seq_params_dict.items()):

                fontweight = "bold" if k in ("important variables") else "normal"

                if np.isscalar(v) and not isinstance(v, str):
                    plt.text(
                        1.0,
                        0.7 - 0.032 * i,
                        f"{k}={v:1.1f}",
                        transform=plt.gca().transAxes,
                        fontsize=15,
                        alpha=0.6,
                        fontweight=fontweight,
                    )

                else:
                    plt.text(
                        1.0,
                        0.7 - 0.032 * i,
                        f"{k}={v}",
                        transform=plt.gca().transAxes,
                        fontsize=15,
                        alpha=0.6,
                        fontweight=fontweight,
                    )

            plt.title(state_name + ": epochs: " + str(self.n_epochs))
            plt.ylim(0.5, 3)
            output_path = get_run_artifact_path(fips, RunArtifact.FORECAST_RESULT)
            state_obj = us.states.lookup(state_name)
            plt.savefig(output_path, bbox_inches="tight")

        return

    def get_forecasts(self, X, Y, scalers_dict, model):
        forecasts = list()
        dates = list()
        for i, j in zip(X, Y):
            i = i.reshape(self.n_batch, i.shape[0], i.shape[1])
            scaled_df = pd.DataFrame(np.squeeze(i))
            thisforecast = scalers_dict[self.predict_variable].inverse_transform(
                model.predict(i, batch_size=self.n_batch)
            )
            forecasts.append(thisforecast)

            last_train_day = np.array(scaled_df.iloc[-1][0]).reshape(1, -1)
            log.info(f"last train day: {last_train_day}")

            unscaled_last_train_day = scalers_dict[self.sim_date_name].inverse_transform(
                last_train_day
            )

            unscaled_first_test_day = unscaled_last_train_day + 1
            unscaled_last_test_day = int(unscaled_first_test_day) + self.predict_days - 1
            # TODO not putting int here creates weird issues that are possibly worth later investigation

            log.info(
                f"unscaled_last_train_day: {unscaled_last_train_day} first test day: {unscaled_first_test_day} last_predict_day: {unscaled_last_test_day}"
            )

            predicted_days = np.arange(unscaled_first_test_day, unscaled_last_test_day + 1.0)
            log.info("predict days")
            log.info(predicted_days)
            dates.append(predicted_days)
        return forecasts, dates

    def get_scaling_dictionary(self, train_scaling_set):
        log.info("getting scaling dictionary")
        scalers_dict = {}
        log.info("training scaling set")
        log.info(train_scaling_set)
        log.info(train_scaling_set.dtypes)
        if self.save_csv_output:
            train_scaling_set.to_csv(self.csv_output_folder + "scalingset_now.csv")
        for columnName, columnData in train_scaling_set.iteritems():
            log.info("column")
            log.info(columnName)
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            log.info("made scaler")
            reshaped_data = columnData.values.reshape(-1, 1)
            log.info("reshaped data")

            scaler = scaler.fit(reshaped_data)
            log.info("fit data")
            scaled_values = scaler.transform(reshaped_data)
            log.info("scaled data")

            scalers_dict.update({columnName: scaler})
            log.info("saved scaler to dictionary")
        log.info("about to return scaling dictionary")
        return scalers_dict

    def get_scaled_X_Y(self, samples, scalers_dict, label):
        sample_list = list()
        for sample in samples:
            for columnName, columnData in sample.iteritems():
                scaled_values = scalers_dict[columnName].transform(columnData.values.reshape(-1, 1))
                # scaled_values = columnData.values.reshape(-1,1)
                sample.loc[:, f"{columnName}{self.scaled_variable_suffix}"] = scaled_values
            sample_list.append(sample)
        X, Y, df_list = self.get_X_Y(sample_list, label)
        log.info(f"lengths: X: {len(X)} Y:{len(Y)} df_list: {len(df_list)}")
        return X, Y, df_list

    def build_model(self, final_train_X, final_train_Y):
        model = Sequential()
        log.info("BUILDING MODEL")
        log.info(final_train_X)
        log.info(type(final_train_X))
        log.info(final_train_X.shape[1])
        log.info(final_train_X.shape[2])

        model.add(
            Masking(
                mask_value=self.mask_value,
                batch_input_shape=(self.n_batch, final_train_X.shape[1], final_train_X.shape[2]),
            )
        )
        model.add(
            LSTM(
                self.n_hidden_layer_dimensions,
                batch_input_shape=(self.n_batch, final_train_X.shape[1], final_train_X.shape[2]),
                stateful=True,
                return_sequences=True,
            )
        )
        model.add(
            LSTM(
                self.n_hidden_layer_dimensions,
                batch_input_shape=(self.n_batch, final_train_X.shape[1], final_train_X.shape[2]),
                stateful=True,
            )
        )
        model.add(Dropout(self.dropout))
        model.add(Dense(final_train_Y.shape[1]))
        es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=self.patience)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs")
        model.compile(loss="mean_squared_error", optimizer="adam")
        history = model.fit(
            final_train_X,
            final_train_Y,
            epochs=self.n_epochs,
            batch_size=self.n_batch,
            verbose=1,
            shuffle=False,
            validation_split=self.validation_split,
            callbacks=[es, tensorboard_callback],
        )
        logging.info("fit")
        logging.info(history.history["loss"])
        logging.info(history.history["val_loss"])
        plot = True
        if self.debug_plots:
            plt.close("all")
            plt.plot(history.history["loss"], color="blue", linestyle="solid", label="Train Set")
            plt.plot(
                history.history["val_loss"],
                color="green",
                linestyle="solid",
                label="Validation Set",
            )
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("RMSE")
            output_path = get_run_artifact_path("01", RunArtifact.FORECAST_LOSS)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")

        return model, history

    def get_X_Y(self, sample_list, label):
        PREDICT_VAR = self.predict_variable + self.scaled_variable_suffix
        X_train_list = list()
        Y_train_list = list()
        df_list = list()
        for i in range(len(sample_list)):
            df = sample_list[i]
            df_list.append(df)
            df = df.filter(regex="scaled")

            X = df.iloc[
                : -self.predict_days, :
            ]  # exclude last n entries of df to use for prediction
            Y = df.iloc[-self.predict_days :, :]

            # fips = X['fips_int'][0]
            # if fips==-1:
            #  X.to_csv(self.csv_output_folder + label + '_X_' + str(fips) + '_' +  str(i) + '.csv')
            #  Y.to_csv(self.csv_output_folder + label + '_Y_' + str(fips) + '_' + str(i) + '.csv')

            n_rows_train = X.shape[0]
            n_rows_to_add = self.sequence_length - n_rows_train
            pad_rows = np.empty((n_rows_to_add, X.shape[1]), float)
            pad_rows[:] = self.mask_value
            padded_train = np.concatenate((pad_rows, X))

            labels = np.array(Y[PREDICT_VAR])

            X_train_list.append(padded_train)
            Y_train_list.append(labels)

        # MAYBE UNCOMMENT NATASHA
        final_test_X = np.array(X_train_list)
        final_test_Y = np.array(Y_train_list)
        final_test_Y = np.squeeze(final_test_Y)
        return final_test_X, final_test_Y, df_list

    def create_samples(self, df):
        df_list = list()
        for index in range(len(df.index) + 1):
            i = index
            if (
                i < self.predict_days + self.min_number_of_days
            ):  # only keep df if it has min number of entries
                continue
            else:
                if self.sample_train_length == -1:  # use all historical data for every sample
                    df_list.append(df[:i].copy())
                else:  # use only SAMPLE_LENGTH historical days of data
                    df_list.append(df[i - self.sample_train_length : i].copy())
        return df_list


def external_run_forecast():
    ForecastRt.run_forecast()
