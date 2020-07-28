import math, random
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

# Hyperparameter Search
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

# Aesthetics
from cycler import cycler

configure(processors=[merge_threadlocal, structlog.processors.KeyValueRenderer()])
log = structlog.get_logger(__name__)


class ForecastRt:
    """
    Write doc string
    """

    def __init__(self, df_all=None):
        self.save_csv_output = False  # do not set to true for github actions run
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
        self.cases_cumulative = True
        self.deaths_cumulative = True
        self.case_var = "cases"
        self.death_var = "deaths"
        self.daily_var_prefix = "new_"
        self.daily_case_var = self.daily_var_prefix + self.case_var
        self.daily_death_var = self.daily_var_prefix + self.death_var
        # self.predict_variable = "Rt_MAP__new_cases"

        self.raw_predict_variable = self.daily_case_var
        self.smooth_predict_variable = f"smooth_{self.raw_predict_variable}"
        self.predict_variable = self.smooth_predict_variable
        self.d_predict_variable = f"d_{self.predict_variable}"
        self.forecast_variables = [
            # self.sim_date_name,  # DO NOT MOVE THIS!!!!! EVA!!!!!
            self.predict_variable,
            self.daily_case_var,
            self.daily_death_var,
            self.fips_var_name_int,
            "positive_tests",
            "negative_tests",
            "raw_search",  # raw google health trends data
            "smoothed_search",  # smoothed google health trends data
            "nmf_day_doc_fbc_fbs_ght",  # delphi combined indicator
            "raw_cli",  # fb raw covid like illness
            "raw_ili",  # fb raw flu like illness
            "contact_tracers_count",
            "nmf_day_doc_fbs_ght",
            "raw_community",
            "raw_hh_cmnty_cli",
            "raw_nohh_cmnty_cli",
            "raw_wcli",
            "raw_wili",
            "smoothed_cli",
            "smoothed_community",
            "smoothed_hh_cmnty_cli",
            "smoothed_ili",
            "smoothed_nohh_cmnty_cli",
            "smoothed_wcli",
            "smoothed_wili",
            "unsmoothed_community",
        ]
        self.scaled_variable_suffix = "_scaled"

        # Seq2Seq Parameters
        self.max_scaling = 2  # multiply max feature values by this number for scaling set
        self.min_scaling = 0.5  # multiply min feature values by this number of scaling set
        self.days_between_samples = 7
        self.mask_value = -10
        self.min_number_of_days = 31
        self.sequence_length = (
            30  # can pad sequence with numbers up to this length if input lenght is variable
        )
        self.sample_train_length = 30  # Set to -1 to use all historical data
        self.predict_days = 7
        self.percent_train = True
        self.train_size = 0.8
        self.n_test_days = 10
        self.n_batch = 100
        self.n_epochs = 2
        self.n_hidden_layer_dimensions = 100
        self.dropout = 0
        self.patience = 50
        self.validation_split = 0  # currently using test set as validation set
        self.hyperparam_search = False

    @classmethod
    def run_forecast(cls, df_all=None):
        engine = cls(df_all)
        return engine.forecast_rt()

    def get_forecast_dfs(self):
        if self.merged_df is None or not self.states_only:
            raise NotImplementedError("Only states are supported.")

        df_merge = pd.read_csv(
            self.csv_path,
            parse_dates=True,
            index_col=self.index_col_name_csv,
            converters={self.fips_var_name: str},
        )

        if self.save_csv_output:
            df_merge.to_csv(self.csv_output_folder + "MERGED_CSV.csv")

        # only store state information
        df_states_merge = df_merge[
            df_merge[self.aggregate_level_name] == self.state_aggregate_level_name
        ]
        # create separate dataframe for each state
        state_df_dictionary = dict(iter(df_states_merge.groupby(self.fips_var_name)))

        # process dataframe
        state_names, df_forecast_list, df_list = [], [], []
        for state in state_df_dictionary:
            df = state_df_dictionary[state]
            state_name = df[self.fips_var_name][0]

            if self.deaths_cumulative:
                df[self.daily_case_var] = df[self.case_var].diff()
            if self.cases_cumulative:
                df[self.daily_death_var] = df[self.death_var].diff()

            # Calculate average of predict variable
            df[self.smooth_predict_variable] = (
                df.iloc[:][self.raw_predict_variable].rolling(window=5).mean()
            )
            # Calculate Rt derivative, exclude first row since-- zero derivative
            df[self.d_predict_variable] = df[self.predict_variable].diff()

            # Only keep data points where predict variable exists
            first_valid_index = df[self.predict_variable].first_valid_index()
            df = df[first_valid_index:].copy()

            # dates.append(df.iloc[-self.predict_days:]['sim_day'])
            # TODO decide if first and last entry need to be removed
            df = df[1:-1]
            df[self.fips_var_name_int] = df[self.fips_var_name].astype(int)
            df[self.sim_date_name] = (df.index - self.ref_date).days + 1

            # df_forecast = df[self.forecast_variables].copy()
            # Fill empty values with mask value
            df_forecast = df.fillna(self.mask_value)
            # ignore last entry = NaN #TODO find a better way to do this!!!
            # Is this necessary? dunno why some states have 0 for last Rt
            # df_forecast = df_forecast.iloc[:-1]

            state_names.append(state_name)
            df_forecast_list.append(df_forecast)
            df_list.append(df)
            if self.save_csv_output:
                df_forecast.to_csv(self.csv_output_folder + df["state"][0] + "_forecast.csv")
                df.to_csv(self.csv_output_folder + df["state"][0] + "_OG_forecast.csv")

        return state_names, df_forecast_list

    def get_train_test_samples(self, df_forecast):
        # True if test samples are constrained to be latest samples
        test_set_end_of_sample = False

        # create list of dataframe samples
        df_samples = self.create_samples(df_forecast)

        # Split sample list into training and testing
        if test_set_end_of_sample:
            # Determine size of train set to split sample list into training and testing
            if self.percent_train:
                train_set_length = int(len(df_samples) * self.train_size)
            else:
                train_set_length = int(len(df_samples)) - self.n_test_days
            train_samples_not_spaced = df_samples[:train_set_length]
            first_test_index = (
                self.days_between_samples * ((train_set_length // self.days_between_samples) + 1)
                - 1
            )
            test_samples = df_samples[first_test_index:]

            if self.save_csv_output:
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

        else:  # test and train set randomly selected from sample set
            # require samples to be days_between_samples apart
            df_samples_spaced = df_samples[0 :: self.days_between_samples]

            if self.percent_train:
                train_set_length = int(len(df_samples_spaced) * self.train_size)
            else:
                train_set_length = int(len(df_samples_spaced)) - self.n_test_days
            # shuffle samples before spliting between test and train sets
            random.shuffle(df_samples_spaced)
            # Select train and test sets
            train_samples = df_samples_spaced[:train_set_length]
            test_samples = df_samples_spaced[train_set_length:]

            if self.save_csv_output:
                for i in range(len(train_samples_not_spaced)):
                    df = train_samples_not_spaced[i]
                    if self.save_csv_output:
                        df.to_csv(self.csv_output_folder + "df" + str(i) + "_train-notspaced.csv")

                for i in range(len(test_samples)):
                    df = test_samples[i]
                    if self.save_csv_output:
                        df.to_csv(self.csv_output_folder + "df" + str(i) + "_test-notspaced.csv")

        # Scaling set is the concatenated train_samples
        scaling_set = pd.concat(train_samples)

        return train_samples, test_samples, scaling_set

    def plot_variables(self, df_list, state_fips, scalers_dict):
        col = plt.cm.jet(np.linspace(0, 1, round(len(self.forecast_variables) + 1)))
        BOLD_LINEWIDTH = 3
        for df, state in zip(df_list, state_fips):
            fig, ax = plt.subplots(figsize=(18, 12))
            for var, color in zip(self.forecast_variables, col):
                if var != self.predict_variable:
                    ax.plot(df[var], label=var, color=color, linewidth=1)
                else:
                    ax.plot(df[var], label=var, color="black", linewidth=BOLD_LINEWIDTH)
            ax.legend()
            plt.xticks(rotation=30, fontsize=14)
            plt.grid(which="both", alpha=0.5)
            output_path = get_run_artifact_path(state, RunArtifact.FORECAST_VAR_UNSCALED)
            plt.title(us.states.lookup(state).name)
            plt.savefig(output_path, bbox_inches="tight")

            fig2, ax2 = plt.subplots(figsize=(18, 12))
            # for var, color in zip(self.forecast_variables, col):
            for i in range(len(self.forecast_variables)):
                if i % 2 == 0:
                    lstyle = "solid"
                elif i % 3 == 0:
                    lstyle = "dotted"
                else:
                    lstyle = "dashed"
                var = self.forecast_variables[i]
                color = col[i]
                reshaped_data = df[var].values.reshape(-1, 1)
                scaled_values = scalers_dict[var].transform(reshaped_data)
                # ax2.plot(scaled_values, label=var, color=color)
                if var != self.predict_variable and var != self.daily_case_var:
                    ax2.plot(scaled_values, label=var, color=color, linewidth=1, linestyle=lstyle)
                else:
                    ax2.plot(
                        scaled_values,
                        label=var,
                        color=color,
                        linewidth=BOLD_LINEWIDTH,
                        linestyle=lstyle,
                    )
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
        area_fips, area_df_list = self.get_forecast_dfs()

        # get train, test, and scaling samples per state and append to list
        area_scaling_samples, area_train_samples, area_test_samples = [], [], []
        for df, fips in zip(area_df_list, area_fips):
            train, test, scaling = self.get_train_test_samples(df)
            area_scaling_samples.append(scaling)
            area_train_samples.append(train)
            area_test_samples.append(test)
            area_name = us.states.lookup(fips).name
            log.info(f"{area_name}: train_samples: {len(train)} test_samples: {len(test)}")
        # Get scaling dictionary
        # TODO add max min rows to avoid domain adaption issues
        train_scaling_set = pd.concat(area_scaling_samples)
        scalers_dict = self.get_scaling_dictionary(slim(train_scaling_set, self.forecast_variables))
        if self.debug_plots:
            self.plot_variables(
                slim(area_df_list, self.forecast_variables), area_fips, scalers_dict
            )

        # Create scaled train samples
        list_train_X, list_train_Y, list_test_X, list_test_Y = [], [], [], []
        # iterate over train/test_samples = list[state_dfs_samples]
        for train, test in zip(area_train_samples, area_test_samples):
            train_filter = slim(train, self.forecast_variables)
            test_filter = slim(test, self.forecast_variables)
            train_X, train_Y = self.get_scaled_X_Y(train_filter, scalers_dict, "train")
            test_X, test_Y = self.get_scaled_X_Y(test_filter, scalers_dict, "test")
            list_train_X.append(train_X)
            list_train_Y.append(train_Y)
            list_test_X.append(test_X)
            list_test_Y.append(test_Y)

        final_list_train_X = np.concatenate(list_train_X)
        final_list_train_Y = np.concatenate(list_train_Y)
        final_list_test_X = np.concatenate(list_test_X)
        final_list_test_Y = np.concatenate(list_test_Y)

        skip_train = 380
        skip_test = 58
        if skip_train > 0:
            final_list_train_X = final_list_train_X[:-skip_train]
            final_list_train_Y = final_list_train_Y[:-skip_train]
        if skip_test > 0:
            final_list_test_X = final_list_test_X[:-skip_test]
            final_list_test_Y = final_list_test_Y[:-skip_test]

        log.info(f"train: {len(final_list_train_X)} test: {len(final_list_test_X)}")
        if self.hyperparam_search:
            model, history, tuner = self.build_model(
                final_list_train_X, final_list_train_Y, final_list_test_X, final_list_test_Y,
            )

            # TODO find a better way to do this and change batch size?
            best_hps = tuner.get_best_hyperparameters()[0]
            dropout = best_hps.get("dropout")
            n_hidden_layer_dimensions = best_hps.get("n_hidden_layer_dimensions")
            n_layers = best_hps.get("n_layers")
        dropout = 0
        n_hidden_layer_dimensions = 100
        n_layers = 4
        log.info(f"n_features: {len(self.forecast_variables)}")
        modelClass = MyHyperModel(
            train_sequence_length=self.sequence_length,
            predict_sequence_length=self.predict_days,
            n_features=len(self.forecast_variables),
            mask_value=self.mask_value,
            batch_size=self.n_batch,
        )
        model = modelClass.build(
            tune=False,
            dropout=dropout,
            n_hidden_layer_dimensions=n_hidden_layer_dimensions,
            n_layers=n_layers,
        )
        es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=self.patience)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs")
        history = model.fit(
            final_list_train_X,
            final_list_train_Y,
            epochs=self.n_epochs,
            batch_size=self.n_batch,
            callbacks=[es, tensorboard_callback],
            verbose=1,
            shuffle=True,
            validation_data=(final_list_test_X, final_list_test_Y),
        )

        plt.close("all")
        plt.plot(history.history["loss"], color="blue", linestyle="solid", label="Train Set")
        plt.plot(
            history.history["val_loss"], color="green", linestyle="solid", label="Validation Set",
        )
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        output_path = get_run_artifact_path("01", RunArtifact.FORECAST_LOSS)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close("all")

        forecast_model_skeleton = MyHyperModel(
            train_sequence_length=self.sequence_length,
            predict_sequence_length=self.predict_days,
            n_features=len(self.forecast_variables),
            mask_value=self.mask_value,
            batch_size=1,
        )
        forecast_model = forecast_model_skeleton.build(
            hp=None,
            tune=False,
            n_layers=n_layers,
            dropout=dropout,
            n_hidden_layer_dimensions=n_hidden_layer_dimensions,
        )
        trained_model_weights = model.get_weights()
        forecast_model.set_weights(trained_model_weights)

        DATA_LINEWIDTH = 1
        MODEL_LINEWIDTH = 2
        # plot training predictions
        train_error = 0
        test_error = 0
        for train_df, train_X, train_Y, test_df, test_X, test_Y, area_df in zip(
            area_train_samples,
            list_train_X,
            list_train_Y,
            area_test_samples,
            list_test_X,
            list_test_Y,
            area_df_list,
        ):
            plt.figure(figsize=(18, 12))
            fips = train_df[0]["fips"][0]  # here
            state_name = us.states.lookup(fips).name
            forecasts_train, dates_train, unscaled_forecasts_train = self.get_forecasts(
                train_df, train_X, train_Y, scalers_dict, forecast_model
            )
            train_error += rmse(unscaled_forecasts_train, train_Y)
            log.info(state_name)
            log.info(forecasts_train)
            log.info(train_Y)
            log.info(rmse(forecasts_train, train_Y))

            for n in range(len(dates_train)):
                newdates = dates_train[n]
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

            log.info("now here")
            forecasts_test, dates_test, unscaled_forecasts_test = self.get_forecasts(
                test_df, test_X, test_Y, scalers_dict, forecast_model
            )
            test_error += rmse(unscaled_forecasts_test, test_Y)

            for n in range(len(dates_test)):
                newdates = dates_test[n]
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
                area_df.index,
                area_df[self.predict_variable],
                label="Data",
                markersize=3,
                marker=".",
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
            # plt.ylim(0.5, 3)
            output_path = get_run_artifact_path(fips, RunArtifact.FORECAST_RESULT)
            state_obj = us.states.lookup(state_name)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
            log.info(f"train error: {train_error} test error: {test_error}")
        return

    def get_forecasts(self, df_list, X_list, Y_list, scalers_dict, model):
        unscaled_predictions = list()
        forecasts = list()
        dates = list()
        for df, x, y in zip(df_list, X_list, Y_list):
            x = x.reshape(1, x.shape[0], x.shape[1])
            # scaled_df = pd.DataFrame(np.squeeze(x))
            unscaled_prediction = model.predict(x)
            thisforecast = scalers_dict[self.predict_variable].inverse_transform(
                unscaled_prediction
            )
            forecasts.append(thisforecast)
            unscaled_predictions.append(unscaled_prediction)

            # dates.append(df.iloc[-self.predict_days:]['sim_day'])
            dates.append(df.iloc[-self.predict_days :].index)

        return forecasts, dates, unscaled_predictions

    def get_scaling_dictionary(self, train_scaling_set):
        scalers_dict = {}
        if self.save_csv_output:
            train_scaling_set.to_csv(self.csv_output_folder + "scalingset_now.csv")
        for columnName, columnData in train_scaling_set.iteritems():
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            reshaped_data = columnData.values.reshape(-1, 1)

            scaler = scaler.fit(reshaped_data)
            # scaled_values = scaler.transform(reshaped_data)

            scalers_dict.update({columnName: scaler})
        return scalers_dict

    def get_scaled_X_Y(self, samples, scalers_dict, label):
        sample_list = list()
        for sample in samples:
            for columnName, columnData in sample.iteritems():
                scaled_values = scalers_dict[columnName].transform(columnData.values.reshape(-1, 1))
                # scaled_values = columnData.values.reshape(-1,1)
                sample.loc[:, f"{columnName}{self.scaled_variable_suffix}"] = scaled_values
            sample_list.append(sample)
        # X, Y, df_list = self.get_X_Y(sample_list, label)
        X, Y = self.get_X_Y(sample_list, label)
        # return X, Y, df_list
        return X, Y

    def old_specify_model(
        self, n_batch
    ):  # , sample_train_length, n_features, predict_sequence_length):
        model = Sequential()
        model.add(
            Masking(
                mask_value=self.mask_value,
                batch_input_shape=(n_batch, self.sequence_length, len(self.forecast_variables)),
            )
        )
        model.add(
            LSTM(
                self.n_hidden_layer_dimensions,
                batch_input_shape=(n_batch, self.sequence_length, len(self.forecast_variables)),
                stateful=True,
                return_sequences=True,
            )
        )
        model.add(
            LSTM(
                self.n_hidden_layer_dimensions,
                batch_input_shape=(n_batch, self.sequence_length, len(self.forecast_variables)),
                stateful=True,
            )
        )
        model.add(Dropout(self.dropout))
        model.add(Dense(self.predict_days))

        return model

    def build_model(self, final_train_X, final_train_Y, final_test_X, final_test_Y):
        hypermodel = MyHyperModel(
            train_sequence_length=self.sequence_length,
            predict_sequence_length=self.predict_days,
            n_features=len(self.forecast_variables),
            mask_value=self.mask_value,
            batch_size=self.n_batch,
        )
        tuner = RandomSearch(
            hypermodel,
            objective="val_loss",
            max_trials=100,
            directory="hyperparam_search",
            project_name="hyperparam_search",
        )

        # final_train_X = final_train_X[:-2]
        # final_train_Y = final_train_Y[:-2]
        tuner.search(
            final_train_X,
            final_train_Y,
            epochs=self.n_epochs,
            validation_data=(final_test_X, final_test_Y),
        )
        tuner.results_summary()

        model = tuner.get_best_models(num_models=1)[0]
        best_hyperparams = tuner.get_best_hyperparameters()[0]
        log.info(best_hyperparams)
        history = model.fit(
            final_train_X,
            final_train_Y,
            epochs=self.n_epochs,
            batch_size=self.n_batch,
            verbose=1,
            shuffle=True,  # TODO test shuffle
            # callbacks=[es, tensorboard_callback],
            # validation_split=self.validation_split,
            # validation_data=(final_test_X[:-4], final_test_Y[:-4]),
            validation_data=(final_test_X, final_test_Y),
        )
        # if self.debug_plots:
        if True:
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

        return model, history, tuner

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
        return final_test_X, final_test_Y  # , df_list

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


class MyHyperModel(HyperModel):
    def __init__(
        self, train_sequence_length, predict_sequence_length, n_features, mask_value, batch_size
    ):
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.n_features = n_features
        self.mask_value = mask_value
        self.batch_size = batch_size

    def build(self, hp=None, tune=True, n_layers=-1, dropout=-1, n_hidden_layer_dimensions=-1):
        if tune:
            # access hyperparameters from hp
            dropout = hp.Float("dropout", min_value=0, max_value=0.3, step=0.05, default=0)
            n_hidden_layer_dimensions = hp.Int(
                "n_hidden_layer_dimensions", min_value=10, max_value=100, step=5, default=100
            )
            n_layers = hp.Int("n_layers", min_value=2, max_value=6, step=1, default=4)
            # n_batch = hp.Choice('n_batch', values=[10]) #TODO test other values

        model = Sequential()
        model.add(
            Masking(
                mask_value=self.mask_value,
                batch_input_shape=(self.batch_size, self.train_sequence_length, self.n_features),
            )
        )
        for i in range(n_layers - 1):
            model.add(
                LSTM(
                    n_hidden_layer_dimensions,
                    batch_input_shape=(
                        self.batch_size,
                        self.train_sequence_length,
                        self.n_features,
                    ),
                    stateful=True,
                    return_sequences=True,
                )
            )
        model.add(
            LSTM(
                n_hidden_layer_dimensions,
                batch_input_shape=(self.batch_size, self.train_sequence_length, self.n_features),
                stateful=True,
            )
        )
        model.add(Dropout(dropout))
        model.add(Dense(self.predict_sequence_length))
        es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=3)
        model.compile(loss="mean_squared_error", optimizer="adam")

        return model


def slim(dataframe_input, variables):
    if type(dataframe_input) == list:
        df_list = []
        for df in dataframe_input:
            df_list.append(df[variables].copy())
        return df_list
    else:  # assume there is one inputdataframe
        return dataframe_input[variables].copy()


def rmse(prediction, data):
    error_sum = 0
    prediction_np = np.squeeze(np.asarray(prediction))
    for i, j in zip(prediction_np, data):  # iterate over samples
        for k, l in zip(i, j):  # iterate over predictions for a given sample
            error_sum += abs(k - l)
    return error_sum


def external_run_forecast():
    ForecastRt.run_forecast()
