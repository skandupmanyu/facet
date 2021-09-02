"""
Last updated 6/25/2020
This function trains a groups the model predicted outcomes by the
the predicted probabilities and default model binary predictions, as well as the outcome and protected group arrays
"""
# Import libraries for data analysis
import pandas as pd
from sklearn import metrics


class RAIFairnessScenarios:
    def __init__(self, target_rate, bias_detect_thresh) -> None:
        self._target_rate = target_rate
        self._bias_detect_thresh = bias_detect_thresh
        self._scenarios_output_df = pd.DataFrame()

    def fit(self, outcome_array, preds_naive, preds_proba, pg_array):
        """
        Parameters
        ----------
        outcome_array = true outcome array (binary)
        pg_array = protected group indicator (binary)
        preds_proba = predicted probabilities output array from sklearn
        preds_naive = predictions of highest accuracy threshold
        dependent = outcome variable (binary)
        protected_group = variable indicating a protected group (binary)
        target_rate = target_rate = 1 - desired (target) positive rate, i.e., outome rate: percentage of the population that classified as predicted outcome 1


        Returns
        -------
        output_list_scenario -- defined as:
            For all four scenarios, aggregate output metrics
            output_metrics:
                bias_index = bias index defined as outcome rate for protected group / rate for non-protected group
                acc = post-threshold adjustment predictive accuracy
                TP = true positive rate
                FN = false negative rate
                TN = true negative rate
                FP = false positive rate
                non_pg_rate = outcome rate for non-protected group
                (later can calculate pg_rate = bias_index*non_pg_rate)

        """
        # Create DF from input arrays

        self._scenarios_output_df["y_true"] = outcome_array
        self._scenarios_output_df["protected_group"] = pg_array
        self._scenarios_output_df["preds_proba"] = preds_proba
        self._scenarios_output_df["preds_naive"] = preds_naive

        # To test the fairness scenarios, we split the data among the protected and non-protected groups, and we explore
        # the accuracy and confusion matrix outcomes which result from each type of threshold adjustment.

        # Run get bias metrics function
        # Scenario 1
        # print("Scenario 1 - Naive Model")
        output_list_1 = self._run_model_scenario("preds_naive")
        # Scenario 2
        # print("Scenario 2 - Threshold Specific Model")
        output_list_2 = self._run_model_scenario("preds_threshold")
        # Scenario 3
        # print("Scenario 3 - Historic Parity at Threshold")
        output_list_3 = self._run_model_scenario("preds_historic")
        # Scenario 4
        # print("Scenario 4 - Demographic Parity at Threshold")
        output_list_4 = self._run_model_scenario("preds_demographic")

        # Combine DFs
        output_list = pd.DataFrame(
            list(zip(output_list_1, output_list_2, output_list_3, output_list_4))
        )

        # Formatting
        fairness_scenarios = output_list.transpose()
        fairness_scenarios.columns = [
            "Bias_Test",
            "Bias_Index",
            "Accuracy",
            "TP",
            "FN",
            "TN",
            "FP",
            "Non_PG_Outcome_Rate",
        ]
        fairness_scenarios["PG_Outcome_Rate"] = (
            (
                fairness_scenarios["Non_PG_Outcome_Rate"]
                * fairness_scenarios["Bias_Index"]
            )
            .astype(float)
            .round(4)
        )

        fairness_scenarios["Scenario"] = [
            "1 - Naive",
            "2 - Threshold Best",
            "3 - Historic Parity",
            "4 - Demographic Parity",
        ]
        fairness_scenarios.set_index("Scenario", inplace=True)
        # fairness_scenarios["Threshold_Used"] = self._target_rate
        # fairness_scenarios["Protected_Group"] = "protected_group"

        self.preds_naive = self._scenarios_output_df["preds_naive"].values
        self.preds_threshold = self._scenarios_output_df["preds_threshold"].values
        self.preds_historic = self._scenarios_output_df["preds_historic"].values
        self.preds_demographic = self._scenarios_output_df["preds_demographic"].values

        return fairness_scenarios

    def _run_model_scenario(self, predict_array):
        """
        Parameters
        ----------
        scenarios_output_df = for
        predict_array = outcome variable to compare rates for (binary)
        dependent = outcome variable (binary)
        protected_group = variable indicating a protected group (binary)
        target_rate = 1 - desired (target) positive rate, i.e., outome rate: percentage of the population that classified as predicted outcome 1.

        Returns
        -------
        output_list_scenario -- defined as:
            For all four scenarios, aggregate output metrics
            output_metrics:
                bias_index = bias index defined as outcome rate for protected group / rate for non-protected group
                acc = post-threshold adjustment predictive accuracy
                TP = true positive rate
                FN = false negative rate
                TN = true negative rate
                FP = false positive rate
                non_pg_rate = outcome rate for non-protected group
                (later can calculate pg_rate = bias_index*non_pg_rate)

        """
        if predict_array == "preds_naive":
            scenario_select = self._scenarios_output_df

        if predict_array == "preds_threshold":
            # Force threshold to be equal to the population outcome rate (bad loan)
            force_thresh = pd.DataFrame(
                self._scenarios_output_df["preds_proba"]
            ).quantile(self._target_rate)
            threshold = force_thresh[0]
            self.thresh_best = threshold
            self._scenarios_output_df["preds_threshold"] = (
                self._scenarios_output_df["preds_proba"] > threshold
            ).astype("int")
            scenario_select = self._scenarios_output_df

        if predict_array == "preds_historic":
            # create separate DFs for PG and non-PG groups
            X_test_pg = self._scenarios_output_df[
                self._scenarios_output_df["protected_group"] == 1
            ]
            X_test_non_pg = self._scenarios_output_df[
                self._scenarios_output_df["protected_group"] != 1
            ]
            N_pg = X_test_pg.shape[0]  # number of instances in pg group
            N_npg = X_test_non_pg.shape[0]  # number of instances in non-pg group

            # Force thresholds to be equal to the population historical outcome rate
            pg_baseline = (
                self._scenarios_output_df[["protected_group", "y_true"]]
                .groupby(["protected_group"], as_index=False)
                .mean()
            )
            # compute the historic bias index as positive_rate_pg / positive_rate_non_pg
            hist_bias_index = pg_baseline["y_true"][1] / pg_baseline["y_true"][0]

            # computing desired size of positives for pg & non-pg iff we want the new bias index to be equal to historic one
            npg = ((N_pg + N_npg) * (1 - self._target_rate)) / (
                hist_bias_index * N_pg / N_npg + 1
            )
            pg = npg * hist_bias_index * N_pg / N_npg

            # computing desired positive rates
            positive_rate_pg = pg / float(N_pg)
            positive_rate_npg = npg / float(N_npg)

            # chose threshold for pg & non-pg to get the desired positive rates

            pg_thresh = pd.DataFrame(X_test_pg["preds_proba"]).quantile(
                1 - positive_rate_pg
            )
            non_pg_thresh = pd.DataFrame(X_test_non_pg["preds_proba"]).quantile(
                1 - positive_rate_npg
            )

            # Maybe there is a cleaner way to do this, data type issue so include this step
            threshold_pg = pg_thresh[0]
            threshold_non_pg = non_pg_thresh[0]
            self.thresh_hist_pg = threshold_pg
            self.thresh_hist_non_pg = threshold_non_pg

            # Create predictions using model output probabilities
            X_test_pg["predictions_hist"] = (
                X_test_pg["preds_proba"] >= threshold_pg
            ).astype("int")
            X_test_non_pg["predictions_hist"] = (
                X_test_non_pg["preds_proba"] >= threshold_non_pg
            ).astype("int")

            # Combine DFs
            X_test_adj = X_test_pg.append(X_test_non_pg)

            # Create preds
            self._scenarios_output_df["preds_historic"] = X_test_adj["predictions_hist"]
            scenario_select = self._scenarios_output_df

        if predict_array == "preds_demographic":
            # create separate DFs for PG and non-PG groups
            X_test_pg = self._scenarios_output_df[
                self._scenarios_output_df["protected_group"] == 1
            ]
            X_test_non_pg = self._scenarios_output_df[
                self._scenarios_output_df["protected_group"] != 1
            ]
            # Force thresholds to be equal among PG and non-PG
            pg_thresh_eq = pd.DataFrame(X_test_pg["preds_proba"]).quantile(
                self._target_rate
            )
            non_pg_thresh_eq = pd.DataFrame(X_test_non_pg["preds_proba"]).quantile(
                self._target_rate
            )

            # Maybe there is a cleaner way to do this, data type issue so include this step
            threshold_pg = pg_thresh_eq[0]
            threshold_non_pg = non_pg_thresh_eq[0]
            self.thresh_demog_pg = threshold_pg
            self.thresh_demog_non_pg = threshold_non_pg

            # Create predictions using model output probabilities
            X_test_pg["preds_demographic"] = (
                X_test_pg["preds_proba"] > threshold_pg
            ).astype("int")
            X_test_non_pg["preds_demographic"] = (
                X_test_non_pg["preds_proba"] > threshold_non_pg
            ).astype("int")

            # Combine DFs
            X_test_adj = X_test_pg.append(X_test_non_pg)

            # Create preds
            self._scenarios_output_df["preds_demographic"] = X_test_adj[
                "preds_demographic"
            ]
            scenario_select = self._scenarios_output_df

        output_list_scenario = self._get_bias_metrics(
            scenario_select,
            predict_array,
        )

        return output_list_scenario

    """
    Last updated 6/25/2020
    This function takes arrays of model predictions, true outcomes and protected group indicator and outputs
    the bias index and values of the confusion matrix
    """

    def _get_bias_metrics(self, scenario_select, predict_array):
        """
        Parameters
        ----------
        model_input_path = location of model input file - must be parquet
        dependent = outcome variable to compare rates for (binary)

        Returns
        -------
        output_metrics -- defined as:
            bias_index = bias index defined as outcome rate for protected group / rate for non-protected group
            acc = post-threshold adjustment predictive accuracy
            TP = true positive rate
            FN = false negative rate
            TN = true negative rate
            FP = false positive rate
            non_pg_rate = outcome rate for non-protected group
            (later can calculate pg_rate = bias_index*non_pg_rate)

        """
        # Create pivot working class - in population overall (train and test)
        pg_compare = (
            scenario_select[["protected_group", predict_array]]
            .groupby(["protected_group"], as_index=False)
            .mean()
        )
        # print(pg_compare)

        bias_index = pg_compare[predict_array][1] / pg_compare[predict_array][0]
        # print("Disparity in predicted outcome (%):", round(bias_index * 100 - 100, 2))

        if abs(round(bias_index - 1, 3)) > self._bias_detect_thresh:
            bias_test = "Fail"
        else:
            bias_test = "Pass"

        # Profitability
        non_pg_rate = pg_compare[predict_array][0]
        acc = metrics.accuracy_score(
            scenario_select[predict_array], scenario_select["y_true"]
        )
        # print("Accuracy:", round(acc, 4) * 100, "%")

        # Confusion Matrix Values
        cm = pd.crosstab(
            scenario_select[predict_array], scenario_select["y_true"]
        ).apply(lambda r: r / r.sum(), axis=1)
        TN = round(cm[0][0], 4)
        FN = round(cm[0][1], 4)
        FP = round(cm[1][0], 4)
        TP = round(cm[1][1], 4)
        bias_index = round(bias_index, 4)
        acc = round(acc, 4)
        non_pg_rate = round(non_pg_rate, 4)

        # Combine lists into string output
        output_metrics = [bias_test, bias_index, acc, TP, FN, TN, FP, non_pg_rate]

        return output_metrics
