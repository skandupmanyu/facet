import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats
from scipy.stats import chi2_contingency


class RAIDataBaisCheck:
    def __init__(
        self,
        protected_group,
        test_col,
        pvalue_threshold=0.1,
        test_type="z-test",
        is_2_sided=False,
    ) -> None:
        self._protected_group = protected_group
        self._test_col = test_col
        self._pvalue_threshold = pvalue_threshold
        self._is_2_sided = is_2_sided
        self._test_type = test_type

    def fit(self, df):

        if self._test_type == "z-test":
            metrics = self._stat_test_z(df)
        elif self._test_type == "categorical":
            metrics = self._stat_test_z(df)
        elif self._test_type == "welch":
            metrics = self._stat_test_welch(df)
        else:
            metrics = self._stat_test_z(df)

        return metrics

    def _stat_test_categoric(self, df):
        """
        This function tests whether there's difference among classes in protected group regarding to test column.
        :param df: input dataframe
        :param test_name: column to be tested (only works for numeric columns)
        :param pvalue_threshold: alpha level of this test, usually 0.1 or 0.05
        :param is_2_sided: whether this is a 2-sided test or 1-sided test
        :return:
            chi2: the test statistic
            p: the p-value of the test
            dof: degrees of freedom
            expected: the expected frequencies
        """

        ctable = pd.crosstab(df[self._protected_group], df[self._test_col])

        self.chi2, self.p_value, self.dof, self.ex = chi2_contingency(ctable, correction=False)

        if self.p_value < self._pvalue_threshold:
            self.biased = True
        else:
            self.biased = False

        return self.biased, self.p_value

    def _stat_test_z(self, df):
        """
        This function does z test on test variable
        :param df: input dataframe
        :param test_name: column to be tested (only works for numeric columns)
        :param pvalue_threshold: alpha level of this test, usually 0.1 or 0.05
        :param is_2_sided: whether this is a 2-sided test or 1-sided test
        :return: statistics, pvalue
        """

        class_names = df[self._protected_group].unique()

        self.historic_crosstab = pd.crosstab(
            df[self._protected_group], df[self._test_col]
        ).apply(lambda r: r / r.sum(), axis=1)

        var1 = df[df[self._protected_group] == class_names[0]][self._test_col]
        var2 = df[df[self._protected_group] == class_names[1]][self._test_col]

        # equal_val = False makes it a welch test
        self.statistics, self.p_value = stats.ttest_ind(var1, var2, equal_var=True)

        if not self._is_2_sided:
            self.p_value = self.p_value / 2

        if self.p_value < self._pvalue_threshold:
            self.biased = True
        else:
            self.biased = False

        return self.biased, self.p_value

    def _stat_test_welch(self, df):
        """
        This function does welch test on test variable
        :param df: input dataframe
        :param test_name: column to be tested (only works for numeric columns)
        :param pvalue_threshold: alpha level of this test, usually 0.1 or 0.05
        :param is_2_sided: whether this is a 2-sided test or 1-sided test
        :return: statistics, pvalue
        """

        class_names = df[self._protected_group].unique()

        var1 = df[df[self._protected_group] == class_names[0]][self._test_col]
        var2 = df[df[self._protected_group] == class_names[1]][self._test_col]

        # equal_val = False makes it a welch test
        self.statistics, self.p_value = stats.ttest_ind(var1, var2, equal_var=False)

        if not self._is_2_sided:
            self.p_value = self.p_value / 2

        if self.p_value < self._pvalue_threshold:
            self.biased = True
        else:
            self.biased = False

        return self.biased, self.p_value
