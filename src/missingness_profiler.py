import pandas as pd

# Utility class to profile missingness
# in features
class MissingnessProfiler:
    # Initialize the internal state 
    # for further analysis, without affecting
    # original dataset integrity
    def __init__(self, X, y):
        self._X = X.copy()
        self._y = y.copy()

    # Returns a series (Feature -> Missing %)
    def missingness_percentage_per_col(self):
        missingness_series = self._X.isna().mean() * 100
        return missingness_series

    # Builds a summary table to group features by missingness
    def group_missing_cols(self, missingness_series):
        # Groups features by their missingness percentage
        missingness_groups = missingness_series.groupby(missingness_series)
        missingness_groups.groups.keys()

        # Empty list to populate
        rows = []
        
        for missingness_pct, group in missingness_groups:

            # Build the data for the result
            rows.append({
                "Missingness %": missingness_pct,
                "Feature Count": len(group),
                "Features": list(group.index)
            })


        return pd.DataFrame(rows).sort_values(by="Missingness %",ascending=False)

    # Validates a structural pattern in each missingness group,
    # then performs a quantative analysis to measure a 
    # statistical relationship with target
    def explore_missingness_relationship(self, missing_summary):

        # List to populate the summary rows
        rows = []
        
        for idx, row in missing_summary.iterrows():

            # Collect the features from the dataset
            feature_group = self._X[row["Features"]].isna()

            # Create a reference column
            reference = feature_group.iloc[:, -1]

            # Test alignment
            alignment = (
                feature_group
                .eq(reference, axis=0)
                .all()
            ).all()

            # Verify the structural pattern
            if alignment:

                # Start peforming the target
                # relationship analysis
                rows.append(self._explore_subgroup_relationship(feature_group))
        
        return pd.DataFrame(rows)

    # Perform subgroup association with failure rate
    def _explore_subgroup_relationship(self, feature_group):

        # Create a row-level mask to
        # split the target
        missing_mask = feature_group.any(axis=1)
        # Split the target
        y_missing = self._y[missing_mask]
        y_present = self._y[~missing_mask]


        # Present/Missing fail rate
        missing_fail_rate = (y_missing == 1).mean()
        present_fail_rate = (y_present == 1).mean()

        # Delta between rates
        delta_fail_rate = missing_fail_rate - present_fail_rate

        row = {
            "Features": list(feature_group.columns),
            "Missing %": (feature_group.mean() * 100).round(3).iloc[0],
            "N Missing": y_missing.size,
            "N Present": y_present.size,
            "Fail Rate Missing": missing_fail_rate,
            "Fail Rate Present": present_fail_rate,
            "Delta Fail Rate": delta_fail_rate
        }
            
        return row