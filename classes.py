import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from scipy.stats.mstats import winsorize

class DataTransform:
    def __init__(self, df):
        self.df = df

    def convert_employment_to_numeric(self, column_name):
        def convert(value):
            if pd.isna(value):
                return np.nan
            
            value = str(value)
            
            if '<' in value:
                return 0
            elif '+' in value:
                return int(value.split('+')[0])
            else:
                return int(value.split()[0])

        self.df[column_name] = self.df[column_name].apply(convert)
        return self.df

class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def describe_columns(self):
        "Describe all columns in the DataFrame to check their data types"
        print("Column Descriptions:")
        print(self.df.dtypes)
        print("\n")

    def statistical_summary(self):
        "Extract statistical values: median, standard deviation and mean from the columns and the DataFrame"
        print("Statistical Summary:")
        print("Mean:\n", self.df.mean(numeric_only=True))
        print("\nMedian:\n", self.df.median(numeric_only=True))
        print("\nStandard Deviation:\n", self.df.std(numeric_only=True))
        print("\n")

    def count_distinct_values(self):
        "Count distinct values in categorical columns"
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        print("Distinct Values in Categorical Columns:")
        for col in categorical_cols:
            print(f"{col}: {self.df[col].nunique()} distinct values")
        print("\n")

    def print_shape(self):
        "Print out the shape of the DataFrame"
        print(f"The DataFrame has {self.df.shape[0]} rows and {self.df.shape[1]} columns.\n")

    def null_value_summary(self):
        "Generate a count/percentage count of NULL values in each column"
        print("NULL Value Summary:")
        null_counts = self.df.isnull().sum()
        null_percentage = (null_counts / len(self.df)) * 100
        null_summary = pd.DataFrame({
            'Null Count': null_counts,
            'Null Percentage': null_percentage
        }).query("`Null Count` > 0")
        print(null_summary)
        print("\n")

    def identify_skewed_columns(self, threshold=0):
        "Identify columns that have a skewness greater than the given threshold."
        skewness = self.df.skew(numeric_only=True)
        skewed_cols = skewness[abs(skewness) > threshold]
        skewed_dict = skewed_cols.to_dict()
        print(f"Skewed Columns (>|{threshold}| skewness):")
        for col, skew_value in skewed_dict.items():
            print(f" - {col}: {skew_value:.3f}")

       
class Plotter:
    def __init__(self, df):
        self.df = df

    def plot_nulls(self, null_summary_before, null_summary_after):
        "Visualise removal of NULL values with a bar plot."
        plt.figure(figsize=(12, 6))
        width = 0.20
        common_columns = null_summary_before[(null_summary_before > 0)].index.intersection(null_summary_after.index)
        null_summary_before = null_summary_before.loc[common_columns]
        null_summary_after = null_summary_after.loc[common_columns]
             
        indices = np.arange(len(common_columns))

        plt.bar(indices, null_summary_before, width, label='Before NULL Removal', color='skyblue')
        plt.bar(indices + width, null_summary_after, width, label='After NULL Removal', color='salmon')

        plt.xticks(indices + width / 2, common_columns, rotation=90)
        plt.ylabel('Number of NULLs')
        plt.title('Comparison of NULL Counts Before and After Removal')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_skew(self, threshold=1, exclude_cols=None):
        skewness = self.df.skew(numeric_only=True)
        skewed_cols = skewness[abs(skewness) > threshold].index.tolist()
        skewed_cols = [col for col in skewed_cols if col not in exclude_cols]
        "Plot histograms of skewed columns to analyze distribution."
        for col in skewed_cols:
            plt.figure(figsize=(10, 4))

            # Histogram
            plt.subplot(1, 2, 1)
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f"Histogram of {col}")

    def plot_outliers(self, numeric_cols):
        "Plots box plots to visualize outliers in numeric columns."
        plt.figure(figsize=(15, 5 * len(numeric_cols)))
        
        # Box Plots
        for i, col in enumerate(numeric_cols):
            plt.subplot(len(numeric_cols), 1, i + 1)
            sns.boxplot(data=self.df[col])
            plt.title(f"Box Plot of {col}")
            plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        "Computes and visualizes the correlation matrix."
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.df.corr(numeric_only=True)
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

class DataFrameTransform:
    def __init__(self, df):
        self.df = df

    def drop_high_null_columns(self, threshold=50):
        "Drop columns with a NULL percentage above the threshold."
        null_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        columns_to_drop = null_percentage[null_percentage > threshold].index
        self.df.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns with >{threshold}% NULLs: {list(columns_to_drop)}\n")

    def impute_nulls(self):
        "Impute NULL values with the mean, median (for numerical data) or mode (for categorical data)."
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                if abs(self.df[col].skew()) > 1:
                    self.df.fillna({col: self.df[col].median()}, inplace=True)
                    print(f"Imputed column '{col}' with median.")
                else:
                    self.df.fillna({col: self.df[col].mean()}, inplace=True)
                    print(f"Imputed column '{col}' with mean.")

        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()[0]
                self.df.fillna({col: mode_value}, inplace=True)
                print(f"Imputed column '{col}' with mode: '{mode_value}'.")
        print("\n")

    def transform_skewed_columns(self, threshold=1, exclude_cols=None):
        skewness = self.df.skew(numeric_only=True)
        skewed_cols = skewness[abs(skewness) > threshold].index.tolist()
        skewed_cols = [col for col in skewed_cols if col not in exclude_cols]
        "Apply transformations to reduce skewness."
        for col in skewed_cols:
            if (self.df[col] > 0).sum() / len(self.df[col]) > 0.95: 
                log_transformed = np.log1p(self.df[col] + 1e-6)
                sqrt_transformed = np.sqrt(self.df[col] + 1e-6)
                boxcox_transformed, _ = boxcox(self.df[col] + 1)

                # Choose transformation with the least skewness
                transformations = {
                    "log": log_transformed.skew(),
                    "sqrt": sqrt_transformed.skew(),
                    "boxcox": pd.Series(boxcox_transformed).skew(),
                }
                best_transformation = min(transformations, key=lambda k: abs(transformations[k]))

                # Apply the best transformation
                if best_transformation == "log":
                    self.df[col] = log_transformed
                elif best_transformation == "sqrt":
                    self.df[col] = sqrt_transformed
                else:
                    self.df[col] = boxcox_transformed

                print(f"Applied {best_transformation} transformation to {col}. Skewness after: {self.df[col].skew()}")
        print("\n")
    
    def handle_outliers(self, method="remove", threshold=1.5):
            """
            Detects and handles outliers in numeric columns.
            method = "remove" → Removes outliers using the IQR method.
            method = "winsorize" → Caps outliers using Winsorization.
            """
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['id', 'member_id']]

            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)

                if method == "remove":
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                    print(f"Removed outliers from {col}.")
                
                elif method == "winsorize":
                    self.df[col] = winsorize(self.df[col], limits=[0.05, 0.05])  # Winsorize 5% from both ends
                    print(f"Winsorized outliers in {col}.")
    
    def identify_highly_correlated_columns(self, threshold=0.85):
        "Identifies columns that are highly correlated (above a given threshold)."
        correlation_matrix = self.df.corr(numeric_only=True).abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

        highly_correlated_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        print(f"Highly correlated columns (>|{threshold}| correlation): {highly_correlated_cols}\n")
        return(highly_correlated_cols)

    def remove_highly_correlated_columns(self, threshold=0.85):
        "Removes highly correlated columns based on the correlation threshold."
        cols_to_remove = self.identify_highly_correlated_columns(threshold)
        self.df.drop(columns=cols_to_remove, inplace=True)
        print(f"Removed columns: {cols_to_remove}")