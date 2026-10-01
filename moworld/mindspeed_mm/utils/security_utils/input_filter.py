from pandas import DataFrame
import pandas as pd


def sanitize_dataframe(
        df: pd.DataFrame,
        escape_char: str = '"'
) -> DataFrame:
    """
    Wrap all cell values in the DataFrame with specified quotation marks

    Parameters:
        df: Original DataFrame
        escape_char: Quotation mark to use for wrapping (default: double quote)

    Returns:
        New DataFrame with all cells wrapped in quotation marks
    """
    result_df = df.copy()

    # Add quotes to column names
    result_df.columns = [f"{escape_char}{col}{escape_char}" for col in result_df.columns]

    # Apply wrapping to all columns
    for col in result_df.columns:
        result_df[col] = result_df[col].apply(
            lambda x: f"{escape_char}{x}{escape_char}"
        )

    return result_df
