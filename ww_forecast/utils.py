import numpy as np 
import pandas as pd
from typing import Generator, Tuple


def simpsons_diversity(df: pd.Series) -> float:
    """
    Calculate Simpson's Diversity Index
    """
    return sum(df**2)


def simpsons_evenness(df: pd.Series) -> float:
    """
    Calculate Simpson's Evenness Index
    """
    diversity = simpsons_diversity(df)
    if diversity != 0:
        return 1 / diversity
    else:
        return np.nan


def get_window_dates(start: pd.Timestamp, end: pd.Timestamp, slide: int, width: int) -> Generator[Tuple[pd.Timestamp, pd.Timestamp], None, None]:
    """
    Generator function to yield the start and end dates of a window given its width and the slide length.

    Parameters:
    start (pd.Timestamp): The overall start date.
    end (pd.Timestamp): The overall end date.
    slide (int): The length of the slide (in days).
    width (int): The width of the window (in days).

    Yields:
    Tuple[pd.Timestamp, pd.Timestamp]: The start and end dates of each window.
    """
    window_start = start
    window_end = start + pd.DateOffset(days=width)

    while window_end <= end:
        yield window_start, window_end
        window_start += pd.DateOffset(days=slide)
        window_end += pd.DateOffset(days=slide)


def logistic_growth(ndays: float, b: float, r: float) -> float:
    """
    Adapted from Freyja v1.3.1.
    
    Calculates logistic growth.

    Parameters:
    ndays (float): The number of days.
    b (float): The upper limit of the logistic function.
    r (float): The growth rate of the logistic function.

    Returns:
    float: The logistic growth at ndays.
    """
    return 1 / (1 + (b * np.exp(-1 * r * ndays)))


def cv(x):
    return np.nanstd(x) / np.nanmean(x)
