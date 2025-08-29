#  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣦⣴⣶⣾⣿⣶⣶⣶⣶⣦⣤⣄⠀⠀⠀⠀⠀⠀⠀
#  ⠀⠀⠀⠀⠀⠀⠀⢠⡶⠻⠛⠟⠋⠉⠀⠈⠤⠴⠶⠶⢾⣿⣿⣿⣷⣦⠄⠀⠀⠀        𓐓  preprocessing.py 𓐔           
#  ⠀⠀⠀⠀⠀⢀⠔⠋⠀⠀⠤⠒⠒⢲⠀⠀⠀⢀⣠⣤⣤⣬⣽⣿⣿⣿⣷⣄⠀⠀
#  ⠀⠀⠀⣀⣎⢤⣶⣾⠅⠀⠀⢀⡤⠏⠀⠀⠀⠠⣄⣈⡙⠻⢿⣿⣿⣿⣿⣿⣦⠀       Eng: oezzaou <oussama.ezzaou@gmail.com>
#  ⢀⠔⠉⠀⠊⠿⠿⣿⠂⠠⠢⣤⠤⣤⣼⣿⣶⣶⣤⣝⣻⣷⣦⣍⡻⣿⣿⣿⣿⡀
#  ⢾⣾⣆⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠉⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇
#  ⠀⠈⢋⢹⠋⠉⠙⢦⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇       Created: 2025/07/24 13:45:35 by oezzaou
#  ⠀⠀⠀⠑⠀⠀⠀⠈⡇⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇       Updated: 2025/08/29 10:37:02 by oezzaou
#  ⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢀⣾⣿⣿⠿⠟⠛⠋⠛⢿⣿⣿⠻⣿⣿⣿⣿⡿⠀
#  ⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⢠⣿⣟⣭⣤⣶⣦⣄⡀⠀⠀⠈⠻⠀⠘⣿⣿⣿⠇⠀
#  ⠀⠀⠀⠀⠀⠱⠤⠊⠀⢀⣿⡿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠘⣿⠏⠀⠀                             𓆩♕𓆪
#  ⠀⠀⠀⠀⠀⡄⠀⠀⠀⠘⢧⡀⠀⠀⠸⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠐⠋⠀⠀⠀                     𓄂 oussama ezzaou𓆃
#  ⠀⠀⠀⠀⠀⠘⠄⣀⡀⠸⠓⠀⠀⠀⠠⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

# ===[ Imports: ]==============================================================
from utils.logger import getLogger
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import month_plot, quarter_plot
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.seasonal import seasonal_decompose


# ===[ clean_data: ]===========================================================
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    logger = getLogger(__name__)
    logger.info("-> Cleaning data ...")
    # INFO:---------------------------------------------------------------
    # - The process of cleaning data depends on the dataset that are given
    return data


# ===[ EDA ]==================================================================
def exploratory_data_analysis(data: pd.DataFrame):
    logger = getLogger(__name__)
    logger.info("-> EDA (Exploratory Data Analysis) ...")
    return data


# ===[ transform_data ]========================================================
def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    logger = getLogger(__name__)
    logger.info("-> Data Transformation ...")
    return data


# ===[ create_feature ]========================================================
def create_feature(data: pd.DataFrame) -> pd.DataFrame:
    logger = getLogger(__name__)
    logger.info("-> Creating Features ...")
    return data


# ===[ preprocess_data: ]======================================================
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # 1. Cleaning dataset (sum + date creation)
    data = clean_data(data)
    # 2. Data Transformation
    data = transform_data(data)
    # 3. EDA (Exploratory Data Analysis) => Summarize & Visualize the data
    exploratory_data_analysis(data)
    # 4. Feature Engineering (Create Features)
    return create_feature(data)
