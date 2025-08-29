#  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣦⣴⣶⣾⣿⣶⣶⣶⣶⣦⣤⣄⠀⠀⠀⠀⠀⠀⠀
#  ⠀⠀⠀⠀⠀⠀⠀⢠⡶⠻⠛⠟⠋⠉⠀⠈⠤⠴⠶⠶⢾⣿⣿⣿⣷⣦⠄⠀⠀⠀             𓐓  evaluate.py 𓐔           
#  ⠀⠀⠀⠀⠀⢀⠔⠋⠀⠀⠤⠒⠒⢲⠀⠀⠀⢀⣠⣤⣤⣬⣽⣿⣿⣿⣷⣄⠀⠀
#  ⠀⠀⠀⣀⣎⢤⣶⣾⠅⠀⠀⢀⡤⠏⠀⠀⠀⠠⣄⣈⡙⠻⢿⣿⣿⣿⣿⣿⣦⠀      Eng: oezzaou <oussama.ezzaou@gmail.com>
#  ⢀⠔⠉⠀⠊⠿⠿⣿⠂⠠⠢⣤⠤⣤⣼⣿⣶⣶⣤⣝⣻⣷⣦⣍⡻⣿⣿⣿⣿⡀
#  ⢾⣾⣆⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠉⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇
#  ⠀⠈⢋⢹⠋⠉⠙⢦⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇       Created: 2025/08/27 15:05:40 by oezzaou
#  ⠀⠀⠀⠑⠀⠀⠀⠈⡇⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇       Updated: 2025/08/29 10:36:21 by oezzaou
#  ⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢀⣾⣿⣿⠿⠟⠛⠋⠛⢿⣿⣿⠻⣿⣿⣿⣿⡿⠀
#  ⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⢠⣿⣟⣭⣤⣶⣦⣄⡀⠀⠀⠈⠻⠀⠘⣿⣿⣿⠇⠀
#  ⠀⠀⠀⠀⠀⠱⠤⠊⠀⢀⣿⡿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠘⣿⠏⠀⠀                             𓆩♕𓆪
#  ⠀⠀⠀⠀⠀⡄⠀⠀⠀⠘⢧⡀⠀⠀⠸⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠐⠋⠀⠀⠀                     𓄂 oussama ezzaou𓆃
#  ⠀⠀⠀⠀⠀⠘⠄⣀⡀⠸⠓⠀⠀⠀⠠⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

# ===[ Imports ]===============================================================
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
# from utils.logger import getLogger
import matplotlib.pyplot as plt
import config


# ===[ plot_forecasts ]========================================================
def plot_forecasts(train, test, forecasts, title):
    '''
      - This function used to plot `train`, `test` and 'forecasts/predictions'
    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train, label="Train")
    ax.plot(test, label="Test")
    ax.plot(forecasts, label="Forecasts/Predictions")
    ax.set_title(title)
    plt.legend()
    plt.show()


# ===[ compute_forecast_metrics ]==============================================
def compute_forecast_metrics(forecasts, actuals):
    '''
        - This function measure the error and tells you how good your model are
          using different metrics like:
            . MAE: Mean Absolute Error
            . RMSE: Root Mean Squared Error
            . MAPE: Mean Absolute Percentage Error
    '''
    mae = mean_absolute_error(actuals, forecasts)
    rmse = root_mean_squared_error(actuals, forecasts)
    mape = mean_absolute_percentage_error(actuals, forecasts)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape * 100  # convert to percentage
        # NOTE:> You can add or replace metrices
    }
    return metrics


# ===[ evaluate_model ]========================================================
def evaluate_model(model, train_set, test_set, plot=False):
    '''
      - This function evaluate the model, this process include:
        1. Train the given model
        2. Test the model using the `test_set`
        3. Compute metrics using different approaches (RMSE, MAE, MAPE, etc)
        4. Visualize `train_set`, `test_set` and `forecasts/predictions` if
           plot=True
    '''
    # logger.getLogger(__name__)
    # logger.info("Model Evaluation ...")
    train_model = model.fit()
    forecasts = train_model.forecast(config.period)
    # logger.debug(f"Forecasts: {forecasts}")
    metrics = compute_forecast_metrics(test_set, forecasts)
    if plot is True:
        plot_forecasts(train_set, None, forecasts, "Holt winter forecasts")
    return forecasts, metrics


# INFO:[ `evaluate.py` ]-------------------------------------------------------
# - This file does:
#   1. 'regression_metrics': computes numerical evaluation metrics (MAE, RMSE
#      , R^2)
#   2. 'plot_predictions': plots actual vs predicted (especially useful for
#       time series)
#   3. 'evaluate_model': a wrapper that ties everything together:
#      [ prediction + metrics + visualization ]
# - This makes 'evaluate.py' reusable: you just call:
#   > evaluate_model(train_modelmodel, X_test, y_test) from anywhere in your
#     pipeline
