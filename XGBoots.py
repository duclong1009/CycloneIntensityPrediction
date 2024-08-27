import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

data_dir = "data07/cropped_data"

# Load training data
data_dir_train = f"{data_dir}/train/data.npz"
arr_train = np.load(data_dir_train)
x_train, y_train = arr_train['x_arr'], arr_train['groundtruth']
y_train = y_train * 0.5

# Load testing data
data_dir_test = f"{data_dir}/test/data.npz"
arr_test = np.load(data_dir_test)
x_test, y_test = arr_test['x_arr'], arr_test['groundtruth']
y_test = y_test * 0.5

# Extract the specific slice of the arrays
x_train = x_train[:, :, 51, 51]
x_test = x_test[:, :, 51, 51]

# Reshape the data to 2D arrays for scaling
x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1)

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
x_train_scaled = scaler.fit_transform(x_train_reshaped)
x_test_scaled = scaler.transform(x_test_reshaped)

# Initialize and train the model
model = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                     max_depth=5, alpha=10, n_estimators=100)

model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
from repo import model_utils

import wandb

wandb.login(key='ab2505638ca8fabd9114e88f3449ddb51e15a942')
wandb.init(
    entity="aiotlab",
    project="Cyclone intensity prediction",
    group="Data2",
    name=f"XGBoots",
    config={},
)
mae, mse, mape, rmse, r2, corr = model_utils.cal_acc(y_pred, y_test)


print(f"MSE: {mse} MAE:{mae} MAPE:{mape} RMSE:{rmse} R2:{r2} Corr:{corr}")
data = [[pred, gt] for pred, gt in zip(list(y_pred), list(y_test))]
table = wandb.Table(data=data, columns=["Prediction", "Ground Truth"])
wandb.log({"predictions_vs_groundtruths_table": table})

# Optionally, create a plot to visualize the predictions vs ground truths
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))
plt.plot(list(y_pred), label='Predictions', marker='o')
plt.plot(list(y_test), label='Ground Truths', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Predictions vs Ground Truths')
plt.legend()
plt.grid(True)

# Log the plot to W&B
wandb.log({"predictions_vs_groundtruths_plot": wandb.Image(plt)})
plt.close()

# Finish the W&B run
wandb.finish()