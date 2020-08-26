#%%
import pandas as pd 
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.backend import batch_normalization, dtype 

# %%
data = pd.read_csv("train_set.csv")
print(data.columns)
# %%
print(data.head())
# %%
#task: find optimal stator and rotor temperatore using given data
#to predict: pm, stator_yoke, stator_tooth, stator_winding
#time stamp: each row is recorded every 0.5 seconds. 
# Individual measurement sessions last between 1 and 6 hours and can be found using profile_id


print("checking if each profile contains data for 1 to 6 hours")
for i in data['profile_id'].unique():
    len(data[data["profile_id"] == i])
    #each second, you get 2 values, for 1 hour, it should be count/(3600*2)
    hours = len(data[data["profile_id"] == i])/(3600*2)
    if(hours >= 6 or hours <= 1):
        print(f"potential anomaly in session {i} with {hours:.2f} hours")
    print(f'session {i} lasted {hours:.2f} hours')
# %%
#for profile 4
data_p4 = data[data.profile_id == 81]
plt.plot(data_p4.pm, label="rotor temp")
# plt.plot(data_p4.stator_yoke, label="stator yoke temp")
# plt.plot(data_p4.stator_tooth, label="stator tooth temp")
# plt.plot(data_p4.stator_winding, label="stator winding temp")
plt.plot(data_p4.ambient, label="ambient temp")
plt.legend()
plt.show()
# %%
data_p4 = data[data.profile_id == 4]
plt.plot(data_p4.pm, label="rotor temp")
# plt.plot(data_p4.stator_yoke, label="stator yoke temp")
# plt.plot(data_p4.stator_tooth, label="stator tooth temp")
# plt.plot(data_p4.stator_winding, label="stator winding temp")
plt.plot(data_p4.ambient, label="ambient temp")
plt.legend()
plt.show()
# %%
def get_data(data):
    data_X = data[['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d',
        'i_q', 'stator_yoke', 'stator_tooth', 'stator_winding','pm'
        ]]
    data_Y = data[['pm']]
    count = round(len(data)*0.7)
    X_train, Y_train,  = data_X.to_numpy()[:count,:], data_Y.to_numpy()[:count,:]
    X_valid, Y_valid,  = data_X.to_numpy()[count:,:], data_Y.to_numpy()[count:,:]
    print(f"X_train_shape initial = {X_train.shape} {X_valid.shape}")
    #TODO normalize data across row dimension
    def normalize_rows(x: np.ndarray):
        """
        function that normalizes each row of the matrix x to have unit length.

        Args:
        ``x``: A numpy matrix of shape (n, m)

        Returns:
        ``x``: The normalized (by row) numpy matrix.
        """
        return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    X_train = normalize_rows(X_train)
    X_valid = normalize_rows(X_valid)
    print(f"X_train_shape after normalize = {X_train.shape} {X_valid.shape}")
    X_train = X_train[:,:,np.newaxis]
    X_valid = X_valid[:,:,np.newaxis]

    return X_train, Y_train, X_valid, Y_valid
X_train, Y_train, X_valid, Y_valid = get_data(data_p4)
print(f"X_train_shape after adding axis = {X_train.shape} {X_valid.shape}")
print(f"Y_train_shape after adding axis = {Y_train.shape} {Y_valid.shape}")
# %%
def last_time_step_mse(y_true,y_pred):
    return keras.metrics.mean_squared_error(y_true[:,-1],y_pred[:,-1])
#%%
tf.random.set_seed(42)
model = keras.models.Sequential(
    [keras.layers.GRU(10, return_sequences=True, input_shape=[None,1]),
    keras.layers.GRU(10, return_sequences=True), 
    keras.layers.TimeDistributed(keras.layers.Dense(1, activation=None))
    ]
)
print(f"X_train_shape before processing = {X_train.shape}")
model.compile(loss="mse", optimizer="adam")
print(model.summary())
#model.predict(X_train[0])
history = model.fit(X_train,Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# %%
import matplotlib as mpl
def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss))+0.5,loss,"b.-",label="Training Loss")
    plt.plot(np.arange(len(val_loss))+0.5,val_loss,"r.-",label="Validation Loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
plot_learning_curves(history.history['loss'], history.history['val_loss'])
# %%
tf.random.set_seed(42)
model = keras.models.Sequential(
    [keras.layers.LSTM(20, return_sequences=True, input_shape=[None,1]),
    keras.layers.LSTM(20, return_sequences=True), 
    keras.layers.BatchNormalization(),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
    ]
)
print(f"X_train_shape before processing = {X_train.shape}")
model.compile(loss="mse", optimizer="adam")
print(model.summary())
history = model.fit(X_train,Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# %%
plot_learning_curves(history.history['loss'], history.history['val_loss'])
# %%
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None,1]))
for rate  in (1,2,4,8) *2:
    model.add(keras.layers.Conv1D(filters=40, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))
model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
model.compile(loss="mse", optimizer="adam",  metrics=[keras.metrics.RootMeanSquaredError()])
history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid))
# %%
plot_learning_curves(history.history['loss'], history.history['val_loss'])
# %%
plot_learning_curves(history.history['root_mean_squared_error'], history.history['val_root_mean_squared_error'])
#%%
pred_values = model.predict(X_valid)
rms = keras.metrics.mean_squared_error(pred_values, Y_valid)

# %%
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[None,1]),
    keras.layers.Dense(40,activation='selu', kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(40, activation='selu', kernel_initializer='lecun_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1)]
)
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)
model.compile(loss=tf.losses.mean_squared_error, optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError()])
print(model.summary())
# %%
history = model.fit(X_train, Y_train, epochs=40, validation_data=(X_valid, Y_valid))
# %%
plot_learning_curves(history.history['loss'], history.history['val_loss'])
# %%
plot_learning_curves(history.history['root_mean_squared_error'], history.history['val_root_mean_squared_error'])

# %%
'''predict multiple items ''' 
def get_data(data):
    data_X = data[['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d',
        'i_q',
        ]]
    data_Y = data[['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']]
    count = round(len(data)*0.7)
    X_train, Y_train,  = data_X.to_numpy()[:count,:], data_Y.to_numpy()[:count,:]
    X_valid, Y_valid,  = data_X.to_numpy()[count:,:], data_Y.to_numpy()[count:,:]
    print(f"X_train_shape initial = {X_train.shape} {X_valid.shape}")
    #TODO normalize data across row dimension
    def normalize_rows(x: np.ndarray):
        """
        function that normalizes each row of the matrix x to have unit length.

        Args:
        ``x``: A numpy matrix of shape (n, m)

        Returns:
        ``x``: The normalized (by row) numpy matrix.
        """
        return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    X_train = normalize_rows(X_train)
    X_valid = normalize_rows(X_valid)
    print(f"X_train_shape after normalize = {X_train.shape} {X_valid.shape}")
    X_train = X_train[:,:,np.newaxis]
    X_valid = X_valid[:,:,np.newaxis]
    Y_train = Y_train[:, np.newaxis,:]
    Y_valid = Y_valid[:, np.newaxis,:]
    return X_train, Y_train, X_valid, Y_valid
X_train, Y_train, X_valid, Y_valid = get_data(data_p4)
# %%
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[None,1]),
    keras.layers.Dense(40,activation='selu', kernel_initializer="lecun_normal"),
    keras.layers.Dense(40, activation='selu', kernel_initializer='lecun_normal'),
    keras.layers.Dense(4)]
)

model.compile(loss=tf.losses.mean_squared_error, optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError()])
print(model.summary())
# %%
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# %%
plot_learning_curves(history.history['loss'], history.history['val_loss'])

#%%
tf.random.set_seed(42)
model = keras.models.Sequential(
    [keras.layers.LSTM(15, return_sequences=True, input_shape=[None,1]),

    keras.layers.LSTM(15, return_sequences=True), 
    keras.layers.TimeDistributed(keras.layers.Dense(4))
    ]
)
print(f"X_train_shape before processing = {X_train.shape}")
model.compile(loss="mse", optimizer=optimizer)
print(model.summary())
X_train, Y_train, X_valid, Y_valid = get_data(data)
history = model.fit(X_train,Y_train, epochs=10, validation_data=(X_valid, Y_valid))
# %%
plot_learning_curves(history.history['loss'], history.history['val_loss'])
# %%


# %%
#using a sequence to sequence model to predict only PM 1 window ahead
data = pd.read_csv("train_set.csv")
X_train = data[["pm","stator_tooth","stator_yoke","stator_winding"]]
#X_train, Y_train, X_valid, Y_valid = get_data(data_p4)
dataset_size = X_train.shape[0]
train_size = dataset_size * 90//100

dataset = tf.data.Dataset.from_tensor_slices(X_train[:train_size])
#dataset = tf.data.Dataset.from_tensor_slices(X_train[:])
# %%
#convert into window
n_steps = 100
window_length = n_steps + 1 #target is now input shifted 1 row ahread
dataset = dataset.window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
# %%
batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:,:,:-1], windows[:,:,1:]))
dataset = dataset.map(lambda X_batch, Y_batch:( X_batch[:, :] ,Y_batch[:, :]))
# %%
#dataset = dataset.map(X_batch, Y_batch : X_batch, Y_batch)
model = keras.models.Sequential([
    keras.layers.GRU(32,return_sequences=True, input_shape=[None, 3], dropout=0.2),
    keras.layers.GRU(32, return_sequences=True, dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(3))
])


model.compile(loss="mse", optimizer="adam", )
print(model.summary())
# %%
history = model.fit(dataset, epochs=10,steps_per_epoch=1000)
# %%
plt.plot(history.history['loss'])
# %%
