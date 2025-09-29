import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix
from joblib import Parallel, delayed
import timeit

# from src
from src.data_loader import VoiceDataLoader

# --- Functions from the notebook ---

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)



def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_features(path,duration=2.5, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset)
    aud=extract_features(data)
    audio=np.array(aud)
    
    noised_audio=noise(data)
    aud2=extract_features(noised_audio)
    audio=np.vstack((audio,aud2))
    

    
    return audio

def process_feature(path, emotion):
    features = get_features(path)
    X = []
    Y = []
    for ele in features:
        X.append(ele)
        Y.append(emotion)
    return X, Y

if __name__ == '__main__':
    # 1. Load RAVDESS data
    print("--- Loading RAVDESS Data ---")
    data_loader = VoiceDataLoader(data_path='e:/Assignment_data/voice processing/data/raw')
    ravdess_files = data_loader.load_ravdess_data()
    data_path = pd.DataFrame(ravdess_files)

    # 2. Feature Extraction and Augmentation
    print("--- Feature Extraction and Augmentation ---")
    start = timeit.default_timer()
    
    paths = data_path.path
    emotions = data_path.emotion

    results = Parallel(n_jobs=-1)(delayed(process_feature)(path, emotion) for (path, emotion) in zip(paths, emotions))

    X = []
    Y = []
    for result in results:
        x_chunk, y_chunk = result
        X.extend(x_chunk)
        Y.extend(y_chunk)

    stop = timeit.default_timer()
    print(f'Feature extraction time: {stop - start:.2f}s')

    # 3. Data Preparation
    print("--- Data Preparation ---")
    Emotions = pd.DataFrame(X)
    Emotions['Emotions'] = Y
    Emotions = Emotions.fillna(0)

    X = Emotions.iloc[: ,:-1].values
    Y = Emotions['Emotions'].values

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

    # 4. Train/Test split
    print("--- Splitting Data ---")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42,test_size=0.2, shuffle=True)

    # 5. Scaling
    print("--- Scaling Data ---")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 6. Reshape for CNN
    print("--- Reshaping data for CNN ---")
    x_traincnn =np.expand_dims(x_train, axis=2)
    x_testcnn= np.expand_dims(x_test, axis=2)

    # 7. Build and Train CNN model
    print("--- Building and Training CNN model ---")
    model = Sequential([
        Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(x_train.shape[1],1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=5,strides=2,padding='same'),
        
        Conv1D(512,kernel_size=5,strides=1,padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=5,strides=2,padding='same'),
        Dropout(0.2),
        
        Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=5,strides=2,padding='same'),
        
        Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=5,strides=2,padding='same'),
        Dropout(0.2),
        
        Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3,strides=2,padding='same'),
        Dropout(0.2),
        
        Flatten(),
        Dense(512,activation='relu'),
        BatchNormalization(),
        Dense(y_train.shape[1],activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()

    history=model.fit(x_traincnn, y_train, epochs=50, validation_data=(x_testcnn, y_test), batch_size=64)

    # 8. Evaluation
    print("--- Model Evaluation ---")
    print(f"Accuracy of our model on test data : {model.evaluate(x_testcnn,y_test)[1]*100} %")
    pred_test = model.predict(x_testcnn)
    y_pred = encoder.inverse_transform(pred_test)
    y_test_labels = encoder.inverse_transform(y_test)

    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred))
