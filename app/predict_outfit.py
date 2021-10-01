import pymongo
from pymongo import ReturnDocument
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from bson.json_util import dumps
from matplotlib import pyplot as plt
import random
import os, sys, io
import base64

client = pymongo.MongoClient("localhost", 27017)
db = client.outfitPlanner

dataMapper = lambda outfit, addCustomFeatures: {
    "temp": outfit["weather"]["temp"],
    "worn_times": getWornTimes(outfit) if addCustomFeatures == True else 0,
    #"clouds": outfit["weather"]["clouds"],
    "humidity": outfit["weather"]["humidity"],
    #"pressure": outfit["weather"]["pressure"],
    "wind_speed": outfit["weather"]["wind_speed"]
}

def getWornTimes(outfit):
    if outfit.get("outfitId") == None:
        return 1
    else:
        return db.worns.count_documents({"outfitId": outfit["outfitId"]})

def scale_features(X):
    #X[:, 0] *= 1
    X[:, 1] *= 2
    return X

def getData():
    X, Y = getXY()
    X_scaled, scaler = scale_data(X)
    
    return (X_scaled, Y, scaler)

def scale_data(X):
    scaler = preprocessing.MinMaxScaler().fit(X)
    #scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_scaled = scale_features(X_scaled)
    return X_scaled, scaler

def get_XY(email, addCustomFeatures):
    outfits = list(db.worns.find({"email": email}))
    data = [dataMapper(outfit, addCustomFeatures) for outfit in outfits ]
    y_data = pd.DataFrame(outfits)
    Y = y_data.loc[:, y_data.columns == "outfitId"]

    df = pd.DataFrame(data)
    X = df.loc[:, df.columns != 'outfitId']
    return X, Y

def predict_new_case(neigh, scaler, weather, addCustomFeatures):
    print("WEATHER: ", weather)
    newOutfit = dict({
        "weather": weather
    })
    newOutfitWeighted = dataMapper(newOutfit, addCustomFeatures)
    newCase = [newOutfitWeighted]
    newCaseDf = pd.DataFrame(newCase)
    scaled = scaler.transform(newCaseDf[0:1])
    scaled = scale_features(scaled)

    pred_proba = neigh.predict_proba(scaled)
    return pred_proba[0]

def predict_probabilities(weather, X, Y, addCustomFeatures):
    X_scaled, scaler = scale_data(X)

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_scaled, Y.values.ravel())

    pred_proba = predict_new_case(neigh, scaler, weather, addCustomFeatures)

    zipped = zip(neigh.classes_, pred_proba)
    sortedOutfits = sorted(zipped, key = lambda t: t[1])
    print("Sorted outfits: ", sortedOutfits)
    return sortedOutfits

def predict(email, weather):
    X, Y = get_XY(email, True)
    sortedOutfits = predict_probabilities(weather, X, Y, True)

    wornIds = [int(x[0]) for x in sortedOutfits]
    res = np.array([list(db.outfits.find({"id": x})) for x in wornIds]).flatten()
    outfitsNeverWorn = np.array(list(db.outfits.find({"id": {"$nin": wornIds}, "email": email})))
    return np.concatenate((outfitsNeverWorn, res))


def plot(email, weather):
    X, Y = get_XY(email, False)
    sortedOutfits = predict_probabilities(weather, X, Y, False)
    wornIds = [int(x[0]) for x in sortedOutfits]
    outfitId = wornIds[-1]

    print(wornIds)

    colors = [np.where(wornIds == y['outfitId'])[0] for i, y in Y.iterrows()]

    X = X.append(weather, ignore_index=True)
    Y = Y.append({'outfitId':'new'}, ignore_index=True)
    colors.append(len(wornIds)-1)

    print(colors, 'colors')
    colors = np.array(colors) / 100

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X["temp"], X["humidity"], X["wind_speed"], c=colors)
    for i, row in X.iterrows():
        y = Y.iloc[i]['outfitId']
        x_offset = 0
        y_offset = 0
        z_offset = 0
        ax.text(row["temp"] + x_offset, row["humidity"] + y_offset, row["wind_speed"] + z_offset, '%s' % (y), size=10, zorder=1, color='k')

    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Humidity [%]')
    ax.set_zlabel('Wind speed [m/s]')

    plt.savefig("foo.png")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    base64_data = base64.b64encode(buf.read())

    return outfitId, base64_data.decode('ascii')