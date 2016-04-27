#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
NaiveBayes Example.
"""
from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import numpy as np

#from pyspark.mllib._common import _dot

# def parseLine(line):
#     parts = line.split(',')
#     label = float(parts[2])
#     features = Vectors.dense(map(float,[parts[0], parts[1], parts[3]]))
#     return LabeledPoint(label, features)

# $example off$

# Parse Fn for OneHot encoding
def parseLine(line):
    parts = line.split(',')
    label = float(parts[0])
    feature_vector = map(float, parts[1:]) 
    # for feat in parts[1:]:
    #     feature_vector.append(float(feat))
    features = Vectors.dense(feature_vector)
    # features = Vectors.dense([float(x) for x in parts[1].split(' ')])
    return LabeledPoint(label, features)


def predict_probability(model, x):
    unnormalized_log = model.pi + x.dot(model.theta.transpose())
    return np.exp(unnormalized_log)/np.sum(np.exp(unnormalized_log))

def log_loss(y_act, y_pred, num_classes):
    y_act_one_hot = np.eye(num_classes)[y_act]
    return -1*np.sum(y_act_one_hot*np.log(y_pred))

if __name__ == "__main__":

    sc = SparkContext(appName="PythonNaiveBayesExample")

    # $example on$
    #data = sc.textFile('data/mllib/sample_naive_bayes_data.txt').map(parseLine)
    # data = sc.textFile('/usr/sfd_0.csv').map(parseLine)
    data = sc.textFile('/usr/sfd_train_one_hot_2.csv').map(parseLine)
    # Split data aproximately into training (60%) and test (40%)
    training, test = data.randomSplit([0.7, 0.3], seed=0)

    num_training = training.count()
    num_test = test.count()
    # Train a naive Bayes model.
    model = NaiveBayes.train(training, 1.0)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
    
    # print("Accuracy: " + str(accuracy))

    probabilityAndLabel = test.map(lambda p: (p.label, predict_probability(model, p.features), 10))
    # for _,p,_ in probabilityAndLabel.collect():
    #     print(p)

    logloss = 1.0 * probabilityAndLabel.map(lambda p: log_loss(p[0],p[1],p[2])).reduce(lambda x,y: x+y)/num_test
    print("Logloss: ", logloss)
    #model.predictProbabilities()
    # Save and load model
    #model.save(sc, "target/tmp/myNaiveBayesModel")
    #sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
    # $example off$
