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

# $example on$
from __future__ import print_function

from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint


import numpy as np

def predict_probabilities(model, x):
    margin = []
    if x.size + 1 == model._dataWithBiasSize:
        for i in range(0, model._numClasses - 1):
            margin.append(x.dot(model._weightsMatrix[i][0:x.size]) + model._weightsMatrix[i][x.size])

    else:
        for i in range(0, model._numClasses - 1):
            margin.append(x.dot(model._weightsMatrix[i]))
    
    # Here, K-1 Binary logistic regression models are created 
    # and class 0 is chosen as reference model
    denom = 1 + np.sum(np.exp(margin))
    # for i in range(1,model._numClasses - 1):
        # denom += np.exp(margin)

    #return [x.dot(model._weightsMatrix[i]) for i in range(0, model._numClasses-1)]
    
    # proba is a list of individual probabilities for classes 1 to (numClasses - 1)
    proba = [np.exp(margin[i])/denom for i in range(0,model._numClasses - 1)]    

    # Return a list of probabilities including that of class 0 
    # I assume the probability of class 0 is "1-sum(proba)" 
    # This makes sense while running using the sample data.
    # Do check for mathematical correctness.
    return [1-np.sum(proba)] + proba

def log_loss(y_act, y_pred_proba, num_classes):
    y_act_one_hot = np.eye(num_classes)[y_act]
    return -1*np.sum(y_act_one_hot*np.log(y_pred_proba))  # Divide this value by no of records on return


# Parse fn for Crime-category prediction WITH ONE HOT ENCODING and without 
# (it's the same for crime category prediction)
def parseLine(line):
    parts = line.split(',')
    label = float(parts[0])
    features = map(float, parts[1:])
    # features = Vectors.dense(features)
    return LabeledPoint(label, features)

# $example on$
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
# $example off$

from pyspark import SparkContext

if __name__ == "__main__":
    sc = SparkContext(appName="MultiClassMetrics_CrimeCategoryPrediction_WithOneHotEncoding")

    # Several of the methods available in scala are currently missing from pyspark
    # $example on$
    # Load training data in LIBSVM format
    # data = MLUtils.loadLibSVMFile(sc, "")

    # USE THIS FOR ONE-HOT
    data = sc.textFile('/usr/sfd_train_one_hot.csv').map(parseLine)
    
    # data = sc.textFile('/usr/sfd_0.csv').map(parseLine)

    # Split data into training (60%) and test (40%)
    training, test = data.randomSplit([0.85, 0.15], seed=11L)
    training.cache()

    # Run training algorithm to build the model
    model = LogisticRegressionWithLBFGS.train(training, numClasses=39)

    # Compute raw scores on the test set
    predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))

    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)
    
    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    #accuracy = metrics.accuracy
    accuracy = 1.0 * predictionAndLabels.filter(lambda (x, v): x == v).count() / test.count()
    # print("Summary Stats")
    # print("Precision = %s" % precision)
    # print("Recall = %s" % recall)
    # print("F1 Score = %s" % f1Score)    
    # print("Accuracy = %s" % accuracy)
    # Statistics by class
    labels = data.map(lambda lp: lp.label).distinct().collect()
    # for label in sorted(labels):
    #     print("Class %s precision = %s" % (label, metrics.precision(label)))
    #     print("Class %s recall = %s" % (label, metrics.recall(label)))
    #     print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

    # Weighted stats
    # print("Weighted recall = %s" % metrics.weightedRecall)
    # print("Weighted precision = %s" % metrics.weightedPrecision)
    # print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    # print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    # print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
    # $example off$
    
    probabilitiesAndLabels = test.map(lambda lp: (predict_probabilities(model, lp.features), lp.label, float(model.predict(lp.features))))

    logloss = probabilitiesAndLabels.map(lambda pl : log_loss(pl[1], pl[0], 39)).reduce(lambda x,y: x+y)/test.count()
    print ("Logloss : ", logloss)