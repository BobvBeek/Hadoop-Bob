
# Hoi Thijs,

# Ik heb het deze keer in Visual Studio Code gemaakt ipv Jupyer Notebook. 
# Liep in Anaconda tegen veel problemen aan met versions van Spark. 
# Laadt het bestand in VScode in en zorg dat alles van Spark tot numpy gedownload is en het moet het doen!

# Importing Packages
import findspark
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression

findspark.init("C:/Users/Bob/Documents/spark")
 
spark = SparkSession.builder.master("local").appName(
    "Spark Assignment (3)").config("spark.executor.memory", "1gb").getOrCreate()
 
# Read the dataset
df = spark.read.csv("C:/Users/Bob/Downloads/titanic.csv",
                    inferSchema=True, header=True)
 
# Columns Age and Sex need to be integers to be fit for analysis.
df = df.withColumn("Age", df.Age.cast(IntegerType()))
df = df.withColumn("Sex", when(df.Sex == "male", 0).when(df.Sex == "female", 1))

print('\n')
# Question a
print('QUESTION ONE (a)')
print('Lets first calculate all the different probabilities for each statistic')

print('\n')
male_probability = len(df.toPandas()[df.toPandas().Sex == 0]) / len(df.toPandas())
female_probability = len(df.toPandas()[df.toPandas().Sex == 1]) / len(df.toPandas())
print('Male probability: ' + str(male_probability))
print('Female probability: ' + str(female_probability))
print('\n')
 
pc1_prob = len(df.toPandas()[df.toPandas().Pclass == 1]) / len(df.toPandas())
pc2_prob = len(df.toPandas()[df.toPandas().Pclass == 2]) / len(df.toPandas())
pc3_prob = len(df.toPandas()[df.toPandas().Pclass == 3]) / len(df.toPandas())
print('Passenger class 1 probability: ' + str(pc1_prob))
print('Passenger class 2 probability: ' + str(pc2_prob))
print('Passenger class 3 probability: ' + str(pc3_prob))
print('\n')
 
sv0_prob = len(df.toPandas()[df.toPandas().Survived == 0]) / len(df.toPandas())
sv1_prob = len(df.toPandas()[df.toPandas().Survived == 1]) / len(df.toPandas())
print('Did not survive probability: ' + str(sv0_prob))
print('Survived probability: ' + str(sv1_prob))
print('\n')
 
print('Lets now put it all together')
print('\n')

q1a = round((sv1_prob * female_probability * pc1_prob) * 100, 2)
q1b = round((sv1_prob * female_probability * pc2_prob) * 100, 2)
q1c = round((sv1_prob * female_probability * pc3_prob) * 100, 2)
q1d = round((sv1_prob * male_probability * pc1_prob) * 100, 2)
q1e = round((sv1_prob * male_probability * pc2_prob) * 100, 2)
q1f = round((sv1_prob * male_probability * pc3_prob) * 100, 2)
print('Question a1: P(S = true | G = female, C = 1) = ' + str(q1a) + '%')
print('Question a2: P(S = true | G = female, C = 2) = ' + str(q1b) + '%')
print('Question a3: P(S = true | G = female, C = 3) = ' + str(q1c) + '%')
print('Question a4: P(S = true | G = male, C = 1) = ' + str(q1d) + '%')
print('Question a5: P(S = true | G = male, C = 2) = ' + str(q1e) + '%')
print('Question a6: P(S = true | G = male, C = 3) = ' + str(q1f) + '%')
print('\n')

# Question b
print('QUESTION TWO (b)')
print('\n')

age10oryounger_prob = len(df.toPandas()[df.toPandas().Age <= 10]) / len(df.toPandas())
print('Probability of being age 10 or younger: ' + str(age10oryounger_prob))
print('\n')

q2 = round((sv1_prob * age10oryounger_prob * pc3_prob) * 100, 2)
print('Question b: P(S = true | A <= 10, C = 3) = ' + str(q2) + '%')
print('\n')

# Question c
print('QUESTION THREE (c)')
print('\n')
 
df_select = df.select("Fare", "Pclass")
input_data = df_select.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
df_vector = spark.createDataFrame(input_data, ["labels", "features"])
standardScaler = StandardScaler(
    inputCol="features", outputCol="features_scaled")
scaler = standardScaler.fit(df_vector)
df_scaled = scaler.transform(df_vector)
 
train_data, test_data = df_scaled.randomSplit([.8, .2], seed=1234)
 
lr = LinearRegression(labelCol="labels", maxIter=10,
                      regParam=0.3, elasticNetParam=0.8)
linearModel = lr.fit(train_data)
 
print('\n')
# Print the coefficient and intercept
print("The Coefficient: %s" % str(linearModel.coefficients))
print("The Intercept: %s" % str(linearModel.intercept))
print('\n')
 
# Print out some metrics of the summary of the model
trainingSummary = linearModel.summary
print("number of Iterations: %d" % trainingSummary.totalIterations)
print("objective History: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
print('\n')

print('The answers to question 3 (c)')
print('\n')

pc1 = round(linearModel.intercept + (linearModel.coefficients[0] * 1), 2)
pc2 = round(linearModel.intercept + (linearModel.coefficients[0] * 2), 2)
pc3 = round(linearModel.intercept + (linearModel.coefficients[0] * 3), 2)
print('Question c1: Class 1 predicted fare: ' + str(pc1) + ' pound')
print('Question c2: Class 2 predicted fare: ' + str(pc2) + ' pound')
print('Question c3: Class 3 predicted fare: ' + str(pc3) + ' pound')
print('\n')