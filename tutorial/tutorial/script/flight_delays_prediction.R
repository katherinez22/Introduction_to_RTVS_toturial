## Flight Delays Prediction with Microsoft R Server

# In this example, we use historical on-time performance and 
# weather data to predict whether the arrival of a scheduled passenger 
# flight will be delayed by more than 15 minutes.

# We approach this problem as a classification problem, 
# predicting two classes -- whether the flight will be delayed, 
# or whether it will be on time. 

# Broadly speaking, in machine learning and 
# statistics, classification is the task of identifying 
# the class or category 
# to which a new observation belongs, on the basis of 
# a training set of data containing observations with known categories. 
# Classification is generally a supervised learning problem. 
# Since this is a binary classification task, there are only two classes.

# In this example, we train a model using a subset of examples 
# from historic flight data, along with an outcome measure that 
# indicates the appropriate category or class for each example. 
# The two classes are labeled 1 if a flight was delayed, and labeled 0 
# if the flight was on time.

# The following scripts include five basic steps of building 
# this example using Microsoft R Server.
# This execution might require several minutes.



## Tutorial Starts from here ##

### Step 0: Get Started

# Initial the input datasets.
inputFileFlight <- "./data/Flight_Delays_Sample.csv"
inputFileWeather <- "./data/Weather_Sample.csv"

# Create a temporary directory to store the intermediate .xdf files.
# Those intermediate .xdf files will be automatically removed once
# the current R session is over. 
td <- tempdir()
outFileFlightView <- paste0(td, "/flightView.xdf")
outFileFlightView2 <- paste0(td, "/flightView2.xdf")
outFileFlight <- paste0(td, "/flight.xdf")
outFileWeather <- paste0(td, "/weather.xdf")
outFileOrigin <- paste0(td, "/originData.xdf")
outFileDest <- paste0(td, "/destData.xdf")
outFileFinal <- paste0(td, "/finalData.xdf")

# Change compute context to local parallel computing
rxSetComputeContext("localpar")
# You can also try the default local sequential computing on your own. 
#rxSetComputeContext("local")


### Step 1: Exlore and Import Data

# Import the flight data from a .csv file and store it as .xdf.
flightView <- rxImport(inData = inputFileFlight,
                       outFile = outFileFlightView, overwrite = TRUE,
                       missingValueString = "M", stringsAsFactors = FALSE)

# Explore the flight data. 
rxGetInfo(flightView, numRows = 10, getVarInfo = TRUE)
# rxGetVarInfo(flightView)  # will also review the variable informaiton.
# head(flightView, n = 10)  # will also return the first 10 records.

# Another way to explore the data in the data viewer window. 
View(rxReadXdf(flightView, numRows = 1000))

# After the exploration, we know some columns are not useful or are 
# possible target leakers, and "CRSDepTime" can be rounded down to full hours.
# So we will use "rxDataStep()" for this type of transformation.
flightView2 <- rxDataStep(inData = flightView, outFile = outFileFlightView2,
                          # Remove unuseful columns and columns that are possible target leakers.
                          varsToDrop = c("Year", 
                                         "DepDelay", 
                                         "DepDel15", 
                                         "CRSArrTime", 
                                         "ArrDelay", 
                                         "Cancelled"),
                          # Round down scheduled departure time to full hour.
                          transforms = list(CRSDepTime = floor(CRSDepTime / 100)),
                          overwrite = TRUE)

# Also, "Carrier" can be treated as a categorical feature.
# "rxFactors()" will help us convert non-factor feature into a factor.
flight <- rxFactors(inData = flightView2, outFile = outFileFlight, 
                    sortLevels = TRUE,
                    factorInfo = "Carrier",
                    overwrite = TRUE)

# The second option of importing the flight data:
# Using this option if you are familar with the dataset and 
# know what data preparation needs to do.
# This way of import the data set is more efficient. 
#flight <- rxImport(inData = inputFileFlight, outFile = outFileFlight,
                #missingValueString = "M", stringsAsFactors = FALSE,
                ## Remove unuseful columns and columns that are possible target leakers.
                #varsToDrop = c("Year", 
                                #"DepDelay", 
                                #"DepDel15", 
                                #"CRSArrTime", 
                                #"ArrDelay", 
                                #"Cancelled"),
                ## Define "Carrier" as categorical.
                #colInfo = list(Carrier = list(type = "factor")),
                ## Round down scheduled departure time to full hour.
                #transforms = list(CRSDepTime = floor(CRSDepTime / 100)),
                #overwrite = TRUE)

# Let's quickly check the pre-processing of flight data is done correctly.
rxGetVarInfo(flight)  

# This time, we will import the weather data using the efficient approach. 
# We know some weather features are integer and their scales are irrelevent.
# So we will normalize those numeric features when importing the dataset.
xform <- function(dataList) {
  # Create a function to normalize some numeric features.
  featureNames <- c("Visibility",
                    "DryBulbCelsius",
                    "DewPointCelsius",
                    "RelativeHumidity",
                    "WindSpeed",
                    "Altimeter")
  # Apply "scale()" to each column using "lapply()".
  dataList[featureNames] <- lapply(dataList[featureNames], scale)
  return(dataList)
}

# While we are importing the weather data, we also want to take
# care of some simply tranformations at the same time. 
weather <- rxImport(inData = inputFileWeather, outFile = outFileWeather,
                    missingValueString = "M", stringsAsFactors = FALSE,
                    # Eliminate some features due to redundance.
                    varsToDrop = c("Year", "Timezone",
                                   "DryBulbFarenheit", "DewPointFarenheit"),
                    # Create a new column "DestAirportID" in weather data.
                    transforms = list(DestAirportID = AirportID),
                    # Apply the normalization function.
                    transformFunc = xform,
                    transformVars = c("Visibility",
                                      "DryBulbCelsius",
                                      "DewPointCelsius",
                                      "RelativeHumidity",
                                      "WindSpeed",
                                      "Altimeter"),
                    overwrite = TRUE)

# Review the feature information of weather data.
rxGetInfo(weather, numRows = 10, getVarInfo = TRUE)

# Check out the summaries of weather features.
rxSummary(formula = ~., data = weather)


### Step 2: Data Manipulation

# In order to merge the flight and weather data, we need to 
# rename some column names in the weather to be the same
# as they are in the flight. The name mapping will be:
# AdjustedMonth -> Month
# AdjustedDay -> DayofMonth
# AirportID -> OriginAirportID
# AdjustedHour -> CRSDepTime
newVarInfo <- list(AdjustedMonth = list(newName = "Month"),
                   AdjustedDay = list(newName = "DayofMonth"),
                   AirportID = list(newName = "OriginAirportID"),
                   AdjustedHour = list(newName = "CRSDepTime"))

# "rxSetVarInfo()" can set variable information, including variable names, 
# descriptions, and value labels.
rxSetVarInfo(varInfo = newVarInfo, data = weather)

# Two steps to merge flight and weather data.
# 1). Join flight records and weather data at origin of flight (OriginAirportID).
originData <- rxMerge(inData1 = flight, inData2 = weather, 
                      outFile = outFileOrigin,
                      type = "inner", autoSort = TRUE,
                      matchVars = c("Month", 
                                    "DayofMonth", 
                                    "OriginAirportID", 
                                    "CRSDepTime"),
                      varsToDrop2 = "DestAirportID",
                      overwrite = TRUE)

# 2). Join the merged records from step 1 and weather data 
#     at destination of flight (DestAirportID).
destData <- rxMerge(inData1 = originData, inData2 = weather, 
                    outFile = outFileDest,
                    type = "inner", autoSort = TRUE,
                    matchVars = c("Month", 
                                  "DayofMonth", 
                                  "DestAirportID", 
                                  "CRSDepTime"),
                    varsToDrop2 = "OriginAirportID",
                    duplicateVarExt = c("Origin", "Destination"),
                    overwrite = TRUE)

# Check the dimension of the merged dataset.
dim(destData)

# It's time to convert some numeric features to factors. 
rxFactors(inData = destData, outFile = outFileFinal, sortLevels = TRUE,
          factorInfo = c("Month", 
                         "DayofMonth", 
                         "DayOfWeek", 
                         "CRSDepTime",
                         "OriginAirportID", 
                         "DestAirportID"),
          overwrite = TRUE)


### Step 3: Prepare Training and Test Datasets

# Randomly split 80% data as training set and the remaining 20% as test set.
# To randomly split the data, we need to define a split variable.
# "splitVar" carries two values "Train" and "Test".
rxSplit(inData = outFileFinal,
        outFilesBase = paste0(td, "/modelData"),
        outFileSuffixes = c("Train", "Test"),
        splitByFactor = "splitVar",
        overwrite = TRUE,
        transforms = list(splitVar = factor(sample(c("Train", "Test"),
                                                   size = .rxNumRows,
                                                   replace = TRUE,
                                                   prob = c(.80, .20)),
                                            levels = c("Train", "Test"))),
        rngSeed = 17,
        consoleOutput = TRUE)

# Point the output .xdf files to the training and test set.
train <- RxXdfData(paste0(td, "/modelData.splitVar.Train.xdf"))
test <- RxXdfData(paste0(td, "/modelData.splitVar.Test.xdf"))


### Step 4A: Choose and apply a learning algorithm (Logistic Regression)

# Build the model formula, define "ArrDel15" as dependent variable
# and remove the split variable "splitVar".
modelFormula <- formula(train, depVars = "ArrDel15",
                        varsToDrop = c("splitVar"))
modelFormula

# Now, let's fit a Logistic Regression model on the training data.
logitModel <- rxLogit(modelFormula, data = train)

# Review the model results.
summary(logitModel)


### Step 5A: Predict over new data (Logistic Regression)

# Use the trained Logistic Regression to predict the probability of 
# flights will be delayed over 15 mins on the test data.
rxPredict(logitModel, data = test,
          type = "response",
          predVarNames = "ArrDel15_Pred_Logit",
          overwrite = TRUE)
          
# Let's take a look of the predicted probabilities. 
head(test)

# Calculate Area Under the Curve (AUC).
# The AUC is part of performance metric of a logistic regression, 
# and is a commonly used evaluation metric for binary 
# classification problems.
# 
# A perfect model will score an AUC of 1, while random guessing 
# will score an AUC of around 0.5, a meager 50% chance on each other.
paste0("AUC of Logistic Regression Model: ",
       rxAuc(rxRoc(actualVarName = "ArrDel15", 
                   predVarNames = "ArrDel15_Pred_Logit", 
                   data = test)))

# Plot the ROC curve.
# True Positive Rate (Sensitivity), tpr=TP(TP+FN).
# False Positive Rate (Specificity): fpr=FP/(FP+TN).
rxRocCurve(actualVarName = "ArrDel15", 
           predVarNames = "ArrDel15_Pred_Logit", 
           data = test,
           title = "ROC curve - Logistic regression")


### Step 4B: Choose and apply a learning algorithm (Decision Tree)

# First, let's build a very basic decision tree model.
dTree1 <- rxDTree(modelFormula, data = train, reportProgress = 2)

# If "rpart" library is installed, we can plot the Error vs. cp
# as a guide to pruning.
(if (!require("rpart", quietly = TRUE)) install.packages("rpart"))
plotcp(rxAddInheritance(dTree1))

# To further pruning the trees, we want to find the best value of "cp".
# "cp" is a complexity parameter that specifies how the cost of a tree
# is penalized by the numer of terminal nodes.
# In another words, small "cp" results in larger trees and potential 
# overfitting, large "cp" results in small trees and potential 
# underfitting. So we want to find the best "cp" value.
treeCp <- rxDTreeBestCp(dTree1)
treeCp

# Once we get the best "cp" value, we can now pruning the trees and 
# return the smaller tree.
dTree2 <- prune.rxDTree(dTree1, cp = treeCp)

# Want to see an interactive decission tree in your browser?
# Let's do it!
# First, we need to use zip the current tree to a .zip file
# by using the "RevoTreeView" library.
# Note: if you don't have a zip program installed on your machine,
# the following command may not work.
library("RevoTreeView")
zipTreeView(createTreeView(dTree2), 
            "myDecisionTree.zip", 
            flags = "a", 
            zip = "C:/Program Files/7-Zip/7z.exe") 

# Now, let's switch to the folder and unzip the file.
# once the .zip file is unzipped, double click on the .html file 
# to open the ineractive plot in your browser.


### Step 5B: Predict over new data (Decision Tree)

# Predict the probability of flight delays on the test data.
rxPredict(dTree2, data = test,
          predVarNames = "ArrDel15_Pred_Tree",
          overwrite = TRUE)

# Calculate Area Under the Curve (AUC).
paste0("AUC of Decision Tree Model: ",
       rxAuc(rxRoc(actualVarName = "ArrDel15", 
                   predVarNames = "ArrDel15_Pred_Tree", 
                   data = test)))

# Plot the ROC curve.
rxRocCurve(actualVarName = "ArrDel15", 
           predVarNames = c("ArrDel15_Pred_Tree", "ArrDel15_Pred_Logit"),
           data = test,
           title = "ROC curve: Logistic Regression vs. Decision Tree")