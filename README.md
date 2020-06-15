# Prediction-of-Burned-Area-of-Bushfire
This project is an analysis on the forest fire data in Portugal. Statistical evidences (p values and correlations) were used to identify the subset and assessed with cross-validation.  R's dummy, log and best normalise  function were utilised to transform the data which is then predicted with linear regression, XGBoost and XGBoostLasso models.

Programming Language: R 3.6.1 in Jupyter Notebook

R Libraries used:
- psych
- ggplot2
- reshape2
- lattice
- dummies
- caret
- e1071
- cowplot
- bestNormalize
- leaps
- purrr
- simEd
- dplyr
- caTools
- Metrics
- xgboost

## Table of Contents

[1. Introduction](#sec_1)



[2.Data Exploration](#sec_2)

[3.Model Development](#sec_3)

[4.Model Comparison](#sec_4)

[5.Variable Identification and Explanation](#sec_5)

[6.Conclusion](#sec_6)


## 1. Introduction <a class="anchor" id="sec_1"></a>
This notebook includes results of the data analysis performed on a set of forest fires data that is obtained from a UCI website. The purpose of the data analysis is to build models from the data that can be used to predict the burned area using collected attributes. 

The second section of the notebook shows the exploratory data analysis (EDA) performed to explore and understand the data. It looks at each attribute (variables) in the data to understand the nature and distribution of the attribute values. It also examines the correlation between the variables through visual analysis. A summary at the end highlights the key findings of the EDA.

The third section shows the development of the models namely linear regression, XGBoost and XGBoost Lasso. It details the process used to transform and build the model. During the transformation phase, outliers are removed and variables are normalised prior to building the model. After building the models,the final models are then presented along with an analysis and interpretation of the model. This section concludes with the results of using the models to predict burned area.

The fourth section provides comparisons of the performance of the models developed. This gives an indication as to which model built performs best with the forest fires data. Statistical methods such as root mean square error (RMSE) and mean squared error(MSE) are used to compare the performances of the models.

The fifth section looks into variable identification and explanation. In this subsection, important variables which affects the model selected and its performance are explained in greater details.

## Load the libraries used in the notebook
```{r }
library(psych) # for statistics
library(ggplot2) # used for visualisation
library(reshape2) # give new shape to array without changing its data
library(lattice) # used for data visualisation
library(dummies) # used for one hot encoding 
library(caret) # machine learning library
library(e1071) # to find skewness of data
library(cowplot) # used to display visualisation side by side 
library(bestNormalize) # used to normalise data
library(leaps) # used for computing best subsets regression
library(purrr) # used for map function
library(simEd) # for seed function 
library(dplyr) # used for data manipulation
library(caTools) # used to split data into train and test
library(Metrics) # library to measure MSE
library(xgboost) # to use xgboost algorithm 
```
## 2. Data Exploration<a class="anchor" id="sec_2"></a>
For this subsection, we are exploring the relationship between the predictor and response variables in the forest fire dataset with correlation analysis. The distributions of each of the variables are also investigated to determine whether transformation of the variables are required. Through this, we are able to get some basic intuition as to which variables are important. 

### 2.1 Overview of the Bushfire Dataset
```{r }
# Loading the bushfire dataset
fire_data <- read.csv("forestfires.csv")
```
```{r }
cat("The bushfire dataset has", dim(fire_data)[1], "observation records, each with", dim(fire_data)[2],
    "attributes.")
```
```{r }
cat("The structure of the bushfire data is:\n\n")
str(fire_data)
```
From the data above, we can see that there are different type of variables such as integer, factor and number variables. 
Integer variables consist of X,Y and RH. 
Factor variables consist of month and day which need to be transformed to dummy variables to allow easy comparison.
Number variables consists of FFMC, DMC, DC, ISI, temp, wind, rain and area.
Here, the details of each property will be explained:

- **X**: x-axis spatial coordinate within the Montesinho park map: 1 to 9
- **Y**: y-axis spatial coordinate within the Montesinho park map: 2 to 9
- **month**: month of the year: "jan" to "dec" 
- **day**: day of the week: "mon" to "sun"
- **FFMC**: FFMC index from the FWI system: 18.7 to 96.20
- **DMC**: DMC index from the FWI system: 1.1 to 291.3 
- **DC**: DC index from the FWI system: 7.9 to 860.6 
- **ISI**: ISI index from the FWI system: 0.0 to 56.10
- **temp**: temperature in Celsius degrees: 2.2 to 33.30
-  **RH**: relative humidity in %: 15.0 to 100
- **wind**: wind speed in km/h: 0.40 to 9.40 
- **rain**: outside rain in mm/m2 : 0.0 to 6.4 
- **area**: the burned area of the forest (in ha): 0.00 to 1090.84 

The details of the property are described in [Cortez and Morais, 2007]. 
```{r }
cat("The descriptive statistics of the variables in the dataset are: \n")
summary(fire_data)

cat("The advanced descriptive statistics of the variables in the dataset are:\n  ")
round(describe(fire_data), 3)
```

#### Summary of Attributes:



The following table identifies which attributes are numerical and whether they are continuous or discrete, and which
are categorical and whether they are nominal or ordinal. In addition, it includes some initial observations about the ranges and 
common values of the attributes.

|Attribute  |Type       |Sub-type  |Comments                                                                              |
|-----------|-----------|----------|--------------------------------------------------------------------------------------|
|X          |Numerical  |Discrete   |value ranges from 1 to 9.|
|Y          |Numerical  |Discrete   |value ranges from 2 to 9.|
|month      |Categorical|Ordinal  |August and September have higher records of bushfire compared to other months.|
|day        |Categorical |Ordinal  |Sunday has the highest record of bushfire compared to other days.|
|Fine Fuel Moisture Code (FFMC)|Numerical |Continuous  |values range from 18.7 to 96.2. Probably has outliers - especially for the low values.|
|Duff Moisture Code (DMC)          |Numerical |Continuous  |values range from 1.1 to 291.3. Probably has extreme outliers - especially for the low values.|
|Drought Code (DC)          |Numerical |Continuous  |values range from 7.9 to 860.6. Probably has extreme outliers - especially for the low values.|
|Initial Spread Index (ISI)        |Numerical  |Continuous|values range from 0 to 56.1. Probably has outliers - especially for the high values.|
|Outside Temperature (Temp)       |Numerical  |Continuous|values range from 2.20 to 33.30 celcius. Average temperature is around 18.89 celcius.|
|Outside Relative Humidity (RH)          |Numerical |Discrete  |values range from 15 to 100. Probably has outliers for high values.|
|Outside Wind Speed (Wind)       |Numerical |Continuous  |values range from 0.4 to 9.4.|
|Outside Rain (Rain)       |Numerical |Continuous  |values range from  0 to 6.4. Probably has outliers for high values.|
|Total Burned Area (Area)       |Numerical |Continuous  |values range from 0 to 1090.84.Probably has outliers for high values. |

### 2.2 Investigate distribution of each variable
Below we are generating boxplots for numerical variables to view their distribution.
```{r }
#plot size
options(repr.plot.width=9, repr.plot.height=8)
#convert fire_data excluding x,y, month and day with several measurement columns into a data frame in this canonical format
melt_data <- melt(as.data.frame(fire_data[,c(-1,-2,-3,-4)]))

#plotting boxplots
ggplot(melt_data,aes(x = variable,y = value)) +
facet_wrap(~variable, scales="free") +
geom_boxplot(fill = "darkmagenta", color = 'black') +
scale_y_continuous(labels=function (n) {format(n, scientific=FALSE)}) +
ggtitle("Figure 1: Boxplot for numerical variables in Fire Data")
```
- The boxplot above indicate that outliers for all 9 numerical variables.
- FFMC, DC and temperature have outliers which are of lower values while the remaining have outliers of high values.

Below we are generating barchart and histogram for categorical and numerical variables respectively to view their distribution.
```{r }
# rearranging the days and months based so that the x labels are populated in orderly manner for graph

fire_data$month <- factor(fire_data$month, levels=c("jan", "feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))
fire_data$day <- factor(fire_data$day, levels = c("mon","tue","wed","thu","fri","sat","sun"))
```
```{r }
# plot size
par(mfrow = c(3,2))
# plotting bar chart for months
plot(as.factor(fire_data$month),main="Bar Chart of Months",col="darkmagenta" )
# plotting bar chart for days 
plot(as.factor(fire_data$day),main="Bar Chart of Days",col="darkmagenta")
# plotting histogram for FMC
hist(fire_data$FFMC,main="Histogram of Fine Fuel Moisture Code(FFMC)",xlab="FFMC",col="darkmagenta")
# plotting histogram for DMC
hist(fire_data$DMC,main="Histogram of Duff Moisture Code(DMC)",xlab="DMC",col="darkmagenta")
# plotting histogram for DC
hist(fire_data$DC,main="Histogram of Drought Code(DC)",xlab="DC",col="darkmagenta")
# plotting histogram for ISI
hist(fire_data$ISI,main="Histogram of Initial Spread Index (ISI)",xlab="ISI",col="darkmagenta")
# plotting histogram for Temp
hist(fire_data$temp,main="Histogram of Outside Temperature(Temp) in ◦C",xlab="Temp",col="darkmagenta")
# plotting histogram for RH
hist(fire_data$RH,main="Histogram of Outside Relative Humidity (RH) in % ",xlab="RH",col="darkmagenta")
# plotting histogram for Wind
hist(fire_data$wind,main="Histogram of Outside Wind Speed (Wind) in km/h ",xlab="Wind",col="darkmagenta")
# plotting histogram for Rain
hist(fire_data$rain,main="Histogram of Outside Rain (Rain) in mm/m^2",xlab="Rain",col="darkmagenta")
# plotting histogram for Area
hist(fire_data$area,main="Histogram of Total Burned Area (Area)in ha",xlab="Area",col="darkmagenta")
```
The graphs above indicate that:

**For categorical variables**: 
- August and September are months that have the relatively highest bushfire occurence compared to other months.
- Sundays is when there is the highest occurence of bushfire, followed by friday and saturday. 

**For non categorical variables**:
- The histogram of ISI, DMC, RH, Wind are right skewed. 
- The histogram of FFMC and DC are left skewed.
- The histogram of Temp is normally distributed.
- The histogram of Rain and Area are highly skewed towards 0.0 indicating the need to perform logarithm transformation on both these variables to make the pattern in the data more interpretable.

We can see that the range differs for most of the attributes, hence in order to make comparisons in variance we need to ensure they are in the same scale using Normalization techniques.

### 2.3 Investigate Pairs of Variables
```{r }
#### Correlation Plot Function

# function for correlation plot
range_color <- c('#69091e', '#e37f65', 'white', '#58fc9a', '#036e2e')
## colorRamp() returns a function which takes as an argument a number
## on [0,1] and returns a color in the gradient in colorRange
Color_RampFunc <- colorRamp(range_color)

cor_panel <- function(w, z, ...) {
    correlation <- cor(w, z)

    col <- rgb(Color_RampFunc((1 + correlation) / 2 ) / 255 )

    ## square it to avoid visual bias due to "area vs diameter"
    radius <- sqrt(abs(correlation))
    radians <- seq(0, 2*pi, len = 50) # 50 is arbitrary
    x <- radius * cos(radians)
    y <- radius * sin(radians)
    ## make them full loops
    x <- c(x, tail(x,n=1))
    y <- c(y, tail(y,n=1))

    par(new=TRUE)
    plot(0, type='n', xlim=c(-1,1), ylim=c(-1,1), axes=FALSE, asp=1)
    polygon(x, y, border=col, col=col)
}
```
```{r }
# to exclude x and y from being plotted in correlation plot
pairs(fire_data[c(-1,-2)], upper.panel = cor_panel) 
```

The correlation matrix shows:
- FFMC, DMC, DC , ISI and temp have correlation to each other - in particular both DMC and DC have a stronger positive correlation to each other.
- There is a strong positive correlation between month and DC.
- There is very little correlation between day, rain and area with the other variables.
- RH is negatively correlated to temp and FFMC - there is a stronger negative correlation between temp and RH.
- There is a weak negative correlation between wind and temperature as well as between wind and DC.

#### Correlation coefficient 
```{r }
# function for correlation matrix
panel_func <- function(x, y, z, ...) {
    panel.levelplot(x,y,z,...)
    panel.text(x, y, round(z, 2))
}
#Define the color scheme
cols = colorRampPalette(c("blue","green"))
#Plot the correlation matrix.
levelplot(cor(fire_data[c(-1,-2,-3,-4)]), col.regions = cols(100), main = "Correlation between variables",
          scales = list(x = list(rot = 90)), panel = panel_func)

# plot size
options(repr.plot.width=9, repr.plot.height=8)
```
The higher positive correlations are between:
- DC and DMC

The only significant negative correlation is between RH and temp.

#### Investigating relationship between month and area burnt 
```{r }
# barplot for count of months 
month_count <- ggplot(fire_data, aes(x = month)) + geom_bar(stat = "count", fill = 'darkmagenta') + 
labs(title="Figure A: Total count of months",x ="Month", y = "Total count per month")
# barplot for total area burnt for every month 
total_area <- ggplot(as.data.frame(fire_data), aes(month,area, fill = month)) + geom_col(position = 'dodge')+ 
labs(title="Figure B: Total area burnt for every month ",x ="Month", y = "Total Area")

# plotting figures side by side
cowplot :: plot_grid(month_count,total_area)
```
Based on figure A, we can see that the data has the highest count for August, September and March. We would expect that the total area burnt would be higher for the months with the higher count. However, surprisingly, in figure B, the highest total area is in September, followed by August and July. These months could possibly have outlier values which affect the total area burnt.
#### Investigating relationship between days and area burnt 
```{r }
# barplot for count of days
day_count <- ggplot(fire_data, aes(x = day)) + geom_bar(stat = "count", fill = 'darkmagenta') + 
labs(title="Figure A: Total count of days",x ="Days", y = "Total count per day")
# barplot for total area burnt for days respectively  
total_area <- ggplot(as.data.frame(fire_data), aes(day,area, fill = day)) + geom_col(position = 'dodge')+ 
labs(title="Figure B: Total area burnt based on days ",x ="Days", y = "Total Area")

# plotting figures side by side
cowplot :: plot_grid(day_count,total_area)
```
Based on figure A, we can see that the data has the highest count for sudays, fridays and satursdays. We would expect that the total area burnt would be higher for days with the higher count. However, surprisingly, in figure B, the highest total area is on saturdays, thursdays and mondays. These days could possibly have outlier values which affect the total area burnt.
#### 2.4 Insights derived from Exploratory Data Analysis

- **High skewness for predictor and response variable**: There seem to be high skewness for 'area' (response variable) and 'rain'(predictor variable) with data mainly skewed towards 0 (right skewed). This creates a problem as the tail region may act as an outlier which will affect the model's performance if not normalized.

- **Low correlation between predictors and response variable**: There are low correlation between the predictor and response variable. However, we can see that several of the attributes are correlated, with a high correlation between month and DC, as well as DMC and DC. In addition, several of the predictor variables like FFMC, DMC, DC , ISI and temp seem to be correlated,therefore it make sense to apply some sort of feature selection. 

## 3. Model Development<a class="anchor" id="sec_3"></a>
For this subsection, we are implementing the transformations required in order to improve the accuracy of the models. Then, the feature selection is done to investigate which features are essential and are incorporated to the built model. 
####  Initial model with all features 
A model with all the features is developed to compare the models with transformed model and lesser features.
```{r }
# creating function to calculate mean squared error 
mse_model <- function(model)
    mean(model$residuals^2)
```
```{r }
# ensure results are repeatable
set.seed(5)
# linear model
lmfit3 = lm(area~., data = fire_data)
summary(lmfit3)
```
**From the summary of the model**:

- The adjusted R-squared ($R^2$) value of -0.006905 indicates this model does not explain the variation in area.

- The F-statistic 0.8689 has a p-value < 0.6581 - so cannot reject the null hypothesis (the model explains nothing) - the model is not useful

- The p-values for the coefficients show that only DMC and DC are significant at the 0.05 level.

Some transformation is needed to improve the model which will be done later.
```{r }
mse_model(lmfit3) # MSE for model with all the features which are not transformed
```
As shown above, we can see that MSE is high which indicates that the data values are dispersed widely around its mean and that the data is skewed. Therefore, we need to build a model with lower MSE.
#### Using plot() function to produce diagnostic plots of the linear regression fit

The purpose of this is to check the following assumptions:

- Constant variance

- Linearity

- Normality
```{r }
# plotting residual vs fitted, scale - location, normal q-q and residual vs leverage plots
options(repr.plot.width=9, repr.plot.height=8) 
par(mfcol=c(2,2))  
plot(lmfit3)
```

The following conclusions are derived from the plots above:


* The **residual vs fitted plot**: This plot is used to check linear assumption. This is to indicate whether residuals have non linear patterns. The first plot above shows that there could be a non-linear relationship between area and all the predictors, as there is an obvious pattern in this plot given that the residuals are not scattered evenly. 

* The normal **Q-Q plot**: In the case of linear regression analysis, we assume that residual is normally distributed with constant variance and mean equal to zero. The normal Q-Q plot shows if residuals are normally distributed. In this case we can see that the residuals are lined well on the straight dashed line, therefore the residuals are most likely distributed normally.

* The **scale-location plot**: This plot is used to check the assumption of equal variance by showing if residuals are spread equally along the ranges of predictors. The scale-location plot shows that the residuals appear randomly spread.


* The **residual-leverage plot**: This plot is used to identify influential data sample. As observed, the forth plot shows outliers such as 480,416 and 239. We note that point 239 and 416 is close to Cook's distance of 0.5. However, these outlier points are not outside of the Cook's distance lines. Therefore there are no influential cases observed.


### 3.1 Model Transformation
Data transformation is done in order to generate symmetric distribution instead of the original skewed distribution to make it is easier to interpret and generate inferences. The relationship between variables are clearer when we re-express the variables, especially when converting non-linear relationship between variables to linear ones. Below transformations are done to both categorical and numerical variables.

#### 3.1.1. Categorical variable transformation
##### Creating dummies for categorical variables
Dummies are created for categorical variables namely month and day from the dataset to indicate the absence or presence of some categorical effect that may be expected to shift the outcome.
```{r }
# new dataframe for dummy
fire.new <- dummy.data.frame(fire_data,sep = ".")
```
```{r }
# creating dummy variable for months
dummy(fire_data$month, sep = ".")
```
```{r }
# creating dummy variable for days
dummy(fire_data$day, sep = ".")
```
```{r }
# incorporating dummy variables to fire_data1 dataset
fire_data1 <- dummy.data.frame(names = c("month","day"), fire_data,sep = ".") 
fire_data1
```
#### 3.1.2. Numeric Variable Transformation 
##### Logarithmic transformation

Skewed data makes it difficult to check as most of the observations are constricted in a small part of the range of the data. Hence, logarithmic transformation is performed on 'area' and 'rain' to adjust the data distribution to make it less skewed. Given that both these features have zero values which will create an error value, log(x+1) transformation is used to avoid the errors.

```{r }
# check skewness for area 
skewness(fire_data1$area) 
```
From the result above, we can see that the skewness of area is high, therefore, we will need to perform a log transformation. The skewness can also be observed with the QQ and histogram plot below. 
```{r }
## qq plot for area
qqnorm(fire_data1[,13], main = "")
qqline(fire_data1[,13], col = 2,lwd=2,lty=2, main = "")
```
```{r }
# histogram plot for area to observe the distribution 
hist(fire_data1$area, xlab="Area", main = "",col="darkmagenta")
```
As shown by the 2 plots above, we can see that the area variable is highly skewed to the right highly skewed towards 0.0. Therefore, a log transformation is performed in the next step.
```{r }
# log transformation for area
fire_data1$area <- log(fire_data1$area +1) 
qqnorm(fire_data1$area, main = "")
qqline(fire_data1$area, col = 2, lwd=2,lty=2)
```
```{r }
hist(fire_data1$area, xlab="Area", main = "",col="darkmagenta")
```
As shown by the 2 plots above, it is observed that the QQ plot and histogram are more normal after transformation compared to before transformation. 
```{r }
# skewness value after transformation
skewness(fire_data1$area)
```
It is also observed that the skewness for area is recorded lower at 1.21078001224549 after transformation compared to 12.7724826585002
which was recorded before the transformation.

Next, we are exploring the skewness of the rain.
```{r }
# skewness of rain 
skewness(fire_data1$rain)
```
From the result above, we can see that the skewness of rain is high, therefore, we will need to perform a log transformation. The skewness can also be observed with the QQ and histogram plot below. 
```{r }
qqnorm(fire_data1[,12], main = ""); qqline(fire_data1[,12], col = 2,lwd=2,lty=2, main = "")
```
```{r }
hist(fire_data1$rain, xlab="Rain", main = "", col="darkmagenta")
```
As shown by the 2 plots above, we can see that the rain variable is highly skewed to the right highly skewed towards 0.0. Therefore, a log transformation is performed in the next step.
```{r }
fire_data1$rain <- log(fire_data1$rain +1) # log transformation for area
```
```{r }
qqnorm(fire_data1$rain, main = "")
qqline(fire_data1$rain, col = 2, lwd=2,lty=2)
```
```{r }
hist(fire_data1$rain, xlab="Rain", main = "", col="darkmagenta")
```
As shown by the 2 plots above, it is observed that the QQ plot and histogram are still highly skewed after transformation. We will check with the skewness function to observe as to whhether the skewness decreased after transformation.
```{r }
skewness(fire_data1$rain)
```
It is observed that the skewness for area is recorded lower at 14.0908918136066 after transformation compared to 19.7015038029727 which was recorded before the transformation, however it is still high.

Next, we observe whether the transformation of area and rain as well as adding of dummy variables have contributed to the improvement in model.   

```{r }
# ensure results are repeatable
set.seed(5)
# linear model after transformation
lmwithtrans = lm(area~., data = fire_data1)
summary(lmwithtrans)
```
**From the summary of the model**:

- The adjusted R-squared ($R^2$) of 0.02354 indicates that this model explains 2.35% of the variation in area. This is better than the previous R- squared of -0.006905 which did not explain the variation in area.

- The F-statistic 1.461 has a p-value 0.06487 - so cannot reject the null hypothesis (the model explains nothing) - the model is not useful

- The p-values for the coefficients show that Y, month, DMC and temp are significant at the 0.05 level.

Given an improvement in the adjusted R-squared value from -0.006905 to 0.02354  as well as a lower p value from 
0.6581 to 0.06487 from the previous transformation,  we can indicate that the model is performing better after applying the log transformation on area and rain.
```{r }
mse_model(lmwithtrans) # checking mse after log transformation on area and rain
```
As shown above, we can see that MSE is low and much lower compared to before transformation (3859.07245262511) which indicates that the data values are dispersed closely around its mean and that the data is not skewed.
#### Best normalize function

Best normalize function is performed to normalize transformations and selects the best on the basis of the Pearson P test statistic for normality. The tranformation which has the lowest P calculated on the transformed data is selected. 
```{r }
# performing best normalize function on FFMC variable
fire_FFMCnorm <- bestNormalize(fire_data1$FFMC)
fire_FFMCnorm
```
Based on the transformation above, we can see that the estimated normality statistic for the OrderNorm transformation is close to one, so we know it is performing quite well. It is also performing better than all of the other transformations.
```{r }
# performing best normalize function on DMC variable
fire_DMCnorm <- bestNormalize(fire_data1$DMC)
fire_DMCnorm
```
Based on the transformation above, we can see that the estimated normality statistic for the OrderNorm transformation is the closest to one, so we know it is performing quite well. It is also performing better than all of the other transformations.
```{r }
# performing best normalize function on DC variable
fire_DCnorm <- bestNormalize(fire_data1$DC)
fire_DCnorm
```

Based on the transformation above, we can see that the estimated normality statistic for the OrderNorm transformation is the closest to one, so we know it is performing quite well. We also note that it is also performing better in comparison to the other transformations.
```{r }
# performing best normalize function on ISI variable
fire_ISInorm <- bestNormalize(fire_data1$ISI)
fire_ISInorm
```
Based on the transformation above, we can see that the estimated normality statistic for the OrderNorm transformation is the closest to one, so we know it is performing quite well. We also note that it is also performing better in comparison to the other transformations.
```{r }
# performing best normalize function on temp variable
fire_tempnorm <- bestNormalize(fire_data1$temp)
fire_tempnorm
```

Based on the transformation above, we can see that the estimated normality statistic for the Yeo-Johnson transformation is the closest to one, so we know it is performing quite well. We also note that it is also performing better in comparison to the other transformations.
```{r }
# performing best normalize function on RH variable
fire_RHnorm <- bestNormalize(fire_data1$RH)
fire_RHnorm
```{r }

Based on the transformation above, we can see that the estimated normality statistic for the OrderNorm transformation is the closest to one, so we know it is performing quite well. We also note that it is also performing better in comparison to the other transformations.
```{r }
# performing best normalize function on wind variable
fire_windnorm <- bestNormalize(fire_data1$wind)
fire_windnorm
```

Based on the transformation above, we can see that the estimated normality statistic for the OrderNorm transformation is the closest to one, so we know it is performing quite well. We also note that it is also performing better in comparison to the other transformations.
```{r }
# performing best normalize function on rain variable
fire_rainnorm <- bestNormalize(fire_data1$rain)
fire_rainnorm
```

As shown above, none of the normalizing transformations performed well according to the normality statistics. The frequency of ties in this case makes it very difficult to find a normalizing transformation. However, orderNorm is chosen as it has the lowest estimated P/df statistic.
```{r }
# performing best normalize function on area variable
fire_areanorm <- bestNormalize(fire_data1$area)
fire_areanorm
```

Based on the transformation above, we can see that the estimated normality statistic for the no transformation method is the closest to one, so we know it is performing quite well. It is also performing better than all of the other transformations. This indicates that we've performed the relevant trasnformation of log (area + 1) in the earlier trasnformation.

Next, we observe as to whether the transformations performed with the bestNormalize function has improved the distribution of the variables with the histogram plots.  
```{r }
# layout of graphs
par(mfrow = c(3, 2))
#plotting the histogram for original FFMC 
MASS::truehist(fire_data1$FFMC, col="darkmagenta")
#plotting the histogram for FFMC after transformation 
MASS::truehist(fire_FFMCnorm$x.t, col="darkmagenta")
#plotting the histogram for original DMC
MASS::truehist(fire_data$DMC, col="darkmagenta")
#plotting the histogram for DMC after transformation
MASS::truehist(fire_DMCnorm$x.t, col="darkmagenta")
#plotting the histogram for original DC
MASS::truehist(fire_data1$DC, col="darkmagenta")
#plotting the histogram for DC after transformation
MASS::truehist(fire_DCnorm$x.t, col="darkmagenta")
#plotting the histogram for original ISI
MASS::truehist(fire_data1$ISI, col="darkmagenta")
#plotting the histogram for ISI after transformation
MASS::truehist(fire_ISInorm$x.t, col="darkmagenta")
#plotting the histogram for original temp
MASS::truehist(fire_data1$temp, col="darkmagenta")
#plotting the histogram for temp after transformation
MASS::truehist(fire_tempnorm$x.t,col="darkmagenta")
#plotting the histogram for original RH
MASS::truehist(fire_data1$RH, col="darkmagenta")
#plotting the histogram for RH after transformation
MASS::truehist(fire_RHnorm$x.t, col="darkmagenta")
#plotting the histogram for original wind
MASS::truehist(fire_data1$wind,col="darkmagenta")
#plotting the histogram for wind after transformation
MASS::truehist(fire_windnorm$x.t,col="darkmagenta")
#plotting the histogram for original rain
MASS::truehist(fire_data1$rain,col="darkmagenta")
#plotting the histogram for rain after transformation
MASS::truehist(fire_rainnorm$x.t,col="darkmagenta")
#plotting the histogram for original area
MASS::truehist(fire_data1$area,col="darkmagenta")
#plotting the histogram for area after transformation
MASS::truehist(fire_areanorm$x.t,col="darkmagenta")
```
As shown above, we can see that there are improvements in the distributions for FFMC, DMC, DC, ISI, temp, RH and wind as the distributions look more normalized than before. The distribution for rain and area remained the same.

Given the improvement in the distribution, we would like to discover whether the transformed variables explains the variation in area better.
```{r}
# creating new column for transformed FFMC
fire_data1$FFMC.t <- fire_FFMCnorm$x.t
# creating new column for transformed DMC
fire_data1$DMC.t <- fire_DMCnorm$x.t
# creating new column for transformed DC
fire_data1$DC.t <- fire_DCnorm$x.t
# creating new column for transformed ISI
fire_data1$ISI.t <- fire_ISInorm$x.t
# creating new column for transformed temp
fire_data1$temp.t <- fire_tempnorm$x.t
# creating new column for transformed RH
fire_data1$RH.t <- fire_RHnorm$x.t
# creating new column for transformed wind
fire_data1$wind.t <- fire_windnorm$x.t
# creating new column for transformed rain
fire_data1$rain.t <- fire_rainnorm$x.t
# creating new column for transformed area
fire_data1$area.t <- fire_areanorm$x.t
```
```{r}
# checking colummn names for dataset
colnames(fire_data1)
# updating the dataset by removing the non transformed variables
fire_data1 <- fire_data1[,-c(22:30)] # minus non transformed variables 
# ensure results are repeatable
set.seed(5)
# create linear model with transformed variables
linear_trans <- lm(area.t~.,data = fire_data1)
summary(linear_trans)
```
**From the summary of the model**:

- The adjusted R-squared ($R^2$) of 0.02652 indicates that this model explains 2.65% of the variation in area. This is better than the previous R- squared of 0.02354 (with just log transformation of area and rain).

- The F-statistic 1.521 has a p-value 0.04677 - so we can reject the null hypothesis (the model explains nothing) with significance level of 0.05 - the model is useful.

- The p-values for the coefficients show that DMC,DC, temp, month and wind  are significant at the 0.05 level.

Given an improvement in the adjusted R-squared value from 0.02354 to 0.02652 as well as a lower p value from 0.06487 to 0.04677 from the previous transformation,  we can indicate that the model is performing better after applying the bestnormalize function.

```{r}
mse_model(linear_trans)
```
As shown above, we can see that MSE is low and much lower compared to just transformation of log area and rain (1.80617079683761) which indicates that the data values are dispersed closely around its mean and that the data is not skewed.


Let's observe the residuals vs fitted plot, scale-location plot, normal q-q plot and residuals vs leverage plot to visualise the effect of transformation. 

```{r}
# getting residual vs fitted plot, scale- location plot, normal q-q plot, residual vs leverage plot
options(repr.plot.width=9, repr.plot.height=8) 
par(mfcol=c(2,2))  
plot(linear_trans)
```
The following conclusions are derived from the plots above:

* The **residual vs fitted plot**: This plot is used to check linear assumption. This is to indicate whether residuals have non linear patterns. The first plot above shows that there could be a non-linear relationship between area and all the predictors, as there is an obvious pattern in this plot given that the residuals are not scattered evenly. This is still similiar to before transformation as residuals are still not scattered evenly. 

* The normal **Q-Q plot**: In the case of linear regression analysis, we assume that residual is normally distributed with constant variance and mean equal to zero. The normal Q-Q plot shows if residuals are normally distributed. In this case we can see that the residuals are lined well on the straight dashed line, therefore the residuals are most likely distributed normally. This is more normalised compared to before the transformation. 

* The **scale-location plot**: This plot is used to check the assumption of equal variance by showing if residuals are spread equally along the ranges of predictors. The scale-location plot shows that the residuals appear randomly spread. This is similar to before the transformation where residuals are still randomly spread.

* The **residual-leverage plot**: This plot is used to identify influential data sample. As observed, the forth plot shows outliers such as 472,305 and 500. We note that point 500 is outside the Cook's distance lines. Therefore point 500 is an influential point and needs to be removed as it could lead to measurement error.

Below is a plot with a close up look of the index with influential points from the data. 
```{r}
#calculating distance of points
cook.sd <- cooks.distance(linear_trans)
# Plot the Cook's Distance using the traditional 4/n criterion
samplesize <- nrow(fire_data1)
# plot cook's distance
plot(cook.sd, pch="*", cex=2, main="Influential Points by Cooks distance")  
 # add cutoff line
abline(h = 4/samplesize, col="blue") 
 # add labels
text(x=1:length(cook.sd)+1, y=cook.sd, labels=ifelse(cook.sd>4/samplesize, names(cook.sd),""), col="red") 
```
As shown in the plot above, we can see that the obvious influential points are at index 500, 305 and 472. These points will be removed below.

```{r}
#Removing the top 3 outliers
top_x_outlier <- 3

#getting top 3 outlier points index
influential <- as.numeric(names(sort(cook.sd, decreasing = TRUE)[1:top_x_outlier]))

# removing top 3 outlier points from dataset
fire_data2 <- fire_data1[-influential,]
```
```{r}
# ensure results are repeatable
set.seed(5)
# building linear model from new dataset with outliers removed
transformed_model <- lm(area.t ~ FFMC.t + DMC.t +  DC.t + ISI.t +temp.t + RH.t + wind.t + rain.t + X + 
           Y + month.jan + month.feb + month.mar + month.apr + month.may + month.jun + month.jul + 
           month.aug + month.sep + month.oct + month.nov + month.dec + day.mon + day.tue + day.wed +
           day.thu + day.fri + day.sat + day.sun , data= fire_data2)

summary(transformed_model)
```

**From the summary of the model**:

- The adjusted R-squared ($R^2$) of 0.03291 indicates that this model explains 3.23% of the variation in area.This is better than the previous R- squared of 0.02504.

- The F-statistic 1.671 has a p-value 0.02106 - so we cannot reject the null hypothesis (the model explains nothing) with significance level of 0.05. This model is useful.

- The p-values for the coefficients show that month, DMC, temp, wind, rain and X are significant at the 0.05 level.

Given an improvement in the adjusted R-squared value from 0.02652 to 0.03291 as well as a lower p value from 0.04677 to 0.02106 prior to removing the influential point,  we can indicate that the model is performing better after removing the influential point.

MSE of the model is calculated below to investigate the distribution and skewness of the model.

```{r}
mse_model(transformed_model)
```

As shown above, we can see that MSE is low and much lower compared to prior removal of outliers (0.920755890560974) which indicates that the data values are dispersed closely around its mean and that the data is not skewed.

Given that transformation is done and has improved the MSE of the model, we can move on to feature selection.

### 3.2 Feature Selection 

Feature selection is done by performing hard selection which consist of best subset and hybrid selection. Thereafter, cross validation is then performed on the hard selections to choose the optimal number of predictors required for the model to be optimised.

#### 3.2.1 Performing Hybrid Subset Selection 

Hybrid selection is a combination of both forward and backward selection in which the model adds and removes features one by one to reach the optimal model.
```{r}
# ensure results are repeatable
set.seed(5)
#performing hybrid selection on data 
regfit.both = regsubsets(area.t ~ .,data = fire_data2, method = 'seqrep')
#getting summary results for hybrid selection
reg.summary.both <- summary(regfit.both)
reg.summary.both
```

The above reports the best set of variables for each model size with asterisk indicating that a given variable is included in the corresponding model. Based on the analysis above, it is difficult to know which models to choose for our predictive analysis. Therefore, Cp, BIC, adjusted R square and residual sum of square (RSS) are used as indicators to help obtain the best model selection.


**Cp** acts as a penalty which tries to minimize overfitting which is created by our model during training the model. Penalty increases as the number of predictors increases. The model with lowest Cp is the best model.

**BIC** is similiar to Cp. The model with least value is the best model as it indicates a low test error. 

**Adjusted R Square** measures the correct variables and voice variable in the variable. A higher adjusted r square is preferable as it  has more correct variables and lesser noise variable into it.

**Residual Sum of Square (RSS)** is used to measure the amount of variance in the data that is not explained by the model. A lower RSS is preferred as it indicates a lower variance and error. 

A set of plots are generated to visualise the best overall model based on Cp, BIC, adjusted r squared and RSS as shown below:

```{r }
par(mfrow = c(2, 2))
#plotting cp for hybrid selection
plot(reg.summary.both$cp, xlab = "Number of variables", ylab = "Cp", type = "l")
#plotting minimum point of cp for hybrid selection
points(which.min(reg.summary.both$cp), reg.summary.both$cp[which.min(reg.summary.both$cp)], col = "red", cex = 2, pch = 20)
#plotting bic for hybrid selection
plot(reg.summary.both$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
#plotting minimum point of bic for hybrid selection
points(which.min(reg.summary.both$bic), reg.summary.both$bic[which.min(reg.summary.both$bic)], col = "red", cex = 2, pch = 20)
#plotting adjusted r squared for hybrid selection
plot(reg.summary.both$adjr2, xlab = "Number of variables", ylab = "Adjusted R Squared", type = "l")
#plotting maximum point of adjusted r square for hybrid selection
points(which.max(reg.summary.both$adjr2), reg.summary.both$adjr2[which.max(reg.summary.both$adjr2)], col = "red", cex = 2, pch = 20)

plot(reg.summary.both$rss, xlab = "Number of variables", ylab = "RSS", type = "l")
mtext("Plots of Cp, BIC, Adjusted R Squared and RSS for hybrid stepwise selection", side = 3, line = -2, outer = TRUE)
```
As shown in the plots above, we can see that based on Cp,adjusted r squared, the best model is estimated to have 9 predictor variables. For BIC on the other hand, it estimated that the best model have 1 predictor variable.

The best model according to each adjusted r squared, Cp and BIC for hybrid stepwise selection are extracted to confirm and are shown as below:  ```

```{r }
# plotting dataframe for best model based on maximum adjusted r square, minimum cp and minimum bic.
data.frame(
adj.r2 = which.max(reg.summary.both$adjr2),
                  cp = which.min(reg.summary.both$cp),
bic = which.min(reg.summary.both$bic))
```

We can see from the above that based on adjusted r square and cp the best model is the one with 9 predictor variables. However, using the BIC criteria, we should go for the model with 1 variable. Given that we have different "best" models depending on which metrics we consider. Therefore, a more vigorous approach is to select a model based on the prediction error computed on a new test data using k-fold cross validation techniques.
```{r }
# helper function to allow easy access to formula of models returned by the function regsubsets()
get_model_formula <- function(id,object) {
models <- summary(object)$which[id,-1]
form <- as.formula(object$call[[2]])
outcome <- all.vars(form)[1]
predictors <- names(which(models == TRUE))
predictors <- paste(predictors, collapse="+")
as.formula(paste0(outcome,"~",predictors))
     }
```
We will use the defined helper function above to compute the prediction error for the different best models returned by the regsubsets() function  
```{r }
# computing cross validation error
model.ids <- 1:9
cv.errors <- map(model.ids, get_model_formula,regfit.both) %>%
map(get_cv_error, data = fire_data2) %>%
unlist()
cv.errors
```
Cross validation errors are shown above based on each of the respective models which have different variable numbers. The which.min() function is performed to indicate the minimum cross validation error from the models as shown below:

```{r }
# selecting the best model that minimizes the cross validation error
which.min(cv.errors)
```

As shown above, we can see that the model with 9 variables is the best model. This is so as it has the lowest prediction error.

The regression coefficients of this model are shown as below:
```{r }
coef(regfit.both,9)
```
Based on coefficient above, we can see that the 9 variables that are important are X, month.feb, month.oct, day.tue, FFMC,temp, rain, month.dec and day.sun. 

Next, we explore the optimal model from best subset selection to see if it's similiar to the hybrid selection above.

#### 3.2.2. Performing best subset selection
```{r }
# ensure results are repeatable
set.seed(5)
#performing best subset selection on data 
regfit.full = regsubsets(area.t~.,data = fire_data2)
reg.summary.full <- summary(regfit.full)
reg.summary.full
```
The above reports the best set of variables for each model size with asterisk indicating that a given variable is included in the corresponding model. Based on the analysis above, it is difficult to know which models to choose for our predictive analysis. Therefore, Cp, BIC, adjusted R square and residual sum of square (RSS) are used as indicators to help obtain the best model selection.


**Cp** acts as a penalty which tries to minimize overfitting which is created by our model during training the model. Penalty increases as the number of predictors increases. The model with lowest Cp is the best model.

**BIC** is similiar to Cp. The model with least value is the best model as it indicates a low test error. 

**Adjusted R Square** measures the correct variables and voice variable in the variable. A higher adjusted r square is preferable as it  has more correct variables and lesser noise variable into it.

**Residual Sum of Square (RSS)** is used to measure the amount of variance in the data that is not explained by the model. A lower RSS is preferred as it indicates a lower variance and error. 

A set of plots are generated to visualise the best overall model based on Cp, BIC, adjusted r squared and RSS as shown below:


```{r }
par(mfrow = c(2, 2))
plot(reg.summary.full$cp, xlab = "Number of variables", ylab = "Cp", type = "l")
points(which.min(reg.summary.full$cp), reg.summary.full$cp[which.min(reg.summary.full$cp)], col = "red", cex = 2, pch = 20)
plot(reg.summary.full$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(reg.summary.full$bic), reg.summary.full$bic[which.min(reg.summary.full$bic)], col = "red", cex = 2, pch = 20)
plot(reg.summary.full$adjr2, xlab = "Number of variables", ylab = "Adjusted R Square", type = "l")
points(which.max(reg.summary.full$adjr2), reg.summary.full$adjr2[which.max(reg.summary.full$adjr2)], col = "red", cex = 2, pch = 20)
plot(reg.summary.full$rss, xlab = "Number of variables", ylab = "RSS", type = "l")
mtext("Plots of Cp, BIC, adjusted r sqaure and RSS for best subset selection", side = 3, line = -2, outer = TRUE)
```

As shown in the plots above, we can see that based on Cp,adjusted r squared and RSS, the best model is estimated to have 9 predictor variables. For BIC on the other hand, it estimated that the best model have 1 predictor variable.

The best model according to each adjusted r squared, Cp and BIC for best subset selection are extracted to confirm and are shown as below:  
```{r }
# plotting dataframe for best model based on maximum adjusted r square, minimum cp and minimum bic.
data.frame(
adj.r2 = which.max(reg.summary.full$adjr2),
                  cp = which.min(reg.summary.full$cp),
bic = which.min(reg.summary.full$bic))
```
We can see from the above that based on adjusted r square and cp the best model is the one with 9 predictor variables. However, using the BIC criteria, we should go for the model with 1 variable. Given that we have different "best" models depending on which metrics we consider. Therefore, a more vigorous approach is to select a model based on the prediction error computed on a new test data using k-fold cross validation techniques.

We will use the defined helper function above to compute the prediction error for the different best models returned by the regsubsets() function below:
```{r }
model.ids <- 1:9
cv.errors1 <- map(model.ids, get_model_formula,regfit.full) %>%
map(get_cv_error, data = fire_data2) %>%
unlist()
cv.errors1
```
Cross validation errors are shown above based on each of the respective models which have different variable numbers. The which.min() function is performed to indicate the minimum cross validation error from the models as shown below:

```{r }

which.min(cv.errors1)
```

As shown above, we can see that the model with 9 variables is the best model as it has lowest prediction error. This is similiar to the number of variables for the best model selected from hybrid subset selection.

The regression coefficients of this model are shown as below:

```{r }
coef(regfit.full,9)

```

Based on coefficient above, we can see that the 9 variables that are important are X, month.feb, month.oct, day.tue, FFMC,temp, rain, month.dec and day.sun. These are the same variables as the hybrid selection.
```{r }
# storing important variables obtained from cross validation for both hybrid and best subset selection to dataframe named imp_features
imp_features <- fire_data1[,c("X", "month.feb", "month.oct", "day.tue","FFMC.t","temp.t","rain.t","month.dec","day.sun","area.t")]
head(imp_features)
```

#### Splitting dataset into train and test

The data is split into 80% ratio for training and 20% for testing. The model is trained to with the training dataset and the predictions are tested on the testing dataset. Thereafer, the mean squared error (MSE) is then computed to obtain the accuracy of the model. 
```{r }

# ensure results are repeatable
set.seed(5) 
# splitting data with 80 to 20 ratio
sample = sample.split(imp_features, SplitRatio = .80)
# making 80% of the data to train data
train_firedata = subset(imp_features, sample == TRUE)
# making 20% of the data to test data
test_firedata  = subset(imp_features, sample == FALSE)
```
### 3.3 Model 1- Linear model

Below a linear model is built based upon the important features obtained from subset selection. 
```{r }
# ensure results are repeatable
set.seed(5)
# linear model based on important features obtained from feature selection
lmlate = lm(area.t~.,  data = imp_features)
summary(lmlate)
```

**From the summary of the model**:

- The adjusted R-squared ($R^2$) of 0.02182 indicates that this model explains 2.18% of the variation in area. This is lower than the previous R- squared of 0.03291 prior to feature selection. The lower adjusted r square could be due to a reduction in the features. However, the model built with lesser features compared to with full features would be more computationally efficient and have higher interpretebility.  

- The F-statistic 2.279 has a p-value 0.01646 - so we cannot reject the null hypothesis (the model explains nothing) with significance level of 0.05. This model is useful. We note that the p value is lower compared to the p value of the linear model before feature selection of 0.02106. This indicates and improvement in the significance of the model.

- The p-values for the coefficients show that temp and month are significant at the 0.05 level.

```{r }
# getting mse value for model
cat("The MSE for the linear model with variables from feature selection is:",mse_model(lmlate))
```

#### Building linear model based on train dataset
```{r }
# ensure results are repeatable
set.seed(5)
# building linear model from train dataset
linear_model <- lm(area.t ~. , data = train_firedata)
summary(linear_model)
```
**From the summary of the model**:

- The adjusted R-squared ($R^2$) of 0.01419 indicates that this model explains 1.42% of the variation in area.

- The F-statistic 1.66 has a p-value 0.09657 - so we cannot reject the null hypothesis (the model explains nothing) with significance level of 0.05. This model is useful.

- The p-values for the coefficients show that temp and month are significant at the 0.05 level.


#### Predicting on test data:
```{r }
predictions_linear <- linear_model %>% predict(test_firedata)

# storing predictions to compare_lm dataframe for comparison
compare_lm <- as.data.frame(predictions_linear)

# selecting target value
actual_area <- dplyr:: select(test_firedata,area.t)

# merging predicted values with actual values
compare_lm <- cbind(compare_lm, actual_area)
```

```{r }
# printing first few records of actual vs predicted
head(compare_lm)

```
```{r }
# obtaining r squared value for model
cat("The R squared value for the linear model is:",R2(predictions_linear, test_firedata$area.t))
```

```{r }
# obtaining MSE for model
MSE_linear <- mse(predictions_linear, test_firedata$area.t)
cat("The mean squared error (MSE) for the linear model is",MSE_linear)
```
```{r }
# obtaining RMSE for model
RMSE_linear <- RMSE(predictions_linear, test_firedata$area.t)
cat("The root mean square error (RMSE) for the linear model is",RMSE_linear)
```

### 3.4 Model 2 - XGBoost 

- The second model is built on the extreme gradient boosting algorithm, also known as XGBoost.
- The dataset is the final subset of the selected features which was used to implement linear regression. MSE is used as a benchmark to compare the accuracy of the models, in which a lower MSE indicates that the model provides better predictions.
- This model uses an ensemble technique called boosting whereby new models are added to correct the errors made by existing models.
- Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. A gradient descent algorithm is utilized to minimize the loss when adding new models.

#### Creating training and test dataframe for XGBoost Model
```{r }
# creating train dataframe without area column
trainm <- train_firedata[-10]
# creating train data frame with just area column
train_label <- train_firedata[,"area.t"]
# converting training data set to matrix format to use the xgboost algorithm
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)


# creating test dataframe without area column
testm <- test_firedata[-10]
# creating test dataframe with just area column
test_label <- test_firedata[,"area.t"]
# converting test data set to matrix format to use the xgboost algorithm
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)

```
#### Builidng XGBoost model based on train dataset
```{r }
# setting up paramaters to be used by xgboost model
xgb_params <- list("objective" = "reg:linear", # for linear regression
                   "eval_metric" = "rmse") # using rmse for evaluation metric

# creating watchlist to input into XGBoost model, to see how much error exist in each iteration
watchlist <- list(train = train_matrix , test = test_matrix)

# creating XGBoost model for 100 iterations for train model
bst_model <- xgb.train(params = xgb_params, data = train_matrix,nrounds = 100, watchlist = watchlist)
```

As shown above, we can see the train and test RMSE for the XGBoost model for 100 iterations.
#### Training and test error plot for XGBoost
```{r }
e <- data.frame(bst_model$evaluation_log)
# plotting training data's rmse for each iteration
plot(e$iter, e$train_rmse, col = "blue", xlab = "Iteration Number",ylab = "RMSE for train data") 
# plotting test data's rmse for each iteration
lines(e$iter, e$test_rmse, col = 'red')

mtext("Plot of Iteration Number vs RMSE for train data-XGBoost", side = 3, line = -2, outer = TRUE)
```
As shown above, we can see that initially the RMSE for test data decreased, however it quickly increases. This is so as the reduction in error for the training data is causing significant overfitting. The plot for test data indicates that the right model is not found yet.

We can obtain the minimum value for the test RMSE from the plot above below by using the min() function. The iteration number for the minimum value of test RMSE is also investigated.

```{r }
# getting minimum value for test rmse
min(e$test_rmse)
# getting iteration of minimum rmse value
e[e$test_rmse == 1.030477, ]
```
As shown above, we can see that iteration number 4 is where the minimum value of test RMSE is obtained for the training and test plot.

```{r }
# adding on lower eta on XGB Boost train model
bst_model <- xgb.train(params = xgb_params, data = train_matrix,nrounds = 100, watchlist = watchlist, eta = 0.1)
```

The ETA variable is added on to the model to improve the learning rate of the model. The changes are visualised in the training and test error plot below.

```{r }
e <- data.frame(bst_model$evaluation_log)
# plotting training data's rmse for each iteration
plot(e$iter, e$train_rmse, col = "blue", xlab = "Iteration Number", ylab = "RMSE for train data")
# plotting test data's rmse for each iteration
lines(e$iter, e$test_rmse, col = 'red')

mtext("Plot of Iteration Number vs RMSE for train data -XGBoost", side = 3, line = -2, outer = TRUE)
```

As shown above, we can see an improvement in the RMSE for test data as it did not increase as quickly as before. The reduction in error for the training in data is also more gradual now.


The minimum value for the test RMSE from the plot above below is obtained by using the min() function. The iteration number for the minimum value of test RMSE is also further investigated.
```{r}
# getting minimum value for test rmse
min(e$test_rmse)

# getting iteration of minimum rmse value
e[e$test_rmse == 1.027956, ]


```

We can see an improvement in RMSE from  1.030477 to 1.027956 with  a lower eta (learning rate). Iteration number 21 is where the minimum value of test RMSE is obtained for the updated training and test plot.

This would be incorporated into the XGBoost train model as nrounds =21 (to indicate 21 iterations) in order to obtain a minimum error for test data.

```{r}
# performing XGB boost based on iteration number 21 given low rmse to get minimum error for test data
bst_model <- xgb.train(params = xgb_params, data = train_matrix,nrounds = 21, watchlist = watchlist, eta = 0.1)
```

```{r}
e <- data.frame(bst_model$evaluation_log)
# plotting training data's rmse for each iteration
plot(e$iter, e$train_rmse, col = "blue", xlab = "Iteration Number" ,ylab = "RMSE for train data" )
# plotting test data's rmse for each iteration
lines(e$iter, e$test_rmse, col = 'red')

mtext("Plot of Iteration Number vs RMSE for train data -XGBoost", side = 3, line = -2, outer = TRUE)
```

As shown above, we can see an improvement in the RMSE for test data as it did not increase as before and is declining gradually. We can also see that the reduction in error for the training in data is still persistent.

Given a reduction in the RMSE, we can now obtain the important features for the data. 
```{r}
# geting feature importance from importance data table
imp <- xgb.importance(colnames(train_matrix), bst_model)
print(imp)
```
Based on the table above, we can see that the temperature variable is the most important variable followed by FFMC and X. This is based on the relative number of observations and number of times the feature occurs in the trees of the model as calculated by cover and frequency from the importance data table.

We can visualise the ranking of the important variables below as well.
```{r}
# plotting feature importance variables
xgb.plot.importance(imp)
```
As shown above, temperature variable is the most important variable followed by FFMC and X.
#### Predicting on test data:
```{r}
# prediction on test data
predicted_xg <- predict(bst_model, newdata = test_matrix)
```

```{r}
#storing predictions to compare_xg dataframe for comparison
compare_xg <- as.data.frame(predicted_xg)
# selecting target value
actual_area <- test_label
# merging predicted values with actual values
compare_xg <- cbind(compare_xg, actual_area)
# getting first few rows of predicted values and actual values comparison
head(compare_xg)
```
```{r}
# obtaining r squared value for model
cat("The R squared value for the XGBoost model is:",R2(predicted_xg, test_label))
```

```{r}
# obtaining MSE for model
MSE_xg <- mse(predicted_xg, test_label)
cat("The mean squared error (MSE) for the XGBoost model is",MSE_xg)
```

```{r}
# obtaining RMSE for model
RMSE_xg <- RMSE(predicted_xg, test_label)
cat("The root mean square error (RMSE) for the XGBoost model is",RMSE_xg)
```

#### Reason for Choosing XGBoost:

- Tuning of parameters in XGBoost allows improvement in accuracy of prediction and reduces the mean square errors.
- XGBoost penalizes complex models with both ridge and lasso regularization to prevent overfitting of data.
- XGBoost utilizes the power of parallel processing which makes the computation of the model fast.
- Boosting in the model makes use of trees with lesser splits therefore making them more efficient and with better prediction accuracy.

### 3.5 Model 3 - XG Boost lasso
- This model is similiar to the initial XGBoost model performed for model 2 which uses ensemble technique to correct the errors made by existing models. However,this model is enhanced as we are adding on the lasso parameter to the model. 
- The dataset for this model differs from the first and second model as well as it contains initial variables without any feature selection. The dataset for model 1 and 2 is the final subset of the selected features.
- The lasso parameter in this model performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces.

#### Creating training and test dataframe for XGBoost Lasso Model

```{r }
# ensure results are repeatable
set.seed(5) 
# splitting original data before feature selection to 80 to 20 ratio  
sample_xg = sample.split(fire_data2, SplitRatio = .80)
# making 80% of the data to train data
train_firedata_xg = subset(fire_data2, sample_xg == TRUE)
# making 20% of the data to test data
test_firedata_xg  = subset(fire_data2, sample_xg == FALSE)
```

```{r }
# creating train dataframe without area column
trainm_xg <- train_firedata_xg[-30]
# creating train data frame with just area column
train_label_xg <- train_firedata_xg[,"area.t"]
# converting training data set to matrix format to use the xgboost algorithm
train_matrix_xg <- xgb.DMatrix(data = as.matrix(trainm_xg), label = train_label_xg)

# creating test dataframe without area column
testm_xg <- test_firedata_xg[-30]
# creating test data frame with just area column
test_label_xg <- test_firedata_xg[,"area.t"]
# converting test data set to matrix format to use the xgboost algorithm
test_matrix_xg <- xgb.DMatrix(data = as.matrix(testm_xg), label = test_label_xg)
```
#### Builidng XGBoost Lasso model based on train dataset

```{r }
# setting up paramaters to be used by xgboost model
xgb_params <- list("objective" = "reg:linear",# for linear regression
                   "eval_metric" = "rmse", # using rmse for evaluation metric
                   "alpha" = 1) # apply lasso regression

# creating watchlist to input into XGBoost model, to see how much error exist in each iteration
watchlist_xg <- list(train = train_matrix_xg , test = test_matrix_xg)

# creating XGBoost model for 100 iterations
bst_model_xg <- xgb.train(params = xgb_params, data = train_matrix_xg,nrounds = 100, watchlist = watchlist_xg)

```
As shown above, we can see the train and test RMSE for the XGBoost model for 100 iterations.
#### Training and test error plot for XGBoost Lasso

```{r }
e_xg <- data.frame(bst_model_xg$evaluation_log)
# plotting training data's rmse for each iteration
plot(e_xg$iter, e_xg$train_rmse, col = "blue",xlab = "Iteration Number" ,ylab = "RMSE for train data")
# plotting test data's rmse for each iteration
lines(e_xg$iter, e_xg$test_rmse, col = 'red')
mtext("Plot of Iteration Number vs RMSE for train data -XGBoost", side = 3, line = -2, outer = TRUE)

```

As shown above, we can't really see the plot for RMSE for test data. However, we can observe that the RMSE for training data has been declining till roughly around iteration 40. Thereafter, the RMSE remains constant till iteration 100 for training data.

We can obtain the minimum value for the test RMSE from the plot above below by using the min() function. The iteration number for the minimum value of test RMSE is also investigated.

```{r }
# getting minimum value for test rmse
min(e_xg$test_rmse)
# getting iteration of minimum rmse value
e_xg[e_xg$test_rmse == 1.030912, ]
```

As shown above, we can see that iteration number 4 is where the minimum value of 1.030912 for test RMSE is obtained for the training and test plot.
```{r }
# adding on lower eta 
bst_model_xg <- xgb.train(params = xgb_params, data = train_matrix_xg,nrounds = 100, watchlist = watchlist_xg, eta = 0.1)
```

The ETA variable is added on to the model to improve the learning rate of the model. The changes are visualised in the training and test error plot below.
```{r }
e_xg <- data.frame(bst_model_xg$evaluation_log)
# plotting training data's rmse for each iteration
plot(e_xg$iter, e_xg$train_rmse, col = "blue",xlab = "Iteration Number" ,ylab = "RMSE for train data")
# plotting test data's rmse for each iteration
lines(e_xg$iter, e_xg$test_rmse, col = 'red')
mtext("Plot of Iteration Number vs RMSE for train data -XGBoost", side = 3, line = -2, outer = TRUE)
```
As shown above, we can see now see the RMSE for test data which gradually declines before iteration 20 and then increases. The reduction in error for the training in data is also more gradual now and is not stagnant as before.

The minimum value for the test RMSE from the plot above below is obtained by using the min() function. The iteration number for the minimum value of test RMSE is also further investigated.

```{r }
min(e_xg$test_rmse)
e_xg[e_xg$test_rmse == 1.005882, ]
```
We can see an improvement in RMSE from 1.030912 to 1.005882 with  a lower eta (learning rate). Iteration number 13 is where the minimum value of test RMSE is obtained for the updated training and test plot.

This would be incorporated into the XGBoost train model as nrounds =13 (to indicate 13 iterations) in order to obtain a minimum error for test data.

```{r }
# performing XGB boost based on iteration number 13 given low rmse to get minimum error for test data
bst_model_xg <- xgb.train(params = xgb_params, data = train_matrix_xg,nrounds = 13, watchlist = watchlist_xg, eta = 0.1)
```
As shown above, we can see the train and test RMSE for the XGBoost model for 13 iterations.
    
The RMSE for both training and test data are visualised as below.

```{r }
e_xg <- data.frame(bst_model_xg$evaluation_log)
plot(e_xg$iter, e_xg$train_rmse, col = "blue", xlab = "Iteration Number" ,ylab = "RMSE for train data")
lines(e_xg$iter, e_xg$test_rmse, col = 'red')
mtext("Plot of Iteration Number vs RMSE for train data -XGBoost", side = 3, line = -2, outer = TRUE)
```

As shown above, we can see an improvement in the RMSE for test data as it did not increase as before and is declining gradually. We can also see that the reduction in error for the training in data is more spread up.

Given a reduction in the RMSE from , we can now obtain the important features for the data. 
```{r }
# getting important variables 
imp_var <- xgb.importance(colnames(train_matrix_xg), bst_model_xg)
print(imp_var)
```

Based on the table above, we can see that the temperature variable is the most important variable followed by wind and DMC. This is based on the relative number of observations and number of times the feature occurs in the trees of the model as calculated by cover and frequency from the importance data table.

We can visualise the ranking of the important variables below as well.
```{r }
#plotting important variables
xgb.plot.importance(imp_var)
```

As shown above, temperature variable is the most important variable followed by FFMC and X.
#### Predicting on test data:
```{r }
# prediction on test data
predicted_xg2 <- predict(bst_model_xg, newdata = test_matrix_xg)
```

```{r }
#storing predictions to compare_xg2 dataframe for comparison
compare_xg2 <- as.data.frame(predicted_xg2)
# selecting target value
actual_area_xg <- test_label_xg
# merging predicted values with actual values
compare_xg2 <- cbind(compare_xg2, actual_area_xg)
# getting first few rows of predicted values and actual values comparison
head(compare_xg2)
```
```{r }
# obtaining r squared value for model
cat("The R squared value for the XGBoost Lasso model is:",R2(predicted_xg2, test_label_xg))
```
```{r }
# obtaining MSE for model
MSE_xg2 <- mse(predicted_xg2, test_label_xg)
cat("The mean squared error (MSE) for the XGBoost Lasso model is",MSE_xg2)
```
```{r }
# obtaining RMSE for model
RMSE_xg2 <- RMSE(predicted_xg2, test_label_xg)
cat("The root mean square error (RMSE) for the XGBoost Lasso model is",RMSE_xg2)
```
#### Reason for Choosing XGBoost Lasso:

- Able to avoid overfitting and narrow down the features of the data in a more accurate manner with the use of lasso parameter from XGBoost.
- Tuning of parameters in XGBoost allows improvement in accuracy of prediction and reduces the mean square errors.
- The model offer efficient estimates of the test error without incurring the cost of repeated model training associated with cross-validation. 

## 4. Model Comparison<a class="anchor" id="sec_4"></a>
A scatter plot of the predicted vs actual values of the area is displayed to analyse which model has the highest prediction accuracy.
### 4.1 Model 1 - Linear Model:

```{r }
cat("Correlation between predicted and actual values for linear model: ",cor(compare_lm$predictions_linear,compare_lm$area.t) )
```
```{r }
ggplot(compare_lm, aes(x=predictions_linear, y=area.t)) +
  geom_point()+
  geom_smooth(method=lm)+
    ggtitle("Predicted vs test data for Model 1: Linear Model") +
      xlab("Predicted values for area") +
      ylab("Actual values for area ")
      
```
### 4.2 Model 2 - XG Boost
```{r }
cat("Correlation between predicted and actual values for XGBoost Model: ", cor(compare_xg$predicted_xg,compare_xg$actual_area))
```
```{r }
ggplot(compare_xg, aes(x=predicted_xg, y=actual_area)) +
  geom_point()+
  geom_smooth(method=lm)+
    ggtitle("Predicted vs test data for Model 2: XGBoost Model") +
      xlab("Predicted values for area") +
      ylab("Actual values for area ")
```
### 4.3 Model 3 - XGBoost Lasso
```{r }
cat("Correlation between predicted and actual values for XGBoost Lasso Model: ",cor(compare_xg2$predicted_xg2,compare_xg2$actual_area_xg))

```
```{r }
ggplot(compare_xg2, aes(x=predicted_xg2, y=actual_area_xg)) +
  geom_point()+
  geom_smooth(method=lm)+
    ggtitle("Predicted vs test data for Model 3: XGBoost Lasso Model") +
      xlab("Predicted values for area") +
      ylab("Actual values for area")
```
Comparing between the 3 models, it is observed that the fit is better for Model 1 in comparison to model 2 and 3. There is a more linear relationship between the actual and predicted values for model 1 which was built on linear regression. The correlation coefficient between the actual and predicted values are relatively low for the models, with linear regression having the highest correlation value of 0.2158215, followed by XGBoost Lasso with value of 0.08270509 and XGBoost with value of   0.01967348. However, we note that correlation doesnt mean causation. The correlation co-efficient was tabulated to investigate how closely the actual and predicted values are related to each other.

### 4.4 MSE and RMSE of models

- MSE measures the mean of the squares of the errors which is the difference between the predicted values and the actual values and squaring them. The mean of this value provides us an indication of the error in the model in terms of its prediction capability. A lower MSE indicates that the model is performing better in terms of prediction, however a very low MSE could sometimes lead to over fitting. 

- RMSE on the other hand measures the standard deviation of the residuals which indicates how spread out these residuals are. A lower RMSE value indicate a better fit, however like MSE a very low MSE value could lead to over fitting of the data.

The MSE and RMSE calculated after prediction are performed on test data set and is provided as below
```{r }
cat("The mean square error(MSE) of linear model is :", MSE_linear)
```
```{r }
cat("The root mean square error(RMSE) of linear model is :", RMSE_linear)
```
```{r }
cat("The mean square error(MSE) of XGBoost model is :", MSE_xg)
```
```{r }
cat("The root mean square error(RMSE) of XGBoost Model is :", RMSE_xg)
```
```{r }
cat("The mean square error(MSE) of XGBoost Lasso model is :", MSE_xg2)
```
```{r }
cat("The mean square error(MSE) of XGBoost Lasso model is :", MSE_xg2)
```

Based on the values above, we can see that the linear model has the lowest RMSE and MSE amongst the other models. In addition to the better fit of the model based on the plots,  we can conclude that the linear model is the better model in comparison to the other two models.

## 5. Variable Identification and Explanation <a class="anchor" id="sec_5"></a>
The variables used for linear model provided it with the lowest RMSE. Therefore, they are selected as the most important variables.
```{r }
cat("Name of important variables:",colnames(imp_features)[1:8])
```

The chosen subset of attributes that might have a significant impact on the prediction of area are shown above. There were initially 12 predictor variables. Based on the hybrid selection, we've narrowed down our feature selection to X, month.feb, month.oct, day.tue, FFMC, temp, rain and month.dec. It is noted that the month variable in particular February,October and December is when the area of forest is affected most. For the day variable, tuesday is important variable among the other days. Given that month and day are individual variable, the total number of important features are 6 variables namely, X, month,day,FFMC, temp and rain.
### 5.1 Month
Among all the variables in the final subset, one of the influencing attribute is month. In particular, the month of december. This can be observed in the linear regression model developed and as shown below. The p value for this feature is extremely low and is below 0.05. This could possibly due to the humid weather in the month of December which could affect the spread of the fire in the area.

```{r }
# ensure results are repeatable
set.seed(5)
# linear model based on important features obtained from feature selection
lmlate = lm(area.t~.,  data = imp_features)
summary(lmlate)
```

### 5.2 Temperature 
Temperature is also another important variable as reflected in both the models XGBoost and XGBoost Lasso as it was ranked highest for  important variables as shown below. It is also observed in the linear regression model above that the p value for this feature is extremely low and is below 0.05. This could possibly be due to the fact that temperature does affect the humidity and could lead to bushfire. 
```{r }
# importance plot for XGBoost
xgb.plot.importance(imp)
```
```{r }
# importance plot for XGBoost Lasso
xgb.plot.importance(imp_var)
```

### 5.3 FFMC

FFMC is the abbreviation for Fine Fuel Moisture Code. FFMC is one of the important features and as shown in the correlation plot below it is correlated to temperature and month which are also the most important variables in feature selection. Therefore, it could possibly have a greater impact on the response variable due to its interactions with the other 2 important variables.
```{r }
pairs(fire_data[c(-1,-2)], upper.panel = cor_panel) 
```
## 6. Conclusion <a class="anchor" id="sec_6"></a>
- The feature selection methods were utilized to identify the optimum number of features which can explain the changes in the target variable.
- EDA was performed to obtain insights as to which key factors should be considered while selecting features for the model.
- The reduced subset of features was used to build 2 different models which are linear regression and XGBoost. The full subset of features on the other hand was applied on XGBoost Lasso.
- It was observed that R-squared and MSE for a linear model is higher in comparison to the other 2 models.
- The final set of features was 6 variables which have higher significance in comparison to the other features.


