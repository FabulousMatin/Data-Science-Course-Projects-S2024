# Data Science Course
- [CA0: Data Scrapping](#ca0-data-scrapping)
- [CA1: Statistics](#ca1-statistics)
- [CA2: EDA](#ca2-eda)
- [CA3: PySpark](#ca3-pyspark)
- [CA4: Modeling](#ca4-modeling)
- [CA5: Feature Engineering](#ca5-feature-engineering)
- [CA6: Clustering](#ca6-clustering)
- [CA7: Large Language Model](#ca7-large-language-model)
- [Project - Phase 0:](#project---phase-0---data-retrieval)
- [Project - Phase 1:](#project---phase-1---preprocess-and-eda)
- [Project - Phase 2:](#project---phase-2---prediction)


## CA0: Data Scrapping

### Overview
This project focuses on web scraping and introductory data analysis using Ethereum blockchain transaction data from Etherscan.io.

### Tasks

1. **Data Collection**
    - Using web scraping techniques, we collect transaction data from Etherscan, focusing on transactions from the last 10 blocks.

2. **Data Analysis**
    - **Load the Data**: Import the transaction data into a pandas DataFrame.
    - **Data Cleaning**: Clean the data by converting data types, removing irrelevant information, and handling duplicates.
    - **Statistical Analysis**: Calculate the mean and standard deviation of the population. Plot histograms, normal distribution plots, box plots, and violin plots for transaction values and fees.
    - **Visualization**: Create visual representations to aid in the analysis of transaction values.

3. **Data Sampling**
    - **Simple Random Sampling (SRS)**: Randomly select a subset of data.
    - **Stratified Sampling**: Divide the data into strata based on transaction value and randomly select samples from each stratum.
    - **Comparison**: Compare the mean and standard deviation of the samples with the population statistics.

## CA1: Statistics

### Overview
In this project, we get acquainted with and implement some tools for statistical analysis like Monte Carlo Simulation, CLT and T-Test.


### Tasks


1. **Monte Carlo Simulation**
    - **Pi Calculation**: Estimate Pi by generating random points within a square and counting those inside an inscribed circle. Repeat with different point counts and analyze.
    - **Mensch Game**: Simulate a simplified version of the Mensch game to calculate the probability of winning for each player.


2. **Central Limit Theorem (CLT)**
    - Select three different probability distributions.
    - Generate random samples, calculate their means, and plot histograms overlaid with expected normal distributions.
    - Repeat for increasing sample sizes and observe the changes.


3. **Hypothesis Testing**
    - **Unfair Coin**
      - Simulate a biased coin. Perform hypothesis testing to determine fairness using confidence interval and p-value approaches with sample sizes of 30, 100, and 1000.
    
    - **T-Test**
      - Calculate t-statistic and degrees of freedom for two groups.
      - Determine p-value and report results using manual calculations and the SciPy library.
    
    - **Job Placement**
      - Test if working alongside studying affects grades using a job placement dataset. Perform hypothesis tests manually and with SciPy, then compare results.

</details>


## CA2: EDA

### Overview
In this assignment we investigate open-ended questions. The open-ended questions ask you to think creatively and critically about perform some EDAs on the provided datasets.

### Tasks
1. The provided dataset contains information about the passengers of the sunken ship ‘RMS Lusitania’. In this task, performed some preprocess and usual EDAs like such as executing some queries, plotting and etc using numpy, matplotlib and mighty pandas.
2. This dataset focuses on data scientist salaries across different regions from 2020 to 2024. We needed to do some data cleaning and more preprocess, then applied different techniques to extract some usefull insights about the data.
   
## CA3: PySpark

### Overview
In this assignment, we work with PySpark, a Python API for Apache Spark. For the first step, we do somewarm-up exercises, in order to learn how to use it. Next, we work on a large Spotify parquet dataset to gain different insights from it using various and different methods we've learnt so far.
The Spotify dataset has more than 1.5M data with around 25 features that we performed multiple preprocess, data cleaning and feature engineering to prepare it for EDA.

### Some of our reuslts:
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart0.png" width="520"> 
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart1.png" width="520"> 
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart2.png" width="520"> 
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart10.png" width="520"> 
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart4.png" width="520"> 
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart8.png" width="520"> 
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart9.png" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart7.PNG" width="520">

#### Relation of Each Pair of Features
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart3.png" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart5.png" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart6.png" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart6_2.png" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart6_3.png" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA3/assets/chart6_4.png" width="520">


## CA4: Modeling

### Overview
In this assignment, we explore various loss functions and apply gradient descent methods to optimize these functions. We work with the Diabetes dataset from the scikit-learn library This dataset consists of medical diagnostic measurements from numerous patients and is specifically designed to study diabetes progression. We use these data points to predict the quantitative measure of disease progression one year after baseline, thus practicing the application of regression analysis in a medical context.

### Tasks
1. **Functions’ Implementation**
    - Implementing following functions from scratch:
      - Mean Squared Error (MSE)
      - Mean Absolute Error (MAE)
      - Root Mean Squared Error (RMSE)
      - Coefficient of Determination (R² Score)

2. **Building and Training the Linear Regression Model**
3. **Model Evaluation**
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA4/assets/plot1_2.PNG" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA4/assets/plot1.png" width="520">

4. **Ordinary Least Squares**
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA4/assets/plot2.PNG" width="520">
   
   
## CA5: Feature Engineering

### Overview
In this assignment, first we apply feature engineering techniques to a football-related dataset to analyze the likelihood of scoring a goal through a shot. Next, we delve into regression and cross-validation concepts further by implementing multivariate regression and k-fold cross-validation from scratch and utilize them on a preprocessed dataset related to cars and also compare our outcomes with those attained using Python's built-in libraries.

### Tasks
1. **Preprocess, Feature Engineering and Model Evaluation on an Football Dataset**
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA5/assets/plot2.png" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA5/assets/plot5.png" width="520">

#### Before Feature Engineereing
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA5/assets/plot3.png" width="520">

#### After Feature Engineereing
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA5/assets/plot4.PNG" width="520">

2. **Multivariate Regression Implementation**
We implement multivariate regression from scratch and use the gradient descent algorithm to update the weights. Also we plot the accuracy across different random states for a more robust verification.
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA5/assets/plot6.png" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA5/assets/plot7.png" width="520">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA5/assets/plot6_2.png" width="520">

3. **Manual K-Fold Cross Validation Implementation**
We implement K-Fold cross-validation from scratch. As in the previous section, use the gradient descent algorithm to adjust the weights.
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA5/assets/plot8.PNG" width="300">

4. **Comparison with Built-in Python Libraries**

## CA6: Clustering

### Overview
In this assignment, we delve into dimensionality reduction and unsupervised learning tasks. We work with the database from an artice called [Impact of c1HbA Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.](https://www.hindawi.com/journals/bmri/2014/781670/), with more than 200k items and 50 features.

### Tasks

1. **Preprocess**
2. **Dimensionality Reduction with PCA**
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA6/assets/plot1.png" width="550">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA6/assets/plot2.png" width="550">

4. **Unsupervised Learning**
   - K-Means
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA6/assets/plot3.png" width="550">

  - DBSCAN
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA6/assets/plot4.PNG" width="550">

     
## CA7: Large Language Model

### Overview
In this assignmet, we work with a IMDB review comment dataset and try to train a model to classify them as positive or negetive automatically. First, we use different methods to expand labeled data for training, extract features from sentences, and then we train and evaluate our classifier models. We used usual models such as, Decision Tree, Logistic Regression, Gaussian Naive Bayes, Gradiant Boosting, Random Forest and SVM with different kernels to propagete the labels. At last, we used [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) LLM model to generate labels and compare it with traditional methods.

### Tasks

1. **EDA**
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot1.png" width="720">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot2.png" width="720">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot3.png" width="720">

2. **Feature Engineering**
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot4.png" width="550">

3. **Labeling with Traditional Methods**
  - K-Means
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot5.PNG" width="300">

  - KNN from scratch
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot6.PNG" width="300">
        
- Label Propagation
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot7.PNG" width="300">

- Self Training
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot8.PNG" width="300">

4. **Labeling using LLM**
  - Chain of Thoughts
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot9.PNG" width="500">

  - Labeling Test Data with LLM
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/CA7/assets/plot10.PNG" width="400">
    

## Project - Phase 0 - Data Retrieval:
In this phase we started by gathering housing data through the [Realtor.com](https://www.realtor.com/) API. During this phase, we encountered several challenges:

- **Rate Limitations**: The API's rate limit restricted the number of requests we could make. To overcome this, we utilized multiple systems and IP addresses to distribute the load.
- **Duplicated Data**: Initially, our method for sorting and receiving data led to significant duplication. We refined our approach by implementing better sorting mechanisms to ensure the retrieval of unique listings.

After addressing these issues, we successfully collected 43,000 unique housing records, each with 25 features.

## Project - Phase 1 - Preprocess and EDA: 
This phase was dedicated to data cleaning, feature engineering and EDAs.
- **Data Cleaning**: The data cleaning process was meticulous, focusing on handling missing values contextually for each column:

    - **Location-based Imputation**: For certain features, we employed K-Nearest Neighbors (KNN) to impute missing values based on data from neighboring cities.
    - **Statistical Methods**: For other columns, we used the average or median of the data's distribution, ensuring the imputed values were appropriate for their context.
    - **Tags**: We used the data in tags column to fill some of our columns.

- **Feature Engineering**: We enhanced the dataset by performing feature engineering:

    - **Exploding Tags Columns**: Some columns contained lists of tags or categories. We exploded these columns, using One-Hot encoding so that they better represented the data's structure. We also added new columns based on that tags.

- **EDA**: We performed many EDAs. Here are some of them:
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot12.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot13.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot1.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot2.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot3.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot4.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot5.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot6.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot7.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot8.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot10.png" width="500">
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%201/assets/plot11.png" width="500">
  
## Project - Phase 2 - Prediction:
In this phase after performing some more feature engineerings, we used classic ML methods and a Neural Network to predict the prices of the houses.
- **Feature Engineering**
  
Log Transform Price

<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%202/assets/plot1.png" width="320"> <img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%202/assets/plot2.png" width="320">


- **Dimansion Reduction**: To understand the data's variance and simplify our model, we applied Principal Component Analysis (PCA):

    - Variance Explained: We assessed how much variance was captured by the first two principal components. It was around 20%.
    - Feature Requirement: We determined the number of principal components needed to cover 95% of the dataset's variance. Number of needed dimensions with keeping 95% of data: 32

<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%202/assets/plot3.png" width="600">
  
- **Neural Network**: A Deep learning model was trained to capture complex patterns in the data.
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%202/assets/plot4.png" width="700">

- **Classic ML**: We used these models to predict the target:
  - Decision Tree
  - Gradient Boosting
  - Linear Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - XGBoost
  - Support Vector Machine (SVM)
<img src="https://github.com/FabulousMatin/DataScienceCourse/blob/main/Project%20-%20phase%202/assets/plot5.png" width="600">


---
## Contributors
- [Matin Bazrafshan](https://github.com/FabulousMatin)
- [Mohammad Nemati](https://github.com/mmd-nemati)
- [Parva Sharifi](https://github.com/parvash)
