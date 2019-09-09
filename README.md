# Mall Customer Demographic Segmentation with K-Means Clustering
## Context
<font size="+2">In this project I will explore a dataset on mall customers and perform market basket analysis to understand customer segmentation. This dataset was created for learning purposes of understanding market segmentation and unsupervised learning. I will be using a simple unsupervised learning algorithm called K-Means Clustering to cluster mall customers into different segments. </font>

## Content
<font size="+2">The owners of a commercial mall have some basic information about their customers with a membership card. The dataset contain attributes like CustomerID, age, gender, annual income, and spending score. Spending score is a predefined metric with parameters like consumer behavior and purchasing data.
 </font>

## Preliminary Data Analysis
<font size="+2">In the preliminary data analysis we want to understand the quality, structure and the source of the data. After importing relevant packages and reading our dataset into a data frame, we can use the info and describe methods to have a quick overview of our data. Some common data problems we want to look out for are: missing data, outliers, duplicate rows, columns that needs to be processed, and column types. It's important to recognize data problems early on in the process so we can avoid backtracking later on. Please see below for info and describe method outputs.
<br>
<br>
<img src="/Images/data_info.PNG" width="400" height="350">

<br> Summary: No missing values, consistent data types, no outliers, binarize gender.
Since the data is relatively clean, we don't need to do too much data cleaning. We will just need to binarize the gender column and scale our dataset.

</font>

## Exploratory Data Analysis
<font size="+2">
In this section we want to dive deeper into our data and understand the distribution, correlation, range, and behavior of our data. <br>
Distribution Plot <br>
<img src="/Images/dist_plots.PNG" width="400" height="350">
As we can see, each column is relatively normally distributed. <br>
Gender Count Plot <br>
<img src="/Images/gender_count.PNG" width="400" height="350">
Pairplot Plot <br>
<img src="/Images/pairplot.PNG" width="400" height="350">
The above plots indicates little to no correlations between the variables.
</font>

## Data Preprocessing
<font size="+2">
Before we move onto building the actual model, we need to preprocess our dataset into a format in which the Scikit-Learn algorithms can process. This includes binarizing the gender variable and scaling our variables. 
We will use LabelBinarizer and StandardScaler from Scikit-Learn preprocessing to perform these tasks.
</font>

```python
#Binarize Gender
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(data.Genre)
lb.classes_
data.Genre = lb.transform(data.Genre)

#Scaling dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data[['Gender','Age','Annual_Income','Spending_Score']]))
```

## Building the Model
<font size="+2">
Now that everything is ready to go, we can start building our model. 
We want to segment our customers base on different combinations of the variables so our first step is to create dataframes with these combinations. 
</font>

```python
income_spending = X[['Annual_Income','Spending_Score']]
age_spending = X[['Age','Spending_Score']]
age_income_spending = X[['Age','Spending_Score','Annual_Income']]
```

<font size="+2">
Second, we need to determine the optimal number of clusters for each combination of variables. 
Objective: Minimize Within Cluster Sum of Square (WCSS) with the Elbow method. <br>
As the number of cluster increases WCSS decreases.<br> 
When the number of clusters = number of instances WCSS = 0. <br>
When we graph number of clusters against WCSS we will see an elbow graph.<br>
On this elbow plot, the marginal decrease in WCSS decreases significantly at the elbow.<br>
This point will be our optimal number of clusters. 
</font>

```python
def WCSS(segment):
    WCSS = []
    for n in range(1, 20):
        clustering = (KMeans(n_clusters = n, n_init = 20, tol = 0.0001, random_state = 21, algorithm = 'auto'))
        clustering.fit(segment)
        WCSS.append(clustering.inertia_)
    plt.plot(np.arange(1,20), WCSS, 'o')
    plt.plot(np.arange(1,20), WCSS, '-', alpha = 0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.suptitle('Number of Clusters vs WCSS', color='w')
    plt.show()
    return WCSS
```

Age and 



## Interpreting Model Output
## Conclusion
