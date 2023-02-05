# project1_datascientist_nanodegree
Repository for project 1 of udacity data scientist nanodegree

## 1. Installations


## 4. How to Interact 

### Data understanding

Kaggle Boston Airbnb Open Data have three files: calendar, reviews, and listings. 

#### Calendar

![boston_calendar_head](https://user-images.githubusercontent.com/50749963/215360076-460b34e9-4f3d-467c-922f-025498c98285.jpg)

Calendar data has 1,308,890 rows and consist of 4 columns: the linsting_id that identifies the Airbnb property, the date goes from 2016-09-06 to 2017-09-05, available that have an “f” if the property is not available and a “t” if the property is available, and the price if the listing is available.

#### Reviews

![boston_reviews_head](https://user-images.githubusercontent.com/50749963/215360106-503164dc-163b-4b5f-b27c-ef3344f8e9e4.jpg)

Reviews data has 68,275 rows and consist of 6 columns: the linsting_id  column is the property described by the review, the id identifies the review, the date goes from 2009-03-21 to 2016-09-06, the reviewer_id identifies the Airbnb user who made the review, reviewer_name is the name  of the user who made the review and the comment the review itself .

#### Listings

Listings data is the main data source, it describes the Airbnb property and the host of that property, it contains 95 columns and 3585 rows. Not all columns are relevant for analysis, 49 columns were selected and divided in three dimensions: host, property, and reviews. 
Here is a dictionary with all the select columns and the dimension to which they belong, more information about what each column means can be reviewed [here](http://insideairbnb.com/get-the-data/).

![boston_listing_selectcolumns](https://user-images.githubusercontent.com/50749963/215360160-3a9e9d7e-95da-48ab-bc11-38e9a3954ba0.jpg)

### Data preparation

Calculate the occupancy percentage

```
# Drop price column because we don't need it for our analysis
boston_calendar = boston_calendar.drop('price', axis=1)
# Get one column for each variable on for t(true) and one for f(false)
boston_calendar = pd.concat([boston_calendar.drop('available', axis=1), pd.get_dummies(boston_calendar['available'], prefix='available', prefix_sep='_')], axis=1)
# Group by each list id by adding the number of times each list_id is available and unavailable
boston_occupation = boston_calendar.groupby("listing_id").sum()
# Add a column with the occupancy percentage, which is the number of days occupied divided by the total number of days registered
boston_occupation["occupation_percentage"] = boston_occupation["available_f"]*100/(boston_occupation["available_f"]+boston_occupation["available_t"])
```

Function to transform then occupation percentage to a categorical value

```
def percentage_to_categorical(value,levels):
    '''
    INPUT
    value - A integer or float value from 0 to 100 
    levels -The number of levels or categories into which the 100 percent will be divided 
    
    OUTPUT
    level - A integer that represent the level or category to which the value belongs
    
    This function return the category to which the value belongs,given the number of levels.
    '''
    # Ensures that the value is float
    value = float(value)
    # Defines the upper limit
    up_limit = 100
    # Defines the value that each range will have
    step = 100/levels
    level = levels
    while level > 0:
        if value <= up_limit and value > (up_limit-step):
            return level
        else:
            level -= 1
            up_limit -= step
```
Apply the function and create a consolidate dataframe with all the listings data and the ocupation percentage categoric
```
# Apply function
boston_occupation["occupation_percentage_categoric"] = boston_occupation["occupation_percentage"].apply(lambda x: percentage_to_categorical(x,3))
# Drop columns will no longer be used
boston_occupation = boston_occupation.drop(["available_f","available_t","occupation_percentage"], axis=1)
# Create a consolidate dataframe with all the listings data and the ocupation percentage
df = pd.merge(boston_listings, boston_occupation,left_on="id",right_on="listing_id", how="inner")
```
Function that uses aws comprehend to send the reviews of each property and returns a csv file with the average of each sentiment per property
```
def sentiment_comments_todf(df,client):
    '''
    INPUT
    df - Boston reviews dataframe
    client - Boto3 comprehend client
    
    OUTPUT
    mean_df - A dataframe with the mean sentiment score per property
    '''
    result_dict = {}
    size = len(df)
    general_count,correct_count,fail_count = 1,1,1
    for index, row in df.iterrows():
        general_percentage = round(general_count*100/size,2)
        correct_percentage =round(correct_count*100/general_count,2)
        fail_percentage = round(fail_count*100/general_count,2)
        print("Progress:{}%, Correct:{}%, Fail:{}% ............".format(general_percentage,correct_percentage,
                                                                        fail_percentage),end='\r')
        general_count += 1
        # Extract listing id and comment from row
        listing_id = row[0]
        comment = row[5]
        # Use aws comprehend to extract the sentiment of the comment
        try:
            # Use aws comprehend to detect the sentiment of the select comment
            response = client.detect_sentiment(Text=comment,LanguageCode='en')
            # Add data to the dictionary
            if listing_id not in result_dict.keys():
                result_dict[listing_id]={"Positive":[response["SentimentScore"]["Positive"]],
                                         "Negative":[response["SentimentScore"]["Negative"]],
                                         "Neutral":[response["SentimentScore"]["Neutral"]],
                                         "Mixed":[response["SentimentScore"]["Mixed"]]}
            else:
                result_dict[listing_id]["Positive"].append(response["SentimentScore"]["Positive"])
                result_dict[listing_id]["Negative"].append(response["SentimentScore"]["Negative"])
                result_dict[listing_id]["Neutral"].append(response["SentimentScore"]["Neutral"])
                result_dict[listing_id]["Mixed"].append(response["SentimentScore"]["Mixed"])

            correct_count += 1
        except:
            fail_count += 1
            continue
            
    resume_dict = {}
    # Obtain the mean of each sentiment per property
    for listing_id in result_dict.keys():
        mean_positive = round(sum(result_dict[listing_id]["Positive"])*100/len(result_dict[listing_id]["Positive"]),2)
        mean_negative = round(sum(result_dict[listing_id]["Negative"])*100/len(result_dict[listing_id]["Negative"]),2)
        mean_neutral = round(sum(result_dict[listing_id]["Neutral"])*100/len(result_dict[listing_id]["Neutral"]),2)
        mean_mixed = round(sum(result_dict[listing_id]["Mixed"])*100/len(result_dict[listing_id]["Mixed"]),2)
        number_reviews = len(result_dict[listing_id]["Positive"])

        resume_dict[listing_id]={"mean_positive":mean_positive,"mean_negative":mean_negative,"mean_neutral":mean_neutral,
                                "mean_mixed":mean_mixed,"number_reviews":number_reviews}
        
    # Transform the dictionary with sentiment by property to dataframe  
    mean_df = pd.DataFrame.from_dict(resume_dict,orient='index')
    mean_df.index.name = 'listing_id'
    # Save the dataframe as a csv file
    mean_df.to_csv("mean_sentiment_comments.csv") 
    
    return mean_df
```
Open the csv file with the average of each sentiment per property and merge this data with all the property data and the ocupation percentage categoric
```
# Open the csv file obtained with the function sentiment_comments_todf
boston_sentiment_comments = pd.read_csv("mean_sentiment_comments.csv")
# Rename the columns
boston_sentiment_comments.columns = ['listing_id', 'reviews_sentiment_positive', 'reviews_sentiment_negative',
                                     'reviews_sentiment_neutral','reviews_sentiment_mixed','number_sentiment_reviews']
# Create a consolidate dataframe with all the listings data, the ocupation percentage categoric and the sentiment analysis
df = pd.merge(df, boston_sentiment_comments,left_on="id",right_on="listing_id", how="inner")
```
Filters the consolidated dataframe to ensure that it contains only boston properties and create a dataframe with the column name and the percent of missing values per column
```
# Create a dataframe with the column name and the percent of missing values per column
percent_missing = df.isnull().sum()*100/len(df)

df_nullrows_percent = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing}).reset_index(drop=True)

# Filter and display missing values dataframe
df_nullrows_percent[df_nullrows_percent["percent_missing"]>0].sort_values(by=['percent_missing'],ascending=False)
```
![columns_missing_Values](https://user-images.githubusercontent.com/50749963/216449634-3b3ede72-7f69-49b6-8b5f-c7b95e54d8b5.jpg)

-Drop columns: square_feet, security_deposit and cleaning_fee because they have a very high percentage of missing values, and are not critical variables for the analysis<br>
-Review columns, host_response_rate and host_acceptace_rate have two to nine percent of missing values but they are important columns, so they will be filled with the mean<br>
-Host_response_time have 8.3 percent of missing values but is an important column so I will use a dummy nan column for the missing values<br>
-Columns: zipcode,bathrooms,bedrooms,host_location,beds and property_type have a very low percentage of missing values so for those columns the rows that contain nan values will be removed<br>
```
# Drop columns
df = df.drop(["square_feet","security_deposit","cleaning_fee"],axis=1)
# Extract the % sign and transform the value to float.
df['host_response_rate'] = df['host_response_rate'].str.replace('%', '', regex=False).astype(float)
df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%', '', regex=False).astype(float)
# Extract the $ and "," sign and transform the value to float
df['price'] = df['price'].str.replace('$', '', regex=False)
df['price'] = df['price'].str.replace(',', '', regex=False)
df['price'] = df['price'].astype(float)
df['extra_people'] = df['extra_people'].str.replace('$', '', regex=False)
df['extra_people'] = df['extra_people'].str.replace(',', '', regex=False)
df['extra_people'] = df['extra_people'].astype(float)
# Apply the fill mean function to the select columns
fill_mean_cols = ["review_scores_accuracy","review_scores_location","review_scores_value","review_scores_checkin",
                 "review_scores_communication","review_scores_cleanliness","review_scores_rating",
                 "host_response_rate","host_acceptance_rate"]

fill_mean = lambda col: col.fillna(col.mean())
df[fill_mean_cols] = df[fill_mean_cols].apply(fill_mean)
# Get dummy columns for host_response_time column including nan values
var = "host_response_time"
df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var],prefix=var, prefix_sep='-', drop_first=True,dummy_na=True)], axis=1)
# Drop the remaining nan values
df = df.dropna()
```
Host_location has 146 different the values, the most frequent being the city of Boston. Creating dummy variables for each host location will create 146 new columns, and for an Airbnb user doesn’t matter if the host is in Hawaii or Alaska, the only aspect that a user probably cares, is if the host is in the same city as the Airbnb. Because of this, host_location became host_isin_city, a variable that describes if the host lives in Boston or not.
```
def host_isin_city(host_location):
    '''
    INPUT
    host_location - str value with the location of the host
    
    OUTPUT
    host_isin_city - int 1 or int 0
    
    This function return integer 1 if the host live in Boston, integer 0 if the host does not live in boston.
    '''
    try:
        clean_host_location = host_location.split(",")[0]
        if clean_host_location == "Boston":
            return 1
        else:
            return 0
    except AttributeError:
        return np.nan
    
# Apply function to create new column "host_isin_city"
df['host_isin_city'] = df["host_location"].apply(host_isin_city)
# Drop old column
df = df.drop("host_location",axis=1)
```
Finally, there are two variables: host_verifications and amenities, that have something like a list but that must be cleaned and separated to create dummy columns.
```
def get_dummies_strlist(df,column_name):
    '''
    INPUT
    df - dataframe to which function will be applied
    column_name - name of the column of the dataframe to which the function will be applied
    
    OUTPUT
    df - dataframe with corresponding dummy columns
    
    This function clean a str with a structure similar to a list or dictionary and return an actual list, 
    then generate a dummy column for each unique element of the list and add the binary value.
    '''

    # Creat a list to save all possible values
    all_items = []
    # Characters to be remove
    replacements = ['"',"'","[","]","{","}"]
    # Loop through all rows in the df
    for index, row in df.iterrows():
        # Select the value based on the column being cleaned
        items = row[column_name]
        # Loop through the characters that need to be removed
        for char in replacements:
            if char in items:
                # Remove the character
                items = items.replace(char,"")
        # Convert already cleaned str value to list
        items = items.split(",")
        # Loop through the items in the list
        for item in items:
            # Check that the item has at least one character
            if len(item) != 0:
                # Clears the item in case it has empty spaces
                clean_item = item.strip()
                clean_item = "{}-{}".format(column_name,clean_item)
                # Check that a column already exists for that item
                if clean_item in df.columns:
                    # If exists assign a value of 1 to that row and column
                    df.loc[index, clean_item] = 1
                else:
                    # If it does not exist create the column and then assign the value of 1 to that row and column
                    df[clean_item] = 0
                    df.loc[index, clean_item] = 1
                    
    # Drop the original column that was transformed
    df = df.drop([column_name], axis=1)
    # Return the new dataframe
    return df
```
### Data modeling

Function to use the get_dummies method and generate the dummy columns for all the dataframes of the experiment.
```
def dummi_variables_multipledfs(dfs):
    '''
    INPUT
    dfs - A dictionary with one or more pandas dataframe with categorical variables you want to dummy
    
    OUTPUT
    result_dict - A dictionary with the input pandas dataframes but with dummy columns for each of the categorical columns
    '''
    # Start an empty dictionary
    result_dict = {}
    # Iterate over dictionary keys
    for key in dfs.keys():
        # Extract the pandas dataframe associated with that key
        df = dfs[key]
        # Pull a list of the column names of the categorical variables
        cat_vars = df.select_dtypes(include=['object']).copy().columns
        for var in  cat_vars:
            # For each cat add dummy var, drop original column
            df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var],prefix=var, prefix_sep='-', drop_first=True)], axis=1)
        # Add to the result dictionary the new dataframe with dummy columns 
        result_dict[key]=df
    return result_dict
```
Function to use the random forest classifier algorithm n times per dataframe, obtaining the mean accuracy..
```
def experiment_randomforestclassifier(exp_dict,n_repetitions):
    '''
    INPUT
    exp_dict - A dictionary with one or more pandas dataframe
    n_repetitions -Number of times to run Random forest classifier
 
    OUTPUT
    level - A dictionary with the mean accuracy per dataframe obtain with the random forest classifier
    
    '''
    result_dict = {}
    count = 0
    while count < n_repetitions:
        for key in exp_dict.keys():
            df = exp_dict[key]
            #Split data into an X matrix and a response vector y
            y = df['occupation_percentage_categoric']
            x = df.drop('occupation_percentage_categoric', axis=1)
            # Split dataset into training set and test set
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
            #Create a Gaussian Classifier
            rfc=RandomForestClassifier(n_estimators=100,random_state=42)
            #Train the model using the training sets y_pred=clf.predict(X_test)
            rfc.fit(x_train,y_train)
            # Generate prediction
            y_pred=rfc.predict(x_test)
            # Obtain accuracy
            accuracy = metrics.accuracy_score(y_test, y_pred)
            # features
            feature_importance = pd.Series(rfc.feature_importances_,index=list(x.columns)).sort_values(ascending=False)
            # final dict
            try:
                result_dict[key]["accuracy"] += accuracy
            except KeyError:
                result_dict[key] = {"accuracy":accuracy}
        count += 1
        
    # Get the mean, update the result_dict with that value, and graph the results
    x_axis, y_axis = [],[]
    for key in result_dict.keys():
        accuracy_mean = (result_dict[key]["accuracy"])/n_repetitions
        result_dict[key]["accuracy"] = accuracy_mean
        x_axis.append(key)
        y_axis.append(accuracy_mean)
    # Plot the experiment results
    plt.bar(x_axis, y_axis)
    plt.title('Accuracy per experiment')
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')
    # Get the dowm limit of the y axis
    down_ylim = (math.floor(min(y_axis)*10)/10)
    up_ylim = (math.ceil(max(y_axis)*10)/10)
    plt.ylim(down_ylim,up_ylim)
    plt.show()
    
    return result_dict
```
Experiment related to location variables
```
# Defines the dataframe with only zipcode for location variable
df_with_zipcode = df.drop(['latitude','longitude','neighbourhood_cleansed'],axis=1)
# Defines the dataframe with only coordinates for location variable
df_with_coordinates = df.drop(['zipcode','neighbourhood_cleansed'],axis=1)
# Defines the dataframe with only neighbourhood for location variable
df_with_neighbourhood = df.drop(['zipcode','latitude','longitude'],axis=1)
# Create de experiment dictionary
location_experiment = {"df_with_zipcode":df_with_zipcode,"df_with_coordinates":df_with_coordinates,
                    "df_with_neighbourhood":df_with_neighbourhood}
```
![accuracy_location_exp](https://user-images.githubusercontent.com/50749963/216794571-dc15e6e9-8ddb-4c56-9b13-a3f7f9efae72.jpg)

![accuracy_location_exp_numbers](https://user-images.githubusercontent.com/50749963/216794837-55155b7b-9107-453a-9f1f-b0883d1c39fd.jpg)

Experiment related to reviews using coordinates as the location variable
```
# Dataframe with original review variables, sentiment variables, and coordinates as location variable
df_withsentiment = df.drop(['zipcode','neighbourhood_cleansed'],axis=1)
# Dataframe with original review variables and coordinates as location variable
df_withoutsentiment = df.drop(['reviews_sentiment_positive', 'reviews_sentiment_negative','reviews_sentiment_neutral',
                               'reviews_sentiment_mixed','number_sentiment_reviews','zipcode','neighbourhood_cleansed'],axis=1)
# Dataframe with sentiment variables without original review variables and coordinates as location variable
df_without_originalreviews = df.drop(['number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness',
                                      'review_scores_checkin','review_scores_communication','review_scores_location',
                                      'review_scores_value','zipcode','neighbourhood_cleansed'],axis=1)
# Create de experiment dictionary
sentiment_experiment = {'withsentiment':df_withsentiment,'withoutsentiment':df_withoutsentiment,
                             'without_originalreviews':df_without_originalreviews}
```
![accuracy_sentiment_exp](https://user-images.githubusercontent.com/50749963/216794803-159f1271-c553-4647-b7f8-e16b397f6b2c.jpg)

![accuracy_sentiment_exp_number](https://user-images.githubusercontent.com/50749963/216794892-66c461a4-d86a-4deb-b68d-fbce59af06fb.jpg)

The final selected dataframe contains the coordinates as location variable and does not contain the sentiment of the reviews. The following function was used to separate the final dataframe by dimension and perform the last experiment.
```
def get_dataframe_perdimension(select_df,columns_for_analysis):
    '''
    INPUT
    select_df - dataframe with the final select columns
    columns_for_analysis - A dictionary with the columns of interest and the dimensions to which they belong
 
    OUTPUT
    df_per_dimension - A pandas datraframe per dimension
    df_column_dimension - A pandas dataframe with the column and the  dimension to which it belongs
    '''
    # Dictionary to add columns per dimension
    columns_per_dimension = {}
    # Loop through dataframe columns
    for dummy in select_df.columns:
        # Obtain the original name of the column before get_dummies
        column = dummy.split("-")[0]
        # Check if the column is in our columns of interest and get its dimension
        if column in columns_for_analysis.keys():
            column_dimension = columns_for_analysis[column]
            # Add the values to the dictionary
            if column_dimension in columns_per_dimension.keys() and column_dimension != "output":
                columns_per_dimension[column_dimension].append(dummy)
            elif column_dimension not in columns_per_dimension.keys() and column_dimension != "output":
                columns_per_dimension[column_dimension] = [dummy,"occupation_percentage_categoric"]

    # Dictionary to add dataframe per dimension
    df_per_dimension = {}
    # Loop through dimensions, get the columns from that dimension and filter the datafrane to add to the dictionaray
    for dimension in columns_per_dimension.keys():
        columns = columns_per_dimension[dimension]
        df = select_df[columns]
        df_per_dimension[dimension] = df

    # List to add the column and the dimension to which it belongs
    column_dimension = []
    # Loop through dimensions, get the columns from that dimension and add the pair value to the list
    for dimension in columns_per_dimension.keys():
        for column in columns_per_dimension[dimension]:
            column_dimension.append([column,dimension])
    # Create a dataframe with the column name and the dimension to which it belongs      
    df_column_dimension = pd.DataFrame(column_dimension, columns = ['column','dimension'])
    # Add the all dimensions dataframe to the df_per_dimension dictionary
    df_per_dimension["all_dimensions"] = select_df

    return df_per_dimension, df_column_dimension
```
![accuracy_dimension_exp](https://user-images.githubusercontent.com/50749963/216795943-10cc5e56-6130-4df3-ad0a-f713e8211ab3.jpg)
![accuracy_dimension_exp_number](https://user-images.githubusercontent.com/50749963/216795945-2321ed94-8f18-4714-9fae-8d87c8f40d30.jpg)

### Result evaluation

Finally, random forest classifier was applied to the selected dataframe to obtain the final accuracy and the feature importance.

```
#Split data into an X matrix and a response vector y
y = select_df['occupation_percentage_categoric']
x = select_df.drop('occupation_percentage_categoric', axis=1)
# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#Create a Gaussian Classifier
rfc=RandomForestClassifier(n_estimators=100,random_state=42)
#Train the model using the training sets y_pred=clf.predict(X_test)
rfc.fit(x_train,y_train)
# Generate prediction
y_pred=rfc.predict(x_test)
# Obtain accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
# Create a dataframe with the score per column (variable or feature)
feature_importance = pd.DataFrame([list(x.columns),rfc.feature_importances_]).transpose()
feature_importance = feature_importance.rename(columns={1:'score',0:'column'}).sort_values(by='score',ascending=False)
feature_importance = pd.merge(feature_importance,df_column_dimension ,on='column', how="inner")
```
![final_accuracy_and_featureimportance](https://user-images.githubusercontent.com/50749963/216796211-34ec127f-7487-4af2-ab88-8d808346241e.jpg)

To see all the feature importance in detail [click here](https://plotly.com/~jordiagv/1/).

<div>
    <a href="https://plotly.com/~jordiagv/1/?share_key=dPmkofT2Y1sYRFtxulc1kR" target="_blank" title="score_by_feature" style="display: block; text-align: center;"><img src="https://plotly.com/~jordiagv/1.png?share_key=dPmkofT2Y1sYRFtxulc1kR" alt="score_by_feature" style="max-width: 100%;width: 1000px;"  width="1000" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
</div>

