# project1_datascientist_nanodegree
Repository for project 1 of udacity data scientist nanodegree

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

