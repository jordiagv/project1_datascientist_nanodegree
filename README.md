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

calculate the occupancy percentage

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
Apply the function and create a consolidate dataframe with all the listings data and the ocupation percentage
```
# Apply function
boston_occupation["occupation_percentage_categoric"] = boston_occupation["occupation_percentage"].apply(lambda x: percentage_to_categorical(x,3))
# Drop columns will no longer be used
boston_occupation = boston_occupation.drop(["available_f","available_t","occupation_percentage"], axis=1)
# Create a consolidate dataframe with all the listings data and the ocupation percentage
df = pd.merge(boston_listings, boston_occupation,left_on="id",right_on="listing_id", how="inner")
```

