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
