

## Data Preparation

### 1A: Data Analysis
how many records there are: 376912
how many attributes: \[id (string), time (datetime), variable (string), value (float)]

--> ranges of values
    --> id: we have data for 27 users
    --> datetime: the data spans over 113 days, from 2024-02-17 to 2024-06-09
    --> variables: 19
                              count        min        max     mean  median
    --> variable                                                          
    --> mood                   5641      1.000     10.000    6.993   7.000
    --> circumplex.arousal     5597     -2.000      2.000   -0.099   0.000
    --> circumplex.valence     5487     -2.000      2.000    0.688   1.000
    --> activity              22965      0.000      1.000    0.116   0.022
    --> screen                96578      0.035   9867.007   75.335  20.044
    --> call                   5239      1.000      1.000    1.000   1.000
    --> sms                    1798      1.000      1.000    1.000   1.000
    --> appCat.builtin        91288 -82798.871  33960.246   18.538   4.038
    --> appCat.communication  74276      0.006   9830.777   43.344  16.226
    --> appCat.entertainment  27125     -0.011  32148.677   37.576   3.391
    --> appCat.finance          939      0.131    355.513   21.755   8.026
    --> appCat.game             813      1.003   5491.793  128.392  43.168
    --> appCat.office          5642      0.003  32708.818   22.579   3.106
    --> appCat.other           7650      0.014   3892.038   25.811  10.028
    --> appCat.social         19145      0.094  30000.906   72.402  28.466
    --> appCat.travel          2846      0.080  10452.615   45.731  18.144
    --> appCat.unknown          939      0.111   2239.937   45.553  17.190
    --> appCat.utilities       2487      0.246   1802.649   18.538   8.030
    --> appCat.weather          255      1.003    344.863   20.149  15.117

, distribution of values,
relationships between attributes, missing values,

1 - number of records per person per date plot - shows that head and tail of dates can be stripped

2 - number of records where following date has no mood -> shows that we strip many days but do not strip many records

3 - distributions of each feature per day -> shows that on some days for some features, the values are incorrect   


### Data Cleaning

--> which factors have errors, which people have errors, which days have errors ?:
    --> does one person always produce erroneous data
    --> is there one factor which always / never produces erroneous data
    --> is there one day on which there is erroneous data for all / some factors ?

--> how can the errors be fixed?
    --> are they salvageable ? i.e. if all values are off by a factor of exactly 100x, then you can divide by 100 rather 
        than removing entirely
    --> maybe can use pd.DataFrame.clip(lower, higher)
    --> may just have to throw some data away (must report what total % of data was thrown away)