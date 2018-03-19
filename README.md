
## Distinguishing Sentiments

In this exercise I utilized Python libraries - pandas, numpy, matplotlib.pyplot, tweepy, seaborn, datetime, VADER - JSON traversals, and Twitter's API to perform a sentiment analysis on the news mood based on tweets from five different news organizations - __BBC, CBS, CNN, Fox News, and New York times__.

## Three observable trends based on the data below-
1.	The scatterplot reflecting the sentiment for the most recent one hundred tweets on Twitter for five major news organizations was highly variable ranging anywhere from ~-0.95 to +0.95, with -1 being the most negative sentiment, and +1 being the most positive sentiment, based on the VADER (Valence Aware Dictionary and sEntiment Reasoner) Sentiment Analysis. Visually it was difficult to determine which news organizations had the most positive or negative sentiments based on the scatterplot alone. 

2.	Numerous points on the scatterplot were located at the 0 (zero) y-intercept. My first assumption was that these points simply represented an overwhelming number of tweets with neutral sentiment, but a closer look at the tweet text indicated that several of these “neutral” points also represented tweets in languages other than English, which could not be evaluated by VADER, and were, therefore, given a compound score of 0. I added a filter to my code so that only English tweets were counted and evaluated, but a few tweets in other languages still managed to come through in my analysis. 

3.	A bar plot representing the mean tweet sentiment made it easier to interpret the overall sentiment at a specific time for each news organization as being more positive or more negative. Having said that, the sentiment means for the same news organization varied tremendously from hour to hour, and day to day (data not shown). When I ran my code two days ago, which coincided with the release of the book “Fire and Fury: Inside the Trump White House” by Michael Wolff for example, all news organizations presented a negative mean sentiment. The bar plot below represents an analysis performed Sunday night (01/07/2018) with positive mean sentiment values for BBC, CBS and the NY Times (ranging from +0.06 to +0.09, a slightly negative mean for Fox News (- 0.03) and a negative sentiment mean for CNN (-0.1). I noticed that several of the tweets were about the Golden Globe Awards, which may partially explain the overall boost in tweet sentiment this evening, compared to earlier today. Overall, it would be best to sample tweets throughout a couple of months or a year to get a better idea of the overall sentiment for each news organization on Twitter.


```python
# Import dependencies
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import json
import numpy as np
from IPython.display import display
from datetime import datetime

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
#Set up and call config document
import yaml
TWITTER_CONFIG_FILE = 'auth.yaml'

with open (TWITTER_CONFIG_FILE, 'r') as config_file:
    config = yaml.load (config_file)
#print(type(config))
```


```python
# Twitter API Keys
access_token = config ['twitter']['access_token']
access_token_secret = config ['twitter']['access_token_secret']
consumer_key= config['twitter']['consumer_key']
consumer_secret = config ['twitter']['consumer_secret']
#print(access_token, access_token_secret, consumer_key, consumer_secret)
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target Search Term
news_orgs = ("BBC", "CBS", "CNN","FoxNews","nytimes")
    
# Create arrays to hold sentiments for all news organizations
all_sentiments=[]
sentiment_means=[]

# Loop through all target news organizations
for org in news_orgs:
    
    # Reset counter for each news_org loop
    counter=1
    
    # Variables for holding sentiments
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    
    # Run search for each tweet
    public_tweets = api.search(org, count=100, result_type="recent",lang='en')       
    #print(json.dumps(public_tweets["statuses"], indent=4, sort_keys=True, separators=(',',': ')))   
    
    # Loop through all tweets
    for tweet in public_tweets["statuses"]:

        # Run Vader Analysis on each tweet
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]

        # Add each value to the appropriate arrays above
        compound_list.append(compound)
        positive_list.append(pos)
        negative_list.append(neg)
        neutral_list.append(neu)  
        #print(org)
        #print (compound_list, tweets_ago)
        #print(" ")
        
        # Append all sentiments to an array
        all_sentiments.append({" Media" : org,
                           "Date": tweet["created_at"], 
                           "Compound": compound,
                           "Positive": pos,
                           "Neutral": neu,
                           "Negative": neg,
                           "Tweets_Ago": counter
                            })  
        # Add 1 to counter    
        counter+=1
        
    # Store the Average Sentiments into the array created above
    sentiment_means.append({" Media": org,
                    "Compound_Mean": np.mean(compound_list),
                    "Positive": np.mean(positive_list),
                    "Neutral": np.mean(negative_list),
                    "Negative": np.mean(neutral_list),
                    "Count": len(compound_list)
                    })

# Convert all_sentiments to DataFrame
all_sentiments_pd = pd.DataFrame.from_dict(all_sentiments)
all_sentiments_pd.to_csv("sentiments_array_pd.csv")
display(all_sentiments_pd)
#print(all_sentiments_pd.dtypes)

# Convert sentiment_means to DataFrame 
sentiment_means_pd = pd.DataFrame.from_dict(sentiment_means) 
display(sentiment_means_pd)
```


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Media</th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets_Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:04:28 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BBC</td>
      <td>0.5719</td>
      <td>Mon Jan 08 07:04:27 +0000 2018</td>
      <td>0.000</td>
      <td>0.850</td>
      <td>0.150</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BBC</td>
      <td>-0.6597</td>
      <td>Mon Jan 08 07:04:27 +0000 2018</td>
      <td>0.306</td>
      <td>0.694</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BBC</td>
      <td>0.7906</td>
      <td>Mon Jan 08 07:04:26 +0000 2018</td>
      <td>0.000</td>
      <td>0.750</td>
      <td>0.250</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BBC</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:04:25 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BBC</td>
      <td>0.5994</td>
      <td>Mon Jan 08 07:04:25 +0000 2018</td>
      <td>0.075</td>
      <td>0.717</td>
      <td>0.208</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BBC</td>
      <td>-0.5423</td>
      <td>Mon Jan 08 07:04:25 +0000 2018</td>
      <td>0.218</td>
      <td>0.691</td>
      <td>0.091</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BBC</td>
      <td>0.5719</td>
      <td>Mon Jan 08 07:04:24 +0000 2018</td>
      <td>0.000</td>
      <td>0.575</td>
      <td>0.425</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BBC</td>
      <td>0.5719</td>
      <td>Mon Jan 08 07:04:23 +0000 2018</td>
      <td>0.000</td>
      <td>0.850</td>
      <td>0.150</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BBC</td>
      <td>0.6369</td>
      <td>Mon Jan 08 07:04:23 +0000 2018</td>
      <td>0.000</td>
      <td>0.755</td>
      <td>0.245</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BBC</td>
      <td>0.6369</td>
      <td>Mon Jan 08 07:04:23 +0000 2018</td>
      <td>0.000</td>
      <td>0.802</td>
      <td>0.198</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BBC</td>
      <td>-0.4404</td>
      <td>Mon Jan 08 07:04:22 +0000 2018</td>
      <td>0.253</td>
      <td>0.642</td>
      <td>0.106</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BBC</td>
      <td>0.5106</td>
      <td>Mon Jan 08 07:04:22 +0000 2018</td>
      <td>0.092</td>
      <td>0.683</td>
      <td>0.225</td>
      <td>13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BBC</td>
      <td>-0.4767</td>
      <td>Mon Jan 08 07:04:22 +0000 2018</td>
      <td>0.256</td>
      <td>0.744</td>
      <td>0.000</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BBC</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:04:20 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BBC</td>
      <td>0.5719</td>
      <td>Mon Jan 08 07:04:20 +0000 2018</td>
      <td>0.000</td>
      <td>0.850</td>
      <td>0.150</td>
      <td>16</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BBC</td>
      <td>-0.2263</td>
      <td>Mon Jan 08 07:04:19 +0000 2018</td>
      <td>0.087</td>
      <td>0.913</td>
      <td>0.000</td>
      <td>17</td>
    </tr>
    <tr>
      <th>17</th>
      <td>BBC</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:04:19 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>18</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BBC</td>
      <td>0.3612</td>
      <td>Mon Jan 08 07:04:19 +0000 2018</td>
      <td>0.000</td>
      <td>0.898</td>
      <td>0.102</td>
      <td>19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BBC</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:04:18 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>20</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BBC</td>
      <td>0.5719</td>
      <td>Mon Jan 08 07:04:18 +0000 2018</td>
      <td>0.000</td>
      <td>0.850</td>
      <td>0.150</td>
      <td>21</td>
    </tr>
    <tr>
      <th>21</th>
      <td>BBC</td>
      <td>0.5719</td>
      <td>Mon Jan 08 07:04:18 +0000 2018</td>
      <td>0.000</td>
      <td>0.850</td>
      <td>0.150</td>
      <td>22</td>
    </tr>
    <tr>
      <th>22</th>
      <td>BBC</td>
      <td>-0.2617</td>
      <td>Mon Jan 08 07:04:18 +0000 2018</td>
      <td>0.127</td>
      <td>0.785</td>
      <td>0.088</td>
      <td>23</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BBC</td>
      <td>-0.6486</td>
      <td>Mon Jan 08 07:04:18 +0000 2018</td>
      <td>0.374</td>
      <td>0.485</td>
      <td>0.141</td>
      <td>24</td>
    </tr>
    <tr>
      <th>24</th>
      <td>BBC</td>
      <td>-0.3595</td>
      <td>Mon Jan 08 07:04:18 +0000 2018</td>
      <td>0.161</td>
      <td>0.839</td>
      <td>0.000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BBC</td>
      <td>-0.2732</td>
      <td>Mon Jan 08 07:04:17 +0000 2018</td>
      <td>0.123</td>
      <td>0.877</td>
      <td>0.000</td>
      <td>26</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BBC</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:04:17 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>27</td>
    </tr>
    <tr>
      <th>27</th>
      <td>BBC</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:04:16 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>28</td>
    </tr>
    <tr>
      <th>28</th>
      <td>BBC</td>
      <td>0.5267</td>
      <td>Mon Jan 08 07:04:15 +0000 2018</td>
      <td>0.094</td>
      <td>0.644</td>
      <td>0.262</td>
      <td>29</td>
    </tr>
    <tr>
      <th>29</th>
      <td>BBC</td>
      <td>-0.1027</td>
      <td>Mon Jan 08 07:04:15 +0000 2018</td>
      <td>0.123</td>
      <td>0.877</td>
      <td>0.000</td>
      <td>30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>nytimes</td>
      <td>0.5719</td>
      <td>Mon Jan 08 07:03:39 +0000 2018</td>
      <td>0.000</td>
      <td>0.861</td>
      <td>0.139</td>
      <td>71</td>
    </tr>
    <tr>
      <th>471</th>
      <td>nytimes</td>
      <td>-0.8519</td>
      <td>Mon Jan 08 07:03:39 +0000 2018</td>
      <td>0.283</td>
      <td>0.717</td>
      <td>0.000</td>
      <td>72</td>
    </tr>
    <tr>
      <th>472</th>
      <td>nytimes</td>
      <td>0.2732</td>
      <td>Mon Jan 08 07:03:39 +0000 2018</td>
      <td>0.107</td>
      <td>0.741</td>
      <td>0.152</td>
      <td>73</td>
    </tr>
    <tr>
      <th>473</th>
      <td>nytimes</td>
      <td>0.2732</td>
      <td>Mon Jan 08 07:03:39 +0000 2018</td>
      <td>0.000</td>
      <td>0.806</td>
      <td>0.194</td>
      <td>74</td>
    </tr>
    <tr>
      <th>474</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:39 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>75</td>
    </tr>
    <tr>
      <th>475</th>
      <td>nytimes</td>
      <td>-0.4767</td>
      <td>Mon Jan 08 07:03:38 +0000 2018</td>
      <td>0.147</td>
      <td>0.853</td>
      <td>0.000</td>
      <td>76</td>
    </tr>
    <tr>
      <th>476</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:38 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>77</td>
    </tr>
    <tr>
      <th>477</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:37 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>78</td>
    </tr>
    <tr>
      <th>478</th>
      <td>nytimes</td>
      <td>-0.4215</td>
      <td>Mon Jan 08 07:03:37 +0000 2018</td>
      <td>0.109</td>
      <td>0.891</td>
      <td>0.000</td>
      <td>79</td>
    </tr>
    <tr>
      <th>479</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:36 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>80</td>
    </tr>
    <tr>
      <th>480</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:35 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>81</td>
    </tr>
    <tr>
      <th>481</th>
      <td>nytimes</td>
      <td>-0.1695</td>
      <td>Mon Jan 08 07:03:35 +0000 2018</td>
      <td>0.180</td>
      <td>0.702</td>
      <td>0.118</td>
      <td>82</td>
    </tr>
    <tr>
      <th>482</th>
      <td>nytimes</td>
      <td>-0.3182</td>
      <td>Mon Jan 08 07:03:35 +0000 2018</td>
      <td>0.091</td>
      <td>0.909</td>
      <td>0.000</td>
      <td>83</td>
    </tr>
    <tr>
      <th>483</th>
      <td>nytimes</td>
      <td>0.4939</td>
      <td>Mon Jan 08 07:03:35 +0000 2018</td>
      <td>0.000</td>
      <td>0.802</td>
      <td>0.198</td>
      <td>84</td>
    </tr>
    <tr>
      <th>484</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:35 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>85</td>
    </tr>
    <tr>
      <th>485</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:33 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>86</td>
    </tr>
    <tr>
      <th>486</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:31 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>87</td>
    </tr>
    <tr>
      <th>487</th>
      <td>nytimes</td>
      <td>0.3818</td>
      <td>Mon Jan 08 07:03:31 +0000 2018</td>
      <td>0.000</td>
      <td>0.885</td>
      <td>0.115</td>
      <td>88</td>
    </tr>
    <tr>
      <th>488</th>
      <td>nytimes</td>
      <td>-0.5106</td>
      <td>Mon Jan 08 07:03:31 +0000 2018</td>
      <td>0.320</td>
      <td>0.680</td>
      <td>0.000</td>
      <td>89</td>
    </tr>
    <tr>
      <th>489</th>
      <td>nytimes</td>
      <td>-0.5122</td>
      <td>Mon Jan 08 07:03:31 +0000 2018</td>
      <td>0.212</td>
      <td>0.788</td>
      <td>0.000</td>
      <td>90</td>
    </tr>
    <tr>
      <th>490</th>
      <td>nytimes</td>
      <td>-0.4215</td>
      <td>Mon Jan 08 07:03:30 +0000 2018</td>
      <td>0.109</td>
      <td>0.891</td>
      <td>0.000</td>
      <td>91</td>
    </tr>
    <tr>
      <th>491</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:30 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>92</td>
    </tr>
    <tr>
      <th>492</th>
      <td>nytimes</td>
      <td>0.5719</td>
      <td>Mon Jan 08 07:03:30 +0000 2018</td>
      <td>0.000</td>
      <td>0.861</td>
      <td>0.139</td>
      <td>93</td>
    </tr>
    <tr>
      <th>493</th>
      <td>nytimes</td>
      <td>-0.5574</td>
      <td>Mon Jan 08 07:03:29 +0000 2018</td>
      <td>0.375</td>
      <td>0.625</td>
      <td>0.000</td>
      <td>94</td>
    </tr>
    <tr>
      <th>494</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:29 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>95</td>
    </tr>
    <tr>
      <th>495</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:28 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>96</td>
    </tr>
    <tr>
      <th>496</th>
      <td>nytimes</td>
      <td>-0.0516</td>
      <td>Mon Jan 08 07:03:28 +0000 2018</td>
      <td>0.239</td>
      <td>0.606</td>
      <td>0.155</td>
      <td>97</td>
    </tr>
    <tr>
      <th>497</th>
      <td>nytimes</td>
      <td>-0.4215</td>
      <td>Mon Jan 08 07:03:27 +0000 2018</td>
      <td>0.109</td>
      <td>0.891</td>
      <td>0.000</td>
      <td>98</td>
    </tr>
    <tr>
      <th>498</th>
      <td>nytimes</td>
      <td>0.0000</td>
      <td>Mon Jan 08 07:03:27 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>99</td>
    </tr>
    <tr>
      <th>499</th>
      <td>nytimes</td>
      <td>0.4767</td>
      <td>Mon Jan 08 07:03:25 +0000 2018</td>
      <td>0.000</td>
      <td>0.795</td>
      <td>0.205</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 7 columns</p>
</div>



<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Media</th>
      <th>Compound_Mean</th>
      <th>Count</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC</td>
      <td>0.083353</td>
      <td>100</td>
      <td>0.84301</td>
      <td>0.06990</td>
      <td>0.08707</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBS</td>
      <td>0.061578</td>
      <td>100</td>
      <td>0.86744</td>
      <td>0.05553</td>
      <td>0.07701</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>-0.107187</td>
      <td>100</td>
      <td>0.82274</td>
      <td>0.10897</td>
      <td>0.06833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FoxNews</td>
      <td>-0.028154</td>
      <td>100</td>
      <td>0.82891</td>
      <td>0.09015</td>
      <td>0.08093</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nytimes</td>
      <td>0.057799</td>
      <td>100</td>
      <td>0.86923</td>
      <td>0.05773</td>
      <td>0.07304</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Create a scatterplot
all_sentiments_pd.set_index('Tweets_Ago', inplace=True)
all_sentiments_pd.groupby(' Media')['Compound'].plot(legend=True, marker = 'o', linewidth=0)

# Customize scatterplot features
plt.style.use('ggplot')
plt.axhline(c='k', alpha=0.2, linestyle= 'dashed')
plt.axis([0,6,-1.1,1.1])
plt.xlim(0,101)
plt.ylim(-1,1)
plt.xlabel("Tweets Ago", fontsize=15)
plt.ylabel("Tweet Polarity", fontsize=15)
plt.legend(loc=(1.0, 0.75),edgecolor='black')
plt.grid(True, ls='dashed')
plt.title("Sentiment Analysis per Media Source" + " "+ "(" + datetime.now().strftime('%m/%d/%Y') + ")")
plt.savefig("Sentiment Analysis of Media Tweets.png",bbox_inches='tight')
plt.show()
```


![png](README_files/README_6_0.png)



```python
# Create a barplot
ax=sns.barplot(x=' Media', y='Compound_Mean', data=sentiment_means_pd)

# Customize barplot features
ax.set_xlabel('Media', fontsize=15)
ax.set_ylabel('Tweet Polarity', fontsize=15)
ax.set_title("Overall Media Sentiment based on Twitter"+ " "+ "(" + datetime.now().strftime('%m/%d/%Y') + ")")
ax.set_ylim(-0.12, 0.12)
ax.grid(True, ls='dashed')
ax.hlines(0, -1, 10, colors='k', alpha=0.4)
plt.savefig("Overall Sentiment based on Twitter.png")
plt.show()
```


![png](README_files/README_7_0.png)



```python

```
