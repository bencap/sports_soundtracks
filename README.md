## Sports Soundtracks

This project utilizes the Spotify API and various machine learning implementations to predict which sports game a song belongs to. It uses available soundtrack data dating back to 2001 for the four most popular sports games, FIFA, NBA2K, NHL, and Madden. The Spotify API is searched for songs matching the title of those present in the dataset, and then statistics for those songs are saved to a CSV file. This file is used to train an algorithms that attempt to predict the game that a song is from. The best result for this dataset currently achieved with the functions implemented in this repository is 40%, 15% better than a random oracle.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

```
Python3+
NumPy, Spotipy, Pandas, SciKitLearn
```

### Installing and Running

A step by step series of examples that tell you how to get a development env running

To get all the data and code needed to recreate my results, just clone the repository to your local machine. Then, running

```
python convertnba.py nba2k.txt
```

will create a csv file of the 2k track data. You should then concatenate the csv files together, and run 

```
python get_songs.py allsongs.csv
```

This will create another csv file with the Spotify song ID's for each track in the data set.

Finally, running

```
python analyze.py songIDs.csv
```

will walk you through analysis of the data using various ML workflows

## Built With

* [Python3](https://www.python.org/) - Used for all Project Scripts

## Authors

* **Ben Capodanno** - *Initial work* - [bencap](https://github.com/bencap)

See also the list of [contributors](https://github.com/bencap/nba-vis/contributors) who participated in this project.

