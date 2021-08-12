# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:59:19 2021

@author: andre
"""
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment import SentimentIntensityAnalyzer


def main():
    file_path = pathlib.Path(r'C:\Users\andre\Documents\Python\discord message analysis\PUT THAT CANDY BACK - Text - general [111237149793222656].csv')
    save_path = pathlib.Path(r'C:\Users\andre\Documents\Python\discord message analysis\WSB Artifacts')
    data = pd.read_csv(file_path, parse_dates=['Date'])
    comment_words = ''
    stopwords = set(STOPWORDS)
    stopwords.update(["youtube","now","nan","vote","required","local",
                                  "appdata","person","skip","song","want","time",
                                  "song","com","imgur","@","one", "got",
                                  "dont","don't","go","gonna","im","https",
                                  "make","11","still","server",
                                  "new","def","player","mag",
                                  "year","getting","need","position","team",
                                  "tomorrow","grid","home","something",
                                  "give","able","top","ton","numOfRuns",
                                  "start","take","watch", "https", "v", "play", 'u'])
    ############## contribution distribution #######################
    author_count = data['Author'].value_counts()
    author_count_top = author_count[0:20]
    authors = author_count_top.index.values
    authors_cleaned = []
    for author in authors:
        authors_cleaned.append(author.split("#")[0])

    fig = plt.figure(figsize = (10,6))
    plt.bar(authors_cleaned, author_count_top.values, zorder = 3)
    plt.xticks(rotation = 'vertical')
    plt.title('Discord Message Distribution by Author')
    plt.xlabel('Author')
    plt.ylabel('Messages Sent')
    plt.grid(zorder = 0)
    save_path.mkdir(parents = True, exist_ok = True)
    plt.savefig(save_path / 'contribution_distribution.png', dpi = 400, bbox_inches = 'tight')
    plt.close()
    #############################activity by date######################################
    activity = {}
    for row in data.iterrows():
        year = row[1]['Date'].year
        month = row[1]['Date'].month
        day = row[1]['Date'].day
        key = datetime(year, month, day)
        if key in activity.keys():
            activity[key] += 1
        else:
            activity[key] = 1
            
    fig = plt.figure(figsize = (10,6))
    plt.plot(activity.keys(), activity.values(), zorder = 3)
    plt.xticks(rotation = 'vertical')
    plt.title('Discord Message Activity by Day')
    plt.xlabel('Time')
    plt.ylabel('Number of Messages Sent')
    plt.grid(zorder = 0)
    save_path.mkdir(parents = True, exist_ok = True)
    plt.savefig(save_path / 'message_activity.png', dpi = 400, bbox_inches = 'tight')
    plt.close()
    #######################word cloud all#######################################
    # iterate through the csv file
    for val in data.Content:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i, token in enumerate(tokens):
            tokens[i] = token.lower()
        comment_words += " ".join(tokens)+" "
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10,
                    collocations=False).generate(comment_words)
    # plot the WordCloud image
    plt.figure(figsize=(10, 10), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path / 'word_cloud_all.png', dpi = 400, bbox_inches = 'tight')
    plt.close()
    
    ################################word cloud by user###########################################
    author_count = data['Author'].value_counts()
    author_count_top = author_count[0:10]
    top_authors = author_count_top.index.values
    # for author in authors:
    data_reindexed = data.groupby(data.Author,as_index=False)    
    for user, user_data in data_reindexed:
        if user in top_authors:
            for val in user_data.Content:
                val = str(val)
                # split the value
                tokens = val.split()
                # Converts each token into lowercase
                for i, token in enumerate(tokens):
                    tokens[i] = token.lower()
                comment_words += " ".join(tokens)+" "
            wordcloud = WordCloud(width = 800, height = 800,
                            background_color ='white',
                            stopwords = stopwords,
                            min_font_size = 10,
                            collocations=False).generate(comment_words)
            # plot the WordCloud image
            plt.figure(figsize=(10, 10), facecolor=None)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.tight_layout(pad=0)
            user_name = user.split('#')[0]
            plt.savefig(save_path / f'word_cloud_{user_name}.png', dpi = 400, bbox_inches = 'tight')
            plt.close()
    ###################### sentiment by user ################################
    author_count = data['Author'].value_counts()
    author_count_top = author_count[0:10]
    top_authors = author_count_top.index.values
    sia = SentimentIntensityAnalyzer()
    author_scores = {}
    for author in top_authors:
        author_scores[author] = {}
    # for author in authors:
    data_reindexed = data.groupby(data.Author,as_index=False)    
    for user, user_data in data_reindexed:
        if user in top_authors:
            popularity_scores_total = {
                'negative': 0, 
                'neutral': 0,
                'positive': 0}
            for message in user_data.Content:
                try:
                    popularity_scores = sia.polarity_scores(message)
                    popularity_scores_total['negative'] += popularity_scores['neg']
                    popularity_scores_total['neutral'] += popularity_scores['neu']
                    popularity_scores_total['positive'] += popularity_scores['pos']
                except:
                    pass
            author_scores[user] = popularity_scores_total
    authors_cleaned = []
    for author in top_authors:
        authors_cleaned.append(author.split("#")[0])
    positives = []
    neutrals = []
    negatives = []
    for author, values in author_scores.items(): 
        total = values['negative'] + values['neutral'] + values['positive']
        positives.append(values['positive']/total)
        neutrals.append(values['neutral']/total)
        negatives.append(values['negative']/total)
  
    x = np.arange(len(top_authors))
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, positives, width, label='Positive')
    rects2 = ax.bar(x + width/2, negatives, width, label='Negative')
    ax.set_ylabel('Positivity/Negativity')
    ax.set_title('Sentiment Analysis by User for All-time Data using VADER model')
    ax.set_xticks(x)
    ax.set_xticklabels(authors_cleaned)
    ax.legend()
    
if __name__ == "__main__":
    main()
