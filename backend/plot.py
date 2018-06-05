import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from json import dumps
from backend import db
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64
from wordcloud import WordCloud,STOPWORDS
import tempfile
from flask import send_file


def star_ratings_plot_reviews():
    rev = list(db.find_all('Italian_Reviews'))

    reviews_df = pd.DataFrame(rev)

    fig, ax = plt.subplots()

    counts = reviews_df.stars.value_counts()

    sns.barplot(x=counts.index, y=counts, ax=ax)

    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Percentage")
    plt.plot()


    figfile = io.BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')

    return figdata_png


def wordcloud_draw(data, color = 'black'):
    plt.figure(figsize=(12, 10))

    wordcloud = WordCloud(background_color=color,
                          width=1200,
                          stopwords=STOPWORDS,
                          height=1000
                          ).generate(str(data['Preprocessed_Text']))

    plt.imshow(wordcloud)
    plt.axis('off')

    plt.plot()

    return plt



def wordcloud_plot(type):
    rev = list(db.find_all('Italian_Reviews'))

    reviews_df = pd.DataFrame(rev)
    if(type == 'neg'):
        reviews_df = reviews_df[(reviews_df['stars'] == 1) | (reviews_df['stars'] == 2)]
        wordcloud = wordcloud_draw(reviews_df)
    else:
        reviews_df = reviews_df[(reviews_df['stars'] == 5) | (reviews_df['stars'] == 4)]
        wordcloud = wordcloud_draw(reviews_df, 'white')

    figfile = io.BytesIO()
    wordcloud.savefig(figfile, format='png')
    figfile.seek(0)
    wordcloud_png = base64.b64encode(figfile.getvalue()).decode('ascii')

    return wordcloud_png