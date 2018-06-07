import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from json import dumps
from backend import db
import collections
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
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


def Union(lst1, lst2):
    final_list = lst1 + lst2
    return final_list


def top_restaurant_categories():
    rest = []
    for r in db.find_all('restaurants'):
        rest.append(r)

    restaurants = pd.DataFrame(rest)

    cat = []
    for list in restaurants.categories.values:
        cat = Union(cat, list)
    cat = sorted(set(cat))

    category_count = {}

    for category in sorted(cat):
        count = db.find_restaurant_count_by_category(category)
        category_count[category] = count

    category_count = collections.Counter(category_count)

    d = {}
    for c in category_count.most_common(20):
        d[c[0]] = c[1]

    x = []
    y = []
    for k in d.keys():
        x.append(k)
        y.append(d[k])

    sns.barplot(y, x)
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

def average_review_by_month(business_id):
    cursor = db.find_reviews_by_business_id(str(business_id))
    train  = pd.DataFrame(list(cursor))
    data = {'date': train['date'],'stars':train['stars']}
    review_time = pd.DataFrame(data)
    review_time = review_time.set_index('date')

    # calcuate average review by month by creating one dict {'date', stars} and one dict_count{'date',count}
    dic = {}
    dic_count = {}
    for d,s in review_time.iterrows():
        datetemp = pd.to_datetime(d)
        timestamp = datetemp.year*1000 + datetemp.month


        if timestamp in dic:
            dic[timestamp] += s
            dic_count[timestamp] += 1

        else:
            dic[timestamp] = s
            dic_count[timestamp] = 1

    for i in dic:
        dic[i] = dic[i] / dic_count[i]

    rate = []
    for i in list(dic.values()):
        rate.append(float(i))
    date = []
    for i in dic.keys():
        date.append(str(i))
        date = sorted(date)

    plt.figure(figsize=(15,5))
    my_xticks = date
    plt.xticks(range(0,len(date)),date,rotation = 70,fontsize = 12)
    plt.plot(range(0,len(date)),rate,linewidth=3.0,color = 'blue')

    figfile = io.BytesIO()

    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')

    return figdata_png