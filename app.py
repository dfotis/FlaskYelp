from flask import Flask, render_template, Markup, make_response, request
from backend import db, plot, model
import pandas as pd
import json



app = Flask(__name__, template_folder='views')


@app.route("/")
def home():
    num_of_restaurants = db.find_count('Italian_Restaurants')
    num_of_users = db.find_count('Italian_Users')
    num_of_reviews = db.find_count('Italian_Reviews')

    return render_template('home.html', num_of_restaurants=num_of_restaurants, num_of_users=num_of_users,
                           num_of_reviews=num_of_reviews)


@app.route("/restaurants")
def restaurants():

    top_10_restaurants = db.find_top_restaurants(10)
    top_10_restaurants = list(top_10_restaurants)

    top_categories_plot = plot.top_restaurant_categories()

    random_restaurant = db.find_random_restaurants(1)
    month_rating_plot = plot.average_review_by_month(random_restaurant[0]['restaurant_id'])

    random_restaurant_name = random_restaurant[0]['restaurant_name']

    restaurant_stars = plot.star_ratings_by_restaurant_plot()

    return render_template('restaurants.html', top_10_restaurants=top_10_restaurants,
                           top_categories_plot=top_categories_plot, month_rating_plot=month_rating_plot,
                            random_restaurant_name=random_restaurant_name, restaurant_stars = restaurant_stars)

@app.route("/users")
def customers():
    # random_reviews = db.find_random_reviews(5)
    usefull_users = plot.usefull_users_plot()
    return render_template('users.html', usefull_users=usefull_users)

@app.route("/reviews")
def reviews():
    star_ratings_plot = plot.star_ratings_plot_reviews()
    pos_wordcloud = plot.wordcloud_plot('pos')
    neg_wordcloud = plot.wordcloud_plot('neg')

    return render_template('reviews.html', star_ratings_plot=star_ratings_plot,
                           pos_wordcloud=pos_wordcloud , neg_wordcloud=neg_wordcloud)


def Union(lst1, lst2):
    final_list = lst1 + lst2
    return final_list

@app.route("/map")
def map():
    restaurants = db.find_all('Italian_Restaurants')
    restaurants = list(restaurants)

    rest = pd.read_json(json.dumps(list(db.find_all('Italian_Restaurants'))))

    categories = []
    for temp in rest.categories.values:
        categories = Union(categories, temp)
    categories = sorted(set(categories))

    return render_template('map.html', restaurants=restaurants, categories=categories)



@app.route("/sentiment", methods=['GET', 'POST'])
def sentiment():
    sentiment_result = ''
    if request.method == 'POST':  # this block is only entered when the form is submitted
        sentence = request.form.get('sentence')
        #framework = request.form['framework']
        m = model.model_creation()

        result = m.predict([sentence])
        if(result[0] == 0):
            sentiment_result = 'Sentiment: Negative'
        else:
            sentiment_result = 'Sentiment: Positive'

        return render_template('sentiment.html', sentiment_result=sentiment_result)
    return render_template('sentiment.html', sentiment_result='')


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5110, threaded=True)
