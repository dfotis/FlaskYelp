from flask import Flask, render_template, Markup, make_response
from backend import db, plot
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

    return render_template('restaurants.html', top_10_restaurants=top_10_restaurants,
                           top_categories_plot=top_categories_plot, month_rating_plot=month_rating_plot,
                            random_restaurant_name=random_restaurant_name)


@app.route("/users")
def customers():
    random_reviews = db.find_random_reviews(5)
    return render_template('users.html', random_reviews=random_reviews)

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


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5110)
