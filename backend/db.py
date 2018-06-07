import pymongo

client = pymongo.MongoClient('localhost', 27017)
db = client['yelpdb']


def find_all(collection_name):
    return db[collection_name].find({}, {'_id': False})


def find_count(collection_name):
    return db[collection_name].find({}).count()

def find_reviews_by_business_id(id):
    return db['Italian_Reviews'].find({'business_id': id})

def find_restaurant_count_by_category(category):
    return db['restaurants'].find({'categories': category}).count()


def find_top_restaurants(num):
    return db['Italian_Restaurants'].find({}, {'_id': False}).sort([('review_count', pymongo.DESCENDING)]).limit(num)


def find_random_reviews(num):
    random_reviews = db['Italian_Reviews'].aggregate([{'$sample': {'size': num}}])
    random_reviews = list(random_reviews)

    for review in random_reviews:
        review['restaurant_name'] = db['Italian_Restaurants'].find_one({'business_id': review['business_id']})['name']
        review['user_name'] = db['users'].find_one({'user_id': review['user_id']})['name']
    return random_reviews


def find_random_restaurants(num):
    random_rest = db['Italian_Restaurants'].aggregate([{'$sample': {'size': num}}])
    random_rest = list(random_rest)

    for rest in random_rest:
        rest['restaurant_name'] = db['Italian_Restaurants'].find_one({'business_id': rest['business_id']})['name']
        rest['restaurant_id'] = rest['business_id']
    return random_rest


def find_10(collection_name):
    return db[collection_name].find({}, {'name': True, 'stars': True, '_id': False}).limit(10)

def find_most_useful_users(num):
    return db['Italian_Users'].find({}, {'name': True, 'useful': True, '_id': False}).sort([('useful', pymongo.DESCENDING)]).limit(num)
