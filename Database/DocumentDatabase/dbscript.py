import pymongo
import os

# User credentials will be saved in 'new' database

MONGO_DB_CONN_STR = os.getenv("MONGO_DB_CONN_STR")

# Prerequisite : MongoDB Compass
# client = pymongo.MongoClient("localhost", 27017)

# In case you want to use MongoDB Atlas instead of MongoDB Compass
client = pymongo.MongoClient(MONGO_DB_CONN_STR)

# DocuMindz ( DB Name )
db = client.new 

# Users collection ( Collection Name)
collection = db.new