import pymongo

# User credentials will be saved in 'new' database 

# Prerequisite : MongoDB Compass
client = pymongo.MongoClient("localhost", 27017)

# In case you want to use MongoDB Atlas instead of MongoDB Compass
# client = pymongo.MongoClient("mongo-db-connection-string")

db = client.new

collection = db.new