import pymongo
from dotenv import load_dotenv
import os
load_dotenv(override=True)
# User credentials will be saved in 'new' database

MONGO_DB_CONN_STR = os.getenv("MONGO_DB_CONN_STR")
# Prerequisite : MongoDB Compass
# client = pymongo.MongoClient("localhost", 27017)

# In case you want to use MongoDB Atlas instead of MongoDB Compass
# password = os.getenv("DB_PASSWORD")
# connection_string = "mongodb+srv://gaurav:"+password+"@cluster0.g3cuu.mongodb.net/new?retryWrites=true&w=majority"
client = pymongo.MongoClient(MONGO_DB_CONN_STR)

# DocuMindz
db = client.new 

# Users collection
collection = db.new