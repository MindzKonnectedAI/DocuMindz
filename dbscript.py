import pymongo
from dotenv import load_dotenv
import os
load_dotenv()
# User credentials will be saved in 'new' database 

# Prerequisite : MongoDB Compass
# client = pymongo.MongoClient("localhost", 27017)

# In case you want to use MongoDB Atlas instead of MongoDB Compass
password = os.getenv("DB_PASSWORD")
connection_string = "mongodb+srv://gaurav:"+password+"@cluster0.g3cuu.mongodb.net/new?retryWrites=true&w=majority"
client = pymongo.MongoClient(connection_string)

db = client.new
collection = db.new