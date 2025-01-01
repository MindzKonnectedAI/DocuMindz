import pymongo
from dotenv import load_dotenv
import os
load_dotenv()
# User credentials will be saved in 'new' database 

# Prerequisite : MongoDB Compass
# client = pymongo.MongoClient("localhost", 27017)

# In case you want to use MongoDB Atlas instead of MongoDB Compass
password = os.getenv("DB_PASSWORD")
connection_string = "mongodb+srv://shikharcrpf:PE9BVvFUnIkFT9ya@cluster0.myqwb.mongodb.net/"
client = pymongo.MongoClient(connection_string)

db = client.new
collection = db.new