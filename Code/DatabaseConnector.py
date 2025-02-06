import openai
import json
import re
import os
import sys
import sqlite3

class DatabaseConnector:
    def __init__(self):
        print("db inst")

    def readJSON(self, file_path="generatedFiles/query_results.json"):
        try:
            data = []
            #Load the existing data from the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except Exception as e:
            print(f"An error occurred: {e}")
        return data
    
    def runSQLQuery(self, sql_query, data):
        #print("running SQL query: ",sql_query)
        try:
            db_name = data['Database']
            db_path = db_name
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            ###########
            # Remove the ´´´sql formatting of OPENAI
            clean_sql = re.sub(r"^```[a-zA-Z]*\n", "", sql_query)
            # Remove the closing triple backticks at the end.
            clean_sql = re.sub(r"\n```$", "", clean_sql)
            cursor.execute(clean_sql)
            result = cursor.fetchall()
            #result will be a list of row tuples
        except sqlite3.Error as e:
            #print(f"An error occurred when executing the query on {data['Database']}: {e}")
            return f"An error occurred when executing the query on {data['Database']}: {e}"
        except sqlite3.Warning as w:
            #print(f"A warning occurred when executing the query on {data['Database']}: {w}")
            return f"A warning occurred when executing the query on {data['Database']}: {w}"
        return result