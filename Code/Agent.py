import Broker
import DatabaseConnector
from my_openai_utils import openai_execute
from Utils import construct_request_dummy
import json
import os
with open ("/home/timo/ExpertNetwork/environmentVariables.env") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()

# TODOs:
# TODO Self-fixing loop of prompts -> Matching to question then necessary.
# TODO Restating and resubmission of question into prompt.


class Agent:
    def __init__(self, agent_id, db_path):
        self.agent_id = agent_id
        self.db_path = db_path
        self.db_connector = DatabaseConnector.DatabaseConnector()
        self.schema_information = {}
        self.schema_information_nl = "Schema information not set yet in NL"
        self.model = "gpt-4o-mini-2024-07-18"
        self.unhandled_questions = []
        self.entry = {
        "Database": "/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite",
        "SQL": "SELECT T4.department_name FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id INNER JOIN department AS T4 ON T2.department_id = T4.department_id WHERE T3.person_name = 'Marcia Ross' AND T1.title = 'Reign of Fire'",
        "join_count": 3,
        "question": "For the movie \"Reign of Fire\", which department was Marcia Ross in?",
        "evidence": "movie \"Reign of Fire\" refers to title = 'Reign of Fire'; which department refers to department_name",
        "RelatedTables": {
            "country": [
                "country_id",
                "country_iso_code",
                "country_name"
            ],
            "department": [
                "department_id",
                "department_name"
            ],
            "gender": [
                "gender_id",
                "gender"
            ],
            "genre": [
                "genre_id",
                "genre_name"
            ],
            "keyword": [
                "keyword_id",
                "keyword_name"
            ],
            "language": [
                "language_id",
                "language_code",
                "language_name"
            ],
            "language_role": [
                "role_id",
                "language_role"
            ],
            "movie": [
                "movie_id",
                "title",
                "budget",
                "homepage",
                "overview",
                "popularity",
                "release_date",
                "revenue",
                "runtime",
                "movie_status",
                "tagline",
                "vote_average",
                "vote_count"
            ],
            "movie_genres": [
                "movie_id",
                "genre_id"
            ],
            "movie_languages": [
                "movie_id",
                "language_id",
                "language_role_id"
            ],
            "person": [
                "person_id",
                "person_name"
            ],
            "movie_crew": [
                "movie_id",
                "person_id",
                "department_id",
                "job"
            ],
            "production_company": [
                "company_id",
                "company_name"
            ],
            "production_country": [
                "movie_id",
                "country_id"
            ],
            "movie_cast": [
                "movie_id",
                "person_id",
                "character_name",
                "gender_id",
                "cast_order"
            ],
            "movie_keywords": [
                "movie_id",
                "keyword_id"
            ],
            "movie_company": [
                "movie_id",
                "company_id"
            ]
        },
        "Result": [],
        "subquestions": [],
        "subSQLs": [],
        "subResults": [],
        "finalGeneratedQuery": [],
        "oneTimeQuery": [
            "SELECT department.department_name \nFROM movie_crew \nJOIN department ON movie_crew.department_id = department.department_id \nJOIN movie ON movie_crew.movie_id = movie.movie_id \nJOIN person ON movie_crew.person_id = person.person_id \nWHERE movie.title = 'Reign of Fire' AND person.person_name = 'Marcia Ross';"
        ],
        "entry_counter": 0,
        "ForeignKeys": [
            "# movie_genres.movie_id = movie.movie_id",
            "# movie_genres.genre_id = genre.genre_id",
            "# movie_languages.language_role_id = language_role.role_id",
            "# movie_languages.movie_id = movie.movie_id",
            "# movie_languages.language_id = language.language_id",
            "# movie_crew.person_id = person.person_id",
            "# movie_crew.movie_id = movie.movie_id",
            "# movie_crew.department_id = department.department_id",
            "# production_country.movie_id = movie.movie_id",
            "# production_country.country_id = country.country_id",
            "# movie_cast.person_id = person.person_id",
            "# movie_cast.movie_id = movie.movie_id",
            "# movie_cast.gender_id = gender.gender_id",
            "# movie_keywords.keyword_id = keyword.None",
            "# movie_keywords.movie_id = movie.None",
            "# movie_company.company_id = production_company.None",
            "# movie_company.movie_id = movie.None"
        ]
        }
        self.small_entry = {
        "Database": "/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite",
        "SQL": "SELECT T4.department_name FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id INNER JOIN department AS T4 ON T2.department_id = T4.department_id WHERE T3.person_name = 'Marcia Ross' AND T1.title = 'Reign of Fire'",
        "join_count": 3,
        "question": "For the movie \"Reign of Fire\", which department was Marcia Ross in?",
        "evidence": "movie \"Reign of Fire\" refers to title = 'Reign of Fire'; which department refers to department_name",
        "RelatedTables": {
            "country": [
                "country_id",
                "country_iso_code",
                "country_name"
            ],
            "department": [
                "department_id",
                "department_name"
            ],
            "keyword": [
                "keyword_id",
                "keyword_name"
            ],
            "movie": [
                "movie_id",
                "title",
                "budget",
                "homepage",
                "overview",
                "popularity",
                "release_date",
                "revenue",
                "runtime",
                "movie_status",
                "tagline",
                "vote_average",
                "vote_count"
            ],
            "movie_genres": [
                "movie_id",
                "genre_id"
            ],
            "person": [
                "person_id",
                "person_name"
            ],
            "movie_crew": [
                "movie_id",
                "person_id",
                "department_id",
                "job"
            ],
            "production_company": [
                "company_id",
                "company_name"
            ],
            "production_country": [
                "movie_id",
                "country_id"
            ],
            "movie_company": [
                "movie_id",
                "company_id"
            ]
        },
        "entry_counter": 0
        }
        self.entry = self.small_entry

    def init_schema_information(self, manual_test=True, include_fks=False):
        # This should now connect to the database if it is not manually set
        if manual_test == True:
            # I assume that the schema information has already been set and the NL needs to be created now
            nl_schema_prompt_system = """
            # You are an intelligent assistant designed to convert complex Database schema information into a natural lanugage description.
            # You need to summarize the given schema information accurately to distinguish yourself from other database that might have a similar schema but not the same.
            # Clearly describe what you cover and what you do not cover. For this you need to take a close look at the columns.
            # Summarize your schema in a maximum of five sentences.
            """
            nl_schema_prompt = """
            ### Convert the following database schema into natural language:
            """
            nl_schema_prompt = "### SQLite SQL tables, with their properties: " + "\n "
            nl_schema_prompt += "# \n"
            for related_table_name, columns in self.schema_information.items():
                nl_schema_prompt += "# " + related_table_name + "("
                nl_schema_prompt += ", ".join(columns) + ")\n"

            if include_fks == True:
                nl_schema_prompt += "### Foreign Keys: \n"
                for fk in self.schema_information["ForeignKeys"]:
                    nl_schema_prompt += "".join(fk) + "\n"
            
            nl_schema_prompt += """# Answer in this format: {"S_I": "Insert the natural language here"}"""

        else:
            # TODO connect to the db, get the schema information and then generate the nl
            pass

        request_dummy = construct_request_dummy(self.model, nl_schema_prompt_system, nl_schema_prompt)

        llm_answer = openai_execute(request_dummy, force=0.75)
        llm_response_text = llm_answer[0]['choices'][0]['message']['content']
        try:
            json_response = json.loads(llm_response_text)
        except json.JSONDecodeError as e:
            # TODO
            # Here a reprompting is necessary
            print("Error in nl_schema init")
        if "S_I" not in json_response:
            print("S_I not in response")
            # TODO
            # Here a reprompting is necessary
        else:
            self.schema_information_nl = json_response["S_I"]
    
    def answer_questions(self):
        if not self.unhandled_questions:
            return
        answers = []
        api_requests = []
        for question in self.unhandled_questions:
            C3_sys_prompt = """
            ### You are now an excellent SQL writer. You do not make mistakes. 
            ### You format your responses with ```sql SELECT ... ```.
            ### Alyways use "" around table and column names as well as a clear distinction what column of what table is referenced. Example: SELECT "nameTable"."name" FROM "nameTable";
            """
            #Start constructing the related table info prompt
            related_table_info = "### Complete sqlite SQL query only and with no explanation\n"
            related_table_info += " \n#\n"

            #Iterate over related tables and their columns
            for related_table_name, columns in self.schema_information.items():
                related_table_info += "# " + related_table_name + " ("
                related_table_info += ", ".join(columns) + ");\n"

            #Add the SQL query question
            related_table_info += "#\n### " + question + "\n# SELECT "
            request_dummy = construct_request_dummy(self.model, C3_sys_prompt, related_table_info)    
            api_requests.append(request_dummy)

        if len(api_requests) == 1:
            api_requests = api_requests[0] # This is because openai_execute assumes more than one query is it is []

        answers = openai_execute(api_requests, force=0.75)
        results = []
        for answer in answers:
            llm_sql = answer['choices'][0]['message']['content']
            print(llm_sql)
            result = self.db_connector.runSQLQuery(llm_sql, self.entry)
            print(f"Result from Agent{self.agent_id} : {result}")
            if isinstance(result, str) and result.startswith("An error"):
                self.fix_error_results(llm_sql,result, 2)
            results.append(result)
        #self.unhandled_questions = []

        return self.unhandled_questions, results  
    
    def fix_error_results(self, sql_query, result="", max_retries=2):
        updated_res = result
        updated_sql = sql_query
        tries = 1
        fixer_system_prompt = """
            ### You are now an excellent SQL writer. You do not make mistakes. Complete the SQL Lite query without explanation.
            ### You format your responses with ```sql SELECT ... ```.
            ### Alyways use "" around table and column names as well as a clear distinction what column of what table is referenced. Example: SELECT "nameTable"."name" FROM "nameTable";
            """
        while isinstance(updated_res, str) and updated_res.startswith("An error") and tries <= max_retries:
            print(f"Error fixxing No. {tries}")
            fixer_message = f"# The SQL query: {updated_sql} \n"
            fixer_message += f"# Returned the Error: \n"
            fixer_message += f"{result} \n"
            fixer_message += "# Fix the query and return the new query. \n"
            fixer_message += "This is the Database Schema:"
            for related_table_name, columns in self.schema_information.items():
                fixer_message += "# " + related_table_name + "("
                fixer_message += ", ".join(columns) + ")\n"
            #fixer_message += "# SELECT "
            request_dummy = construct_request_dummy(self.model, fixer_system_prompt, fixer_message)
            llm_answer = openai_execute(request_dummy, force=0.75)
            llm_response_text = llm_answer[0]['choices'][0]['message']['content']
            print("fixed Query: ", llm_response_text)
            updated_sql = llm_response_text
            updated_res = self.db_connector.runSQLQuery(updated_sql, self.entry)
            tries += 1



    
            
        
        
