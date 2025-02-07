import Agent, DatabaseConnector
from my_openai_utils import openai_execute
from Utils import construct_request_dummy
import json
import os
import DatabaseConnector
with open ("/home/timo/ExpertNetwork/environmentVariables.env") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()

# TODO Slower breaking of the question. Start with a first question. give answer. Start with second question. ...

class Broker:
    def __init__(self, num_agents, max_queries=20):
        self.agents = []
        self.unhandled_messages = []
        self.total_queries = 0
        self.model = "gpt-4o-mini-2024-07-18"
        #self.model = "o3-mini-2025-01-31"
        self.question_answers = {}
        pass

    def select_agents(self, question):
        broker_selection_system_prompt = """
        You are an intelligent assistant tasked with transforming a complex question into multiple targeted subquestions for different agents. Each agent manages its own database with a unique schema that may or may not contain the specific information required to answer the question. Your goal is to extract as much relevant information as possible by decomposing the complex question into smaller, precise subquestions that can be answered by one or more agents.

        Below are descriptions of the agents and a concrete example based on the question:

        **Agent Descriptions:**

        - **Agent 0:**  
        Agent 0’s database is geared toward handling queries related to movie production and genre information. It includes tables for production companies and movie companies, which provide details on which company produced a movie. Additionally, it contains the movie table (covering titles and popularity) as well as movie genres and genre tables. This setup makes Agent 0 ideal for linking production company data with core movie metadata.

        - **Agent 1:**  
        Agent 1 focuses on core movie data and related details such as cast, country of production, and language information. Its database includes the movie table with key fields like title and popularity, along with tables for movie casts, production countries, countries, movie languages, and languages. This makes Agent 1 well-suited for handling general movie metadata and location or language-based queries.

        - **Agent 2:**  
        Agent 2’s database combines essential movie details with additional context regarding production companies and crew-related information. It features tables for production companies and movies (including title and popularity), as well as tables for departments, gender, keywords, language roles, movie crew, and movie keywords. This structure enables Agent 2 to provide insights not only on basic movie data but also on production roles and related details.

        **Complex Question Example:**

        *Question:*  
        "For all the movies which were produced by Cruel and Unusual Films, which one has the most popularity?"

        *Random Assignment of Tables to Agents:*  
        - **Agent 0:** Covers production_company, movie_company, movie, movie_genres, and genre.
        - **Agent 1:** Covers movie, movie_cast, production_country, country, movie_languages, and language.
        - **Agent 2:** Covers production_company, movie, department, gender, keyword, language_role, movie_crew, and movie_keywords.

        **Steps to Process the Question:**

        1. **Analyze the Complex Question:**  
        - Identify the distinct pieces of information needed: movies produced by "Cruel and Unusual Films" and the popularity metric of those movies.

        2. **Decompose the Question into Subquestions:**  
        - Break the complex question into at least 3 distinct and clear subquestions. For example:
            - "List all movie IDs and titles for movies produced by Cruel and Unusual Films by joining the production_company, movie_company, and movie tables."
            - "For the movie IDs obtained, what are the popularity values from the movie table?"
            - "Which movie, among the listed ones, has the highest popularity value?"

        3. **Examine Schema Information:**  
        - Based on the table distributions, determine which agent(s) can answer each subquestion:
            - Agent 0 is best suited for the first subquestion since it covers production companies and movie metadata.
            - Both Agent 0 and Agent 1 can provide popularity values from the movie table, so cross-verification is possible.
            - Agent 1 is ideal for determining which movie has the highest popularity.

        4. **Select Relevant Agents:**  
        - Assign the first subquestion to Agent 0.
        - Assign the second subquestion to both Agent 0 and Agent 1.
        - Assign the third subquestion to Agent 1.

        5. **Formulate Natural Language Subquestions:**  
        - Write the subquestions in clear, natural language for each selected agent.

        6. **Return Your Output:**  
        - Provide a JSON object where each key is an agent ID (numbers starting from 0) and each value is a list of subquestions for that agent.
        - For example:
            {
                "0": [
                    "List all movie IDs and titles for movies produced by Cruel and Unusual Films by joining the production_company, movie_company, and movie tables.",
                    "For the movie IDs obtained, what are the popularity values from the movie table?"
                ],
                "1": [
                    "For the movie IDs obtained, what are the popularity values from the movie table?",
                    "Which movie, among the listed ones, has the highest popularity value?"
                ]
            }

        **Note:** You may ask the same subquestion to multiple agents if it is relevant.

        # Now, process the complex question as instructed using the steps and example above.
        
        # You have to generate at least 3 distinct subquestions, and assign them to the appropriate agents based on their schema.


        """
        broker_selection_prompt = """
        ### Agent Information:
        """
        for agent in self.agents:
            broker_selection_prompt += f"# Agent{agent.agent_id}: {agent.schema_information_nl} \n"
        broker_selection_prompt = """
            ### Return the agent selection and their respective questions for this question:"""
        broker_selection_prompt += f"\n # Question: {question} \n"
        broker_selection_prompt += """### Answer in the specified format: {"agent_id": [question1, question2, ...], "agent_id": [question3, ...]}."""

        request_dummy = construct_request_dummy(self.model, system_prompt=broker_selection_system_prompt, first_message=broker_selection_prompt)
        llm_answer = openai_execute(request_dummy, force=0.75)
        llm_response_text = llm_answer[0]['choices'][0]['message']['content']
        try:
            json_response = json.loads(llm_response_text)
        except json.JSONDecodeError as e:
            # TODO
            # Here a reprompting is necessary
            print("Error in nl_schema init")
        else:
            for agent_id, questions in json_response.items():
                index = int(agent_id)
                if index < len(self.agents):
                    agent = self.agents[index]
                    if not hasattr(agent, "unhandled_questions"):
                        agent.unhandled_questions = []
                    agent.unhandled_questions.extend(questions)
                    print(f"Agent {agent.agent_id} has been assigned the following questions:")
                    for question in agent.unhandled_questions:
                        print(f"  - {question}")
                else:
                    print(f"Warning: No agent exists with id {agent_id}.")
    
    def decide_result(self):
        broker_result_system_prompt = """
        # You are an intelligent assistant designed to answers a given question that has been broken down into subquestions.
        # These subquestions have been answered by database agents that contain their own databases with information that may or may not be related.
        # You will now:
        # 1. Analyze the initial question.
        # 2. Analyze the sub questions that were answered by the agents and their results.
        # 3. Return the final result or NA as not answerable with these subresults.
        # The format for you response needs to be like this: {"FinalResult": "Insert the final result here or NA"}
        """
        network_messages = """### These questions and results have been answered so far: \n"""
        for question, result in self.question_answers.items():
            network_messages += f"# Question: {question} \n"
            network_messages += f"# Result: {result} \n"
        network_messages = """### Return the final answer to the initial question based on these subresults:"""

        request_dummy = construct_request_dummy(self.model, system_prompt=broker_result_system_prompt, first_message=network_messages)

        llm_answer = openai_execute(request_dummy, force=0.75)
        llm_response_text = llm_answer[0]['choices'][0]['message']['content']
        try:
            json_response = json.loads(llm_response_text)
        except json.JSONDecodeError as e:
            # TODO
            # Here a reprompting is necessary
            print("Error in nl_schema init")
        
        return llm_response_text