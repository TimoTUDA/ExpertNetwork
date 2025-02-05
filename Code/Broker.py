import Agent, DatabaseConnector
from my_openai_utils import openai_execute
import json
import os
import DatabaseConnector
with open ("/home/timo/ExpertNetwork/environmentVariables.env") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()
class Broker:
    def __init__(self, num_agents, max_queries=20):
        self.agents = []
        self.unhandled_messages = []
        self.total_queries = 0
        self.model = "gpt-4o-mini-2024-07-18"
        self.question_answers = {}
        pass

    def select_agents(self, question):
        broker_selection_system_prompt = """
            # You are an intelligent assistant designed to convert complex questions into multiple sub-questions for different agents.
            # You do this because each agent has their own database that may or may not contain information to answer the question.
            # You goal is to gain as much information from smaller subqeustions in order to answer the complex question, given the results of those subquestions from the agents.
            # You do it in these steps:
            # 1. You analyze the complex question break it down into 2-4 subquestions that need to be answered in order to answer this question.
            # 2. You analyze the given schema information of the database agents and their possible relevance to the information needs.
            # 3. You select the agents that have the databases that are needed to answer the question.
            # 4. You the subquestions in natural language for each agent that you have selected.
            # 5. You return your selection and the questions in this format: {"agent_id": [question1, question2, ...], "agent_id": [question3, ...]}
            # Note that you can also ask multiple agents the same question. The agent_ids are numbers starting from 0.
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
        
        request_dummy = [{
        "model": self.model,
        "messages": [
            {"role": "system", "content": broker_selection_system_prompt},
            {"role": "user", "content": broker_selection_prompt},
        ],
        "max_tokens": 10000,
        }]
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

        request_dummy = [{
        "model": self.model,
        "messages": [
            {"role": "system", "content": broker_result_system_prompt},
            {"role": "user", "content": network_messages},
        ],
        "max_tokens": 10000,
        }]
        llm_answer = openai_execute(request_dummy, force=0.75)
        llm_response_text = llm_answer[0]['choices'][0]['message']['content']
        try:
            json_response = json.loads(llm_response_text)
        except json.JSONDecodeError as e:
            # TODO
            # Here a reprompting is necessary
            print("Error in nl_schema init")
        
        return llm_response_text