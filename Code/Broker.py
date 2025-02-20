import Agent
import DatabaseConnector
from LLMprompting.test import Tester
from my_openai_utils import openai_execute
from Utils import construct_request_dummy
from LLMprompting.requestHandling import construct_request_dummy, prepare_for_ollama
import json
import os
import DatabaseConnector
with open ("/home/timo/ExpertNetwork/environmentVariables.env") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()

# TODO Slower breaking of the question. Start with a first question. give answer. Start with second question. ...

class Broker:
    def __init__(self, num_agents, prompt_file, max_queries=20):
        self.agents = []
        self.unhandled_messages = []
        self.total_queries = 0
        self.model = "gpt-4o-mini-2024-07-18"
        #self.model = "deepseek-r1:70b"
        #self.model = "o3-mini-2025-01-31"
        self.question_answers = {}
        self.prompt_file = prompt_file
        pass

    def load_prompt(self, prompt_file):
        with open(prompt_file, 'r') as file:
            return file.read()

    def select_agents(self, question):
        broker_selection_system_prompt = self.load_prompt(self.prompt_file)
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