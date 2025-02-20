import Agent, Broker
import DatabaseConnector
from my_openai_utils import openai_execute
import json
import os
import random
with open ("/home/timo/ExpertNetwork/environmentVariables.env") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()
class Network:
    def __init__(self, num_agents, db_paths, question):
        #if len(db_paths) < num_agents:
            #raise ValueError("Not enough database paths provided for the number of agents.")
        
        self.agents = [Agent.Agent(agent_id=i, db_path=db_paths[i]) for i in range(num_agents)]
        self.broker = Broker.Broker(num_agents, "/home/timo/ExpertNetwork/Code/LLMprompting/brokerPrompt.txt")

        self.json_path = "/home/timo/ExpertNetwork/Data/WorkingJsons/test.json"
        self.entry_counter = 0

        self.network_start(True)
        self.query_network(question)

        

    def network_start(self, manual_test=True):
        # This function starts the network -> The broker gets told the information about the agents etc
        # This is only necessary when i am doing the manual test @alhamza
        self.break_schema(self.json_path, self.entry_counter)
        #
        for agent in self.agents:
            if manual_test == True:
                #self.break_schema(self.json_path, self.entry_counter)
                agent.init_schema_information()
                self.broker.agents.append(agent)
            else:
                # TODO
                pass

            
    def query_network(self, question):
        self.broker.select_agents(question)
        for agent in self.agents:
            if agent.unhandled_questions:
                questions, results = agent.answer_questions()
                for q, r in zip(questions, results):
                    self.broker.question_answers[q] = r
        
        end_result = self.broker.decide_result()
        print("The end result is: ", end_result)

    def break_schema(self, json_file, entry_counter):
        """
        Reads a specific entry from a JSON file and distributes the tables (from the
        "RelatedTables" dictionary) evenly among the agents. Each agent's allocated tables 
        are stored in its self.schema_information attribute.
        
        Parameters:
            json_file (str): Path to the JSON file containing schema entries.
            entry_counter (int): Identifier used to select the specific entry.
            
        Raises:
            ValueError: If no entry with the specified entry_counter is found.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        entry = data
        
        if entry is None:
            raise ValueError(f"No entry with entry_counter {entry_counter} found in the JSON file.")
        
        related_tables = entry.get("RelatedTables", {})
        table_names = list(related_tables.keys())
        random.shuffle(table_names)
        
        # Distribute the tables among agents in a round-robin fashion.
        num_agents = len(self.agents)
        for idx, table in enumerate(table_names):
            agent_index = idx % num_agents
            agent = self.agents[agent_index]
            
            if not hasattr(agent, "schema_information"):
                agent.schema_information = {}
            
            agent.schema_information[table] = related_tables[table]
        
        for agent in self.agents:
            print(f"Agent {agent.agent_id} schema information:")
            print(json.dumps(agent.schema_information, indent=4))

db_paths_8 = ["/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite", "/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite", "/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite","/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite","/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite","/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite","/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite","/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite"]
db_paths_3 = ["/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite", "/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite", "/home/timo/ExpertNetwork/Data/movies_4/movies_4.sqlite"]

n = Network(3, db_paths_3, "For the movie \"Reign of Fire\", which department was Marcia Ross in?")