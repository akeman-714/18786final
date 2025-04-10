import os
import openai
import json
import datetime
import random
import re
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def llm(query, history=[], user_stop_words=[]):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for hist in history:
        messages.append({"role": "user", "content": hist[0]})
        messages.append({"role": "assistant", "content": hist[1]})
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stop=user_stop_words
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

os.environ['TAVILY_API_KEY'] = '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'
tavily = TavilySearchResults(max_results=5)
tavily.description = 'This is a search engine similar to Google and Baidu. It can search knowledge, weather, stocks, movies, novels, encyclopedias, etc. If you are not sure about something, you should search for it.'

class FlightSearchInput(BaseModel):
    """Input for flight search."""
    origin: str = Field(description="Departure city or airport code")
    destination: str = Field(description="Arrival city or airport code")
    date: str = Field(description="Departure date in YYYY-MM-DD format")
    return_date: Optional[str] = Field(description="Return date in YYYY-MM-DD format for round trips", default=None)
    adults: int = Field(description="Number of adult passengers", default=1)
    cabin_class: Optional[str] = Field(description="Cabin class (Economy, Premium Economy, Business, First)", default="Economy")

class FlightSearchTool:
    name = "flight_search"
    description = "Search for flights between an origin and destination on a specific date. This tool can search for one-way or round-trip flights with specified cabin class and number of passengers."
    args = {
        "origin": {
            "type": "string",
            "description": "Departure city or airport code"
        },
        "destination": {
            "type": "string",
            "description": "Arrival city or airport code"
        },
        "date": {
            "type": "string",
            "description": "Departure date in YYYY-MM-DD format"
        },
        "return_date": {
            "type": "string",
            "description": "Return date in YYYY-MM-DD format for round trips (optional)"
        },
        "adults": {
            "type": "integer",
            "description": "Number of adult passengers (default: 1)"
        },
        "cabin_class": {
            "type": "string",
            "description": "Cabin class (Economy, Premium Economy, Business, First) (default: Economy)"
        }
    }

    def __init__(self):
        pass

    def invoke(self, input):
        try:
            try:
                input_data = json.loads(input)
            except json.JSONDecodeError:
                input_data = {}
                input = input.strip()
                if input.startswith('{') and input.endswith('}'):
                    input = input[1:-1]
                
                pairs = re.split(r',\s*', input)
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert to appropriate types
                        if key == 'adults':
                            try:
                                value = int(value)
                            except ValueError:
                                value = 1
                        elif key in ['origin', 'destination', 'date', 'return_date', 'cabin_class']:
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                        
                        input_data[key] = value
            
            origin = input_data.get("origin", "")
            destination = input_data.get("destination", "")
            date = input_data.get("date", "")
            return_date = input_data.get("return_date", None)
            adults = input_data.get("adults", 1)
            cabin_class = input_data.get("cabin_class", "Economy")
            
            if not origin or not destination or not date:
                return "Error: Required fields missing. Please provide origin, destination, and date."
            
            flight_results = self._simulate_flight_search(
                origin, destination, date, return_date, adults, cabin_class
            )
            
            return json.dumps(flight_results, ensure_ascii=False)
        except Exception as e:
            return f"Error performing flight search: {str(e)}"
    
    def _simulate_flight_search(self, origin, destination, date, return_date=None, adults=1, cabin_class="Economy"):
        """Simulate flight search API response with realistic data."""
        airlines = ["United Airlines", "American Airlines", "Delta Air Lines", "Southwest Airlines", 
                    "JetBlue", "Alaska Airlines", "Spirit Airlines", "Frontier Airlines"]
        
        distance_factor = {
            ("SFO", "LAX"): (1, 2),
            ("JFK", "LAX"): (5, 6),
            ("ORD", "MIA"): (3, 4),
        }
        
        duration_range = distance_factor.get((origin, destination), 
                         distance_factor.get((destination, origin), (2, 5)))
        
        outbound_flights = []
        for _ in range(random.randint(3, 8)):
            airline = random.choice(airlines)
            flight_number = f"{airline[:2]}{random.randint(100, 9999)}"
            
            departure_hour = random.randint(6, 21)
            departure_minute = random.choice([0, 15, 30, 45])
            departure_time = f"{departure_hour:02}:{departure_minute:02}"
            
            flight_duration_hours = random.randint(duration_range[0], duration_range[1])
            flight_duration_minutes = random.randint(0, 59)
            
            arrival_hour = (departure_hour + flight_duration_hours + (departure_minute + flight_duration_minutes) // 60) % 24
            arrival_minute = (departure_minute + flight_duration_minutes) % 60
            arrival_time = f"{arrival_hour:02}:{arrival_minute:02}"
            
            base_price = random.randint(150, 400)
            price_multipliers = {
                "Economy": 1,
                "Premium Economy": 1.5,
                "Business": 2.5,
                "First": 4
            }
            price = round(base_price * price_multipliers.get(cabin_class, 1) * adults)
            
            flight = {
                "airline": airline,
                "flight_number": flight_number,
                "origin": origin,
                "destination": destination,
                "departure_date": date,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "duration": f"{flight_duration_hours}h {flight_duration_minutes}m",
                "cabin_class": cabin_class,
                "price": price,
                "currency": "USD",
                "seats_available": random.randint(1, 30)
            }
            outbound_flights.append(flight)
        
        outbound_flights = sorted(outbound_flights, key=lambda x: x["price"])
        
        result = {
            "outbound_flights": outbound_flights,
            "search_parameters": {
                "origin": origin,
                "destination": destination,
                "date": date,
                "adults": adults,
                "cabin_class": cabin_class
            }
        }
        
        if return_date:
            return_flights = []
            for _ in range(random.randint(3, 8)):
                airline = random.choice(airlines)
                flight_number = f"{airline[:2]}{random.randint(100, 9999)}"
                
                departure_hour = random.randint(6, 21)
                departure_minute = random.choice([0, 15, 30, 45])
                departure_time = f"{departure_hour:02}:{departure_minute:02}"
                
                flight_duration_hours = random.randint(duration_range[0], duration_range[1])
                flight_duration_minutes = random.randint(0, 59)
                
                arrival_hour = (departure_hour + flight_duration_hours + (departure_minute + flight_duration_minutes) // 60) % 24
                arrival_minute = (departure_minute + flight_duration_minutes) % 60
                arrival_time = f"{arrival_hour:02}:{arrival_minute:02}"
                
                base_price = random.randint(150, 400)
                price_multipliers = {
                    "Economy": 1,
                    "Premium Economy": 1.5,
                    "Business": 2.5,
                    "First": 4
                }
                price = round(base_price * price_multipliers.get(cabin_class, 1) * adults)
                
                flight = {
                    "airline": airline,
                    "flight_number": flight_number,
                    "origin": destination,
                    "destination": origin,
                    "departure_date": return_date,
                    "departure_time": departure_time,
                    "arrival_time": arrival_time,
                    "duration": f"{flight_duration_hours}h {flight_duration_minutes}m",
                    "cabin_class": cabin_class,
                    "price": price,
                    "currency": "USD",
                    "seats_available": random.randint(1, 30)
                }
                return_flights.append(flight)
            
            return_flights = sorted(return_flights, key=lambda x: x["price"])
            result["return_flights"] = return_flights
            result["search_parameters"]["return_date"] = return_date
            
            for outbound in result["outbound_flights"]:
                outbound["round_trip_from"] = outbound["price"] + return_flights[0]["price"]
        
        return result


flight_search = FlightSearchTool()

tools = [tavily, flight_search]
tool_names = 'or'.join([tool.name for tool in tools])
tool_descs = []
for t in tools:
    args_desc = []
    for name, info in t.args.items():
        args_desc.append({
            'name': name,
            'description': info.get('description', ''),
            'type': info['type']
        })
    args_desc = json.dumps(args_desc, ensure_ascii=False)
    tool_descs.append(f"{t.name}: {t.description}, args: {args_desc}")
tool_descs = '\n'.join(tool_descs)

prompt_tpl = '''Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

These are chat history before:
{chat_history}

IMPORTANT: You MUST use the exact format including "Thought: I now know the final answer" followed by "Final Answer: ..."

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action - use valid JSON format with quotes around string values, e.g., {{"key": "value", "number": 42}}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{agent_scratchpad}
'''

def agent_execute(query, chat_history=[]):
    global tools, tool_names, tool_descs, prompt_tpl, llm

    agent_scratchpad = ''
    while True:
        history = '\n'.join([f"Question:{h[0]}\nAnswer:{h[1]}" for h in chat_history])
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        prompt = prompt_tpl.format(
            today=today,
            chat_history=history,
            tool_descs=tool_descs,
            tool_names=tool_names,
            query=query,
            agent_scratchpad=agent_scratchpad
        )

        print('\033[32m---Waiting for LLM response... ...\n%s\n\033[0m' % prompt, flush=True)
        response = llm(prompt, user_stop_words=['Observation:'])
        print('\033[34m---LLM Response---\n%s\n---\033[34m' % response, flush=True)

        thought_i = response.rfind('Thought:')
        final_answer_i = response.rfind('\nFinal Answer:')
        action_i = response.rfind('\nAction:')
        action_input_i = response.rfind('\nAction Input:')
        observation_i = response.rfind('\nObservation:')

        if final_answer_i != -1 and thought_i < final_answer_i:
            final_answer = response[final_answer_i + len('\nFinal Answer:'):].strip()
            chat_history.append((query, final_answer))
            return True, final_answer, chat_history

        if not (thought_i < action_i < action_input_i):
            return False, 'Abnormal LLM response format', chat_history
        if observation_i == -1:
            observation_i = len(response)
            response = response + 'Observation: '

        thought = response[thought_i + len('Thought:'):action_i].strip()
        action = response[action_i + len('\nAction:'):action_input_i].strip()
        action_input = response[action_input_i + len('\nAction Input:'):observation_i].strip()

        the_tool = next((t for t in tools if t.name == action), None)
        if the_tool is None:
            observation = 'the tool does not exist'
            agent_scratchpad += response + observation + '\n'
            continue

        try:
            tool_ret = the_tool.invoke(input=action_input)
        except Exception as e:
            observation = f'the tool has error: {e}'
        else:
            observation = str(tool_ret)

        agent_scratchpad += response + observation + '\n'

def agent_execute_with_retry(query, chat_history=[], retry_times=3):
    for i in range(retry_times):
        success, result, chat_history = agent_execute(query, chat_history=chat_history)
        if success:
            return success, result, chat_history
    return success, result, chat_history

my_history = []
while True:
    query = input('query: ')
    success, result, my_history = agent_execute_with_retry(query, chat_history=my_history)
    my_history = my_history[-10:]