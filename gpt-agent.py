import os
import openai
import json
import datetime
import re
import requests
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Load environment variables from .env file
load_dotenv()

# Set API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
SKYSCANNER_API_KEY = os.environ.get("SKYSCANNER_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def llm(query, history=[], user_stop_words=[]):
    """LLM function - Use OpenAI API"""
    messages = [{"role": "system", "content": "You are a helpful assistant specialized in flight booking assistance. Help users find the best flights based on their needs."}]
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

# Set up Tavily search tool
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
tavily = TavilySearchResults(max_results=5)
tavily.description = 'This is a search engine for general information. Use it to search for travel guides, visa requirements, travel restrictions, and other travel-related information.'

class SkyscannerFlightSearchTool:
    """Skyscanner Flight Search Tool"""
    name = "flight_search"
    description = "Search for flights between an origin and destination on a specific date using Skyscanner API. This tool can search for one-way or round-trip flights with specified cabin class and number of passengers."
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
        # Set Skyscanner API key from environment variable
        self.api_key = SKYSCANNER_API_KEY
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "skyscanner89.p.rapidapi.com"
        }
        # Cache for location IDs to avoid repeated API calls
        self.location_id_cache = {}

    def invoke(self, input):
        """Call flight search tool"""
        try:
            # Try to parse input
            try:
                input_data = json.loads(input)
            except json.JSONDecodeError:
                # If JSON parsing fails, try parsing key=value pairs
                input_data = {}
                # Remove braces (if present)
                input = input.strip()
                if input.startswith('{') and input.endswith('}'):
                    input = input[1:-1]
                
                # Parse key=value pairs
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
                            # Remove quotes (if present)
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
            
            # Validate required fields
            if not origin or not destination or not date:
                return "Error: Required fields missing. Please provide origin, destination, and date."
            
            # Validate date format and ensure it's in the future
            try:
                date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                today = datetime.datetime.now().date()
                if date_obj < today:
                    return f"Error: The requested departure date {date} is in the past. Please choose a future date."
            except ValueError:
                return f"Error: Invalid date format '{date}'. Please use YYYY-MM-DD format."
                
            # Validate return date if provided
            if return_date:
                try:
                    return_date_obj = datetime.datetime.strptime(return_date, "%Y-%m-%d").date()
                    if return_date_obj < today:
                        return f"Error: The requested return date {return_date} is in the past. Please choose a future date."
                    if return_date_obj < date_obj:
                        return f"Error: The return date {return_date} cannot be before the departure date {date}."
                except ValueError:
                    return f"Error: Invalid return date format '{return_date}'. Please use YYYY-MM-DD format."
            
            # Use Skyscanner API to search for flights
            return self._search_skyscanner(origin, destination, date, return_date, adults, cabin_class)
            
        except Exception as e:
            return f"Error performing flight search: {str(e)}"
    
    def _search_skyscanner(self, origin, destination, date, return_date=None, adults=1, cabin_class="Economy"):
        """Use Skyscanner API to search for flights"""
        try:
            # Convert cabin class to Skyscanner format
            cabin_map = {
                "Economy": "economy",
                "Premium Economy": "premium_economy",
                "Business": "business",
                "First": "business"  # API does not have First class option, using business instead
            }
            skyscanner_cabin = cabin_map.get(cabin_class, "economy")
            
            # Get location IDs for origin and destination
            origin_id = self._get_location_id(origin)
            destination_id = self._get_location_id(destination)
            
            # Use exact format from API documentation
            url = "https://skyscanner89.p.rapidapi.com/flights/one-way/list"
            
            # Format query parameters according to new format
            querystring = {
                "date": date,
                "origin": origin,
                "originId": origin_id,
                "destination": destination,
                "destinationId": destination_id,
                "cabinClass": skyscanner_cabin,
                "adults": str(adults),
                "children": "-7"  # This is required by the API
            }
            
            # Make API request
            print(f"Making API request with parameters: {querystring}")
            response = requests.get(url, headers=self.headers, params=querystring)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") != "success":
                    error_msg = data.get("errors") or "Unknown API error"
                    return f"Skyscanner API returned errors: {error_msg}"
                
                return self._format_skyscanner_response(data, origin, destination, date, return_date, adults, cabin_class)
            else:
                error_text = response.text
                return f"Skyscanner API error: Status code {response.status_code}, Response: {error_text}"
                
        except Exception as e:
            return f"Skyscanner API call failed: {str(e)}"
    
    def _get_location_id(self, location_code):
        """Get location ID for airport code"""
        # Check cache first
        if location_code in self.location_id_cache:
            return self.location_id_cache[location_code]
            
        # Default location IDs for common airports
        default_ids = {
            "NYCA": "27537542",  # New York City area
            "JFK": "27545983",   # JFK Airport
            "LGA": "27544957",   # LaGuardia
            "EWR": "27545306",   # Newark
            "HNL": "95673827",   # Honolulu
            "LAX": "27538634",   # Los Angeles
            "SFO": "27537542",   # San Francisco
            "ORD": "95673827",   # Chicago
            "LHR": "29475375",   # London
            "CDG": "27539733"    # Paris
        }
        
        # If the location code is directly in our defaults, use that
        if location_code in default_ids:
            entity_id = default_ids[location_code]
            self.location_id_cache[location_code] = entity_id
            print(f"Using entity ID for {location_code}: {entity_id}")
            return entity_id
            
        try:
            # Call the auto-complete API to get the correct location ID
            url = "https://skyscanner89.p.rapidapi.com/flights/auto-complete"
            querystring = {"query": location_code}
            
            print(f"Looking up location ID for {location_code}")
            response = requests.get(url, headers=self.headers, params=querystring)
            
            if response.status_code == 200:
                data = response.json()
                # Search for exact match first
                for item in data.get("data", []):
                    if item.get("iata") == location_code:
                        entity_id = item.get("entityId")
                        # Cache the result
                        self.location_id_cache[location_code] = entity_id
                        return entity_id
                
                # If no exact match, use the first result
                if data.get("data") and len(data["data"]) > 0:
                    entity_id = data["data"][0].get("entityId")
                    # Cache the result
                    self.location_id_cache[location_code] = entity_id
                    return entity_id
            
            # Fallback to default mapping
            entity_id = default_ids.get(location_code, "27537542")  # Default to NYC if not found
            self.location_id_cache[location_code] = entity_id
            return entity_id
            
        except Exception as e:
            print(f"Error getting location ID: {str(e)}")
            # Fallback to default mapping
            entity_id = default_ids.get(location_code, "27537542")
            return entity_id
    
    def _format_skyscanner_response(self, data, origin, destination, date, return_date, adults, cabin_class):
        """Format Skyscanner API response into a consistent format"""
        try:
            # Extract flight data from the new structure
            flight_data = data.get("data", {}).get("itineraries", {})
            
            # Get items from the "Best" bucket (most relevant flights)
            best_bucket = None
            for bucket in flight_data.get("buckets", []):
                if bucket.get("id") == "Best":
                    best_bucket = bucket
                    break
                    
            if not best_bucket:
                # If no "Best" bucket, try "Cheapest"
                for bucket in flight_data.get("buckets", []):
                    if bucket.get("id") == "Cheapest":
                        best_bucket = bucket
                        break
            
            if not best_bucket or not best_bucket.get("items"):
                return "No flights found for your search criteria."
            
            # Format outbound flights
            outbound_flights = []
            for item in best_bucket.get("items", []):
                # Get the leg information
                if not item.get("legs"):
                    continue
                
                leg = item["legs"][0]
                segments = leg.get("segments", [])
                if not segments:
                    continue
                
                # Get carrier information
                carrier_info = None
                if "carriers" in leg and "marketing" in leg["carriers"] and leg["carriers"]["marketing"]:
                    carrier_info = leg["carriers"]["marketing"][0]
                
                # Format the flight
                flight = {
                    "airline": carrier_info.get("name", "Unknown Airline") if carrier_info else "Unknown Airline",
                    "flight_number": segments[0].get("flightNumber", "") if segments else "",
                    "origin": leg.get("origin", {}).get("displayCode", origin),
                    "destination": leg.get("destination", {}).get("displayCode", destination),
                    "departure_date": date,
                    "departure_time": segments[0].get("departure", "").split("T")[1][:5] if segments and "T" in segments[0].get("departure", "") else "",
                    "arrival_time": segments[-1].get("arrival", "").split("T")[1][:5] if segments and "T" in segments[-1].get("arrival", "") else "",
                    "duration": f"{leg.get('durationInMinutes', 0) // 60}h {leg.get('durationInMinutes', 0) % 60}m",
                    "cabin_class": cabin_class,
                    "price": item.get("price", {}).get("raw", 0),
                    "formatted_price": item.get("price", {}).get("formatted", "$0"),
                    "currency": "USD",
                    "stops": leg.get("stopCount", 0)
                }
                
                outbound_flights.append(flight)
            
            # Sort by price
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
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            print(f"Error formatting response: {str(e)}")
            return f"Error formatting Skyscanner response: {str(e)}"

# Natural language date parsing
def parse_natural_date(date_str, current_date=None):
    """Parse natural language date expressions into YYYY-MM-DD format"""
    if current_date is None:
        current_date = datetime.datetime.now()
    
    # If date_str is None, return None to avoid errors
    if date_str is None:
        return None
    
    # Convert to lowercase for easier matching
    if isinstance(date_str, str):
        date_str = date_str.lower()
    else:
        return None
    
    # Handle explicit date format first
    if isinstance(date_str, str) and re.match(r'\d{4}-\d{2}-\d{2}', date_str):
        return date_str
    
    # Special handling for "next Friday"
    if "next friday" in date_str:
        # Calculate days until next Friday (weekday 4)
        current_weekday = current_date.weekday()  # 0 is Monday, 4 is Friday
        days_until_friday = (4 - current_weekday) % 7
        
        # If today is Friday or if "next" is specified, add 7 days
        if days_until_friday == 0 or "next" in date_str:
            days_until_friday += 7
            
        next_friday = current_date + datetime.timedelta(days=days_until_friday)
        return next_friday.strftime("%Y-%m-%d")
    
    # Handle common date formats
    if "today" in date_str:
        return current_date.strftime("%Y-%m-%d")
    elif "tomorrow" in date_str:
        return (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    elif "next week" in date_str:
        return (current_date + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Handle days of the week (e.g., "next Monday")
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in date_str:
            # Calculate days until the next occurrence of this day
            target_weekday = i
            current_weekday = current_date.weekday()
            days_ahead = (target_weekday - current_weekday) % 7
            
            # If it's the same day of the week or "next" is specified, add 7 days
            if (days_ahead == 0 or "next" in date_str):
                days_ahead += 7
                
            target_date = current_date + datetime.timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d")
    
    # Handle specific date formats like "April 12, 2025" or "12 April 2025"
    try:
        # Try to parse with various formats
        for fmt in ["%B %d, %Y", "%d %B %Y", "%B %d %Y", "%d %B, %Y"]:
            try:
                parsed_date = datetime.datetime.strptime(date_str, fmt)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        # Handle formats like "April 12" or "12 April" (assume current year)
        months = ["january", "february", "march", "april", "may", "june", 
                  "july", "august", "september", "october", "november", "december"]
        
        for i, month in enumerate(months, 1):
            if month in date_str:
                # Find the day number
                day_match = re.search(r'\d+', date_str)
                if day_match:
                    day = int(day_match.group())
                    # Assume current year
                    year = current_date.year
                    result_date = f"{year}-{i:02d}-{day:02d}"
                    
                    # Check if the date is in the past, if so, use next year
                    parsed_date = datetime.datetime.strptime(result_date, "%Y-%m-%d").date()
                    if parsed_date < current_date.date():
                        result_date = f"{year+1}-{i:02d}-{day:02d}"
                    
                    return result_date
    except Exception as e:
        print(f"Error parsing specific date format: {e}")
    
    # Return None if all parsing attempts fail
    return None

# Extract flight parameters from natural language query
def extract_flight_parameters(query):
    """Extract flight search parameters from natural language query"""
    if query is None:
        return {}  # Return empty dict if query is None
        
    current_date = datetime.datetime.now()
    
    try:
        # Use LLM to extract parameters from natural language
        prompt = f"""You are a flight search assistant. Extract the following information from this query:
        - Origin (airport code if possible)
        - Destination (airport code if possible) 
        - Departure date (keep as natural language, do NOT convert to a specific date)
        - Return date if round trip (keep as natural language, do NOT convert to a specific date)
        - Number of passengers
        - Cabin class (Economy, Premium Economy, Business, First)
        
        TODAY's DATE IS {current_date.strftime('%Y-%m-%d')}.
        
        Query: {query}
        
        Output as JSON:"""
        
        response = llm(prompt)
        
        # Try to parse JSON response
        try:
            # Look for JSON object in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                params = json.loads(json_str)
            else:
                # Fall back to simpler parsing
                params = {}
                for line in response.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        params[key.strip().lower()] = value.strip()
        except Exception as json_error:
            print(f"JSON parsing error: {json_error}")
            # If JSON parsing fails, use regex to extract key parameters
            params = {}
            
            # Airport codes are usually 3 uppercase letters
            origin_match = re.search(r'\b([A-Z]{3})\b.*to', query, re.IGNORECASE)
            if origin_match:
                params['origin'] = origin_match.group(1).upper()
            else:
                # Try to find common city names as origin
                cities = ["New York", "Los Angeles", "Chicago", "San Francisco", "Miami", "Seattle"]
                for city in cities:
                    if city.lower() in query.lower() and "to" in query.lower().split(city.lower())[1]:
                        params['origin'] = city
                        break
            
            dest_match = re.search(r'to\s+\b([A-Z]{3})\b', query, re.IGNORECASE)
            if dest_match:
                params['destination'] = dest_match.group(1).upper()
            else:
                # Try to find common city names as destination
                cities = ["New York", "Los Angeles", "Chicago", "San Francisco", "Miami", "Seattle"]
                for city in cities:
                    if city.lower() in query.lower() and "to" in query.lower().split(city.lower())[0]:
                        params['destination'] = city
                        break
            
            # Check cabin class
            cabin_classes = ['economy', 'premium economy', 'business', 'first']
            for cabin in cabin_classes:
                if cabin.lower() in query.lower():
                    params['cabin_class'] = cabin.title()
                    break
            
            # Check number of adults
            adults_match = re.search(r'(\d+)\s+adult', query, re.IGNORECASE)
            if adults_match:
                params['adults'] = int(adults_match.group(1))
            
            # Check for date keywords
            date_keywords = ["today", "tomorrow", "next week", "monday", "tuesday", "wednesday", 
                             "thursday", "friday", "saturday", "sunday"]
            
            # Store raw date mentions for later processing
            for keyword in date_keywords:
                if keyword.lower() in query.lower():
                    params['date'] = query  # Will be parsed with parse_natural_date later
                    break
        
        # Clean and validate parameters
        result = {}
        
        # Handle origin/destination
        if 'origin' in params and params['origin'] is not None:
            result['origin'] = params['origin'].upper() if len(params['origin']) == 3 else params['origin']
        if 'destination' in params and params['destination'] is not None:
            result['destination'] = params['destination'].upper() if len(params['destination']) == 3 else params['destination']
        
        # Special handling for "next Friday" or similar date expressions
        if 'next friday' in query.lower():
            date_text = "next friday"
        else:
            # Get raw date text for parsing
            date_text = params.get('date') or params.get('departure_date')
            
            # Special handling for specific date format like "April 12, 2025"
            date_match = re.search(r'([A-Za-z]+)\s+(\d{1,2})(?:\s*,\s*|\s+)(\d{4})', query)
            if date_match:
                month, day, year = date_match.groups()
                date_text = f"{month} {day}, {year}"
        
        # Parse departure date
        if date_text:
            parsed_date = parse_natural_date(date_text, current_date)
            if parsed_date:
                result['date'] = parsed_date
        
        # Parse return date if any
        return_text = params.get('return_date')
        if return_text:
            parsed_return = parse_natural_date(return_text, current_date)
            if parsed_return:
                result['return_date'] = parsed_return
        
        # Handle passenger count and cabin class
        if 'adults' in params and params['adults'] is not None:
            try:
                result['adults'] = int(params['adults'])
            except:
                result['adults'] = 1
                
        if 'cabin_class' in params and params['cabin_class'] is not None:
            valid_classes = ['Economy', 'Premium Economy', 'Business', 'First']
            cabin = params['cabin_class'].strip().title()
            if cabin in valid_classes:
                result['cabin_class'] = cabin
            elif 'business' in cabin.lower():
                result['cabin_class'] = 'Business'
            elif 'first' in cabin.lower():
                result['cabin_class'] = 'First'
            elif 'premium' in cabin.lower():
                result['cabin_class'] = 'Premium Economy'
            else:
                result['cabin_class'] = 'Economy'
        
        return result
    except Exception as e:
        print(f"Error extracting flight parameters: {str(e)}")
        return {}

# Create tools
flight_search = SkyscannerFlightSearchTool()
tools = [tavily, flight_search]
tool_names = '|'.join([tool.name for tool in tools])
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

# Prompt template
prompt_tpl = '''Today is {today}. Please help the user with their flight search request. You have access to the following tools:

{tool_descs}

These are chat history before:
{chat_history}

IMPORTANT: You MUST use the exact format including "Thought: I now know the final answer" followed by "Final Answer: ..."

Use the following format:

Question: the input question you must answer
Thought: Think step by step to understand what information the user is looking for. For flight searches, carefully identify:
  1. Origin and destination airports/cities
  2. Travel dates (departure and return if applicable)
  3. Number of passengers 
  4. Cabin class preferences
  5. Any other specific requirements (direct flights, price range, etc.)
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action - use valid JSON format with quotes around string values, e.g., {{"key": "value", "number": 42}}. Do NOT use markdown code blocks (```).
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: Present the flight information in a clear, organized manner. For each flight option, include:
- Airline and flight number
- Departure and arrival times
- Duration
- Cabin class
- Price
- Number of available seats (if known)

For the user, explain if any information was missing from their query and what assumptions you made.

Begin!

Question: {query}
{agent_scratchpad}
'''

# Agent execution logic
def agent_execute(query, chat_history=[]):
    global tools, tool_names, tool_descs, prompt_tpl, llm

    # Preprocess query to extract flight parameters (if it's a flight query)
    flight_keywords = ['flight', 'fly', 'plane', 'airport', 'travel']
    is_flight_query = any(keyword in query.lower() for keyword in flight_keywords)
    
    if is_flight_query:
        # Extract flight parameters to make it easier for the agent to process
        flight_params = extract_flight_parameters(query)
        if flight_params:
            print(f"Extracted flight parameters: {flight_params}")
            # Enhance query with structured data
            query = f"{query}\n\nExtracted flight parameters: {json.dumps(flight_params, indent=2)}"
    
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

        print('\033[32m---Waiting for LLM response...\033[0m', flush=True)
        response = llm(prompt, user_stop_words=['Observation:'])
        print('\033[34m---LLM Response---\n%s\n---\033[34m' % response, flush=True)

        # Parse response content
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
            response = response + '\nObservation: '

        thought = response[thought_i + len('Thought:'):action_i].strip()
        action = response[action_i + len('\nAction:'):action_input_i].strip()
        action_input = response[action_input_i + len('\nAction Input:'):observation_i].strip()

        # Find tool
        the_tool = next((t for t in tools if t.name == action), None)
        if the_tool is None:
            observation = 'The tool does not exist'
            agent_scratchpad += response + observation + '\n'
            continue

        try:
            # Pass action_input as is, let the tool handle parsing
            tool_ret = the_tool.invoke(input=action_input)
        except Exception as e:
            observation = f'The tool encountered an error: {e}'
        else:
            observation = str(tool_ret)

        agent_scratchpad += response + observation + '\n'

# Agent retry function
def agent_execute_with_retry(query, chat_history=[], retry_times=3):
    for i in range(retry_times):
        success, result, chat_history = agent_execute(query, chat_history=chat_history)
        if success:
            return success, result, chat_history
    return success, result, chat_history

# Main function
def main():
    current_date = datetime.datetime.now()
    print(f"🛫 Flight Search AI Agent 🛬")
    print(f"----------------------------")
    print(f"Current system date: {current_date}")
    
    print("Ask me anything about flights, or type 'exit' to quit.")
    print("Example: 'I need to find a business class flight from SFO to ORD next Friday for 2 adults'")
    
    my_history = []
    while True:
        query = input('\n✈️ Query: ')
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using Flight Search AI Agent. Goodbye!")
            break
        else:
            success, result, my_history = agent_execute_with_retry(query, chat_history=my_history)
            my_history = my_history[-10:]  # Only keep last 10 interactions
            print(f"\n{result}")

if __name__ == "__main__":
    main()