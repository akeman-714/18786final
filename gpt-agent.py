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

# 从.env文件加载环境变量
load_dotenv()

# 获取API密钥
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
SKYSCANNER_API_KEY = os.environ.get("SKYSCANNER_API_KEY")

# 初始化OpenAI客户端
client = OpenAI(api_key=OPENAI_API_KEY)

def llm(query, system_prompt=None, history=[], user_stop_words=[], temperature=0.7):
    """使用OpenAI API的LLM函数"""
    if system_prompt is None:
        system_prompt = "You are a helpful assistant specialized in flight booking assistance. Help users find the best flights based on their needs."
    
    messages = [{"role": "system", "content": system_prompt}]
    for hist in history:
        messages.append({"role": "user", "content": hist[0]})
        messages.append({"role": "assistant", "content": hist[1]})
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            stop=user_stop_words
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

# 设置Tavily搜索工具
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
tavily = TavilySearchResults(max_results=3)
tavily.description = 'This is a search engine for general information. Use it to search for travel guides, visa requirements, travel restrictions, and other travel-related information.'

class SkyscannerAPIClient:
    """Skyscanner API接口封装"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "skyscanner89.p.rapidapi.com"
        }
        # 位置ID缓存
        self.location_cache = {}
        
    def auto_complete(self, query, locale="en-US", market="US", currency="USD"):
        """获取Skyscanner位置建议"""
        cache_key = f"{query}_{locale}_{market}_{currency}"
        
        if cache_key in self.location_cache:
            return self.location_cache[cache_key]
            
        url = "https://skyscanner89.p.rapidapi.com/flights/auto-complete"
        querystring = {
            "query": query,
            "locale": locale,
            "market": market,
            "currency": currency
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            if response.status_code == 200:
                data = response.json()
                self.location_cache[cache_key] = data
                return data
            else:
                return {"error": f"API error: {response.status_code}", "message": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def get_location_details(self, query):
        """从自动完成结果中提取位置详情"""
        data = self.auto_complete(query)
        
        if "error" in data:
            return None
            
        results = data.get("inputSuggest", [])
        if not results:
            return None
            
        # 先找精确匹配(精确的城市/机场代码)
        for result in results:
            flight_params = result.get("navigation", {}).get("relevantFlightParams", {})
            sky_id = flight_params.get("skyId", "")
            if sky_id and sky_id.upper() == query.upper():
                return {
                    "entityId": flight_params.get("entityId"),
                    "skyId": sky_id,
                    "name": flight_params.get("localizedName"),
                    "type": flight_params.get("flightPlaceType")
                }
        
        # 没有精确匹配，取第一个结果
        if results:
            flight_params = results[0].get("navigation", {}).get("relevantFlightParams", {})
            return {
                "entityId": flight_params.get("entityId"),
                "skyId": flight_params.get("skyId"),
                "name": flight_params.get("localizedName"),
                "type": flight_params.get("flightPlaceType")
            }
            
        return None
    
    def search_one_way_flights(self, date, origin, origin_id, destination, destination_id, 
                              cabin_class="economy", adults=1, children=0, infants=0):
        """搜索单程航班"""
        url = "https://skyscanner89.p.rapidapi.com/flights/one-way/list"
        
        # 验证舱位等级
        valid_classes = ["economy", "premium_economy", "business"]
        if cabin_class.lower() not in valid_classes:
            cabin_class = "economy"
        
        # 准备查询参数
        querystring = {
            "date": date,
            "origin": origin,
            "originId": origin_id,
            "destination": destination,
            "destinationId": destination_id,
            "cabinClass": cabin_class.lower(),
            "adults": str(adults),
        }
        
        # 添加可选参数
        if children > 0:
            querystring["children"] = str(children)
        if infants > 0:
            querystring["infants"] = str(infants)
            
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"API error: {response.status_code}", "details": response.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class SkyscannerFlightSearchTool:
    """Skyscanner航班搜索工具"""
    name = "flight_search"
    description = "Search for flights between an origin and destination on a specific date using Skyscanner API. This tool can search for one-way flights with specified cabin class and number of passengers."
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
        "adults": {
            "type": "integer",
            "description": "Number of adult passengers (default: 1)"
        },
        "cabin_class": {
            "type": "string",
            "description": "Cabin class (Economy, Premium Economy, Business) (default: Economy)"
        },
        "children": {
            "type": "integer",
            "description": "Number of children (2-12 years) (default: 0)"
        },
        "infants": {
            "type": "integer",
            "description": "Number of infants (under 2 years) (default: 0)"
        }
    }

    def __init__(self):
        self.api_client = SkyscannerAPIClient(SKYSCANNER_API_KEY)

    def invoke(self, input):
        """调用航班搜索工具"""
        try:
            input_data = self._parse_input(input)
            
            origin = input_data.get("origin", "")
            destination = input_data.get("destination", "")
            date = input_data.get("date", "")
            adults = input_data.get("adults", 1)
            cabin_class = input_data.get("cabin_class", "Economy")
            children = input_data.get("children", 0)
            infants = input_data.get("infants", 0)
            
            # 验证必填字段
            if not origin or not destination or not date:
                return "Error: Required fields missing. Please provide origin, destination, and date."
            
            # 验证日期格式
            try:
                date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                today = datetime.datetime.now().date()
                if date_obj < today:
                    return f"Error: The requested departure date {date} is in the past. Please choose a future date."
            except ValueError:
                return f"Error: Invalid date format '{date}'. Please use YYYY-MM-DD format."
            
            # 获取位置详情
            origin_details = self.api_client.get_location_details(origin)
            if not origin_details:
                return f"Error: Could not find location information for origin '{origin}'."
            
            destination_details = self.api_client.get_location_details(destination)
            if not destination_details:
                return f"Error: Could not find location information for destination '{destination}'."
            
            # 映射舱位等级
            cabin_map = {
                "Economy": "economy",
                "Premium Economy": "premium_economy",
                "Business": "business",
                "First": "business"  # API使用business表示头等舱
            }
            skyscanner_cabin = cabin_map.get(cabin_class, "economy")
            
            # 执行航班搜索
            results = self.api_client.search_one_way_flights(
                date=date,
                origin=origin_details.get("skyId", origin),
                origin_id=origin_details.get("entityId", ""),
                destination=destination_details.get("skyId", destination),
                destination_id=destination_details.get("entityId", ""),
                cabin_class=skyscanner_cabin,
                adults=adults,
                children=children,
                infants=infants
            )
            
            if results.get("status") == "error":
                return f"Error performing flight search: {results.get('message', 'Unknown error')}"
            
            return self._format_flight_results(results, origin, destination, date, adults, cabin_class)
            
        except Exception as e:
            return f"Error performing flight search: {str(e)}"
    
    def _parse_input(self, input):
        """解析工具输入"""
        input_data = {}
        
        try:
            input_data = json.loads(input)
        except json.JSONDecodeError:
            input = input.strip()
            if input.startswith('{') and input.endswith('}'):
                input = input[1:-1]
            
            pairs = re.split(r',\s*', input)
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 移除引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # 转换类型
                    if key in ['adults', 'children', 'infants']:
                        try:
                            value = int(value)
                        except ValueError:
                            if key == 'adults':
                                value = 1
                            else:
                                value = 0
                    
                    input_data[key] = value
        
        return input_data
    
    def _format_flight_results(self, results, origin, destination, date, adults, cabin_class):
        """格式化航班搜索结果"""
        try:
            flight_data = results.get("data", {}).get("itineraries", {})
            
            # 获取最佳或最便宜航班项目
            target_buckets = ["Best", "Cheapest"] 
            selected_bucket = None
            
            for bucket_id in target_buckets:
                for bucket in flight_data.get("buckets", []):
                    if bucket.get("id") == bucket_id and bucket.get("items"):
                        selected_bucket = bucket
                        break
                if selected_bucket:
                    break
            
            if not selected_bucket or not selected_bucket.get("items"):
                return "No flights found for your search criteria."
            
            # 格式化结果
            flights = []
            for item in selected_bucket.get("items", [])[:5]:  # 限制为前5个结果
                if not item.get("legs"):
                    continue
                
                leg = item["legs"][0]
                segments = leg.get("segments", [])
                if not segments:
                    continue
                
                # 获取航空公司信息
                carrier_info = None
                if "carriers" in leg and "marketing" in leg["carriers"] and leg["carriers"]["marketing"]:
                    carrier_info = leg["carriers"]["marketing"][0]
                
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
                    "stops": leg.get("stopCount", 0)
                }
                
                flights.append(flight)
            
            # 按价格排序
            flights = sorted(flights, key=lambda x: x["price"])
            
            result = {
                "flights": flights,
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
            return f"Error formatting flight results: {str(e)}"

# 自然语言日期解析
def parse_natural_date(date_str, current_date=None):
    """将自然语言日期表达式解析为YYYY-MM-DD格式"""
    if current_date is None:
        current_date = datetime.datetime.now()
    
    if date_str is None or not isinstance(date_str, str):
        return None
    
    date_str = date_str.lower()
    
    # 处理显式日期格式
    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
        return date_str
    
    # 特殊日期关键词
    if "today" in date_str:
        return current_date.strftime("%Y-%m-%d")
    elif "tomorrow" in date_str:
        return (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    elif "next week" in date_str:
        return (current_date + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    
    # 处理星期几
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in date_str:
            target_weekday = i
            current_weekday = current_date.weekday()
            days_ahead = (target_weekday - current_weekday) % 7
            
            if (days_ahead == 0 or "next" in date_str):
                days_ahead += 7
                
            target_date = current_date + datetime.timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d")
    
    # 处理带月份名称的特定日期格式
    try:
        # 尝试不同日期格式
        for fmt in ["%B %d, %Y", "%d %B %Y", "%B %d %Y", "%d %B, %Y"]:
            try:
                parsed_date = datetime.datetime.strptime(date_str, fmt)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        # 处理"April 12"这样的格式（假设当前年份）
        months = ["january", "february", "march", "april", "may", "june", 
                  "july", "august", "september", "october", "november", "december"]
        
        for i, month in enumerate(months, 1):
            if month in date_str:
                day_match = re.search(r'\d+', date_str)
                if day_match:
                    day = int(day_match.group())
                    year = current_date.year
                    result_date = f"{year}-{i:02d}-{day:02d}"
                    
                    # 如果日期在过去，使用下一年
                    parsed_date = datetime.datetime.strptime(result_date, "%Y-%m-%d").date()
                    if parsed_date < current_date.date():
                        result_date = f"{year+1}-{i:02d}-{day:02d}"
                    
                    return result_date
    except Exception:
        pass
    
    return None

# 航班参数提取
def extract_flight_parameters(query):
    """从自然语言查询中提取航班搜索参数"""
    if query is None:
        return {}
        
    current_date = datetime.datetime.now()
    
    try:
        # 提取参数的系统提示
        system_prompt = """You are a flight parameter extraction system. Your ONLY task is to extract structured flight search parameters from user queries.
        
        DO NOT make up or invent information not explicitly stated.
        DO NOT include commentary or explanations.
        
        Output ONLY a valid JSON object with these fields if mentioned:
        - origin: Airport code or city name
        - destination: Airport code or city name
        - date: The raw date mention (do not convert to a specific format)
        - adults: Number of adult passengers (default to 1 if not specified)
        - cabin_class: One of "Economy", "Premium Economy", or "Business"
        - children: Number of children aged 2-12 (default to 0 if not specified)
        - infants: Number of infants under 2 (default to 0 if not specified)
        
        If a field is not mentioned in the query, DO NOT include it in the output.
        """
        
        prompt = f"Extract flight parameters from this query. Today's date is {current_date.strftime('%Y-%m-%d')}.\n\nQuery: {query}"
        
        response = llm(prompt, system_prompt=system_prompt, temperature=0.1)
        
        # 从响应中提取JSON
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            params = json.loads(json_match.group(1))
        else:
            # 使用正则表达式提取
            params = {}
            
            # 提取出发地和目的地
            if 'from' in query.lower() and 'to' in query.lower():
                parts = query.lower().split('from')[1].split('to')
                if len(parts) >= 2:
                    origin = parts[0].strip()
                    destination = parts[1].split()[0].strip()
                    params['origin'] = origin
                    params['destination'] = destination
            
            # 提取舱位等级
            cabin_classes = ['economy', 'premium economy', 'business', 'first']
            for cabin in cabin_classes:
                if cabin.lower() in query.lower():
                    params['cabin_class'] = cabin.title()
                    break
            
            # 提取成人数量
            adults_match = re.search(r'(\d+)\s+adult', query, re.IGNORECASE)
            if adults_match:
                params['adults'] = int(adults_match.group(1))
            
            # 检查日期关键词
            date_keywords = ["today", "tomorrow", "next week", "monday", "tuesday", "wednesday", 
                           "thursday", "friday", "saturday", "sunday"]
            for keyword in date_keywords:
                if keyword.lower() in query.lower():
                    params['date'] = keyword
                    break
        
        # 清理和验证参数
        result = {}
        
        # 处理出发地/目的地
        if 'origin' in params and params['origin']:
            result['origin'] = params['origin'].upper() if len(params['origin']) == 3 else params['origin']
        if 'destination' in params and params['destination']:
            result['destination'] = params['destination'].upper() if len(params['destination']) == 3 else params['destination']
        
        # 解析日期
        date_text = params.get('date')
        if date_text:
            parsed_date = parse_natural_date(date_text, current_date)
            if parsed_date:
                result['date'] = parsed_date
        
        # 处理乘客数量
        for field in ['adults', 'children', 'infants']:
            if field in params and params[field] is not None:
                try:
                    result[field] = int(params[field])
                except:
                    result[field] = 1 if field == 'adults' else 0
                    
        # 处理舱位等级
        if 'cabin_class' in params and params['cabin_class']:
            valid_classes = ['Economy', 'Premium Economy', 'Business', 'First']
            cabin = params['cabin_class'].strip().title()
            if cabin in valid_classes:
                result['cabin_class'] = cabin
            elif 'business' in cabin.lower():
                result['cabin_class'] = 'Business'
            elif 'first' in cabin.lower():
                result['cabin_class'] = 'Business'  # API不支持First，映射到Business
            elif 'premium' in cabin.lower():
                result['cabin_class'] = 'Premium Economy'
            else:
                result['cabin_class'] = 'Economy'
        
        return result
    except Exception as e:
        print(f"Error extracting flight parameters: {str(e)}")
        return {}

# 创建工具
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

# 提示模板
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
  3. Number of passengers (adults, children, infants)
  4. Cabin class preferences (Economy, Premium Economy, Business)
  5. Any other specific requirements (non-stop flights, price range, etc.)
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action - use valid JSON format with quotes around string values, e.g., {{"key": "value", "number": 42}}. Do NOT use markdown code blocks.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated if needed)
Thought: I now know the final answer
Final Answer: Present the flight information in a clear, organized manner. For each flight option, include:
- Airline and flight number
- Departure and arrival times
- Duration
- Cabin class
- Price
- Number of stops

For the user, explain if any information was missing from their query and what assumptions you made.

Begin!

Question: {query}
{agent_scratchpad}
'''

# 代理执行逻辑
def agent_execute(query, chat_history=[]):
    global tools, tool_names, tool_descs, prompt_tpl, llm

    # 预处理航班查询
    flight_keywords = ['flight', 'fly', 'plane', 'airport', 'travel', 'trip']
    is_flight_query = any(keyword in query.lower() for keyword in flight_keywords)
    
    if is_flight_query:
        # 提取航班参数
        flight_params = extract_flight_parameters(query)
        if flight_params:
            # 用结构化参数增强查询
            query_addition = f"\n\nExtracted parameters: {json.dumps(flight_params, indent=2)}"
            query = f"{query}{query_addition}"
    
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

        response = llm(prompt, user_stop_words=['Observation:'])

        # 解析响应内容
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
            # 格式不正确，生成回退响应
            fallback_response = "I'm having trouble processing your flight search request. Could you please provide more specific details like origin, destination, travel dates, and the number of passengers?"
            chat_history.append((query, fallback_response))
            return True, fallback_response, chat_history
            
        if observation_i == -1:
            observation_i = len(response)
            response = response + '\nObservation: '

        thought = response[thought_i + len('Thought:'):action_i].strip()
        action = response[action_i + len('\nAction:'):action_input_i].strip()
        action_input = response[action_input_i + len('\nAction Input:'):observation_i].strip()

        # 查找工具
        the_tool = next((t for t in tools if t.name == action), None)
        if the_tool is None:
            observation = 'The tool does not exist. Please use one of the available tools.'
            agent_scratchpad += response + observation + '\n'
            continue

        try:
            # 使用工具
            tool_ret = the_tool.invoke(input=action_input)
        except Exception as e:
            observation = f'Tool execution error: {e}'
        else:
            observation = str(tool_ret)

        agent_scratchpad += response + observation + '\n'

# 带错误处理的代理重试
def agent_execute_with_retry(query, chat_history=[], retry_times=2):
    for i in range(retry_times):
        try:
            success, result, chat_history = agent_execute(query, chat_history=chat_history)
            if success:
                return success, result, chat_history
        except Exception as e:
            if i == retry_times - 1:
                error_msg = f"I apologize, but I'm having trouble processing your request due to a technical issue. Could you please try rephrasing your question or providing more specific flight details?"
                chat_history.append((query, error_msg))
                return False, error_msg, chat_history
    
    # 所有重试都失败
    fallback_msg = "I couldn't find the flight information you requested. Could you please provide more details about your trip, including specific airports or cities, travel dates, and passenger information?"
    chat_history.append((query, fallback_msg))
    return False, fallback_msg, chat_history

# 主函数
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
            my_history = my_history[-5:]  # 仅保留最后5个交互
            print(f"\n{result}")

if __name__ == "__main__":
    main()