import os
import openai
import json
import datetime
from langchain_community.tools.tavily_search import TavilySearchResults

# 设置 OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

from openai import OpenAI

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


# 设置 Tavily 搜索工具
os.environ['TAVILY_API_KEY'] = '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'
tavily = TavilySearchResults(max_results=5)
tavily.description = '这是一个类似谷歌和百度的搜索引擎，搜索知识、天气、股票、电影、小说、百科等都是支持的哦，如果你不确定就应该搜索一下，谢谢！'

# 工具相关配置
tools = [tavily]
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

# 提示模板
prompt_tpl = '''Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

These are chat history before:
{chat_history}

IMPORTANT: You MUST use the exact format including "Thought: I now know the final answer" followed by "Final Answer: ..."

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{agent_scratchpad}
'''

# Agent 主逻辑
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

        print('\033[32m---等待LLM返回... ...\n%s\n\033[0m' % prompt, flush=True)
        response = llm(prompt, user_stop_words=['Observation:'])
        print('\033[34m---LLM返回---\n%s\n---\033[34m' % response, flush=True)

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
            return False, 'LLM回复格式异常', chat_history
        if observation_i == -1:
            observation_i = len(response)
            response = response + 'Observation: '

        thought = response[thought_i + len('Thought:'):action_i].strip()
        action = response[action_i + len('\nAction:'):action_input_i].strip()
        action_input = response[action_input_i + len('\nAction Input:'):observation_i].strip()

        # 查找工具
        the_tool = next((t for t in tools if t.name == action), None)
        if the_tool is None:
            observation = 'the tool not exist'
            agent_scratchpad += response + observation + '\n'
            continue

        try:
            action_input = json.loads(action_input)
            tool_ret = the_tool.invoke(input=json.dumps(action_input))
        except Exception as e:
            observation = f'the tool has error: {e}'
        else:
            observation = str(tool_ret)

        agent_scratchpad += response + observation + '\n'

# Agent 自动重试执行
def agent_execute_with_retry(query, chat_history=[], retry_times=3):
    for i in range(retry_times):
        success, result, chat_history = agent_execute(query, chat_history=chat_history)
        if success:
            return success, result, chat_history
    return success, result, chat_history

# 交互入口
my_history = []
while True:
    query = input('query: ')
    success, result, my_history = agent_execute_with_retry(query, chat_history=my_history)
    my_history = my_history[-10:]
