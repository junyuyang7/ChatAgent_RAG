from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional, Union
import asyncio
import json
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from langchain.prompts import PromptTemplate
from server.utils import get_prompt_template
from server.memory.conversation_db_buffer_memory import ConversationBUfferDBMemory
from server.db.repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler

async def chat(query: str = Body(..., description="用户输入", example=["番茄炒鸡蛋怎么做？"]),
               conversation_id: str = Body("", description="对话框ID"),
               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "老虎和狮子打架谁会赢"},
                                                             {"role": "assistant", "content": "人类会赢"}]]
                                                         ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None

        # 负责保存llm response到message db
        message_id = add_message_to_db(conversation_id=conversation_id,
                                       chat_type="llm_chat",
                                       query=query)
        conversation_callback = ConversationCallbackHandler(
            conversation_id=conversation_id,
            message_id=message_id,
            chat_type="llm_chat",
            query=query
        )
        callbacks.append(conversation_callback)

        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

        if history: # 有限使用前端传入的历史信息
            history = [History.from_data(h) for h in history]
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role='user', content=prompt_template).to_msg_template(False)
            # 历史记录加上当前的query
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + input_msg
            )
        elif conversation_id and history_len > 0: # 使用数据库中保存的信息
            # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
            prompt = get_prompt_template(type="llm_chat", name="with_history")
            chat_prompt = ChatPromptTemplate.from_template(prompt)
            # 根据conversation_id 获取message 列表进而拼凑 memory
            memory = ConversationBUfferDBMemory(
                conversation_id=conversation_id,
                llm=model,
                message_limit=history_len
            )
        else:
            prompt_template = get_prompt_template(type="llm_chat", name=prompt_name)
            input_msg = History(role='user', content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_template([input_msg])
        
        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        # 开始创建任务
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done
        ))

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"text": token, "message_id": message_id},
                    ensure_ascii=False
                )
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id},
                ensure_ascii=False
            )

        await task
    
    # 在函数内部，通过生成器 event_generator 生成事件数据，并使用 EventSourceResponse 将其作为响应返回给客户端。客户端就可以通过 SSE 连接接收到实时事件数据，从而实现实时更新的效果。
    return EventSourceResponse(chat_iterator())