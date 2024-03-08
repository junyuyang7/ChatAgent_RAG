import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contextlib import contextmanager

import httpx
from fastchat.conversation import Conversation
from httpx_sse import EventSource

from server.model_workers.base import *
from fastchat import conversation as conv
import sys
from typing import List, Dict, Iterator, Literal, Any
# JWT（JSON Web Tokens）是一种开放标准（RFC 7519），用于在网络上以 JSON 对象的形式安全地传输声明。JWT 可以在用户和服务器之间传递声明，并使用数字签名进行验证。它通常用于身份验证和授权。
import jwt
import time

from server.model_workers.base import ApiChatParams

@contextmanager
def connect_sse(client: httpx.Client, method: str, url: str, **kwargs: Any):
    '''
    定义了一个上下文管理器 connect_sse，用于在使用 SSE（Server-Sent Events）时建立与服务器的连接并返回响应的事件流 EventSource。
    '''
    with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)

def generate_token(apikey: str, exp_seconds: int):
    '''生成身份token'''
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

class ChatGLMWorker(ApiModelWorker):
    # *在这里用于定义关键字参数的开始位置，表示之后的参数必须以关键字参数的形式传递，而不是位置参数。
    def __init__(
            self,
            *,
            model_names: List[str] = ['zhipu-api'],
            controller_addr: str = None,
            worker_addr: str = None,
            version: Literal['glm-4'] = 'glm-4',
            **kwargs,
    ):
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 4096)
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Iterator[Dict]:
        '''调用大模型进行对话'''
        params.load_config(self.model_names[0])
        token = generate_token(params.api_key, 60)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        data = {
            "model": params.version,
            "messages": params.messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stream": False
        }

        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        with httpx.Client(headers=headers) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            chunk = response.json()
            print(chunk)
            yield {"error_code": 0, "text": chunk["choices"][0]["message"]["content"]}

    def get_embeddings(self, params):
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str=None, model_path: str=None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是智谱AI小助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n###",
            stop_str="###",
        )
    
if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    worker = ChatGLMWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21001",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21001)
