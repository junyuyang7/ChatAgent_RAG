from server.db.session import with_session
import uuid
from server.db.models.conversation_model import ConversationModel

@with_session
def add_conversation_to_db(session, chat_type, name="", conversation_id=None):
    '''新增聊天记录'''
    if not conversation_id:
        conversation_id = uuid.uuid4().hex # Python 中生成 UUID（Universally Unique Identifier，通用唯一标识符）的一种方式。
    c = ConversationModel(id=conversation_id, name=name, chat_type=chat_type)

    # add()方法用于将对象添加到会话中，表示将对象添加到数据库会话的待处理队列中。这意味着对象被暂时标记为“待插入”状态，但还没有真正被插入到数据库中。
    session.add(c)
    return c.id
