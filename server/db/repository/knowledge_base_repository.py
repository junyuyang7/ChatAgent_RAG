from server.db.models.knowledge_base_model import KnowledgeBaseModel
from server.db.session import with_session
from typing import List

@with_session
def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model):
    '''创建/更新知识库实例加入数据库'''
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    if not kb:
        kb = KnowledgeBaseModel(kb_name=kb_name, kb_info=kb_info, vs_type=vs_type, embed_model=embed_model)
        session.add(kb)
    else: # 如果已经存在就进行更新即可
        kb.vs_type = vs_type
        kb.kb_info = kb_info
        kb.embed_model = embed_model
    return True

@with_session
def list_kbs_from_db(session, min_file_count: int = -1) -> List:
    '''列出数据库中含有的知识库'''
    kbs = session.query(KnowledgeBaseModel.kb_name).filter(KnowledgeBaseModel.file_count > min_file_count).all()
    kbs = [kb[0] for kb in kbs]
    return kbs

@with_session
def kb_exists(session, kb_name):
    '''判断知识库存不存在'''
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    status = True if kb else False
    return status

@with_session
def load_kb_from_db(session, kb_name):
    '''从数据库中加载对应知识库'''
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model

@with_session
def delete_kb_from_db(session, kb_name):
    '''从数据库中删除对应知识库'''
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    if kb:
        session.delete(kb)
    return True

@with_session
def get_kb_detail(session, kb_name: str) -> dict:
    '''获取知识库的详细信息'''
    kb: KnowledgeBaseModel = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    if kb:
        return {
            "kb_name": kb.kb_name,
            "kb_info": kb.kb_info,
            "vs_type": kb.vs_type,
            "embed_model": kb.embed_model,
            "file_count": kb.file_count,
            "create_time": kb.create_time,
        }
    else:
        return {}
