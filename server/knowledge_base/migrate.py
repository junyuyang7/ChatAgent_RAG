from configs import (
    EMBEDDING_MODEL, DEFAULT_VS_TYPE, ZH_TITLE_ENHANCE,
    CHUNK_SIZE, OVERLAP_SIZE,
    logger, log_verbose
)
from server.knowledge_base.utils import (
    get_file_path, list_kbs_from_folder,
    list_files_from_folder, files2docs_in_thread,
    KnowledgeFile
)
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.models.conversation_model import ConversationModel
from server.db.models.message_model import MessageModel
from server.db.repository.knowledge_file_repository import add_file_to_db # ensure Models are imported
from server.db.repository.knowledge_metadata_repository import add_summary_to_db

from server.db.base import Base, engine
from server.db.session import session_scope
import os
from dateutil.parser import parse
from typing import Literal, List