import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
sys.path.append('/home/yangjy/Study/ChatAgent_RAG/')

import pytest
from fastapi import HTTPException
from server.knowledge_base.kb_doc_api import search_docs_query_fusion
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.model.kb_document_model import DocumentWithVSId, Document
from unittest.mock import patch, MagicMock

# Mocking dependencies
@pytest.fixture
def mock_kb_service():
    with patch.object(KBServiceFactory, 'get_service_by_name') as mock:
        yield mock

@pytest.fixture
def mock_document():
    return MagicMock(spec=Document)

# Happy path tests with various realistic test values
@pytest.mark.parametrize("queries, knowledge_base_name, top_k, score_threshold, file_name, metadata, expected_ids", [
    (["你好"], "samples", 5, 0.5, "", {}, ["id1", "id2"]), # Test ID: HP-1
    (["查询"], "samples", 3, 0.7, "", {}, ["id1", "id2"]), # Test ID: HP-2
    ([], "samples", 5, 0.5, "file_name", {"key": "value"}, ["id1", "id2", 'value']), # Test ID: HP-3
], ids=["HP-1", "HP-2", "HP-3"])
def test_search_docs_query_fusion_happy_path(mock_kb_service, mock_document, queries, knowledge_base_name, top_k, score_threshold, file_name, metadata, expected_ids):
    # Arrange
    mock_service = MagicMock()
    mock_kb_service.return_value = mock_service
    mock_service.search_docs.return_value = [(mock_document, 0.9)]
    mock_service.list_docs.return_value = [mock_document]
    mock_document.page_content = ''
    mock_document.metadata = {'id': ["id1", "id2"]}  # Set the 'metadata' attribute
    
    # Act
    result = search_docs_query_fusion(queries=queries, knowledge_base_name=knowledge_base_name, top_k=top_k, score_threshold=score_threshold, file_name=file_name, metadata=metadata)
    
    # Assert
    print([doc.id for doc in result])
    assert [doc.id for doc in result] == expected_ids

# Edge cases
@pytest.mark.parametrize("queries, knowledge_base_name, top_k, score_threshold, file_name, metadata, expected_exception", [
    (None, "samples", 5, 0.5, "", {}, ValueError), # Test ID: EC-1
    ([""], "", 5, 0.5, "", {}, HTTPException), # Test ID: EC-2
], ids=["EC-1", "EC-2"])
def test_search_docs_query_fusion_edge_cases(queries, knowledge_base_name, top_k, score_threshold, file_name, metadata, expected_exception):
    # Act & Assert
    with pytest.raises(expected_exception):
        search_docs_query_fusion(queries=queries, knowledge_base_name=knowledge_base_name, top_k=top_k, score_threshold=score_threshold, file_name=file_name, metadata=metadata)

# Error cases
@pytest.mark.parametrize("knowledge_base_name, expected_exception", [
    ("invalid_kb", ValueError), # Test ID: ERR-1
], ids=["ERR-1"])
def test_search_docs_query_fusion_error_cases(mock_kb_service, knowledge_base_name, expected_exception):
    # Arrange
    mock_kb_service.return_value = None
    
    # Act & Assert
    with pytest.raises(expected_exception):
        search_docs_query_fusion(knowledge_base_name=knowledge_base_name)


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
from server.chat.knowledge_base_chat import knowledge_base_chat
from server.utils import BaseResponse
from configs import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE, LLM_MODELS
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.chat.utils import History

# Setup FastAPI app for testing
app = FastAPI()
app.post("/knowledge_base_chat")(knowledge_base_chat)
client = TestClient(app)

@pytest.mark.parametrize(
    "query, knowledge_base_name, top_k, score_threshold, history, stream, model_name, temperature, max_tokens, prompt_name, expected_status, expected_response_contains",
    [
        # Happy path test cases
        ("你好", "samples", VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, [], False, LLM_MODELS[0], TEMPERATURE, None, "default", 200, "answer"),
        ("历史查询", "history", 5, 0.5, [{"role": "user", "content": "历史"}], False, LLM_MODELS[0], 0.7, 100, "default", 200, "answer"),
        # Edge case: Empty query
        ("", "samples", VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, [], False, LLM_MODELS[0], TEMPERATURE, None, "default", 200, "未找到知识库"),
        # Error case: Non-existent knowledge base
        ("你好", "nonexistent", VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, [], False, LLM_MODELS[0], TEMPERATURE, None, "default", 404, "未找到知识库"),
    ],
    ids=["happy-path", "happy-path-with-history", "edge-case-empty-query", "error-nonexistent-kb"]
)
@pytest.mark.asyncio
async def test_knowledge_base_chat(query, knowledge_base_name, top_k, score_threshold, history, stream, model_name, temperature, max_tokens, prompt_name, expected_status, expected_response_contains):
    # Arrange
    with patch.object(KBServiceFactory, 'get_service_by_name', return_value=AsyncMock()) as mock_kb_service:
        mock_kb_service.return_value.search_docs = AsyncMock(return_value=[])
        mock_kb_service.return_value.search_docs_query_fusion = AsyncMock(return_value=[])

    # Act
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/knowledge_base_chat", json={
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name
        })

    # Assert
    assert response.status_code == expected_status
    assert expected_response_contains in response.text
