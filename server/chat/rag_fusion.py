# RAG utils
import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from configs import PROMPT_TEMPLATES, FUSION_K, EMBEDDING_DEVICE
from server.utils import embedding_device

# 定义一个函数来去除字符串开头的数字和句点
def remove_number_and_dot(text):
    if re.match(r'^\d+\.\s*', text):
        return re.sub(r'^\d+\.\s*', '', text)
    else:
        return text

# Function to generate queries using OpenAI's ChatGPT
def generate_queries(original_query, pipe: Pipeline):
    # get HuggingFacePipeline
    local_llm = HuggingFacePipeline(pipeline=pipe)
    # get llm-chain 
    template = PROMPT_TEMPLATES['rag-fusion']['fusion_en']
    template = "You are a helpful assistant that generates multiple search queries based on a single input query. Please Generate multiple search queries related to: {original_query}. OUTPUT ({k_query} queries): \n\n"
    prompt = PromptTemplate.from_template(template=template)

    llm_chain = LLMChain(prompt=prompt,
                        llm=local_llm
                        )
    input_list = [
        {'original_query': original_query, 'k_query': str(FUSION_K)}
    ]
    response = llm_chain.apply(input_list)

    generated_queries = response[0]['text'].split(template[-6:])[-1].split("\n")
    generated_queries = [remove_number_and_dot(query) for query in generated_queries]
    
    return generated_queries

# Mock function to simulate vector search, returning random scores
def vector_search(query, all_documents):
    available_docs = list(all_documents.keys())
    random.shuffle(available_docs)
    selected_docs = available_docs[:random.randint(2, 5)]
    scores = {doc: round(random.uniform(0.7, 0.9), 2) for doc in selected_docs}
    return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}

# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=60) -> dict:
    fused_scores = {}
    print("Initial individual search result ranks:")
    for query, doc_scores in search_results_dict.items():
        print(f"For query '{query}': {doc_scores}")
        
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    print("Final reranked results:", reranked_results)
    return reranked_results

# Dummy function to simulate generative output
def generate_output(reranked_results, queries):
    return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"


# Predefined set of documents (usually these would be from your search database)
all_documents = {
    "doc1": "Climate change and economic impact.",
    "doc2": "Public health concerns due to climate change.",
    "doc3": "Climate change: A social perspective.",
    "doc4": "Technological solutions to climate change.",
    "doc5": "Policy changes needed to combat climate change.",
    "doc6": "Climate change and its impact on biodiversity.",
    "doc7": "Climate change: The science and models.",
    "doc8": "Global warming: A subset of climate change.",
    "doc9": "How climate change affects daily weather.",
    "doc10": "The history of climate change activism."
}

# Test Code
if __name__ == "__main__":
    # get Model
    tokenizer = AutoTokenizer.from_pretrained('llm_models/chatglm3-6b', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('llm_models/chatglm3-6b', trust_remote_code=True)
    pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, 
                    max_length=256, temperature=0.6, top_p=0.95, repetition_penalty=1.2, device='cuda:1')


    original_query = "impact of climate change"
    generated_queries = generate_queries(original_query, pipe)
    
    # all_results = {}
    # for query in generated_queries:
    #     search_results = vector_search(query, all_documents)
    #     all_results[query] = search_results
    
    # reranked_results = reciprocal_rank_fusion(all_results)
    
    # final_output = generate_output(reranked_results, generated_queries)
    
    # print(final_output)

