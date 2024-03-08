from langchain.docstore.document import Document
import re

def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    '''
    检查文本片段中非字母字符的比例是否超过给定的阈值。
    它用于防止像 "-----------BREAK---------" 这样的文本被错误地标记为标题或叙述性文本。
    '''
    if len(text) == 0:
        return False
    
    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    try:
        ratio = alpha_count / total_count
        return ratio < threshold
    except:
        return False
    
def is_possible_title(
        text: str,
        title_max_word_length: int = 20,
        non_alpha_threshold: float = 0.5,
) -> bool:
    '''判断该text是不是标题'''
    if len(text) == 0:
        print('Not a title. Text is empty')
        return False
    
    # 如果存在标点符号，肯定不是标题
    ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
    ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
    if ENDS_IN_PUNCT_RE.search(text) is not None:
        return False
    
    # 文本长度不能超过设定值，默认20
    if len(text) > title_max_word_length:
        return False   

    # 文本中数字的占比不能太高，否则不是title
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    # 不能以下面这些符号结尾
    if text.endswith((",", ".", "，", "。")):
        return False
    
    # 全部都是数字不是title
    if text.isnumeric():
        print(f"Not a title. Text is all numeric:\n\n{text}")  # type: ignore
        return False

    # 开头的字符内应该有数字，默认5个字符内
    if len(text) < 5:
        text_5 = text
    else:
        text_5 = text[:5]
    alpha_in_text_5 = sum(list(map(lambda x: x.isnumeric(), list(text_5))))
    if not alpha_in_text_5:
        return False

    return True

def zh_title_enhance(docs: Document) -> Document:
    title = None
    if len(docs) > 0:
        for doc in docs:
            if is_possible_title(doc.page_content):
                doc.metadata['category'] = 'cn_Title'
                title = doc.page_content
            elif title:
                doc.page_content = f"下文与({title})有关。{doc.page_content}"
        return docs
    else:
        print("文件不存在")
