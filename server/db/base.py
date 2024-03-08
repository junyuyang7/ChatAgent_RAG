from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
# declarative_base()用于创建基本的类，这些类可以映射到数据库中的表格，使得通过对象来进行数据库操作更加方便。
# DeclarativeMeta是declarative_base()返回的类型 Base = declarative_base()
from sqlalchemy.orm import sessionmaker

from configs import SQLALCHEMY_DATABASE_URI
import json

# json.dumps 函数作用是将对象 obj 转换为 JSON 字符串。
engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base: DeclarativeMeta = declarative_base()
