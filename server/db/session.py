from functools import wraps
from contextlib import contextmanager
from server.db.base import SessionLocal
from sqlalchemy.orm import Session

# 上下文管理器是指在一段代码执行之前执行一段代码，用于一些预处理工作；执行之后再执行一段代码，用于一些清理工作。
# 装饰器contextmanager。该装饰器将一个函数中yield语句之前的代码当做__enter__方法执行，yield语句之后的代码当做__exit__方法执行。同时yield返回值赋值给as后的变量。
@contextmanager
def session_scope() -> Session: # type: ignore
    '''上下文管理器用于自动获取 Session, 避免发生异常'''
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


# 定义一个与数据库交互的装饰器，减少代码冗余
def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except:
                session.rollback()
                raise
    
    return wrapper


def get_db() -> SessionLocal: # type: ignore
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

