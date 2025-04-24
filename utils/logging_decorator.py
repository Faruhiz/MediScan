import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_method(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Entering: {func.__name__} with args: {args} and kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"Exiting: {func.__name__} with result: {result}")
            return result
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper

def auto_log_methods(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, log_method(attr_value))
    return cls