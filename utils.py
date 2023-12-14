import logging
import time

# init logger
logging.basicConfig(filename="logs.log",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('urbanGUI')
logger.addHandler(logging.StreamHandler())

# timing decorator
def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time-start_time:.2f} sec")

        return result
    return wrapper

