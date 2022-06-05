"""
Model storage
"""
from abc import abstractmethod
import redis
import pickle

REDIS_HOST = '0.0.0.0'
REDIS_PORT = 6379
REDIS_DB = 1

class ModelStorage(object):
    """The object to store the model."""
    @abstractmethod
    def get_model(self,model_id=None):
        """Get model"""
        pass

    @abstractmethod
    def save_model(self,model_id=None):
        """Save model"""
        pass


class MemoryModelStorage(ModelStorage):
    """Store the model in memory."""
    def __init__(self):
        self._model = None

    def get_model(self,model_id=None):
        return self._model

    def save_model(self,model_id=None,model=None):
        self._model = model

class RedisModelStorage(ModelStorage):
    """Store the model in redis."""
    def __init__(self,host=None,port=None,db=None,client=None): 
        if client:
            self.r_server =  client
        else:
            if host is None:
                host = REDIS_HOST
                port = REDIS_PORT
                db = REDIS_DB
            self.r_server = redis.StrictRedis(host=host, port=port, db=db)
            
    def get_model(self,model_id):
        model = self.r_server.hget("streamingbandit:{0}".format(model_id),"onlinemodel")
        if model is None:
            return None
        else:
            return pickle.loads(model)
        
    def save_model(self,model_id,model):
        _model = pickle.dumps(model)
        self.r_server.hset("streamingbandit:{0}".format(model_id),"onlinemodel",_model)
        