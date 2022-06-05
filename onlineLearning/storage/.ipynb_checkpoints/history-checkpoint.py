"""
History storage
"""
from abc import abstractmethod
from datetime import datetime
import pickle
import redis
from .disk_cache import get_data_by_cache,set_data_by_cache

REDIS_HOST = '0.0.0.0'
REDIS_PORT = 6379
REDIS_DB = 1



class History(object):
    """action/reward history entry.
    Parameters
    ----------
    history_id : int
    context : {dict of list of float, None}
    recommendations : {Recommendation, list of Recommendation}
    created_at : datetime
    rewards : {float, dict of float, None}
    rewarded_at : {datetime, None}
    """

    def __init__(self, history_id, context, recommendations, created_at,
                 rewarded_at=None):
        self.history_id = history_id
        self.context = context
        self.recommendations = recommendations
        self.created_at = created_at
        self.rewarded_at = rewarded_at

    def update_reward(self, rewards, rewarded_at):
        """Update reward_time and rewards.
        Parameters
        ----------
        rewards : {float, dict of float, None}
        rewarded_at : {datetime, None}
        """
        if not hasattr(self.recommendations, '__iter__'):
            recommendations = (self.recommendations,)
        else:
            recommendations = self.recommendations

        for rec in recommendations:
            try:
                rec.reward = rewards[rec.action.id]
            except KeyError:
                pass
        self.rewarded_at = rewarded_at

    @property
    def rewards(self):
        if not hasattr(self.recommendations, '__iter__'):
            recommendations = (self.recommendations,)
        else:
            recommendations = self.recommendations
        rewards = {}
        for rec in recommendations:
            if rec.reward is None:
                continue
            rewards[rec.action.id] = rec.reward
        return rewards


class HistoryStorage(object):
    """The object to store the history of context, recommendations and rewards.
    """
    @abstractmethod
    def get_history(self, history_id):
        """Get the previous context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        pass

    @abstractmethod
    def get_unrewarded_history(self, history_id):
        """Get the previous unrewarded context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        pass

    @abstractmethod
    def add_history(self, context, recommendations, rewards=None):
        """Add a history record.
        Parameters
        ----------
        context : {dict of list of float, None}
        recommendations : {Recommendation, list of Recommendation}
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        pass

    @abstractmethod
    def add_reward(self, history_id, rewards):
        """Add reward to a history record.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        pass


class MemoryHistoryStorage(HistoryStorage):
    """HistoryStorage that store History objects in memory."""

    def __init__(self):
        self.histories = {}
        self.unrewarded_histories = {}
        self.n_histories = 0

    def get_history(self, history_id):
        """Get the previous context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        return self.histories[history_id]

    def get_unrewarded_history(self, history_id):
        """Get the previous unrewarded context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        return self.unrewarded_histories[history_id]

    def add_history(self, context, recommendations, rewards=None):
        """Add a history record.
        Parameters
        ----------
        context : {dict of list of float, None}
        recommendations : {Recommendation, list of Recommendation}
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        created_at = datetime.now()
        history_id = self.n_histories
        if rewards is None:
            history = History(history_id, context, recommendations, created_at)
            self.unrewarded_histories[history_id] = history
        else:
            rewarded_at = created_at
            history = History(history_id, context, recommendations, created_at,
                              rewards, rewarded_at)
            self.histories[history_id] = history
        self.n_histories += 1
        return history_id

    def add_reward(self, history_id, rewards):
        """Add reward to a history record.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        rewarded_at = datetime.now()
        history = self.unrewarded_histories.pop(history_id)
        history.update_reward(rewards, rewarded_at)
        self.histories[history.history_id] = history
        
class RedisHistoryStorage(HistoryStorage):
    """HistoryStorage that store History objects in redis."""

    def __init__(self,r_server=None):
        if r_server is None:
            host = REDIS_HOST
            port = REDIS_PORT
            db = REDIS_DB
            self.r_server = redis.StrictRedis(host=host, port=port, db=db)
        self.histories = {}
        self.unrewarded_histories = {}
        self.n_histories = 0

    def get_history(self, history_id,model_id=None):
        """Get the previous context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        histories = self.r_server.hget("streamingbandit:{0}:{1}".format(model_id,history_id),
                                       "histories")
        if histories is None:
            histories={}
            return histories
        return pickle.loads(histories)

    def get_unrewarded_history(self, history_id,model_id=None):
        """Get the previous unrewarded context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        unrewarded_histories = self.r_server.hget("streamingbandit:{0}:{1}"
                                                  .format(model_id,history_id),
                                                  "unrewarded_histories")
        if unrewarded_histories is None:
            unrewarded_histories = {}
            return unrewarded_histories
        return pickle.loads(unrewarded_histories)

    def add_history(self, context,
                    recommendations,
                    rewards=None, 
                    request_id = None,
                    model_id=None):
        """Add a history record.
        Parameters
        ----------
        context : {dict of list of float, None}
        recommendations : {Recommendation, list of Recommendation}
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        created_at = datetime.now()
        #n_histories = self.r_server.hget("streamingbandit:{0}:{1}"
        #                                 .format(model_id,history_id),
        #                                 "n_histories")
        #history_id = self.n_histories
        if rewards is None:
            history = History(request_id, context, recommendations, created_at)
            _history = pickle.dumps(history)
            self.r_server.hset("streamingbandit:{0}:{1}".format(model_id,request_id),
                               "unrewarded_histories",_history)
            #self.unrewarded_histories[history_id] = history
        else:
            rewarded_at = created_at
            history = History(request_id, context, recommendations, created_at,
                              rewards, rewarded_at)
            _history = pickle.dumps(history)
            self.r_server.hset("streamingbandit:{0}:{1}".format(model_id,request_id),
                               "histories",_history)
            #self.histories[history_id] = history
        #self.n_histories += 1
        return request_id

    def add_reward(self, history_id, rewards,model_id=None):
        """Add reward to a history record.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        rewarded_at = datetime.now()
        #history = self.unrewarded_histories.pop(history_id)
        _history = self.r_server.hget("streamingbandit:{0}:{1}".format(model_id,history_id),
                                     "unrewarded_histories")
        history = pickle.loads(_history)
        history.update_reward(rewards, rewarded_at)
        #self.histories[history.history_id] = history
        self.r_server.hset("streamingbandit:{0}:{1}".format(model_id,history_id),
                           "histories",pickle.dumps(history))
        
        
class DiskCacheHistoryStorage(HistoryStorage):
    """HistoryStorage that store History objects in redis."""

    def __init__(self,dbpath=None):
        if dbpath == None:
            dbpath = "./"
        self.dbpath = dbpath
            
        self.histories = {}
        self.unrewarded_histories = {}
        self.n_histories = 0

    def get_history(self, history_id,model_id=None):
        """Get the previous context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        key = "streamingbandit:{0}:{1}:histories".format(model_id,history_id)
        histories = get_data_by_cache(key,self.dbpath)
        if histories is None:
            histories={}
            return histories
        return histories

    def get_unrewarded_history(self, history_id,model_id=None):
        """Get the previous unrewarded context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        key = "streamingbandit:{0}:{1}:unrewarded_histories".format(model_id,history_id)
        unrewarded_histories = get_data_by_cache(key,self.dbpath)
        if unrewarded_histories is None:
            unrewarded_histories = {}
            return unrewarded_histories
        return unrewarded_histories

    def add_history(self, context,
                    recommendations,
                    rewards=None, 
                    request_id = None,
                    model_id=None):
        """Add a history record.
        Parameters
        ----------
        context : {dict of list of float, None}
        recommendations : {Recommendation, list of Recommendation}
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        created_at = datetime.now()
        unrewarded_key = "streamingbandit:{0}:{1}:unrewarded_histories".format(model_id,request_id)
        histories_key = "streamingbandit:{0}:{1}:histories".format(model_id,request_id)
        #n_histories = self.r_server.hget("streamingbandit:{0}:{1}"
        #                                 .format(model_id,history_id),
        #                                 "n_histories")
        #history_id = self.n_histories
        if rewards is None:
            history = History(request_id, context, recommendations, created_at)
            #_history = pickle.dumps(history)
            set_data_by_cache(unrewarded_key,history,self.dbpath)
            #self.unrewarded_histories[history_id] = history
        else:
            rewarded_at = created_at
            history = History(request_id, context, recommendations, created_at,
                              rewards, rewarded_at)
            #_history = pickle.dumps(history)
            set_data_by_cache(histories_key,history,self.dbpath)

            #self.histories[history_id] = history
        #self.n_histories += 1
        return request_id

    def add_reward(self, history_id, rewards,model_id=None):
        """Add reward to a history record.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        rewarded_at = datetime.now()
        unrewarded_key="streamingbandit:{0}:{1}:unrewarded_histories".format(model_id,history_id)
        #history = self.unrewarded_histories.pop(history_id)
        history = get_data_by_cache(unrewarded_key,self.dbpath)
        #history = pickle.loads(_history)
        history.update_reward(rewards, rewarded_at)
        #self.histories[history.history_id] = history
        histories_key = "streamingbandit:{0}:{1}:histories".format(model_id,history_id)
        set_data_by_cache(histories_key,history,self.dbpath)
        #self.r_server.hset("streamingbandit:{0}:{1}".format(model_id,history_id),
        #                   "histories",pickle.dumps(history))