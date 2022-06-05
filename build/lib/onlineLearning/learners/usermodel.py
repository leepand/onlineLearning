from typing import List

import random
import numpy as np
import redis
import pickle
from onlineLearning.utils import load_save_config_file


# 'ml-redis.tlqeea.0001.usw2.cache.amazonaws.com'

class EGreedy:
    def __init__(self, config_file):
        self.config = load_save_config_file(config_file)
        self.r_server  = redis.StrictRedis(host= self.config.get("redis_host","localhost"), 
                                           port= self.config.get("redis_port",6379), 
                                           db= self.config.get("redis_db",1)

    def _increment_model_tries(self, model: str, uid: str, model_id: str) -> None:
        key_tries="onlinemodel:{0}:{1}:{2}:tries".format(uid,model_id,model)
        model_tries = self.r_server.get(key_tries)
        if model_tries is None:
            model_tries = 0
            
        model_tries = float(model_tries)
        model_tries+=1
        self.r_server.set(key_tries,model_tries)
        return model_tries

    def get_mostpop(self,uid,model_id,topN,withscores=False):

        score_key = "onlinemodel:{0}:{1}:score_ee".format(uid,model_id)
        return self.r_server.zrange(score_key,0,topN-1,withscores=withscores,desc=True)

    def _epsilon_greedy_selection(self,uid, model_id,topN):
        models = self.get_models(model_id)
        epsilon_key = "onlinemodel:{0}:epsilon".format(model_id)
        epsilon = self.r_server.get(epsilon_key)
        if epsilon is None:
            epsilon = 0.2
        else:
            epsilon = float(epsilon)

        if random.random() < epsilon:
            if topN > len(models):
                res = random.sample(models, len(models))
            else:
                res = random.sample(models, topN)
            recommendation_ids = [str(v,encoding='utf-8') for v in res]
            print(recommendation_ids,topN,models)
        else:
            res = self.get_mostpop(uid,model_id,topN)
            recommendation_ids = [str(v,encoding='utf-8') for v in res]
        for item_id in recommendation_ids:
            #更新曝光ID的 score
            item_id = str(item_id)
            model_tries = self._increment_model_tries(item_id, uid, model_id)
            success_key = "onlinemodel:{0}:{1}:{2}:reward_successes".format(uid,model_id,item_id)
            #key_tries="onlinemodel:{0}:{1}:{2}:tries".format(uid,model_id,item_id)
            _reward = self.r_server.get(success_key)
            if _reward is None:
                _reward = 0.0
            else:
                _reward = float(_reward)
            _model_score = _reward/(model_tries + 0.00000001)
            score_key = "onlinemodel:{0}:{1}:score_ee".format(uid,model_id)
            print({item_id: _model_score},"rrrr")
            with self.r_server.pipeline() as pipe:
                pipe.zadd(score_key,{item_id: _model_score})
                pipe.execute()
            
        return recommendation_ids,"model"

    def select_model(self,uid,model_id,topN=3) -> str:
        epsilon_greedy_selection = self._epsilon_greedy_selection(uid,model_id,topN=topN)
        #self._increment_model_tries(epsilon_greedy_selection)
        return epsilon_greedy_selection

    def reward_model(self, model: str,uid:str,model_id:str,reward:float=None) -> None:

        success_key = "onlinemodel:{0}:{1}:{2}:reward_successes".format(uid,model_id,model)
        score_key = "onlinemodel:{0}:{1}:score_ee".format(uid,model_id)
        key_tries="onlinemodel:{0}:{1}:{2}:tries".format(uid,model_id,model)

        if reward is None:
            reward = 1.0
        _reward = self.r_server.get(success_key)
        if _reward is None:
            _reward = 0.0
        else:
            _reward = float(_reward)
        _reward+= reward
        self.r_server.set(success_key,_reward)

        model_tries = self.r_server.get(key_tries)
        if model_tries is None:
            model_tries = 1.0  
        else:
            model_tries = float(model_tries)
        _model_score = _reward/(model_tries + 0.00000001)

        with self.r_server.pipeline() as pipe:
            pipe.zadd(score_key,{model: _model_score})
            pipe.execute()

    def set_epsilon(self,model_id:str=None,epsilon:float=0.2):
        epsilon_key = "onlinemodel:{0}:epsilon".format(model_id)
        self.r_server.set(epsilon_key,epsilon)
        return True

    def add_model(self,model_id:str=None,model:str=None):
        # model = item_id
        key_models = "onlinemodel:{0}:models".format(model_id)        
        with self.r_server.pipeline() as pipe:
            pipe.sadd(key_models, model)
            ## 初始化每个玩家的模型分数
            # pipe.zadd(score_key,{model: 0.0})
            pipe.execute()

        return True

    def get_models(self,model_id:str=None):
        key_models = "onlinemodel:{0}:models".format(model_id)
        models = self.r_server.smembers(key_models)
        return models
