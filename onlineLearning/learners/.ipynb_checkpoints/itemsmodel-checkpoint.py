from typing import Any, Dict, Sequence, Optional, cast, List
import random
import numpy as np
import redis

from onlineLearning.utils import load_save_config_file
from onlineLearning.storage import dbEngine


class EEItemsModel:
    def __init__(self, config_file):
        self.config = load_save_config_file(config_file)
        self.r_server  = redis.StrictRedis(host= self.config.get("redis_host","localhost"), 
                                           port= self.config.get("redis_port",6379), 
                                           db= self.config.get("redis_db",1))
        #self.db = dbEngine(self.config.get("db_path","/"))

    @property
    def params(self) -> Dict[str, Any]:

        return { "family": "EEItemsModel" }

    def _increment_model_tries(self, model: str, model_id: str=None) -> None:
        key_tries="onlinemodel:eeitemmodel:{0}:{1}:tries".format(model_id,model)
        model_tries = self.r_server.get(key_tries)
        if model_tries is None:
            model_tries = 0
            
        model_tries = float(model_tries)
        model_tries+=1
        self.r_server.set(key_tries,model_tries)
        return model_tries

    def get_best_model_so_far(self, model_id,topN,withscores=False):

        score_key = "onlinemodel:{0}:{1}:score_ee".format("eeitemmodel",model_id)
        return self.r_server.zrange(score_key,0,topN-1,withscores=withscores,desc=True)

    def _epsilon_greedy_selection(self, model_id:str,topN: int):
        models = self.get_models(model_id)
        epsilon_key = "onlinemodel:eeitemmodel:{0}:epsilon".format(model_id)
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
        else:
            res = self.get_best_model_so_far(model_id,topN)
            recommendation_ids = [str(v,encoding='utf-8') for v in res]

        return recommendation_ids

    def select_model(self,model_id,topN, ifexpose='yes',ifrecord= 'no') -> list:
        epsilon_greedy_selection = self._epsilon_greedy_selection(model_id,topN=topN)
        if epsilon_greedy_selection is None or len(epsilon_greedy_selection)<1:
            return []
        if ifexpose=='yes':
            for item_id in epsilon_greedy_selection:
                #更新曝光ID的 score
                item_id = str(item_id)
                model_tries = self._increment_model_tries(item_id, model_id)
                success_key = "onlinemodel:{0}:{1}:{2}:reward_successes".format('eeitemmodel',
                                                                                model_id,
                                                                                item_id)
                _reward = self.r_server.get(success_key)
                if _reward is None:
                    _reward = 0.0
                else:
                    _reward = float(_reward)
                _model_score = _reward/(model_tries + 0.00000001)
                score_key = "onlinemodel:{0}:{1}:score_ee".format("eeitemmodel",model_id)

                with self.r_server.pipeline() as pipe:
                    pipe.zadd(score_key,{item_id: _model_score})
                    pipe.execute()

        return epsilon_greedy_selection

    def reward_model(self, model: str,model_id:str,reward:float=None,init_model="no") -> None:
        success_key = "onlinemodel:{0}:{1}:{2}:reward_successes".format("eeitemmodel",model_id,model)
        score_key = "onlinemodel:{0}:{1}:score_ee".format("eeitemmodel",model_id)
        key_tries="onlinemodel:{0}:{1}:{2}:tries".format("eeitemmodel",model_id,model)

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
        if init_model == 'yes':
            # 初始化模型时默认曝光
            model_tries+=1
            self.add_model(model_id,model)
            self.r_server.set(key_tries,model_tries)
    
        _model_score = _reward/(model_tries + 0.00000001)

        with self.r_server.pipeline() as pipe:
            pipe.zadd(score_key,{model: _model_score})
            pipe.execute()

        return True

    def set_epsilon(self,model_id:str=None,epsilon:float=0.2):
        epsilon_key = "onlinemodel:eeitemmodel:{0}:epsilon".format(model_id)
        self.r_server.set(epsilon_key,epsilon)
        return True

    def add_model(self,model_id:str=None,model:str=None):
        # model = item_id
        key_models = "onlinemodel:eeiemmodel:{0}:models".format(model_id)        
        with self.r_server.pipeline() as pipe:
            pipe.sadd(key_models, model)
            ## 初始化每个玩家的模型分数
            # pipe.zadd(score_key,{model: 0.0})
            pipe.execute()

        return True
    def del_model(self,model_id:str,model:str):
        key_models = "onlinemodel:eeiemmodel:{0}:models".format(model_id)        
        with self.r_server.pipeline() as pipe:
            pipe.srem(key_models, model)
            ## 初始化每个玩家的模型分数
            # pipe.zadd(score_key,{model: 0.0})
            pipe.execute()

    def get_models(self,model_id:str=None):
        key_models = "onlinemodel:eeiemmodel:{0}:models".format(model_id)
        models = self.r_server.smembers(key_models)
        return models
    
    def filter_model(self,model:str, model_id:str=None):
        score_key = "onlinemodel:{0}:{1}:score_ee".format("eeitemmodel",model_id)
        with self.r_server.pipeline() as pipe:
            pipe.zadd(score_key,{model: -100.0})
            pipe.execute()
        return True
        

class UCB1ItemsModel:
    def __init__(self, config_file):
        self.config = load_save_config_file(config_file)
        self.r_server  = redis.StrictRedis(host= self.config.get("redis_host","localhost"), 
                                           port= self.config.get("redis_port",6379), 
                                           db= self.config.get("redis_db",1)
                                          )
        #self.db = dbEngine(self.config.get("db_path","/"))

    @property
    def params(self) -> Dict[str, Any]:

        return { "family": "UCB1ItemsModel" }

    def _increment_model_tries(self, model: str, model_id: str=None) -> None:
        key_tries="onlinemodel:ucb1itemmodel:{0}:{1}:tries".format(model_id,model)
        key_alltries="onlinemodel:ucb1itemmodel:{0}:tries".format(model_id)
        model_tries = self.r_server.get(key_tries)
        all_model_tries = self.r_server.get(key_alltries)
        if model_tries is None:
            model_tries = 0
        if all_model_tries is None:
            all_model_tries = 0
            
        model_tries = float(model_tries)
        all_model_tries = float(all_model_tries)
        model_tries+=1
        all_model_tries+=1
        self.r_server.set(key_tries,model_tries)
        self.r_server.set(key_alltries,all_model_tries)
        return model_tries,all_model_tries

    def get_best_model_so_far(self, model_id,topN,withscores=False):
        score_key = "onlinemodel:{0}:{1}:score_ee".format("ucb1itemmodel",model_id)
        return self.r_server.zrange(score_key,0,topN-1,withscores=withscores,desc=True)

    def _get_model_with_max_ucb(self, model_id:str, topN: int):
        res = self.get_best_model_so_far(model_id,topN)
        recommendation_ids = [str(v,encoding='utf-8') for v in res]
        return recommendation_ids

    def select_model(self, model_id,topN, ifexpose='yes',ifrecord= 'no') -> list:
        ucb1_selection = self._get_model_with_max_ucb(model_id,topN=topN)
        if len(ucb1_selection)<1:
            return []
        if ifexpose=='yes':
            
            for item_id in ucb1_selection:
                success_key = "onlinemodel:{0}:{1}:{2}:reward_successes".format('ucb1itemmodel',
                                                                                model_id,
                                                                                item_id)
                _reward = self.r_server.get(success_key)
                if _reward is None:
                    _reward = 0.0
                else:
                    _reward = float(_reward)
                #更新曝光ID的 score
                item_id = str(item_id)
                model_tries,all_model_tries = self._increment_model_tries(item_id, model_id)

                ucb_numerator = 2 * np.log(np.sum(all_model_tries))
                per_model_means = _reward / (model_tries+0.0000001)
                ucb1_estimates = per_model_means + np.sqrt(ucb_numerator / (model_tries+0.0000001))
                
                score_key = "onlinemodel:{0}:{1}:score_ee".format("ucb1itemmodel",model_id)

                with self.r_server.pipeline() as pipe:
                    pipe.zadd(score_key,{item_id: ucb1_estimates})
                    pipe.execute()

        return ucb1_selection

    def reward_model(self, model: str,model_id:str,reward:float=None,init_model="no") -> None:
        success_key = "onlinemodel:{0}:{1}:{2}:reward_successes".format("ucb1itemmodel",
                                                                        model_id,
                                                                        model)
        score_key = "onlinemodel:{0}:{1}:score_ee".format("ucb1itemmodel",model_id)
        key_tries="onlinemodel:{0}:{1}:{2}:tries".format("ucb1itemmodel",model_id,model)
        key_alltries="onlinemodel:ucb1itemmodel:{0}:tries".format(model_id)

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
        all_model_tries = self.r_server.get(key_alltries)
        if all_model_tries is None:
            all_model_tries = 1.0
        else:
            all_model_tries = float(all_model_tries)
        if model_tries is None:
            model_tries = 1.0  
        else:
            model_tries = float(model_tries)
        if init_model == 'yes':
            # 初始化模型时默认曝光
            model_tries+=1
            all_model_tries+=1
            self.r_server.set(key_tries,model_tries)
            self.r_server.set(key_alltries,all_model_tries)
        ucb_numerator = 2 * np.log(np.sum(all_model_tries))
        per_model_means = _reward / (model_tries+0.0000001)
        ucb1_estimates = per_model_means + np.sqrt(ucb_numerator / (model_tries+0.0000001))

        with self.r_server.pipeline() as pipe:
            pipe.zadd(score_key,{model: ucb1_estimates})
            pipe.execute()

        return True
    def filter_model(self,model:str, model_id:str=None):
        score_key = "onlinemodel:{0}:{1}:score_ee".format("ucb1itemmodel",model_id)
        with self.r_server.pipeline() as pipe:
            pipe.zadd(score_key,{model: -100.0})
            pipe.execute()
        return True