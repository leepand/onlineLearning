import math
import six

from collections import defaultdict
from typing import Any, Dict, Sequence, Optional, cast, Hashable

from onlineLearning.statistics import OnlineVariance
from onlineLearning.learners.base import Learner, Probs, Info,Context, Action
from onlineLearning.pipes import Flatten
from onlineLearning.encodings import InteractionsEncoder
from onlineLearning.exceptions import BanditException
from onlineLearning.storage import Action as Action_cls

class LinUCBLearner(Learner):
    """一个表示预期奖励的Linucb学习器是一个语境(上下文)和行动特征的线性函数。探索是根据置信度上限估计来进行的。
    模型的求解采用`Sherman-Morrison formula`计算矩阵的逆，算法建立在 `Chu et al. (2011) LinUCB algorithm`的基础上，
    该实现的计算复杂度与特征数呈线性关系.
    Remarks:
        The Sherman-Morrsion implementation used below is given in long form `here`__.
    References:
        Chu, Wei, Lihong Li, Lev Reyzin, and Robert Schapire. "Contextual bandits
        with linear payoff functions." In Proceedings of the Fourteenth International
        Conference on Artificial Intelligence and Statistics, pp. 208-214. JMLR Workshop
        and Conference Proceedings, 2011.
    __ https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    __ https://research.navigating-the-edge.net/assets/publications/linucb_alternate_formulation.pdf
    """

    def __init__(self, alpha: float = 0.2, 
                 features: Sequence[str] = [1, 'a', 'ax'],
                 history_storage=None, 
                 model_storage=None, 
                 action_storage=None,
                 recommendation_cls=None,
                 model_id=None) -> None:
        """Instantiate a LinUCBLearner.
        Args:
            alpha: This parameter controls the exploration rate of the algorithm. A value of 0 will cause actions
                to be selected based on the current best point estimate (i.e., no exploration) while a value of inf
                means that actions will be selected based solely on the bounds of the action point estimates (i.e.,
                we will always take actions that have the largest bound on their point estimate).
            features: Feature set interactions to use when calculating action value estimates. Context features
                are indicated by x's while action features are indicated by a's. For example, xaa means to cross the
                features between context and actions and actions.
        """
        #PackageChecker.numpy("LinUCBLearner.__init__")
        super(LinUCBLearner, self).__init__(history_storage, model_storage,
                                     action_storage, recommendation_cls)

        self._alpha = alpha

        self._X = features
        self._model_id = model_id
        self._X_encoder = InteractionsEncoder(features)

    def init_action_model(self, action_list,context=[], model_id = None):

        _action_list =[Action_cls(action) for action in action_list]
        self._action_storage.add(_action_list)
        model = self._model_storage.get_model(model_id)
        save_model=0
        save_model_param=0

        if model is None:
            model = {'A_inv': {}, 'theta': {}, 'action_tries':{}}
            _A_inv = {}
            _theta = {}
            _action_tries = {}
            save_model = 1
        else:
            _A_inv = model.get('A_inv')  # pylint: disable=invalid-name
            _theta = model.get('theta')
            _action_tries = model.get('action_tries')
            if _A_inv is None:
                _A_inv  = {}
                _theta = {}
                _action_tries = {}

        action_wgt = None
        for _action_id in _action_list:
            action_id = _action_id.id
            if isinstance(context, dict):
                _action_wgt = action_wgt.get(action_id,None)
                action_context = np.array([self._X_encoder.encode(x=context[action_id],a=_action_wgt)]).T
            else:
                if action_wgt is None:
                    _action_wgt = 1
                else:
                    _action_wgt = action_wgt.get(action_id,None)
                action_context = np.array([self._X_encoder.encode(x=context,a=_action_wgt)]).T

            if(_A_inv.get(action_id) is None):
                _theta[action_id] = np.zeros((action_context.shape[0],1))
                _A_inv[action_id] = np.identity(action_context.shape[0])
                _action_tries[action_id] = 0
                model["A_inv"][action_id] = _A_inv[action_id]
                model["theta"][action_id] = _theta[action_id]
                model["action_tries"][action_id] = _action_tries[action_id]
                
                save_model_param = 1

        if save_model==1 or save_model_param==1:
            self._model_storage.save_model(model_id=model_id,model=model)
            
        return True

    @property
    def params(self) -> Dict[str, Any]:
        return {'family': 'LinUCB', 
                'alpha': self._alpha, 
                'features': self._X,
                'model_id':self._model_id}

    def _increment_action_tries(self, action: str,model_id: str) -> None:
        # 记录曝光，以便给曝光的action（不反馈的情况）降权
        model = self._model_storage.get_model(model_id)
        _action_tries = model.get('action_tries')
        if _action_tries is None:
            _action_tries = 0
        else:
            _action_tries+=1
        model["action_tries"] =  _action_tries

        self._model_storage.save_model(model_id=model_id,model=model)

    def _linucb_score(self, context,action_wgt=None, model_id=None):
        """disjoint LINUCB algorithm.
        """
        import numpy as np #type: ignore

        model = self._model_storage.get_model(model_id)
        save_model=0
        save_model_param=0
        if model is None:
            model = {'A_inv': {}, 'theta': {}, 'action_tries':{}}
            _A_inv = {}
            _theta = {}
            _action_tries = {}
            save_model = 1
        else:
            _A_inv = model.get('A_inv')  # pylint: disable=invalid-name
            _theta = model.get('theta')
            _action_tries = model.get('action_tries')
            if _A_inv is None:
                _A_inv  = {}
                _theta = {}
                _action_tries = {}

        # The recommended actions should maximize the Linear UCB.
        r = {}
        p = {}
        uncertainty = {}
        for action_id in self._action_storage.iterids():

            if action_wgt is None:
                _action_wgt = 1
            else:
                _action_wgt = action_wgt.get(action_id,None)

            # context可以是dict或list，
            if context is None:
                self._X_encoder = InteractionsEncoder(
                    list(
                        set(
                            filter(
                                None,[ f.replace('x','') if isinstance(f,str) 
                                      else f for f in self._X ]
                            )
                        )
                    )
                )
                action_context = np.array([self._X_encoder.encode(x=context,a=_action_wgt)]).T
            if isinstance(context, dict):
                #_action_wgt = action_wgt.get(action_id,None)
                action_context = np.array([self._X_encoder.encode(x=context[action_id],a=_action_wgt)]).T
            else:
                # Reshape covariates input into (d x 1) shape vector
                action_context = np.array([self._X_encoder.encode(x=context,a=_action_wgt)]).T

            if(_A_inv.get(action_id) is None):
                _theta[action_id] = np.zeros((action_context.shape[0],1))
                _A_inv[action_id] = np.identity(action_context.shape[0])
                _action_tries[action_id] = 0
                model["A_inv"][action_id] = _A_inv[action_id]
                model["theta"][action_id] = _theta[action_id]
                model["action_tries"][action_id] = _action_tries[action_id]
                
                save_model_param = 1

            r[action_id] = float(_theta[action_id].T @ action_context)
            #w = float(np.diagonal(action_context.T @ _A_inv[action_id] @ action_context))
            w = _A_inv[action_id]@ action_context
            v = float(action_context.T@ w)
            uncertainty[action_id] = float(self._alpha*np.sqrt(v))
            # Find ucb based on p formulation (mean + std_dev) 
            # p is (1 x 1) dimension vector, use float transfer to single value
            p[action_id] = r[action_id]+ uncertainty[action_id]
        
            #uncertainty[action_id] = float(self._alpha*np.sqrt(point_bounds))

            #score[action_id] = estimated_reward[action_id] + uncertainty[action_id]

        if save_model==1 or save_model_param==1:
            print("save_model_param",model)
            self._model_storage.save_model(model_id=model_id,model=model)

        return r, uncertainty, p

    def _get_action_with_empty_action_storage(self, context, n_actions):
        if n_actions is None:
            recommendations = None
        else:
            recommendations = []
        history_id = self._history_storage.add_history(context,
                                                       recommendations)
        return history_id, recommendations

    def predict(self, context: Context, 
                actions: Sequence[Action],
                n_actions=None,
                request_id=None,
                model_id = None) -> Probs:
        """Return the action to perform
        Parameters
        ----------
        context : dict
            Contexts {action_id: context} or [context] of different actions.
        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.
        Returns
        -------
        history_id : int
            The history id of the action.
        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        if self._action_storage.count() == 0:
            return self._get_action_with_empty_action_storage(context,
                                                              n_actions)

        if n_actions == -1:
            n_actions = self._action_storage.count()

        estimated_reward, uncertainty, score = self._linucb_score(context,actions,model_id)
        
        if n_actions is None:
            recommendation_id = max(score, key=score.get)
            recommendations = [self._recommendation_cls(
                action=self._action_storage.get(recommendation_id),
                estimated_reward=estimated_reward[recommendation_id],
                uncertainty=uncertainty[recommendation_id],
                score=score[recommendation_id],
            )]
        else:
            recommendation_ids = sorted(score, key=score.get,
                                        reverse=True)[:n_actions]
            recommendations = []  # pylint: disable=redefined-variable-type
            for action_id in recommendation_ids:
                recommendations.append(self._recommendation_cls(
                    action=self._action_storage.get(action_id),
                    estimated_reward=estimated_reward[action_id],
                    uncertainty=uncertainty[action_id],
                    score=score[action_id],
                ))
        #_increment_action_tries(self, action: str,model_id: str)
        return recommendations

    def learn(self, history_id, rewards,model_id=None) -> None:
        """Reward the previous action with reward.
        Parameters
        ----------
        history_id : int
            The history id of the action to reward.
        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        import numpy as np
        #_context = (self._history_storage
        #           .get_unrewarded_history(history_id,model_id)
        #           .context)
        
        #context = list(Flatten().filter([list(context)]))[0] if context else []
        context = [1,2,3]
        # Reshape covariates input into (d x 1) shape vector
        action_context = np.array([self._X_encoder.encode(x=context,a=1)]).T
        #features: np.ndarray = np.array(self._X_encoder.encode(x=context,a=action)).T
        
        # Update the model
        model = self._model_storage.get_model(model_id)
        _A_inv = model["A_inv"]
        _theta = model["theta"]
        _action_tries = model.get('action_tries')

        for action_id, reward in six.viewitems(rewards):
            r = float(_theta[action_id].T @ action_context)
            w = _A_inv[action_id] @ action_context

            v = float(action_context.T @ w)

            # Find A inverse for ridge regression(use Sherman-Morrison Matrix Inverse Update)
            _A_inv[action_id] = _A_inv[action_id] - np.outer(w,w)/(1+v)
            print (v,"ff",((reward-r)/(1+v)),"d",w)
            _theta[action_id] = _theta[action_id] + w.dot((reward-r)/(1+v))
            
        print({
            'A_inv': _A_inv,
            'theta': _theta,
            'action_tries': _action_tries
        })
        self._model_storage.save_model(model_id=model_id,model={
            'A_inv': _A_inv,
            'theta': _theta,
            'action_tries': _action_tries
        })

    def get_models(self,model_id):
        model = self._model_storage.get_model(model_id)
        return model
