import math

from collections import defaultdict
from typing import Any, Dict, Sequence, Optional, cast, Hashable

from onlineLearning.statistics import OnlineVariance
from onlineLearning.learners.base import Learner, Probs, Info,Context, Action
from onlineLearning.pipes import Flatten
from onlineLearning.encodings import InteractionsEncoder
from onlineLearning.exceptions import BanditException


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

        self._theta = None
        self._A_inv = None
        for action_id in self._action_storage.iterids():
            print(action_id)
            #self._init_action_model(model, action_id,model_id)

    @property
    def params(self) -> Dict[str, Any]:
        return {'family': 'LinUCB', 
                'alpha': self._alpha, 
                'features': self._X,
                'model_id':self._model_id}

    def _linucb_score(self, context,model_id=None):
        """disjoint LINUCB algorithm.
        """
        model = self._model_storage.get_model(model_id)

        A_inv = model['A_inv']  # pylint: disable=invalid-name
        theta = model['theta']

        # The recommended actions should maximize the Linear UCB.
        estimated_reward = {}
        uncertainty = {}
        score = {}
        for action_id in self._action_storage.iterids():
            action_context = np.reshape(context[action_id], (-1, 1))
            estimated_reward[action_id] = float(
                theta[action_id].T.dot(action_context))
            uncertainty[action_id] = float(
                self.alpha * np.sqrt(action_context.T
                                     .dot(A_inv[action_id])
                                     .dot(action_context)))
            score[action_id] = (estimated_reward[action_id]
                                + uncertainty[action_id])
        return estimated_reward, uncertainty, score

    def predict(self, context: Context, 
                actions: Sequence[Action],
                n_actions=None,
                request_id=None) -> Probs:

        import numpy as np #type: ignore

        if isinstance(actions[0], dict) or isinstance(context, dict):
            raise BanditException("Sparse data cannot be handled by this algorithm.")

        if not context:
            self._X_encoder = InteractionsEncoder(list(set(filter(None,[ f.replace('x','') if isinstance(f,str) else f for f in self._X ]))))
            print(self._X_encoder)
            
        context = list(Flatten().filter([list(context)]))[0] if context else []
        features: np.ndarray = np.array([self._X_encoder.encode(x=context,a=action)]).T

    def learn(self) -> None:
        pass
