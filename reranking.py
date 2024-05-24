import numpy as np
from time import time
from typing import Any
from abc import ABC, abstractmethod

from mip import Model, xsum, maximize
from ortools.linear_solver import pywraplp


class ReRankingStrategy(ABC):
    @abstractmethod
    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> Any:
        """
        Optimize the recommend result.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        The optimized result. The return type is depend on the strategy.
            W: The binary matrix of shape (n_users, n_items). The value of W[i][j].x is 1 if the item j is recommended to user i.
            group_items_cnt: (Optional) The number of items in each group.
        """
        raise NotImplementedError


class ReRankingStrategyFractory():
    @staticmethod
    def create(strategy_name: str) -> ReRankingStrategy:
        if strategy_name == "group_fairness":
            return GroupFairnessReRanking()
        elif strategy_name == "worst_off_number_of_item":
            return WorstOffNumberOfItemReRankingMIP()
        elif strategy_name == "worst_off_number_of_item_and_group_fairness":
            return WorstOffNumberOfItemAndGroupFairnessReRankingMIP()
        elif strategy_name == "worst_of_mdg_of_item":
            return WorstOfMDGOfItemReRankingMIP()
        elif strategy_name == "worst_off_number_of_item_or_tools":
            return WorstOffNumberOfItemReRankingORTools()
        else:
            raise ValueError(f"Invalid strategy name: {strategy_name}")


class GroupFairnessReRanking(ReRankingStrategy):
    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> Any:
        """
        Optimize the recommend result with group fairness.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.
        kargs:
            group_items (List[np.ndarray]): The number of items in each group.
            group_items_filter_matrix (np.ndarray): The filter matrix of shape (n_users, n_items, n_group_items).
            epsilon (float): The hyper parameter for the group fairness. Default is 0.1.

        Returns:
        W: The binary matrix of shape (n_users, n_items). The value of W[i][j].x is 1 if the item j is recommended to user i.
        """
        for key in ["group_items", "group_items_filter_matrix"]:
            raise ValueError(f"{key} must be provided in kargs.")
        kargs.setdefault("epsilon", 0.1)

        n_user = S.shape[0]
        n_item = S.shape[1]

        group_items = kargs["group_items"]
        n_group_items = len(group_items)

        assert n_group_items == 2, "The number of groups must be 2."

        model = Model()

        W = [[model.add_var() for j in range(n_item)] for i in range(n_user)]
        group_items_cnt = [model.add_var() for k in range(n_group_items)]

        model.objective = maximize(
            xsum((S[i][j] * W[i][j]) for i in range(n_user) for j in range(n_item))
            - kargs["epsilon"] * (group_items[0] - group_items[1])
        )

        for i in range(n_user):
            model += xsum(W[i][j] for j in range(n_item)) == k

        for k in range(n_group_items):
            model += group_items_cnt[k] == xsum(
                W[i][j] * kargs["group_items_filter_matrix"][i][j][k]
                for i in range(n_user)
                for j in range(n_item)
            )

        for i in range(n_user):
            for j in range(n_item):
                model += W[i][j] <= 1

        model.optimize()

        return W, group_items_cnt


class WorstOffNumberOfItemReRankingMIP(ReRankingStrategy):
    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> np.ndarray:
        """
        Optimize the recommend result with the worst-off number of items.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.
        kargs:
            epsilon (int): The minimum number of recommended times for each item. Default is 15.

        Returns:
        W: The binary matrix of shape (n_users, n_items).
        """
        kargs.setdefault("epsilon", 15)

        n_user = S.shape[0]
        n_item = S.shape[1]

        model = Model()

        W = [[model.add_var() for j in range(n_item)] for i in range(n_user)]

        model.objective = maximize(
            xsum((S[i][j] * W[i][j]) for i in range(n_user) for j in range(n_item))
        )

        for i in range(n_user):
            model += xsum(W[i][j] for j in range(n_item)) == k

        for j in range(n_item):
            model += kargs["epsilon"] <= xsum(W[i][j] for i in range(n_user))

        for i in range(n_user):
            for j in range(n_item):
                model += W[i][j] <= 1

        model.optimize()

        W_np = np.array([[W[i][j].x for j in range(S.shape[1])] for i in range(S.shape[0])])
        return W_np


class WorstOffNumberOfItemAndGroupFairnessReRankingMIP(ReRankingStrategy):
    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> Any:
        """
        Optimize the recommend result with the worst-off number of items and group fairness.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.
        kargs:
            group_items (List[np.ndarray]): The number of items in each group.
            group_items_filter_matrix (np.ndarray): The filter matrix of shape (n_users, n_items, n_group_items).
            g_epsilon (float): The hyper parameter for the group fairness. Default is 0.1.
            i_epsilon (int): The minimum number of recommended times for each item. Default is 20.

        Returns:
        W: The binary matrix of shape (n_users, n_items). The value of W[i][j].x is 1 if the item j is recommended to user i.
        group_items_cnt: The number of items in each group.
        """
        for key in ["group_items", "group_items_filter_matrix"]:
            if key not in kargs:
                raise ValueError(f"{key} must be provided in kargs.")
        kargs.setdefault("g_epsilon", 0.1)
        kargs.setdefault(kargs["i_epsilon"], 20)

        n_user = S.shape[0]
        n_item = S.shape[1]

        group_items = kargs["group_items"]
        n_group_items = len(group_items)

        assert n_group_items == 2, "The number of groups must be 2."

        model = Model()

        W = [[model.add_var() for j in range(n_item)] for i in range(n_user)]
        group_items_cnt = [model.add_var() for k in range(n_group_items)]

        model.objective = maximize(
            xsum((S[i][j] * W[i][j]) for i in range(n_user) for j in range(n_item))
            - kargs["g_epsilon"] * (group_items[0] - group_items[1])
        )

        for i in range(n_user):
            model += xsum(W[i][j] for j in range(n_item)) == k

        for j in range(n_item):
            model += kargs["i_epsilon"] <= xsum(W[i][j] for i in range(n_user))

        for k in range(n_group_items):
            model += group_items_cnt[k] == xsum(
                W[i][j] * kargs["group_items_filter_matrix"][i][j][k]
                for i in range(n_user)
                for j in range(n_item)
            )

        for i in range(n_user):
            for j in range(n_item):
                model += W[i][j] <= 1

        model.optimize()

        return W, group_items_cnt


class WorstOfMDGOfItemReRankingMIP(ReRankingStrategy):
    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> Any:
        """
        Optimize the recommend result with the worst-off mean discounted gain (MDG) of items.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.
        kargs:
            epsilon (float): The minimum MDG for each item. Default is 0.001.

        Returns:
        W: The binary matrix of shape (n_users, n_items). The value of W[i][j].x is 1 if the item j is recommended to user i.
        """
        kargs.setdefault("epsilon", 0.001)

        n_user = S.shape[0]
        n_item = S.shape[1]

        model = Model()

        W = [
            [[model.add_var() for j in range(n_item)] for i in range(n_user)]
            for rank in range(k)
        ]

        model.objective = maximize(
            xsum(
                (S[i][j] * W[rank][i][j])
                for i in range(n_user)
                for j in range(n_item)
                for rank in range(k)
            )
        )

        for i in range(n_user):
            for rank in range(k):
                model += xsum(W[rank][i][j] for j in range(n_item)) == 1
            for j in range(n_item):
                model += xsum(W[rank][i][j] for rank in range(k)) == 1

        for rank in range(k):
            for i in range(n_user):
                for j in range(n_item):
                    model += W[rank][i][j] <= 1

        for j in range(n_item):
            model += kargs["epsilon"] <= xsum(
                W[rank][i][j] * 1 / np.log2(2 + rank)
                for rank in range(k)
                for i in range(n_user)
            )

        model.optimize()

        return W


class WorstOffNumberOfItemReRankingORTools(ReRankingStrategy):
    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> np.ndarray:
        """
        Optimize the recommend result with the worst-off number of items using OR-Tools.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.
        kargs:
            epsilon (int): The minimum number of recommended times for each item. Default is 15.
            strategy_type (str): The strategy type of the solver. Default is "PDLP".

        Returns:
        W: The binary matrix of shape (n_users, n_items).
        """
        kargs.setdefault("strategy_type", "PDLP")
        kargs.setdefault("epsilon", 15)

        solver = pywraplp.Solver.CreateSolver(kargs["strategy_type"])

        if kargs["strategy_type"] == "PDLP":
            termination_criteria_str = "termination_criteria { eps_primal_infeasible: 1e-8 eps_dual_infeasible: 1e-8 }"
            solver.SetSolverSpecificParametersAsString(termination_criteria_str)

        if solver is None:
            return

        n_users, n_items = S.shape[:2]

        x = {}
        for i in range(n_users):
            for j in range(n_items):
                x[i, j] = solver.BoolVar(f"x_{i}_{j}")

        for i in range(n_users):
            solver.Add(sum(x[i, j] for j in range(n_items)) == k)

        for j in range(n_items):
            solver.Add(sum(x[i, j] for i in range(n_users)) >= kargs["epsilon"])

        objective = solver.Objective()
        for i in range(n_users):
            for j in range(n_items):
                objective.SetCoefficient(x[i, j], int(S[i, j]))
        objective.SetMaximization()

        # print("Start solving...")
        solver.Solve()
        # print("Solved!")

        W = np.array([[x[i, j].solution_value() for j in range(n_items)] for i in range(n_users)])
        return W


class UMMFReRanking(ReRankingStrategy):
    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> np.ndarray:
        pass


class ReRanking:
    def __init__(self, strategy: ReRankingStrategy = None):
        self.strategy = strategy

    def set_strategy(self, strategy: ReRankingStrategy):
        self.strategy = strategy

    def optimize(self, S: np.ndarray, k: int = 30, *args, **kargs) -> np.ndarray:
        start_time = time()
        W = self.strategy.optimize(S, k, *args, **kargs)
        total_time = time() - start_time
        print(f"Optimize time: {total_time}")
        return W, total_time

    def apply_reranking_matrix(self, S: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Apply the reranking matrix to the score matrix.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        W (np.ndarray): The binary matrix of shape (n_users, n_items). The value of W[i][j].x is 1 if the item j is recommended to user i.

        Returns:
        np.ndarray: The reranked score matrix of shape (n_users, n_items).
        """
        return (S + 1e-6) * B
