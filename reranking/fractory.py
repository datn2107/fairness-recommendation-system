from reranking.base import ReRankingStrategy
from reranking.u_mmf import UMMFReRanking
from reranking.worst_off import (
    GroupFairnessReRanking,
    WorstOffNumberOfItemReRankingMIP,
    WorstOffNumberOfItemAndGroupFairnessReRankingMIP,
    WorstOfMDGOfItemReRankingMIP,
    WorstOffNumberOfItemReRankingORTools,
)

class ReRankingStrategyFractory:
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
        elif strategy_name == "ummf":
            return UMMFReRanking()
        else:
            raise ValueError(f"Invalid strategy name: {strategy_name}")
