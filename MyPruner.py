import logging

from nni.algorithms.compression.pytorch.pruning.weight_masker import WeightMasker
from schema import And, Optional, SchemaError
from nni.common.graph_utils import TorchModuleGraph
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency
from nni.algorithms.compression.pytorch.pruning.constants import MASKER_DICT
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.compressor import Pruner


__all__ = ['IMPruner']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MyMasker(WeightMasker):
    def __init__(self, model, pruner):
        super().__init__(model, pruner)
        # 此处可初始化，如为算法收集计算权重所需要的统计信息。
        # 如果你的算法需要计算掩码

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        # 根据 wrapper.weight, 和 sparsity,
        # 及其它信息来计算掩码
        model = self.model
        for i in 10:
            for layer in model:
                for weight in layer:
                    if abs(weight) < 5:
                        weight=0
        return {'weight_mask': mask}

class MyPruner(Pruner):
    def __init__(self, model, config_list, optimizer):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        # 创建权重掩码实例
        self.masker = MyMasker(model, self)

    def calc_mask(self, wrapper, wrapper_idx=None):
        sparsity = wrapper.config['sparsity']
        if wrapper.if_calculated:
            # 如果是一次性剪枝算法，不需要再次剪枝
            return None
        else:
            # 调用掩码函数来实际计算当前层的掩码
            masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)
            wrapper.if_calculated = True
            return masks



class OneshotPruner(Pruner):
    """
    Prune model to an exact pruning level for one time.
    """

    def __init__(self, model, config_list, pruning_algorithm='IMP', optimizer=None, **algo_kwargs):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        pruning_algorithm: str
            algorithms being used to prune model
        optimizer: torch.optim.Optimizer
            Optimizer used to train model
        algo_kwargs: dict
            Additional parameters passed to pruning algorithm masker class
        """

        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.masker = MASKER_DICT[pruning_algorithm](
            model, self, **algo_kwargs)

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        """
        schema = CompressorSchema([{
            'sparsity': And(float, lambda n: 0 < n < 1),
            Optional('op_types'): [str],
            Optional('op_names'): [str]
        }], model, logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, wrapper_idx=None):
        """
        Calculate the mask of given layer
        Parameters
        ----------
        wrapper : Module
            the module to instrument the compression operation
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        dict
            dictionary for storing masks, keys of the dict:
            'weight_mask':  weight mask tensor
            'bias_mask': bias mask tensor (optional)
        """
        if wrapper.if_calculated:
            return None

        sparsity = wrapper.config['sparsity']
        if not wrapper.if_calculated:
            masks = self.masker.calc_mask(
                sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)

            # masker.calc_mask returns None means calc_mask is not calculated sucessfully, can try later
            if masks is not None:
                wrapper.if_calculated = True
            return masks
        else:
            return None


class IMPruner(OneshotPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : list
        Supported keys:
            - sparsity : This is to specify the sparsity operations to be compressed to.
            - op_types : Operation types to prune.
    optimizer: torch.optim.Optimizer
            Optimizer used to train model
    """

    def __init__(self, model, config_list, optimizer=None):
        super().__init__(model, config_list, pruning_algorithm='level', optimizer=optimizer)