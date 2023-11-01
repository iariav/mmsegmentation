from mmengine.registry import LOOPS, HOOKS
from mmengine.runner import BaseLoop, IterBasedTrainLoop
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.utils import is_list_of
from mmengine.evaluator import Evaluator
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch.utils.data import DataLoader
import logging
import time
import bisect
import torch

def calc_dynamic_intervals(
    start_interval: int,
    dynamic_interval_list: Optional[List[Tuple[int, int]]] = None
) -> Tuple[List[int], List[int]]:
    """Calculate dynamic intervals.
    Args:
        start_interval (int): The interval used in the beginning.
        dynamic_interval_list (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    Returns:
        Tuple[List[int], List[int]]: a list of milestone and its corresponding
        intervals.
    """
    if dynamic_interval_list is None:
        return [0], [start_interval]

    assert is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals

class _InfiniteDataloaderIterator:
    """An infinite dataloader iterator wrapper for IterBasedTrainLoop.
    It resets the dataloader to continue iterating when the iterator has
    iterated over all the data. However, this approach is not efficient, as the
    workers need to be restarted every time the dataloader is reset. It is
    recommended to use `mmengine.dataset.InfiniteSampler` to enable the
    dataloader to iterate infinitely.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self._iterator = iter(self._dataloader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self) -> Sequence[dict]:
        try:
            data = next(self._iterator)
        except StopIteration:
            print_log(
                'Reach the end of the dataloader, it will be '
                'restarted and continue to iterate. It is '
                'recommended to use '
                '`mmengine.dataset.InfiniteSampler` to enable the '
                'dataloader to iterate infinitely.',
                logger='current',
                level=logging.WARNING)
            self._epoch += 1
            if hasattr(self._dataloader, 'sampler') and hasattr(
                    self._dataloader.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self._dataloader.sampler.set_epoch(self._epoch)

            elif hasattr(self._dataloader, 'batch_sampler') and hasattr(
                    self._dataloader.batch_sampler.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no batch
                # sampler. batch sampler in pytorch warps the sampler as its
                # attributes.
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self._iterator = iter(self._dataloader)
            data = next(self._iterator)
        return data

@LOOPS.register_module()
class TwoDataloadersIterBasedTrainLoop(BaseLoop):
    """Loop for iter-based training.
    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            DepthDataloader: Union[DataLoader, Dict],
            max_iters: int,
            val_begin: int = 1,
            val_interval: int = 1000,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_iters = int(max_iters)
        assert self._max_iters == max_iters, \
            f'`max_iters` should be a integer number, but get {max_iters}'
        self._max_epochs = 1  # for compatibility with EpochBasedTrainLoop
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.dataloader2 = runner.build_dataloader(DepthDataloader)
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)
        # get the iterator of the dataloader
        self.dataloader_iterator = _InfiniteDataloaderIterator(self.dataloader)
        self.dataloader_iterator2 = _InfiniteDataloaderIterator(self.dataloader2)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            # segmentation batch
            # print('run segmentation batch')
            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            # depth batch
            # print('run depth batch')
            data_batch = next(self.dataloader_iterator2)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.
        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self._iter + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


@LOOPS.register_module()
class TwoDataloadersValLoop(BaseLoop):
    """Loop for validation.
    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 DepthDataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 DepthEvaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader)

        self.DepthDataloader = runner.build_dataloader(DepthDataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore

        if isinstance(DepthEvaluator, (dict, list)):
            self.DepthEvaluator = runner.build_evaluator(DepthEvaluator)  # type: ignore
        else:
            assert isinstance(DepthEvaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(DepthEvaluator)}.')
            self.DepthEvaluator = DepthEvaluator  # type: ignore

        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)

        if hasattr(self.DepthDataloader.dataset, 'metainfo'):
            self.DepthEvaluator.dataset_meta = self.DepthDataloader.dataset.metainfo
            # self.runner.visualizer.dataset_meta = \
            #     self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.DepthDataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        print('Run eval on Depth estimation:')
        for idx, data_batch in enumerate(self.DepthDataloader):
            self.run_iter(idx, data_batch, Depth=True)
        #
        # # compute metrics
        metrics = self.DepthEvaluator.evaluate(len(self.DepthDataloader.dataset))

        print('Run eval on segmentation:')
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch,Depth=False)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        # self.runner.call_hook('after_val_epoch', metrics=metrics)

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict],Depth: bool):
        """Iterate one mini-batch.
        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        # with autocast(enabled=self.fp16):
        outputs = self.runner.model.val_step(data_batch)
        if Depth:
            self.DepthEvaluator.process(data_samples=outputs, data_batch=data_batch)
        else:
            self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        # print(outputs)
        # print(data_batch)
        # print(data_batch['inputs'][0].shape)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)