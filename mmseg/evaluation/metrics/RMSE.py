from typing import Dict, List, Optional, Sequence
from collections import OrderedDict
import torch
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmseg.registry import METRICS
from prettytable import PrettyTable

@METRICS.register_module()
class DepthMetrics(BaseMetric):

    def __init__(self,
                 min_depth: float = 0.1,
                 max_depth: float = 65.0,
                 depth_metrics: List[str] = ['RMSE'],
                 nan_to_num: Optional[int] = None,
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        """
        The metric first processes each batch of data_samples and predictions,
        and appends the processed results to the results list. Then it
        collects all results together from all ranks if distributed training
        is used. Finally, it computes the metrics of the entire dataset.
        """

        self.metrics = depth_metrics
        self.nan_to_num = nan_to_num
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_depth = data_sample['pred_depth']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                gt_depth = data_sample['gt_depth']['data'].squeeze().to(
                    pred_depth)
                self.results.append(
                    self.valid_mask(pred_depth, gt_depth,self.min_depth,self.max_depth))

    @staticmethod
    def valid_mask(pred_depth: torch.tensor, gt_depth: torch.tensor, min_depth: float, max_depth: float):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        gt_depth = gt_depth.cpu().numpy()
        pred_depth = pred_depth.cpu().numpy()

        mask_1 = gt_depth > min_depth
        mask_2 = gt_depth < max_depth
        mask = np.logical_and(mask_1, mask_2)

        gt_depth = gt_depth[mask]
        pred_depth = pred_depth[mask]

        return gt_depth, pred_depth

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()

        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 2

        a1 = []
        a2 = []
        a3 = []
        abs_rel = []
        sq_rel = []
        rmse = []
        rmse_log = []
        silog = []
        log_10 = []

        for i in range(len(results[0])):

            gt = results[0][i]
            pred = results[1][i]

            thresh = np.maximum((gt / pred), (pred / gt))
            a1.append((thresh < 1.25).mean())
            a2.append((thresh < 1.25 ** 2).mean())
            a3.append((thresh < 1.25 ** 3).mean())

            abs_rel.append(np.mean(np.abs(gt - pred) / gt))
            sq_rel.append(np.mean(((gt - pred) ** 2) / gt))

            rmse_t = (gt - pred) ** 2
            rmse.append(np.sqrt(rmse_t.mean()))

            rmse_log_t = (np.log(gt) - np.log(pred)) ** 2
            rmse_log.append(np.sqrt(rmse_log_t.mean()))

            err = np.log(pred) - np.log(gt)
            silog.append(np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100)

            log_10.append((np.abs(np.log10(gt) - np.log10(pred))).mean())

            if isinstance(self.metrics, str):
                metrics = [self.metrics]

        ret_metrics = OrderedDict({'a1': np.mean(a1)})
        ret_metrics['a2'] = np.mean(a2)
        ret_metrics['a3'] = np.mean(a3)
        ret_metrics['abs_rel'] = np.mean(abs_rel)
        ret_metrics['sq_rel'] = np.mean(sq_rel)
        ret_metrics['rmse'] = np.mean(rmse)
        ret_metrics['rmse_log'] = np.mean(rmse_log)
        ret_metrics['silog'] = np.mean(silog)
        ret_metrics['log_10'] = np.mean(log_10)

        ret_metrics = {
            metric: value
            for metric, value in ret_metrics.items()
        }
        if self.nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=self.nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })

        # # summary table
        # class_table_data = PrettyTable()
        # for key, val in ret_metrics.items():
        #     class_table_data.add_column(key, val)
        metrics = dict()
        for key, val in ret_metrics.items():
            metrics[key] = val

        print_log('Depth estimation results:', logger)
        print_log('\n' + 'a1: {:.2f}, a2: {:.2f}, a3: {:.2f}, abs_rel: {:.2f}, rmse: {:.2f}'.format(
            ret_metrics['a1'],ret_metrics['a2'],ret_metrics['a3'],ret_metrics['abs_rel'],ret_metrics['rmse']), logger=logger)

        return metrics