import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
import terranavalabels.MaterialMorphologyLabels as MaterialMorphologyLabels
import terranavalabels.MaterialLabels as MaterialLabels
# from terranavalabels.MaterialMorphologyLayer import MaterialMorphologyLayer
# from terranavalabels.MaterialLayer import MaterialLayer
from terranavalabels.SemanticExtendedLayer import SemanticExtendedLayer
# from terranavalabels.SemanticLayer import SemanticLayer
from os import path, sep, makedirs
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import classification_report
from DataParamsHandler import DataParamsAPI
import seaborn as sns


def read_image_list_from_txt(data_list, n_lines=-1, offset_line=1):
    images = []
    with open(data_list, 'r') as f:
        for line_index, line in enumerate(f.readlines(), 1):
            if line_index < offset_line:
                continue
            if n_lines != -1 and line_index >= n_lines + offset_line:
                break
            images.append(line.strip("\n"))
    return list(sorted(images))


def get_boarders(binary_mask, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    dilation = cv2.dilate(binary_mask, kernel, iterations=1)
    return dilation - binary_mask


def get_borders_mask(curr_mask, curr_layer, verbose=False):
    borders_masks = []

    for label in curr_layer.labels(valid=True):

        binary_mask = np.array(curr_mask == label.label_id_for_training).astype(np.uint8)

        if np.sum(binary_mask) == 0:
            if verbose:
                print('no pxls for {}'.format(label.label_name))
            continue

        borders = get_boarders(binary_mask, kernel_size=(5, 5))
        borders_masks.append(borders)

    return (np.logical_or.reduce(borders_masks)).astype(np.uint8)


def convert_mask(curr_mask, curr_layer):
    curr_mask_converted = np.zeros_like(curr_mask)
    for label in curr_layer.labels(valid=True):
        curr_mask_converted[curr_mask == label.label_id] = label.label_id_for_training
    return curr_mask_converted


class Experiment:

    def __init__(self, name, layer, data_list, preds_dir):
        self.name = name
        self.layer = layer
        self.gt_paths = read_image_list_from_txt(data_list=data_list)
        self.preds_dir = preds_dir


verbose = False
plot = False
erode_borders = False


dst_results_dir = '/data/DATA/AutoTerrain/SegmentationEvaluation/MadaanNew/ResultsNoErode'
dst_report_path = path.join(dst_results_dir, 'report.txt')
dst_confusion_matrix_path = path.join(dst_results_dir, 'CM_{}.png')

makedirs(dst_results_dir, exist_ok=True)


experiment = Experiment(
    name='Semantic-Flat-Original-Forget',
    layer=SemanticExtendedLayer,
    data_list='/data/TerraNavaRepositories/TerraNavaSegmentationDeploy/scripts/semantics_gt_mat.txt',
    preds_dir='/data/DATA/AutoTerrain/SegmentationEvaluation/MadaanNew/2023-02-21_20-04-41_miou-0.495/original/prediction/1/'
)


with open(dst_report_path, 'w+') as report_file:

    print('Generate report for {}..'.format(experiment.name))

    report_file.write('{}:\n\n'.format(experiment.name))

    dataset_api = DataParamsAPI.by_name(experiment.layer.layer_hierarchy)
    relevant_classes = dataset_api.classes
    relevant_classes_names = [dataset_api.i2l[curr_class] for curr_class in relevant_classes]
    num_classes = len(relevant_classes)

    preds_paths = [path.join(experiment.preds_dir, x.split(sep)[-1]) for x in experiment.gt_paths]

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

    all_labels_gt = []
    all_labels_preds = []

    for pred_path, gt_path in zip(preds_paths, experiment.gt_paths):

        if verbose:
            print(pred_path)
            print(gt_path)

        pred_np = io.imread(pred_path)
        gt_np = io.imread(gt_path)

        pred_np_converted = convert_mask(curr_mask=pred_np,
                                         curr_layer=experiment.layer)

        if erode_borders:
            borders_final_mask = get_borders_mask(curr_mask=gt_np,
                                                  curr_layer=experiment.layer,
                                                  verbose=verbose)
            gt_np[borders_final_mask == 1] = 255
            pred_np_converted[borders_final_mask == 1] = 255

        if experiment.layer.layer_hierarchy == 'MaterialMorphology':
            gt_np[gt_np == MaterialMorphologyLabels.Unclassified.label_id] = 255
            pred_np_converted[gt_np == MaterialMorphologyLabels.Unclassified.label_id] = 255

        if experiment.layer.layer_hierarchy == 'Material':
            gt_np[gt_np == MaterialLabels.Unclassified.label_id] = 255
            pred_np_converted[gt_np == MaterialLabels.Unclassified.label_id] = 255

        diff_mask_np = np.array(gt_np != pred_np_converted).astype(np.uint8)

        confusion_matrix += sklearn_confusion_matrix(gt_np.ravel(), pred_np_converted.ravel(), relevant_classes)

        all_labels_gt.extend(gt_np.ravel())
        all_labels_preds.extend(pred_np_converted.ravel())

        if plot:
            plt.figure('pred_np_converted')
            plt.imshow(pred_np_converted)
            plt.figure('gt_np')
            plt.imshow(gt_np)
            plt.figure('diff_mask_np')
            plt.imshow(diff_mask_np)
            plt.show()

    labels_counts = np.sum(confusion_matrix, axis=1)

    valid_classes = np.argwhere(labels_counts != 0)
    invalid_classes = np.argwhere(labels_counts == 0)

    relevant_classes_names = [relevant_classes_names[xx] for xx in np.squeeze(valid_classes, axis=-1)]

    for xx in sorted(np.squeeze(invalid_classes, axis=-1), reverse=True):
        confusion_matrix = np.delete(confusion_matrix, obj=xx, axis=0)
        confusion_matrix = np.delete(confusion_matrix, obj=xx, axis=1)

    normalized_confusion_matrix_recall = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    overall_true_pxls = np.sum(confusion_matrix.diagonal())
    overall_total_pxls = np.sum(confusion_matrix)

    classification_report_str = classification_report(np.array(all_labels_gt),
                                                      np.array(all_labels_preds),
                                                      labels=np.squeeze(valid_classes, axis=-1),
                                                      target_names=relevant_classes_names)

    report_file.write('  Final Acc = {:.2f}\n'.format(overall_true_pxls / overall_total_pxls))
    report_file.write(classification_report_str)
    report_file.write('\n\n\n')

    plt.figure(figsize=(20, 10))

    sns.heatmap(normalized_confusion_matrix_recall,
                annot=True,
                linewidths=0.01,
                fmt='.2f',
                xticklabels=relevant_classes_names,
                yticklabels=relevant_classes_names)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(dst_confusion_matrix_path.format(experiment.name))
    plt.close(fig='all')
