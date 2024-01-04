# Main module for impelementing the code

import argparse
import os
import torch
from tqdm import tqdm

from .dataset import voc
from .model.faster_rcnn import FasterRCNN
from .model import vgg16
from .model import resnet
from .statistics import TrainingStatistics
from .statistics import PrecisionRecallCurveCalculator
from . import state
from . import utils
from . import visualize

def create_optimizer(model):
    params = []
    for key, value in dict(model.named_parameters()).items():
        if not value.requires_grad:
            continue
        if "weight" in key:
            params += [{"params": [value], "weight_decay": options.weight_decay}]
    return torch.optim.SGD(params, lr=options.learning_rate, momentum=options.momentum)
    
def train(model):
    """
    Train mode
    """
    # Set initial weights
    if options.load_from:
        initial_weights = options.load_from
    else:
        initial_weights = "IMAGENET1K_V1"
            
    # Print training paramenters
    print("Training Parameters")
    print("-------------------")
    print("Initial weights   : %s" % initial_weights)
    print("Dataset           : %s" % options.dataset_dir)
    print("Training split    : %s" % options.train_split)
    print("Evaluation split  : %s" % options.eval_split)
    print("Backbone          : %s" % options.backbone)
    print("Epochs            : %d" % options.epochs)
    print("Learning rate     : %f" % options.learning_rate)
    print("Momentum          : %f" % options.momentum)
    print("Weight decay      : %f" % options.weight_decay)
    print("Dropout           : %f" % options.dropout)
    print("Augmentation      : %s" % ("disabled" if options.no_augment else "enabled"))
    print("Edge proposals    : %s" % ("excluded" if options.exclude_edge_proposals else "included"))
    print("Final weights file: %s" % ("none" if not options.save_final_to else options.save_final_to))
    print("Best weights file : %s" % ("none" if not options.save_best_to else options.save_best_to))
    
    # Build train dataset
    train_data = voc.VOCDataset(
        split=options.train_split,
        image_preprocess_params=model._backbone.image_preprocess_params,
        get_backbone_shape=model._backbone.get_backbone_shape,
        feature_pixels=model._backbone.feature_pixels,
        dir=options.dataset_dir,
        augment=not options.no_augment,
        shuffle=True
    )
    
    #Build optimizer
    optimizer = create_optimizer(model=model)
    
    # Set best model tracker
    if options.save_best_to:
        best_weights_tracker = state.BestWeightsTracker(filepath=options.save_best_to)
        
    # Training
    for epoch in range(1, 1 + options.epochs):
        print("Epoch %d/%d" % (epoch, options.epochs))
        stats = TrainingStatistics()
        progbar = tqdm(iterable=iter(train_data), total=train_data._sample_num, postfix=stats.get_progbar_postfix())
        for sample in progbar:
            # One training step on a sample
            loss = model.train_step(
                image_data=torch.from_numpy(sample.image_data).unsqueeze(dim=0).cuda(),
                optimizer=optimizer,
                anchors=sample.anchors,
                anchors_vaild_map=sample.anchors_valid_map,
                gt_rpn_map=torch.from_numpy(sample.gt_rpn_map).unsqueeze(dim=0).cuda(),
                gt_rpn_object_indices=[sample.gt_rpn_object_indices],
                gt_rpn_background_indices=[sample.gt_rpn_background_indices],
                gt_bboxes=[sample.gt_bboxes]
            )
            stats.on_training_step(loss = loss)
            progbar.set_postfix(stats.get_progbar_postfix())
        last_epoch = epoch == options.epochs
        
        # Evaluate some samples in train dataset
        mean_average_precision_train = evaluate(
            model=model,
            split="train",
            sample_num=options.periodic_eval_samples
        )
        
        # Evaluate all samples in validation dataset to check the results of this epoch
        mean_average_precision_val = evaluate(
            model=model,
            split="val"
        )
        # Update best model weights
        if options.save_best_to:
            best_weights_tracker.on_epoch_end(model=model, epoch=epoch, mAP=mean_average_precision_val)
    
    # Record final and best results after training
    if options.save_final_to:
        torch.save({ "epoch": epoch, "model_state_dict": model.state_dict() }, options.save_to)
        print("Saved final model weights to '%s'" % options.save_to)
    if options.save_best_to:
        best_weights_tracker.save_best_weights(model=model)
        
    # Evaluate final or best model on train dataset
    print("Evaluating %s model on all samples in '%s'..." % (("best" if options.save_best_to else "final"), options.train_split))
    evaluate(
        model=model,
        split="train"
    )
    # Evaluate final or best model on validation dataset
    print("Evaluating %s model on all samples in '%s'..." % (("best" if options.save_best_to else "final"), options.eval_split))
    evaluate(
        model=model,
        split="val"
    )
    
def evaluate(model, split, sample_num=None):
    """
    Evaluate model
    """
    
    # Build evaluate dataset
    eval_data = voc.VOCDataset(
        split=split,
        image_preprocess_params=model._backbone.image_preprocess_params,
        get_backbone_shape=model._backbone.get_backbone_shape,
        feature_pixels=model._backbone.feature_pixels,
        dir=options.dataset_dir,
        augment=False,
        shuffle=False
    )
    # If no assigned number, evalute all samples
    if sample_num is None:
        sample_num = eval_data._sample_num
        
    # Evaluating
    precision_recall_curve = PrecisionRecallCurveCalculator()
    i = 0
    print("Evaluating '%s'..." % eval_data._split)
    for sample in tqdm(iterable=iter(eval_data), total=sample_num):
        # Predict on one sample
        scored_bboxes_by_class_id = model.predict(
            image_data=torch.from_numpy(sample.image_data).unsqueeze(dim=0).cuda(),
            score_threshold=0.05  # Lower threshold for evaluation
        )
        precision_recall_curve.add_image_results(
            scored_bboxes_by_class_id=scored_bboxes_by_class_id,
            gt_boxes=sample.gt_bboxes
        )
        i += 1
        if i >= sample_num:
            break
    
    # Print ap of every class
    precision_recall_curve.print_average_precisions(class_id_to_name=voc.VOCDataset.class_id_to_name)
    # Compute map
    mean_average_precision = 100.0 * precision_recall_curve.compute_mean_average_precision()
    print("Mean Average Precision = %1.2f%%" % mean_average_precision)

    return mean_average_precision

def predict(model, split):
    """
    Predict mode
    """
    
    from .dataset import image
    
    # Build directory
    dir_name = "predictions_" + split
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    print("Rendering predictions from '%s' set to '%s'..." % (split, dir_name))
    
    # Build predict directory
    predict_data = voc.VOCDataset(
        split=split,
        image_preprocess_params=model._backbone.image_preprocess_params,
        get_backbone_shape=model._backbone.get_backbone_shape,
        feature_pixels=model._backbone.feature_pixels,
        dir=options.dataset_dir,
        augment=False,
        shuffle=False
    )
    
    # Predictng
    for sample in iter(predict_data):
        save_path = os.path.join(dir_name, os.path.splitext(os.path.basename(sample.filepath))[0] + ".png")
        image_data = torch.from_numpy(sample.image_data).unsqueeze(dim=0).cuda()
        # Predict on one sample
        scored_bboxes_per_class = model.predict(image_data=image_data, score_threshold=0.7)
        # Save visualize results
        visualize.draw_detections(
            save_path=save_path,
            image=sample.image,
            scored_bboxes_by_class_id=scored_bboxes_per_class,
            class_id_to_name=voc.VOCDataset.class_id_to_name
        )
    
if __name__ == "__main__":
    
    # Argument for using this code
    parser = argparse.ArgumentParser("code")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action = "store_true")
    group.add_argument("--eval", action = "store_true")
    group.add_argument("--predict", action = "store_true")
    parser.add_argument("--load-from", metavar = "file", action = "store")
    parser.add_argument("--backbone", metavar = "model", action = "store", default = "vgg16")
    parser.add_argument("--save-final-to", metavar = "file", action = "store")
    parser.add_argument("--save-best-to", metavar = "file", action = "store")
    parser.add_argument("--dataset-dir", metavar = "dir", action = "store", default = "data")
    parser.add_argument("--train-split", metavar = "name", action = "store", default = "train")
    parser.add_argument("--eval-split", metavar = "name", action = "store", default = "val")
    parser.add_argument("--predict-split", metavar = "name", action = "store", default = "val")
    parser.add_argument("--periodic-eval-samples", metavar = "count", action = "store", default = 100)
    parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1)
    parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3)
    parser.add_argument("--momentum", metavar = "value", type = float, action = "store", default = 0.9)
    parser.add_argument("--weight-decay", metavar = "value", type = float, action = "store", default = 5e-4)
    parser.add_argument("--dropout", metavar = "probability", type = float, action = "store", default = 0.0)
    parser.add_argument("--no-augment", action = "store_true")
    parser.add_argument("--exclude-edge-proposals", action = "store_true")
    options = parser.parse_args()
    
    # Set backnone
    vaild_backbones = ["vgg16", "resnet50", "resnet101", "resnet152"]
    assert options.backbone in vaild_backbones
    if options.backbone == "vgg16":
        backbone = vgg16.VGG16Backbone(dropout=options.dropout)
    elif options.backbone == "resnet50":
        backbone = resnet.ResnetBackbone(architecture=resnet.Architecture.resnet50)
    elif options.backbone == "resnet101":
        backbone = resnet.ResnetBackbone(architecture=resnet.Architecture.resnet101)
    elif options.backbone == "resnet152":
        backbone = resnet.ResnetBackbone(architecture=resnet.Architecture.resnet152)
        
    # Constrcut model and load initial weights
    model = FasterRCNN(
        class_num=voc.VOCDataset.class_num,
        backbone=backbone,
        exclude_edge_proposals=options.exclude_edge_proposals
    ).cuda()
    if options.load_from:
        state.load(model=model, filepath=options.load_from)
        
    # Train
    if options.train:
        train(model=model)
    
    # Evaluate
    if options.eval:
        evaluate(model=model, split=options.eval_split)
        
    # Predict all images
    if options.predict:
        predict(model=model, split=options.predict_split)