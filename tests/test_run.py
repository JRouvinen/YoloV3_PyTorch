try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    # from mock import patch
    exit()
import pytest
import sys
sys.path.append("YoloV3_PyTorch")
from train import *

class TestRun:

    def get_setup_file(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg",
                            help="Path to model definition file (.cfg)")
        parser.add_argument("-d", "--data", type=str, default="config/coco.data",
                            help="Path to data config file (.data)")
        parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
        parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
        parser.add_argument("--n_cpu", type=int, default=2, help="Number of cpu threads to use during batch generation")
        parser.add_argument("--pretrained_weights", type=str,
                            help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
        parser.add_argument("--checkpoint_interval", type=int, default=5,
                            help="Interval of epochs between saving model weights")
        parser.add_argument("--evaluation_interval", type=int, default=5,
                            help="Interval of epochs between evaluations on validation set")
        parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
        parser.add_argument("--iou_thres", type=float, default=0.1,
                            help="Evaluation: IOU threshold required to qualify as detected")
        parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.3,
                            help="Evaluation: IOU threshold for non-maximum suppression")
        parser.add_argument("--sync_bn", type=int, default=-1,
                            help="Set use of SyncBatchNorm")
        parser.add_argument("--cos_lr", type=int, default=0,
                            help="Set type of scheduler")
        parser.add_argument("--logdir", type=str, default="logs",
                            help="Directory for training log files (e.g. for TensorBoard)")
        parser.add_argument("-g", "--gpu", type=int, default=-1, help="Define which gpu should be used")
        parser.add_argument("--checkpoint_keep_best", type=bool, default=True,
                            help="Should the best checkpoint be saved")
        parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
        args = parser.parse_args()
        return args

    def test_run(self):
        testargs = ["prog", "-m", "tests/configs/Test-tiny_1.cfg", "-d", "tests/configs/Test.data", "-e", "25", "--n_cpu", "2",
                    "--pretrained_weights","weights/yolov3-tiny.weights","--evaluation_interval","3","-g","-1"]
        with patch.object(sys, 'argv', testargs):
            setup = self.get_setup_file()
            assert run(setup) == "Finished training for 25 epochs"
