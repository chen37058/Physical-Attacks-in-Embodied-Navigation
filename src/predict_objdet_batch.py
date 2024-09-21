"""
Read & render object detection predictions on a batch of images
"""
import os
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from dt2 import save_adv_image_preds, dt2_input
import argparse

dir = 'red_cube'

if __name__ == "__main__":

    parser = argparse.ArgumentParser( \
        description='Example script with default values' \
        ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--input-dir", help="Directory of images to predict on", type=str, default=dir, required=True)
    parser.add_argument("-st", "--scores-thresh", help="ROI scores threshold", type=float, default=0.3)
    args = parser.parse_args()

    # specify an object detector model
    dt2_cfg = get_cfg()
    dt2_cfg.merge_from_file("pretrained-models/faster_rcnn-robust_l2_eps005_imagenet_C2-R_50_FPN_3x/config.yaml")
    dt2_cfg.MODEL.WEIGHTS = "pretrained-models/faster_rcnn-robust_l2_eps005_imagenet_C2-R_50_FPN_3x/model_final.pth"
    dt2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.scores_thresh
    dt2_cfg.MODEL.DEVICE=0
    model = build_model(dt2_cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(dt2_cfg.MODEL.WEIGHTS)    

    # source directory with images we want to predict on
    directory_in_str = f'renders/{args.input_dir}/'
    directory = os.fsencode(directory_in_str)


    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith('.png'):
            # print(filename)
            im_path = os.path.join(directory_in_str, filename)
            input = dt2_input(im_path)
            save_adv_image_preds(model=model, dt2_config=dt2_cfg, input=input, instance_mask_thresh=args.scores_thresh, path=f'preds/{args.input_dir}/{filename}')
