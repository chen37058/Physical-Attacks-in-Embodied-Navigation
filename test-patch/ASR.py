import os
import numpy as np
import torch as ch
import mitsuba as mi
import drjit as dr
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from PIL import Image

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def use_provided_cam_position(scene_file: str, sensor_key: str) -> np.array:
    scene = mi.load_file(scene_file)
    p = mi.traverse(scene)
    sensors = []
    for key in p.keys():
        if key.endswith('to_world'):
            sensor = p[key]
            sensors.append(sensor)
    return np.array(sensors)


def calculate_attack_success_rate(cfg: DictConfig) -> None:
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("calculate_attack_success_rate")

    # Load scene file
    scene_file = cfg.scene.path
    sensor_key = cfg.scene.sensor_key
    mi.set_variant('cuda_ad_rgb')
    scene = mi.load_file(scene_file)
    params = mi.traverse(scene)

    # Get camera configurations
    camera_positions = use_provided_cam_position(scene_file=scene_file, sensor_key=sensor_key)

    # Initialize counters
    total_configs = len(camera_positions)
    attack_successes = 0

    # Load the object detection model
    dt2_config = get_cfg()
    dt2_config.merge_from_file(cfg.model.config)
    dt2_config.MODEL.WEIGHTS = cfg.model.weights_file
    dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.model.score_thresh_test
    dt2_config.MODEL.DEVICE = 'cuda:0'
    model = build_model(dt2_config)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(dt2_config.MODEL.WEIGHTS)
    model.eval()

    # Prepare for rendering
    spp = 4096  # samples per pixel

    # Prepare target class index
    target_class_idx = 5

    # Prepare directories to save images
    rendered_images_dir = 'test-patch/ASR'
    os.makedirs(rendered_images_dir, exist_ok=True)
    predicted_images_dir = 'test-patch/ASR'
    os.makedirs(predicted_images_dir, exist_ok=True)

    # Get class names for visualization
    metadata = MetadataCatalog.get(dt2_config.DATASETS.TRAIN[0])

    # For each camera configuration
    for idx, camera_position in enumerate(camera_positions):
        # Set the camera position in the scene
        sensor_param_key = f"{sensor_key}.to_world"
        params[sensor_param_key] = camera_position
        params.update()

        # Render the image
        img = mi.render(scene, params=params, spp=spp)
        # Convert image to numpy array
        img_np = np.array(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True))
        # img_np has shape (H, W, 3) in RGB format

        # Save the rendered image
        rendered_image_path = os.path.join(rendered_images_dir, f"rendered_{idx}.png")
        mi.util.write_bitmap(rendered_image_path, data=img, write_async=False)

        # Convert to tensor and move to device
        img_tensor = ch.from_numpy(img_np).permute(2, 0, 1).to(dt2_config.MODEL.DEVICE)
        # Convert to float tensor
        img_tensor = img_tensor.float()

        input = {"image": img_tensor, "height": img_np.shape[0], "width": img_np.shape[1]}

        with ch.no_grad():
            outputs = model([input])
            instances = outputs[0]['instances']
            mask = instances.scores > dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST
            filtered_instances = instances[mask]
            detected_classes = filtered_instances.pred_classes.cpu().numpy()

            # If target class is not detected, consider it as an attack success
            if target_class_idx not in detected_classes:
                attack_successes += 1

            # Visualize and save the predicted image
            img_vis = img_np[:, :, ::-1]  # Convert RGB to BGR for visualization if needed
            visualizer = Visualizer(img_vis, metadata=metadata)
            vis_output = visualizer.draw_instance_predictions(filtered_instances.to('cpu'))
            predicted_image = vis_output.get_image()[:, :, ::-1]  # Convert back to RGB
            predicted_image_path = os.path.join(predicted_images_dir, f"predicted_{idx}.png")
            # Image.fromarray(predicted_image).save(predicted_image_path)

        logger.info(f"Processed image {idx + 1}/{total_configs}, Attack Successes: {attack_successes}")

    # Calculate attack success rate
    attack_success_rate = attack_successes / total_configs
    logger.info(f"Attack Success Rate: {attack_success_rate * 100:.2f}%")
    print(f"Attack Success Rate: {attack_success_rate * 100:.2f}%")


@hydra.main(version_base=None, config_path="/home/disk1/cm/Projects/Physical-Attacks-in-Embodied-Navigation/configs", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    calculate_attack_success_rate(cfg)


if __name__ == "__main__":
    run()
