import os, PIL, csv
import gc
import numpy as np
import torch as ch
from torchvision.io import read_image
import mitsuba as mi
import drjit as dr
import time
from omegaconf import DictConfig
import logging

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage
from detectron2.data.detection_utils import read_image
from detectron2.structures import Instances
from detectron2.data.detection_utils import *

def dt2_input(image_path:str)->dict:
    """
    Construct a Detectron2-friendly input for an image
    """
    input = {}
    filename = image_path
    adv_image = read_image(image_path, format="RGB")
    adv_image_tensor = ch.as_tensor(np.ascontiguousarray(adv_image.transpose(2, 0, 1)))
    height = adv_image_tensor.shape[1]
    width = adv_image_tensor.shape[2]
    instances = Instances(image_size=(height,width))
    instances.gt_classes = ch.Tensor([2])
    instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, float(height), float(width)]]))
    input['image'] = adv_image_tensor    
    input['filename'] = filename
    input['height'] = height
    input['width'] = width
    input['instances'] = instances
    return input

def save_adv_image_preds(model \
    , dt2_config \
    , input \
    , instance_mask_thresh=0.7 \
    , target:int=None \
    , untarget:int=None
    , is_targeted:bool=True \
    , format="RGB" \
    , path:str=None):
    """
    Helper fn to save the predictions on an adversarial image
    attacked_image:ch.Tensor An attacked image
    instance_mask_thresh:float threshold pred boxes on confidence score
    path:str where to save image
    """ 
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dt2")
    model.train = False
    model.training = False

    model.proposal_generator.training = False
    model.roi_heads.training = False    
    with ch.no_grad():
        adv_outputs = model([input])
        # print(adv_outputs[0]['instances']._fields['pred_boxes'])
        perturbed_image = input['image'].data.permute((1,2,0)).detach().cpu().numpy()
        pbi = ch.tensor(perturbed_image, requires_grad=False).detach().cpu().numpy()
        if format=="BGR":
            pbi = pbi[:, :, ::-1]
        v = Visualizer(pbi, MetadataCatalog.get(dt2_config.DATASETS.TRAIN[0]),scale=1.0)
        instances = adv_outputs[0]['instances']
        # things = np.array(MetadataCatalog.get(dt2_config.DATASETS.TRAIN[0]).thing_classes) # holds class labels
        categories = ['chair', 'sofa', 'plant', 'bed', 'toilet', 'tv_monitor', 'fireplace', 'bathtub', 'mirror']
        things = np.array(categories)

        predicted_classes = things[instances.pred_classes.cpu().numpy().tolist()] 
        print(f'Predicted Class: {predicted_classes}')        
        mask = instances.scores > instance_mask_thresh
        instances = instances[mask]
        out = v.draw_instance_predictions(instances.to("cpu"))
        target_pred_exists = target in instances.pred_classes.cpu().numpy().tolist()
        untarget_pred_not_exists = untarget not in instances.pred_classes.cpu().numpy().tolist()
        pred = out.get_image()
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True  
    PIL.Image.fromarray(pred).save(path)
    if is_targeted and target_pred_exists:
        return True
    elif (not is_targeted) and (untarget_pred_not_exists):
        return True
    return False

def use_provided_cam_position(scene_file: str, sensor_key:str) -> np.array:
    scene = mi.load_file(scene_file)
    p = mi.traverse(scene)
    sensors = []
    for key in p.keys():
        if key.endswith('to_world'):
            sensor = p[key]
            sensors.append(sensor)
    return np.array(sensors)

def attack_dt2(cfg:DictConfig) -> None:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", default=1)
    DEVICE = "cuda:0"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dt2")
    batch_size = cfg.attack.batch_size
    eps = cfg.attack.eps
    eps_step =  cfg.attack.eps_step
    targeted =  cfg.attack.targeted
    target_class = cfg.attack.target_idx
    target_string = cfg.attack_class
    untargeted_class = cfg.attack.untarget_idx
    untargeted_string = cfg.untargeted_class
    iters = cfg.attack.iters
    spp = cfg.attack.samples_per_pixel
    multi_pass_rendering = cfg.attack.multi_pass_rendering
    multi_pass_spp_divisor = cfg.attack.multi_pass_spp_divisor
    scene_file = cfg.scene.path
    param_keys = cfg.scene.target_param_keys
    sensor_key = cfg.scene.sensor_key
    score_thresh = cfg.model.score_thresh_test
    weights_file = cfg.model.weights_file 
    model_config = cfg.model.config
    randomize_sensors = cfg.scenario.randomize_positions 
    scene_file_dir = os.path.dirname(scene_file)
    tex_paths = cfg.scene.textures
    multicam = cfg.multicam
    tmp_perturbation_path = os.path.join(f"{scene_file_dir}",f"textures/{target_string}_tex","tmp_perturbations")
    if os.path.exists(tmp_perturbation_path) == False:
        os.makedirs(tmp_perturbation_path)
    render_path = os.path.join(f"renders",f"{target_string}")
    if os.path.exists(render_path) == False:
        os.makedirs(render_path)
    preds_path = os.path.join("preds",f"{target_string}")
    if os.path.exists(preds_path) == False:
        os.makedirs(preds_path)    
    if multi_pass_rendering:
        logger.info(f"Using multi-pass rendering with {spp//multi_pass_spp_divisor} passes")
    mi.set_variant('cuda_ad_rgb')
    for tex in tex_paths:
        mitsuba_tex = mi.load_dict({
            'type': 'bitmap',
            'id': 'heightmap_texture',
            'filename': tex,
            'raw': True
        })
        mt = mi.traverse(mitsuba_tex)
    # FIXME - allow variant to be set in the configuration.
    scene = mi.load_file(scene_file)
    p = mi.traverse(scene)    
    k = param_keys
    keep_keys = [k for k in param_keys]
    k1 = f'{sensor_key}.to_world'
    k2 = f'{sensor_key}.film.size'
    keep_keys.append(k1)
    p.keep(keep_keys)
    p.update()
    orig_texs = []
    moves_matrices = use_provided_cam_position(scene_file=scene_file, sensor_key=sensor_key)
    if randomize_sensors:
        np.random.shuffle(moves_matrices)
    # load pre-trained robust faster-rcnn model
    dt2_config = get_cfg()
    dt2_config.merge_from_file(model_config)
    dt2_config.MODEL.WEIGHTS = weights_file
    dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    # FIXME - Get GPU Device form environment variable.
    dt2_config.MODEL.DEVICE = DEVICE
    model = build_model(dt2_config)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(dt2_config.MODEL.WEIGHTS)
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True

    def optim_batch(scene, batch_size, camera_positions, spp, k, label, unlabel, iters, alpha, epsilon, targeted=False):
        if targeted:
            assert(label is not None)
        print('len(camera_positions)=',len(camera_positions))
        assert(batch_size <= len(camera_positions))
        success = False
        # wrapper function that models the input image and returns the loss
        # TODO - 2 model input should accept a batch
        @dr.wrap_ad(source='drjit', target='torch')
        def model_input(x, target, gt_boxes):
            losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]
            target_loss_idx = [0,1,2,3]  # Target only `loss_cls` loss
            loss_weights = [0.7,0.1,0.1,0.1]  # Updated loss weights
            x = ch.permute(x, (0, 3, 1, 2)).requires_grad_()
            x.retain_grad()
            height = x.shape[2]
            width = x.shape[3]
            instances = Instances(image_size=(height, width))
            instances.gt_classes = target.long()
            inputs = []
            for i in range(x.shape[0]):
                instances.gt_boxes = Boxes(ch.tensor([gt_boxes[i]]))
                print(instances.gt_boxes)
                input = {'image': x[i], 'filename': '', 'height': height, 'width': width, 'instances': instances}
                inputs.append(input)
            with EventStorage(0) as storage:
                losses = model(inputs) 
                # 更新损失权重
                weighted_losses = [losses[losses_name[tgt_idx]] * weight for tgt_idx, weight in zip(target_loss_idx, loss_weights)]
                loss = sum(weighted_losses).requires_grad_()
            del x
            return loss

        params = mi.traverse(scene)
        for k in param_keys:
            if isinstance(params[k], dr.cuda.ad.TensorXf):
                # use Float if dealing with just texture colors (not a texture map)
                orig_tex = dr.cuda.ad.TensorXf(params[k])
            elif isinstance(params[k], dr.cuda.ad.Float):
                orig_tex = dr.cuda.ad.Float(params[k])        
            else:
                raise Exception("Unrecognized Differentiable Parameter Data Type.  Should be one of dr.cuda.ad.Float or dr.cuda.ad.TensorXf")
            orig_tex.set_label_(f"{k}_orig_tex")
            orig_texs.append(orig_tex)
        # indicate sensors to use in producing the perturbation
        # e.g., [0,1,2,3] will use sensors 0-3 focus on Taxi/Cement Truck in 'intersection_taxi.xml'
        # sensor 10 is focused on stop sign.
        sensors = [0]
        if iters % len(sensors) != 0:
            print("uneven amount of iterations provided for sensors! Some sensors will be used more than others\
                during attack")
        # if only one camera in the scene, then this idx will be repeated for each iter
        camera_idx = ch.Tensor(np.array(sensors)).repeat(int(iters/len(sensors))).to(dtype=ch.uint8).numpy().tolist()
        # one matrix per camera position that we want to render from, equivalent to batch size
        # e.g., batch size of 5 = 5 required camera positions
        cam_idx = 0
        current_gt_box = [0.0, 0.0, float(512), float(512)]
        # current_gt_box = [428.0, 224.5, float(148), float(151)]
        # 跳过的相机位置索引
        skipped_camera_indices = [] 
        # 初始化相机位置的迭代次数为0
        iter_counts = {i: 0 for i in range(len(moves_matrices))}
        for it in range(iters):
            # keep 2 sets of parameters because we only need to differentiate wrt texture
            diff_params = mi.traverse(scene)
            non_diff_params = mi.traverse(scene)
            diff_params.keep([k for k in param_keys])
            non_diff_params.keep([k1,k2])
            # optimizer is not used but necessary to instantiate to get gradients from diff rendering.
            opt = mi.ad.Adam(lr=0.1, params=diff_params)
            for i,k in enumerate(param_keys):
                dr.enable_grad(orig_texs[i])
                dr.enable_grad(opt[k])
                opt[k].set_label_(f"{k}_bitmap")
            # sample random camera positions (=batch size) for each batch iteration
            if camera_positions.size > 1:
                np.random.seed(it+1)
                sampled_camera_positions_idx = np.random.randint(low=0, high=len(camera_positions)-1,size=batch_size)
            else: sampled_camera_positions_idx = [0]
            if batch_size > 1:
                sampled_camera_positions = camera_positions[sampled_camera_positions_idx]
            else:
                sampled_camera_positions = camera_positions
            if success:
                cam_idx += 1
                logger.info(f"Successful pred, using camera_idx {cam_idx}")
            N, H, W, C = batch_size, non_diff_params[k2][0], non_diff_params[k2][1], 3
            imgs = dr.empty(dr.cuda.ad.Float, N * H * W * C)
            # gt_boxed
            detected_boxes = []
            for b in range(0, batch_size):
                gc.collect()
                ch.cuda.empty_cache()
                # EOT Strategy
                # set the camera position, render & attack
                if cam_idx > len(sampled_camera_positions)-1:
                    logger.info(f"Successfull detections on all {len(sampled_camera_positions)} positions.")
                    return
                if batch_size > 1: # sample from random camera positions
                    cam_idx = b
                if isinstance(sampled_camera_positions[cam_idx], mi.cuda_ad_rgb.Transform4f):
                    non_diff_params[k1].matrix = sampled_camera_positions[cam_idx].matrix
                else:
                    non_diff_params[k1].matrix = mi.cuda_ad_rgb.Matrix4f(sampled_camera_positions[cam_idx])
                non_diff_params.update()
                params.update(opt)        
                prb_integrator = mi.load_dict({'type': 'prb'})
                if multi_pass_rendering:
                    # achieve the affect of rendering at a high sample-per-pixel (spp) value 
                    # by rendering multiple times at a lower spp and averaging the results
                    # render_passes = 16 # TODO - make this a config param
                    mini_pass_spp = spp//multi_pass_spp_divisor
                    render_passes = mini_pass_spp
                    mini_pass_renders = dr.empty(dr.cuda.ad.Float, render_passes * H * W * C)
                    for i in range(render_passes):
                        seed = np.random.randint(0,1000)+i
                        img_i =  mi.render(scene, params=params, spp=mini_pass_spp, sensor=camera_idx[it], seed=seed, integrator=prb_integrator)
                        s_index = i * (H * W * C)
                        e_index = (i+1) * (H * W * C)
                        mini_pass_index = dr.arange(dr.cuda.ad.UInt, s_index, e_index)
                        img_i = dr.ravel(img_i)
                        dr.scatter(mini_pass_renders, img_i, mini_pass_index)
                    @dr.wrap_ad(source='drjit', target='torch')
                    def stack_imgs(imgs):
                        imgs = imgs.reshape((render_passes, H, W, C))
                        imgs = ch.mean(imgs,axis=0)
                        return imgs
                    mini_pass_renders = dr.cuda.ad.TensorXf(mini_pass_renders, dr.shape(mini_pass_renders))
                    img = stack_imgs(mini_pass_renders)
                else: # dont use multi-pass rendering
                    img =  mi.render(scene, params=params, spp=spp, sensor=camera_idx[it], seed=it+1, integrator=prb_integrator)
                img.set_label_(f"image_b{it:03d}_s{b:03d}")
                rendered_img_path = os.path.join(render_path,f"render_b{it:03d}.png")
                mi.util.write_bitmap(rendered_img_path, data=img, write_async=False)
                #########################################################################
                # gt_boxes
                rendered_img_input = dt2_input(rendered_img_path)
                model.train = False
                model.training = False
                model.proposal_generator.training = False
                model.roi_heads.training = False   
                outputs = model([rendered_img_input])
                instances = outputs[0]['instances']
                if targeted:
                    mask = (instances.scores > dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST) & (instances.pred_classes == target_class)
                else: 
                    mask = (instances.scores > dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST) & (instances.pred_classes == untargeted_class)
                filtered_instances = instances[mask]
                if (len(filtered_instances) > 0 and success) or (len(filtered_instances) > 0 and it == 0):
                    current_gt_box = filtered_instances.pred_boxes.tensor[0].tolist()
                detected_boxes.append(current_gt_box)
                #########################################################################
                img = dr.ravel(img)
                # dr.disable_grad(img)
                start_index = b * (H * W * C)
                end_index = (b+1) * (H * W * C)
                index = dr.arange(dr.cuda.ad.UInt, start_index, end_index)                
                dr.scatter(imgs, img, index)
                time.sleep(1.0)
                # Get and Vizualize DT2 Predictions from rendered image
                rendered_img_input = dt2_input(rendered_img_path)
                success = save_adv_image_preds(model \
                    , dt2_config, input=rendered_img_input \
                    , instance_mask_thresh=dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST \
                    , target = label
                    , untarget = unlabel
                    , is_targeted = targeted
                    , path=os.path.join(preds_path,f'render_b{it:03d}.png'))
                if targeted:
                    target = dr.cuda.ad.TensorXf([label], shape=(1,))
                else:
                    target = dr.cuda.ad.TensorXf([unlabel], shape=(1,))
                
                #########################################################################
                # If 20 consecutive iterations have been made and the attack has not been successful, skip this camera position and increase its iteration count
                if iter_counts[cam_idx] >= 20 and not success:
                    logger.info(f"Skipping camera position {cam_idx} after 20 iterations without success.")
                    skipped_camera_indices.append(cam_idx)
                    cam_idx += 1  # Skip the current camera position
                    continue
                
                # Update the iteration count for the camera position
                iter_counts[cam_idx] += 1
                #########################################################################
            
            imgs = dr.cuda.ad.TensorXf(dr.cuda.ad.Float(imgs),shape=(N, H, W, C))
            if (dr.grad_enabled(imgs)==False):
                dr.enable_grad(imgs)
            loss = model_input(imgs, target, detected_boxes)
            sensor_loss = f"[PASS {cfg.sysconfig.pass_idx}] iter: {it} sensor pos: {cam_idx+1}/{len(sampled_camera_positions)}, loss: {str(loss.array[0])[0:7]}"
            logger.info(sensor_loss)
            gc.collect()
            ch.cuda.empty_cache()
            dr.enable_grad(loss)
            dr.backward(loss)
            #########################################################################
            # L-INFattack
            # grad = dr.grad(opt[k])
            # tex = opt[k]
            # eta = alpha * dr.sign(grad)
            # if targeted:
            #     eta = -eta
            # tex = tex + eta
            # eta = dr.clamp(tex - orig_tex, -epsilon, epsilon)
            # tex = orig_tex + eta
            # tex = dr.clamp(tex, 0, 1)
            #########################################################################
            for i, k in enumerate(param_keys):
                HH, WW  = dr.shape(dr.grad(opt[k]))[0], dr.shape(dr.grad(opt[k]))[1]
                C = 1 # when use opacity
                grad = ch.Tensor(dr.grad(opt[k]).array).view((HH, WW, C))
                tex = ch.Tensor(opt[k].array).view((HH, WW, C))
                _orig_tex = ch.Tensor(orig_texs[i].array).view((HH, WW, C))
                l = len(grad.shape) -  1
                g_norm = ch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
                scaled_grad = grad / (g_norm  + 1e-10)
                if targeted:
                    scaled_grad = -scaled_grad # 有目标攻击，取最小化损失的方向
                # step
                tex = tex + scaled_grad * alpha
                delta  = tex - _orig_tex
                # project
                delta =  delta.renorm(p=2, dim=0, maxnorm=epsilon)
                tex = _orig_tex + delta

                # convert back to mitsuba dtypes            
                tex = dr.cuda.ad.TensorXf(tex.to(DEVICE))
                # divide by average brightness
                scaled_img = img / dr.mean(dr.detach(img))
                tex = tex / dr.mean(scaled_img)         
                tex = dr.clamp(tex, 0, 1)
                params[k] = tex     
                dr.enable_grad(params[k])
                params.update()
                perturbed_tex = mi.Bitmap(params[k])
                
                
                mi.util.write_bitmap(os.path.join(tmp_perturbation_path,f"{k}_{it}.png"), data=perturbed_tex, write_async=False)
                # time.sleep(0.2)
                if it==(iters-1) and isinstance(params[k], dr.cuda.ad.TensorXf):
                    perturbed_tex = mi.Bitmap(params[k])
                    mi.util.write_bitmap("perturbed_tex_map.png", data=perturbed_tex, write_async=False)
                    logger.info(f"Skipped camera positions: {skipped_camera_indices}")
                    #time.sleep(0.2) 
                
                gc.collect()
                ch.cuda.empty_cache()
        return scene
    
    samples_per_pixel = spp
    epsilon = eps
    alpha = eps_step #(epsilon / (iters/50))
    label = target_class
    unlabel = untargeted_class
    img = optim_batch(scene\
                      , batch_size=batch_size\
                      , camera_positions =  moves_matrices\
                      , spp=samples_per_pixel\
                      , k=k, label = label\
                      , unlabel=unlabel\
                      , iters=iters\
                      , alpha=alpha\
                      , epsilon=epsilon\
                      , targeted=targeted)
