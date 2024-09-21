# %env CUDA_VISIBLE_DEVICES=2
import os
from pydoc import render_doc
from random import seed
import torchvision.transforms as T
import torch.nn as nn
import torch as ch
from torchvision.io import read_image
import torchvision.models as models
import numpy as np
import time
import matplotlib.pyplot as plt
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
import drjit as dr
from PIL import Image
import graphviz
import json
from main import *

with open("imagenet_class_index.json", "r") as h:
    labels = json.load(h)

# _SELECTED_ IMAGENET LABELS
GREAT_WHITE_SHARK = 2
MICROWAVE = 651
MOUNTAIN_BIKE = 671
AMBULANCE = 407
POLICE_VAN = 734
FIRE_ENGINE = 555


def custom_mesh():
    mesh = mi.load_dict({
    "type": "ply",
    "filename": "scenes/sphere/meshes/Sphere.ply",
    "bsdf": {
        "type": "diffuse",
        "reflectance": {
            "type": "mesh_attribute",
            "name": "vertex_color",  # This will be used to visualize our attribute
            },
        },
    })
    attribute_size = mesh.vertex_count() * 3
    mesh.add_attribute("vertex_color", 3, [0] * attribute_size) 
    return mesh

def audi_mesh():
    mesh = mi.load_dict({
    "type": "ply",
    "filename": "assets/meshes/taxi.ply",
    "bsdf": {
        "type": "diffuse",
        "reflectance": {
            "type": "mesh_attribute",
            "name": "vertex_color",  # This will be used to visualize our attribute
            },
        },
    })
    attribute_size = mesh.vertex_count() * 3
    mesh.add_attribute("vertex_color", 3, [0] * attribute_size) 
    return mesh


def point_emitter():
    emitter = mi.load_dict({
            "type": "point",
            # "position" : [4.07, 5.9, -1.0],
            "position" : [-5.0, 1.0, -1.0],
            "intensity" : {
                "type" : "rgb",
                "value" : [79.5, 79.5, 79.5]
            }
        })    
    return emitter

def spot_emitter():
    emitter = mi.load_dict({
        'type': 'spot',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0.5, 0.5, 0.5],
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'intensity': {
            'type': 'spectrum',
            'value': 1.0,
        }        
    })
    return emitter

# "emitter": {
#     "type": "envmap",
#     "filename": "scenes/sphere/textures/lythwood.exr",   
#     "to_world": mi.ScalarTransform4f.rotate([0, 1, 0], -30.0)
#                 .rotate([0, 0, 1], 30.0),
#     "scale": 0.65
# },

def envmap_emitter():
    emitter = mi.load_dict({
        "type": "envmap",
        "filename" : "scenes/sphere/textures/lythwood.exr"
    })
    return emitter

def bitmap_tex_mesh():
    bitmap_tex_mesh = mi.load_dict({
        "type": "ply",
        "filename": "scenes/sphere/meshes/Sphere.ply",
        'material': {
            'type': 'twosided',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': 'scenes/noise_cube/textures/noise_tex.jpg'
                    }
                }
            }
        })
    return bitmap_tex_mesh

def sphere_scene(mesh, emitter):
    path_integrator = {
        "type": "path",
        "max_depth": 12
    }, 
    prb_reparam_integrator  = {
        'type': 'prb_reparam',
        'max_depth': 8,
        'reparam_rays': 8
    }
    scene = mi.load_dict({  
        "type": "scene",
        "integrator" : prb_reparam_integrator,
        "emitter" : emitter,
    # camera used for bird
	# <sensor type="perspective">
	# 	<string name="fov_axis" value="x"/>
	# 	<float name="fov" value="39.597752"/>
	# 	<float name="principal_point_offset_x" value="0.000000"/>
	# 	<float name="principal_point_offset_y" value="-0.000000"/>
	# 	<float name="near_clip" value="0.100000"/>
	# 	<float name="far_clip" value="1000.000000"/>
	# 	<transform name="to_world">
	# 		<rotate x="1" angle="5.194121514721039"/>
	# 		<rotate y="1" angle="1.7394183425526424"/>
	# 		<rotate z="1" angle="0.515880826277331"/>
	# 		<translate value="-0.056464 0.151855 -3.756493"/>
	# 	</transform>
	# 	<sampler type="independent">
	# 		<integer name="sample_count" value="$spp"/>
	# 	</sampler>
	# 	<film type="hdrfilm">
	# 		<integer name="width" value="$resx"/>
	# 		<integer name="height" value="$resy"/>
	# 	</film>
	# </sensor>        
    # camera used for car
        "sensor": {
            "type": "perspective",
            "fov": 39.597752,
            "fov_axis": "x",
            "principal_point_offset_x": 0.0,
            "principal_point_offset_y": -0.0,
            "near_clip" : 0.1,
            "far_clip" : 100.0,
            "film": {
                "type": "hdrfilm",
                "width": 299,
                "height": 299,
                "sample_border": True,
                "pixel_format": "rgb"                
            },
            "sampler":{
                "type":"independent",
                "sample_count": 32
            },
            # "to_world": mi.ScalarTransform4f.rotate([1, 0, 0], -153.55)
            #                             .rotate([0, 1, 0], -46.69)
            #                             .rotate([0, 0, 1], -179.0)
            #                             .translate([0, 0, 0])
            #                                 .look_at(origin=[0, -1.5, 1.5], target=[0,0,0], up=[0, 0, 1])        
            # },
            "to_world": mi.ScalarTransform4f.rotate([1, 0, 0], -153.55)
                                        .rotate([0, 1, 0], 0.0)
                                        .rotate([0, 0, 1], -179.0)
                                        .translate([0, -0.8, 1.5])
                                        .look_at(origin=[0, -1.5, 1.5], target=[0,0,0], up=[0, 0, 1])        
            },           
        "sphere": mesh
    })

    return scene

def bob_model()->ch.nn.Module:
    # load pre-trained undefended bird-or-bicycle model
    device = ch.device('cpu')
    checkpoint = ch.load('/nvmescratch/mhull32/unrestricted-adversarial-examples/model_zoo/pp_undefended_pytorch_resnet.pth.tar')
    model = getattr(models, 'resnet50')(num_classes=2)
    model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), model)
    # model = ch.nn.DataParallel(model) # stage model for GPUs
    model.load_state_dict(checkpoint)
    # ch.save(model.module.state_dict(), '/nvmescratch/mhull32/unrestricted-adversarial-examples/model_zoo/pp_undefended_pytorch_resnet.pth.tar')    
    # model.cuda()
    model.eval()
    return model

def robust_model_args(arch='resnet50', checkpoint_path:str=None):
    """
    These args are to make the robust model loading happy.
    For this attack, we aren't using any LR or optimization parameters 
    so the only thing that matters is to match the architecture
    and model checkpoint path
    """
    args = argparse.Namespace(additional_hidden=0
        , adv_eval=None
        , adv_train=0
        , arch=arch
        , attack_lr=None
        , attack_steps=None
        , batch_size=64
        , cifar10_cifar10=False
        , config_path=None
        , constraint=None
        , custom_eps_multiplier=None
        , custom_lr_multiplier=None
        , data='/tmp'
        , data_aug=1
        , dataset='cifar100'
        , epochs=150
        , eps=None
        , eval_only=0
        , exp_name='cifar100-transfer-demo'
        , freeze_level=-1
        , log_iters=5
        , lr=0.01
        , lr_interpolation='step'
        , mixed_precision=0
        , model_path=checkpoint_path
        , momentum=0.9
        , no_replace_last_layer=True  # change this to False to replace last layer w/ appropriate class outputs for target dataset
        , no_tqdm=1
        , out_dir='outdir'
        , per_class_accuracy=False
        , pytorch_pretrained=False
        , random_restarts=None
        , random_start=None
        , resume=False
        , resume_optimizer=0
        , save_ckpt_iters=-1
        , step_lr=30
        , step_lr_gamma=0.1
        , subset=None
        , use_best=None
        , weight_decay=0.0005
        , workers=30)
    return args

def robust_model(args):
    device = ch.device('cpu')
    ds, train_loader, validation_loader = get_dataset_and_loaders(args)
    model, checkpoint = get_model(args, ds)
    model.eval()
    return model

# def softmax(x):
#   return np.exp(x)/np.sum(np.exp(x),axis=0)

# def dr_softmax(x):
#     softmax =  dr.exp(x) /dr.sum(dr.exp(x))
#     return dr.llvm.ad.TensorXf(softmax.numpy())

# torchCE = nn.CrossEntropyLoss()

# def cross_entropy_np(y,y_pre):
#   loss=-np.sum(y*np.log(y_pre))
#   return loss

# def dr_xentropy(label, logits):
#     loss = -dr.sum(label * dr.log(logits))
#     return loss

if __name__ == "__main__":

    mi.set_variant('llvm_ad_rgb')
    # mesh = custom_mesh()
    # mesh = audi_mesh()
    # mesh = bitmap_tex_mesh()
    # scene = sphere_scene(mesh, envmap_emitter())
    # scene = mi.load_file('scenes/adv_tex_bird/adv_tex_bird.xml')
    scene = mi.load_file("scenes/intersection_taxi/intersection_taxi.xml")
    # N = mesh.vertex_count()
    p = mi.traverse(scene)
    # k = 'sphere.vertex_color'
    # k = 'sphere.bsdf.brdf_0.reflectance.data'
    # k = 'mat-12214_bird.001.brdf_0.base_color.data'
    # k = 'mat-13914_Taxi_car.002.brdf_0.base_color.data'
    k = 'mat-13914_Taxi_car.001.brdf_0.base_color.data'
    # k = 'mat-dumpster.brdf_0.base_color.data'
    # k = 'mat-TankMaterial.brdf_0.base_color.data'
    # p.update()
    # dims = (3*N)
    # # rand_colors = dr.llvm.ad.Float(ch.rand(dims).numpy())
    # # whites = dr.llvm.ad.Float(ch.ones(dims).numpy())
    # grays = dr.llvm.ad.Float(ch.Tensor([0.5]).repeat(dims).numpy())
    # p[k] = grays
    p.keep(k)
    p.update()
    with dr.suspend_grad():
        img = mi.render(scene, params = p, spp=32)
        mi.util.write_bitmap("renders/render.jpg", data=img)

    # model = bob_model()
    args = robust_model_args(arch='resnet50', checkpoint_path='/nvmescratch/mhull32/robust-models-transfer/pretrained-models/resnet50_linf_ep0.5.ckpt')
    model = robust_model(args)

    
    def optim(scene, k, label, iters, alpha, epsilon, targeted=False):

        if targeted:
            assert(label is not None)

        @dr.wrap_ad(source='drjit', target='torch')
        def model_input(x, target):
            x = ch.permute(x, (2,1,0)).unsqueeze(dim=0).requires_grad_()
            logits = model(x)[0]
            pred = labels[str(ch.argmax(logits).item())][1]
            print(f'Prediction: {pred}')
            target = target.long() 
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, target).requires_grad_()
            return loss

        params = mi.traverse(scene)
        if isinstance(params[k], dr.llvm.ad.TensorXf):
            # use Float if dealing with just texture colors (not a texture map)
            orig_tex = dr.llvm.ad.TensorXf(params[k])
        elif isinstance(params[k], dr.llvm.ad.Float):
            orig_tex = dr.llvm.ad.Float(params[k])        
        else:
            raise Exception("Unrecognized Differentiable Parameter Data Type.  Should be one of dr.llvm.ad.Float or dr.llvm.ad.TensorXf")

        orig_tex.set_label_("orig_tex")
        
        # indicate sensors to use in producing the perturbation
        # e.g., [0,1,2,3] will use sensors 0-3 focus on Taxi/Cement Truck in 'intersection_taxi.xml'
        sensors = [0,1,2,3]
        if iters % len(sensors) != 0:
            print("uneven amount of iterations provided for sensors! Some sensors will be used more than others\
                during attack")
        camera_idx = ch.Tensor(np.array(sensors)).repeat(int(iters/len(sensors))).to(dtype=ch.uint8).numpy().tolist()

        for it in range(iters):
            print(f'iter {it}')
            params = mi.traverse(scene)
            params.keep(k)
            opt = mi.ad.Adam(lr=0.1, params=params)
            dr.enable_grad(orig_tex)
            dr.enable_grad(opt[k])
            opt[k].set_label_("bitmap")
            params.update(opt)            

            img =  mi.render(scene, params=params, spp=256, sensor=camera_idx[it], seed=it+1)
            img.set_label_("image")
            dr.enable_grad(img)
            mi.util.write_bitmap(f"renders/render_{it}_s{camera_idx[it]}.jpg", data=img)
            target = dr.llvm.ad.TensorXf([label], shape=(1,))
            loss = model_input(img, target)
            print(f"sensor: {str(camera_idx[it])}, loss: {str(loss.array[0])[0:5]}")
            dr.enable_grad(loss)
            dr.backward(loss)

            grad = dr.grad(opt[k])
            tex = opt[k]
            eta = alpha * dr.sign(grad)
            if targeted:
                eta = -eta
            tex = tex + eta
            eta = dr.clamp(tex - orig_tex, -epsilon, epsilon)
            tex = orig_tex + eta
            tex = dr.clamp(tex, 0, 1)
            params[k] = tex
            dr.enable_grad(params[k])
            params.update()
            if it==(iters-1) and isinstance(params[k], dr.llvm.ad.TensorXf):
                perturbed_tex = mi.Bitmap(params[k])
                mi.util.write_bitmap("perturbed_tex_map.jpg", data=perturbed_tex)
        return scene

    iters = 40
    epsilon = (128/255)
    alpha = (epsilon / (iters/4))
    label = AMBULANCE
    img = optim(scene, k=k, label = label, iters=iters, alpha=alpha, epsilon=epsilon, targeted=True)
