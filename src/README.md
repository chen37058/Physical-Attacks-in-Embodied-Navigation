### How to use

Use mitsuba compatible scene (see scenes/ dir)

in scripts `dt2.py` and `attack_tex.py`, specify the scene to load and the key containing the texture for the image we want to attack.

Specify the label of the target image (for targeted) attack

After attacking image, take perturbed texture, e.g., the `perturbed_tex_map.jpg` and specify this as the texture for the original object in the mitusba scene (.xml file)

Use `renders.ipynb` to render a series of images for the scene on the adversarial object.

Use `predict_objet_batch.py` to get predictions on a set of images. 
