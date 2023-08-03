# PoMMEs: (Human) *Po*se *Es*timation using *M*acaque *M*onkey Data

PoMMEs was developed as part of the Advanced Masters Research Project course at UoA.

## Quick guide to the repository
TODO(to be expanded)

`preprocsessing.py`:
Performs preprocessing for both macaque monkey data and human data.
Harbors functionality to generate scaling factors to be used during downstream evaluation.

`mrp_baseline_tf.py`:
Loads the pre-trained baseline model from the DeepLabCut ModelZoo and generates predictions on chosen dataset using auxiliary DeepLabCut functions. Currently, this is not working due to variation in image dimensions within the datasets.

`mrp_baseline.py`:
Loads the pre-trained baseline model from the DeepLabCut ModelZoo and generates predictions on chosen dataset using advanced DeepLabCut functionality.

`mrp_tl.py`:
Loads the pre-trained baseline model from the DeepLabCut ModelZoo, performs re-training on human data and generates predictions on chosen dataset using advanced DeepLabCut functionality.

`mrp_evaluation.py`:
Evaluates model performance using pre-generated predictions. 

`plotting.py`:
Provides basic plotting functionality to support model evaluation.
