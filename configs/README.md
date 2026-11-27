# Training recipes

Each tracked .example.yaml file is a portable recipe template. Copy one to an
operator-local YAML under configs/training/, edit the asset paths and run it
through the local trainer or Cloud TPU launcher. Ordinary local YAML files are
ignored by Git.

For the paper recipes, start from `robotwin-{baseline,hog}.example.yaml` or
`libero-{baseline,hog,rgb-to-hog}.example.yaml`. The `full` and `small`
templates illustrate Cloud TPU deployment with the legacy RoboTwin packing
profile; they are not the aligned RoboTwin recipes described in the main README.

The model-selection settings should be explicit:

| Variant | training_towers | visual_input_mode | lambda_hog |
|---|---|---|---:|
| FastWAM baseline | rgb_action | rgb | 0 |
| HOG-WAM | rgb_hog_action | rgb | 1 |
| RGB-to-HOG ablation | rgb_action | rgb_to_hog | 0 |

For the aligned RoboTwin TWP3 profile, retain data.robotwin_packing: aligned.
The loader checks image color order and the zero action shift against spec.json.
For Cloud TPU runs, use a writable GCS output prefix and ensure all model/data
inputs resolve to the same regional bucket. A bucket name is selected by the
operator rather than fixed in this repository.

See README.md for the paper's paired batch sizes and step counts and docs/training.md
for the data contract, checkpoint rules, and launch examples.
