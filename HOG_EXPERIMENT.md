# HOG-WAM training and inference contract

HOG-WAM extends the FastWAM RGB/action policy with an auxiliary HOG visual
prediction branch during training. RGB and HOG visual branches predict their
respective future visual targets; the action branch predicts actions from the
RGB observation, language context and proprioception. Deployment exports retain
the RGB/action policy parameters and omit the auxiliary HOG parameters.

The aligned data path is:

~~~text
RGB observation -> frozen Wan2.2 VAE -> visual DiT -> future-RGB prediction
       |                                   \
       +-> deterministic HOG -> VAE -> HOG DiT -> future-HOG prediction
RGB/proprioception/language -> Action DiT -> action chunk
~~~

RGB-to-HOG is a separate ablation: convert the input pixels to HOG before the
VAE, remove the auxiliary HOG branch, and train the standard visual prediction
path to predict future HOG. Deployment must keep that conversion enabled.

Training recipes, RGB-to-HOG settings, input contracts, checkpoint exports and
benchmark commands are documented in README.md and docs/. HOG is never supplied
to the HOG-WAM inference policy.
