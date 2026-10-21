import os, sys
CUR_DIR = "/home/skye/data/Skye/visualad"
sys.path.insert(0, CUR_DIR)
from run_a4_scratch_8ep import train_scratch_8ep_a4
train_scratch_8ep_a4(
    dataset_name="Bridge512_balance_CODEBRIM",
    short_ds="CODEBRIM",
    save_dir=os.path.join(CUR_DIR, "outputs", "a4_scratch8ep_CODEBRIM"),
    epochs=8,
    beta_edge=0.03,
)
