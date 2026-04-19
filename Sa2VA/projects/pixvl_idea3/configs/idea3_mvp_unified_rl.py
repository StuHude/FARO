from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_base.py"))["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_unified_rl"
config["run_name"] = "idea3_mvp_unified_rl"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/stage3_unified_rl"
config["loss"].update(
    {
        "lambda_ce": 0.3,
        "lambda_rl_seg": 1.0,
        "lambda_rl_cap": 0.1,
        "lambda_opd": 0.0,
        "beta_kl": 0.02,
    }
)
config["rl"] = {
    "group_size": {
        "refseg": 4,
        "maskcap": 2,
    },
    "tau_seg": 0.5,
    "tau_cap": 0.65,
}
config["generation"]["refseg"].update(
    {
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    }
)
config["generation"]["maskcap"].update(
    {
        "temperature": 0.8,
        "top_p": 0.95,
        "do_sample": True,
        "max_new_tokens": 64,
    }
)
