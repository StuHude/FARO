from pathlib import Path
from copy import deepcopy
import runpy


base = runpy.run_path(Path(__file__).with_name("idea1_joint_sft.py"))["config"]
config = deepcopy(base)

config["stage"] = "stage2_joint_opd"
config["run_name"] = "idea1_joint_opd"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage2_joint_opd"
config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/adapter",
}
config["resume"] = {
    "completed_steps": 0,
}
config["teacher"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/adapter",
}
config["memory_optim"]["fsdp"]["enabled"] = False
config["memory_optim"]["gradient_checkpointing"] = True
config["generation"]["refseg"].update({
    "temperature": 0.7,
    "top_p": 0.95,
    "do_sample": True,
})
config["generation"]["maskcap"].update({
    "temperature": 0.7,
    "top_p": 0.95,
    "do_sample": True,
})
config["opd"] = {
    "lambda_opd": 0.3,
    "tau_seg": 0.5,
    "tau_cap": 0.65,
    "all_sample_distill": False,
}
