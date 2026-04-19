from pathlib import Path
from copy import deepcopy
import runpy


base = runpy.run_path(Path(__file__).with_name("idea1_joint_opd.py"))["config"]
config = deepcopy(base)

config["stage"] = "stage3_joint_opd_rl"
config["run_name"] = "idea1_joint_opd_rl"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage3_joint_opd_rl"
config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage2_joint_opd/adapter",
}
config["resume"] = {
    "completed_steps": 0,
}
config["teacher"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/adapter",
}
config["reference"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/adapter",
}
config["memory_optim"]["fsdp"]["enabled"] = False
config["memory_optim"]["gradient_checkpointing"] = True
config["data"]["batch_size"] = 4
config["data"]["task_mix"] = {
    "refseg": 0.75,
    "maskcap": 0.25,
}
config["loss"].update({
    "lambda_ce": 0.3,
    "lambda_rl_seg": 1.0,
    "lambda_rl_cap": 0.1,
    "lambda_opd": 0.3,
    "beta_kl": 0.02,
})
config["rl"] = {
    "group_size": {
        "refseg": 16,
        "maskcap": 8,
    },
    "tau_seg": 0.5,
    "tau_cap": 0.65,
}
config["generation"]["refseg"].update({
    "temperature": 0.7,
    "top_p": 0.95,
    "do_sample": True,
})
config["generation"]["maskcap"].update({
    "temperature": 0.8,
    "top_p": 0.95,
    "do_sample": True,
    "max_new_tokens": 64,
})
