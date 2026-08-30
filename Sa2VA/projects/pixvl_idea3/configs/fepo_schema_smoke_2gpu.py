from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_base.py"))["config"]
config = deepcopy(base)
config["stage"] = "fepo_schema_smoke_2gpu"
config["run_name"] = "fepo_schema_smoke_2gpu"
config["data"]["batch_size"] = 1
config["data"]["num_workers"] = 0
config["data"]["task_mix"] = {"refseg": 0.5, "maskcap": 0.5}
config["data"]["schema_files"] = [
    str(Path(config["paths"]["schema_root"]) / "refseg_train_routed.jsonl"),
    str(Path(config["paths"]["schema_root"]) / "maskcap_train_routed.jsonl"),
]
config["optimizer"]["max_steps"] = 20
config["optimizer"]["grad_accum_steps"] = 1
config["optimizer"]["lr"] = 1e-6
# H200 smoke uses DDP so rollout generation sees a full embedding matrix.  The
# production FSDP path still remains in the base config and will be addressed
# separately with a summon-full-params generation context.
config["memory_optim"]["fsdp"]["enabled"] = False
config["checkpoint"]["output_dir"] = str(Path(config["paths"]["output_root"]) / "fepo_schema_smoke_2gpu")
config["checkpoint"]["save_every"] = 10
config["logging"]["log_every"] = 1
config["logging"]["snapshot_every"] = 5
config["routing"]["mode"] = "predicted_only_evidence"
config["routing"]["predicted_only_evidence"].update({"temperature": 0.25, "min_failure": 0.25, "probe_index": 0})
config["loss"].update({
    "lambda_ce": 0.3,
    "lambda_rl_seg": 1.0,
    "lambda_rl_cap": 0.1,
    "beta_kl": 0.02,
})
# Candidate groups require sampling; the inherited SFT config is greedy and
# Transformers rejects ``num_return_sequences > 1`` in that mode.
for _task in ("refseg", "maskcap"):
    config["generation"][_task].update({
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    })
# The joint trainer always forms an online candidate group for each task,
# including when the smoke run only exercises the supervised/route contract.
config["rl"] = {
    "group_size": {
        "refseg": 2,
        "maskcap": 2,
    }
}
config["loss"]["lambda_opd"] = 0.0
