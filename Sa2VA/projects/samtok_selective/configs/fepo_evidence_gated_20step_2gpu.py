import runpy
from pathlib import Path


config = runpy.run_path(str(Path(__file__).with_name("fepo_evidence_gated_one_step_2gpu.py")))["config"]
config["stage"] = "fepo_evidence_gated_20step_2gpu"
config["optimizer"]["max_steps"] = 20
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).resolve().with_name(
        f"{config['stage']}_{config['gr_cppo']['evidence_gate']['mode']}"
    )
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
