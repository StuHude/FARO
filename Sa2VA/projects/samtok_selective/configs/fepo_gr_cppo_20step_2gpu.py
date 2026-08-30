import runpy
from pathlib import Path


smoke = Path(__file__).resolve().with_name("fepo_gr_cppo_one_step_2gpu.py")
config = runpy.run_path(str(smoke))["config"]
config["stage"] = "fepo_gr_cppo_20step_2gpu"
config["optimizer"]["max_steps"] = 20
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).resolve().with_name(config["stage"])
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
