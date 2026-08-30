from copy import deepcopy
from pathlib import Path
import runpy

config = deepcopy(runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"])
for _task in ("refseg", "maskcap", "existence"):
    config["generation"].setdefault(_task, {})
    config["generation"][_task].update({
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    })
