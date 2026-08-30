from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("samtok_selective_refseg_rl_2gpu.py"))["config"]
)
config["generation"]["refseg"].update({
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
})
