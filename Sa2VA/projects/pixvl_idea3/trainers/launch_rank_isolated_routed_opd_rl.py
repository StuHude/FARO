from __future__ import annotations

import os
import runpy


def main() -> None:
    original_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(original_local_rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    runpy.run_module("projects.pixvl_idea3.trainers.joint_routed_opd_rl_trainer", run_name="__main__")


if __name__ == "__main__":
    main()
