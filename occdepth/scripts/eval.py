from occdepth.runtime import trainer_device, load_model
from occdepth.runtime import config_main
from pytorch_lightning import Trainer
from occdepth.models.OccDepth import OccDepth
from occdepth.data.NYU.nyu_dm import NYUDataModule
from occdepth.data.semantic_kitti.kitti_dm import KittiDataModule
from occdepth.data.tartanair.tartanair_dm import TartanAirDataModule

import hydra
from omegaconf import DictConfig
import torch
import os
from hydra.utils import get_original_cwd

config_path= os.getenv('DATA_CONFIG')

@config_main
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    load_strict = True
    if config.dataset == "kitti":
        config.batch_size_per_gpu = 1
        full_scene_size = tuple(config.full_scene_size)
        data_module = KittiDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            project_scale=config.project_scale,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu),
            pattern_id=config.pattern_id,
            multi_view_mode=config.multi_view_mode,
            use_stereo_depth_gt=config.use_stereo_depth_gt,
            use_lidar_depth_gt=config.use_lidar_depth_gt,
            data_stereo_depth_root=config.data_stereo_depth_root,
            data_lidar_depth_root=config.data_lidar_depth_root,
        )

    elif config.dataset == "NYU":
        config.batch_size_per_gpu = 1
        full_scene_size = tuple(config.full_scene_size)
        data_module = NYUDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu),
            pattern_id=config.pattern_id,
            use_depth_gt=config.use_depth_gt,
        )
    elif config.dataset == "tartanair":
        data_module = TartanAirDataModule(
            config=config,
        )

    trainer = Trainer(
        **trainer_device(config), deterministic=config.deterministic,
        logger=False, enable_checkpointing=False,
        limit_test_batches=config.get("limit_test_batches", 1.0),
    )
    model = load_model(OccDepth, config, full_scene_size=tuple(config.full_scene_size))
    model.eval()
    data_module.setup("validate")
    if config.n_gpus:
        torch.cuda.reset_peak_memory_stats()
    trainer.test(model, dataloaders=data_module.val_dataloader())
    if config.n_gpus:
        print(f"Peak allocated CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
