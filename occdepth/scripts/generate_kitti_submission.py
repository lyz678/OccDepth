from occdepth.runtime import inference_device, load_model, move_to_device
from occdepth.runtime import config_main
from pytorch_lightning import Trainer
from occdepth.models.OccDepth import OccDepth
from occdepth.data.NYU.nyu_dm import NYUDataModule
from occdepth.data.semantic_kitti.kitti_dm import KittiDataModule
from occdepth.data.tartanair.tartanair_dm import TartanAirDataModule
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os
from hydra.utils import get_original_cwd
from tqdm import tqdm
import pickle

from occdepth.data.semantic_kitti.io_data import get_inv_map


config_path= os.getenv('DATA_CONFIG')

@config_main
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    device = inference_device(config)

    # Setup dataloader
    assert config.dataset == "kitti"
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
    data_module.setup("test")
    data_loader = data_module.test_dataloader()

    # Load pretrained models
    model = load_model(OccDepth, config, full_scene_size=tuple(config.full_scene_size)).to(device)
    model.eval()

    inv_map = get_inv_map()

    # Save prediction and additional data
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(config.output_path, config.dataset)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if config.get("max_batches") is not None and batch_idx >= config.max_batches:
                break
            batch = move_to_device(batch, device)

            pred = model(batch)
            y_pred = torch.softmax(pred["ssc_logit"], dim=1).detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            for i in range(y_pred.shape[0]):
                sub_y_pred = inv_map[y_pred[i].reshape(-1)].astype(np.uint16)
                write_path = os.path.join(
                    output_path, "sequences", batch["sequence"][i], "predictions"
                )
                os.makedirs(write_path, exist_ok=True)
                filepath = os.path.join(write_path, batch["frame_id"][i] + ".label")
                sub_y_pred.tofile(filepath)


if __name__ == "__main__":
    main()
