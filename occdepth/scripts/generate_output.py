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


config_path= os.getenv('DATA_CONFIG')

@config_main
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    device = inference_device(config)

    # Setup dataloader
    if config.dataset == "kitti":
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
        data_module.setup("validate")
        data_loader = data_module.val_dataloader()

    elif config.dataset == "NYU":
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
        data_module.setup("validate")
        data_loader = data_module.val_dataloader()
    elif config.dataset == "tartanair":
        data_module = TartanAirDataModule(
            config=config,
        )
        data_module.setup("validate")
        data_loader = data_module.val_dataloader()
    else:
        print("dataset not support")

    # Load pretrained models
    model = load_model(OccDepth, config, full_scene_size=tuple(config.full_scene_size)).to(device)
    model.eval()

    # Save prediction and additional data
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(config.get("output_path", "output"), config.dataset)
    output_path = os.path.abspath(output_path)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if config.get("max_batches") is not None and batch_idx >= config.max_batches:
                break
            batch = move_to_device(batch, device)

            pred = model(batch)
            y_pred = torch.softmax(pred["ssc_logit"], dim=1).detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            for i in range(y_pred.shape[0]):
                out_dict = {"y_pred": y_pred[i].astype(np.uint16)}
                if "target" in batch:
                    out_dict["target"] = (
                        batch["target"][i].detach().cpu().numpy().astype(np.uint16)
                    )

                if config.dataset == "NYU":
                    write_path = output_path
                    filepath = os.path.join(write_path, batch["name"][i] + ".pkl")
                    out_dict["cam_pose"] = batch["cam_pose"][i].detach().cpu().numpy()
                    out_dict["vox_origin"] = (
                        batch["vox_origin"][i].detach().cpu().numpy()
                    )
                elif config.dataset == "tartanair":
                    write_path = os.path.join(output_path, batch["sequence"][i])
                    filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                    out_dict["vox_origin"] = np.array([-6, -3, 0])  # cam coord
                    out_dict["T_velo_2_cam"] = (
                        batch["T_velo_2_cam"][i].detach().cpu().numpy()
                    )
                    out_dict["fov_mask_1"] = (
                        batch["fov_mask_1"][i].detach().cpu().numpy()
                    )
                elif config.dataset == "kitti":
                    write_path = os.path.join(output_path, batch["sequence"][i])
                    filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                    out_dict["fov_mask_1"] = (
                        batch["fov_mask_1"][i].detach().cpu().numpy()
                    )
                    out_dict["cam_k"] = batch["cam_k"][i].detach().cpu().numpy()
                    out_dict["T_velo_2_cam"] = (
                        batch["T_velo_2_cam"][i].detach().cpu().numpy()
                    )

                os.makedirs(write_path, exist_ok=True)
                with open(filepath, "wb") as handle:
                    pickle.dump(out_dict, handle)
                    print("wrote to", filepath)


if __name__ == "__main__":
    main()
