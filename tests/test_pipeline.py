"""Small real-model training/evaluation checks; no mock network or downloads."""
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from occdepth.models.OccDepth import OccDepth
from occdepth.models.flosp_depth import flosp_depth_conf_map
from occdepth.runtime import move_to_device


torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


def make_model_and_batch(dataset, monkeypatch):
    folder = 'NYU' if dataset == 'NYU' else 'semantic_kitti'
    name = 'multicam_flosp_crp_stereodepth_cascadecls_2080ti.yaml' if dataset == 'NYU' else 'multicam_flospdepth_crp_stereodepth_cascadecls_2080ti.yaml'
    config = OmegaConf.load(ROOT/'occdepth/config'/folder/name)
    config = OmegaConf.merge(config, dict(full_scene_size=[16,16,16], feature=8,
        feature_2d_oc=8, n_gpus=0, batch_size_per_gpu=2, pretrained_backbone=False,
        backbone_2d_name='tf_efficientnet_b3_ns', context_prior=False, relation_loss=False,
        fp_loss=False, use_depth_gt=False, use_stereo_depth_gt=False,
        share_2d_backbone_gradient=False))
    # Exercise the actual depth projection on a physically consistent tiny KITTI grid.
    if dataset == 'kitti':
        conf = dict(flosp_depth_conf_map['kitti'])
        conf.update(x_bound=[0,3.2,0.2], y_bound=[-1.6,1.6,0.2],
                    z_bound=[2,5.2,0.2], final_dim=(32,64),
                    d_bound=[1,9,1], depth_net_conf=dict(in_channels=8, mid_channels=8))
        monkeypatch.setitem(flosp_depth_conf_map, 'kitti', conf)
    else:
        config.trans_2d_to_3d = 'flosp'
        config.use_depth_gt = True
    model = OccDepth(class_names=[str(i) for i in range(config.n_classes)],
        class_weights=torch.ones(config.n_classes), class_weights_occ=torch.ones(2),
        full_scene_size=tuple(config.full_scene_size), project_res=['1','2','4','8'], config=config)
    b, views, h, w = 2, 2, 32, 64
    side = 16//config.project_scale
    # A pinhole camera observes a regular front-facing grid; projected pixels
    # stay inside the image for every feature scale.
    xyz = torch.stack(torch.meshgrid(torch.linspace(-1,1,side),
        torch.linspace(-0.5,0.5,side),torch.linspace(2,5,side),indexing='ij'),-1).reshape(-1,3)
    pix = torch.stack((xyz[:,0]/xyz[:,2]*16+32, xyz[:,1]/xyz[:,2]*16+16),-1).long()
    pix = pix[None,:,None,:].repeat(views,1,1,1)
    mask = torch.ones(views,side**3,1,dtype=torch.bool)
    key = str(config.project_scale)
    batch = {'img':torch.randn(b,views,3,h,w),
        'target':torch.randint(0,config.n_classes,(b,16,16,16)),
        f'projected_pix_{key}':[pix.clone() for _ in range(b)],
        f'fov_mask_{key}':[mask.clone() for _ in range(b)],
        'cam_k':[torch.tensor([[16.,0,32],[0,16,16],[0,0,1]],dtype=torch.float64).repeat(views,1,1) for _ in range(b)],
        'T_velo_2_cam':[torch.eye(4).repeat(views,1,1) for _ in range(b)],
        'ida_mats':[torch.eye(4).repeat(views,1,1) for _ in range(b)]}
    if dataset == 'NYU':
        batch['vox_origin'] = torch.zeros(b,3)
        batch['img'] = batch['img'][:, :1]
        batch['gt_depth'] = torch.full((b,1,h,w), 4.)
        batch['virtual_bf'] = [torch.tensor(2.)]
    batch['target'][0,0,0,0] = 255
    return model,batch


@pytest.mark.parametrize('dataset',['kitti','NYU'])
def test_train_evaluate_resume(dataset,monkeypatch,tmp_path):
    model,batch = make_model_and_batch(dataset,monkeypatch)
    # Prebatched samples preserve variable-length geometry containers.
    loader = DataLoader([batch],batch_size=None)
    before = model.net_rgb.decoder.resize_output_1_1.weight.detach().clone()
    trainer = Trainer(accelerator='cpu',devices=1,max_epochs=1,logger=False,
                      enable_checkpointing=False,enable_progress_bar=False,
                      enable_model_summary=False,num_sanity_val_steps=0)
    trainer.fit(model,train_dataloaders=loader,val_dataloaders=loader)
    assert torch.isfinite(trainer.callback_metrics['train/loss'])
    assert not torch.equal(before,model.net_rgb.decoder.resize_output_1_1.weight)
    assert 'val/mIoU' in trainer.callback_metrics
    trainer.test(model,dataloaders=loader)
    assert torch.isfinite(trainer.callback_metrics['test/loss'])
    path = tmp_path/'resume.ckpt'
    trainer.save_checkpoint(path)
    restored = OccDepth.load_from_checkpoint(path,map_location='cpu',weights_only=False,strict=True)
    model.eval(); restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch)['ssc_logit'],restored(batch)['ssc_logit'])
    resumed = Trainer(accelerator='cpu',devices=1,max_epochs=2,logger=False,
                      enable_checkpointing=False,enable_progress_bar=False,
                      enable_model_summary=False,num_sanity_val_steps=0)
    restored.train()
    resumed.fit(restored,train_dataloaders=loader,val_dataloaders=loader,ckpt_path=path)
    assert resumed.global_step == 2
    assert torch.isfinite(resumed.callback_metrics['train/loss'])
