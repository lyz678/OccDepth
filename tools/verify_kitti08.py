#!/usr/bin/env python3
"""Verify downloaded manifests, frame pairs, calibration and preprocessed labels."""
import argparse
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT, crc=False):
    data = root/'data'
    for name in ('color', 'calib', 'voxels'):
        manifest = json.loads((data/'downloads'/f'{name}-08-manifest.json').read_text())
        assert manifest['complete'], name
        for entry in manifest['files']:
            path = data/entry['path']
            assert path.is_file() and path.stat().st_size == entry['bytes'], path
            if crc:
                value = 0
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(1024**2),b''):
                        value=zlib.crc32(block,value)
                assert f'{value & 0xffffffff:08x}' == entry['crc32'], path
    seq = data/'semantic_kitti/dataset/sequences/08'
    left = {p.stem for p in (seq/'image_2').glob('*.png')}
    right = {p.stem for p in (seq/'image_3').glob('*.png')}
    assert left == right and len(left) == 4071
    frames = {p.stem for p in (seq/'voxels').glob('*.bin')}
    assert len(frames) == 815 and frames <= left
    for suffix in ('.label', '.invalid', '.occluded'):
        assert {p.stem for p in (seq/'voxels').glob('*'+suffix)} == frames
    calibration = {line.split(':')[0] for line in (seq/'calib.txt').read_text().splitlines() if ':' in line}
    assert {'P2','P3','Tr'} <= calibration
    allowed = set(range(20)) | {255}
    for frame in sorted(frames):
        for scale, shape in (('1_1',(256,256,32)),('1_8',(32,32,4))):
            path = data/f'kitti_semantic_preprocess/labels/08/{frame}_{scale}.npy'
            label = np.load(path, mmap_mode='r', allow_pickle=False)
            assert label.shape == shape and set(np.unique(label)) <= allowed, path
    checkpoint = root/'trained_models/kitti_multicam_flospdepth_crp_stereodepth_cascadecls_2080ti_mIoU12.8.ckpt'
    sha = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            sha.update(block)
    report = dict(sequence='08', stereo_pairs=len(left), labeled_frames=len(frames),
        preprocessed_labels=2*len(frames), archive_crc_checked=crc,
        checkpoint={'file':str(checkpoint.relative_to(root)), 'bytes':checkpoint.stat().st_size,
                    'sha256':sha.hexdigest(), 'source':'https://drive.google.com/file/d/1MGJ_HZcuW5UpULpOeJV0M5ZrT-98j7OE/view'},
        depth='Not published for 08 in author archive; not required by validation loader.')
    (data/'downloads/verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--crc',action='store_true',help='Re-read files and verify original ZIP CRC32')
    args=p.parse_args()
    verify(args.root,args.crc)
