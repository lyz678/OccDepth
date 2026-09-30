#!/usr/bin/env python3
"""Extract sequence 08 directly from official ZIPs using verified HTTP ranges.

Only selected files are retained. Existing files are checked against ZIP CRC32;
completed files make reruns resumable. No credentials or unofficial mirrors.
"""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import zlib

import requests

SOURCES = {
    'color': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip',
    'calib': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip',
    'voxels': 'https://www.semantic-kitti.org/assets/data_odometry_voxels.zip',
}


class RemoteZipFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        r = self.session.head(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        self.size = int(r.headers['Content-Length'])
        self.pos = 0
        self.start = 0
        self.cache = b''
        self.downloaded = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos

    def read(self, n=-1):
        n = self.size-self.pos if n < 0 else min(n, self.size-self.pos)
        if n <= 0:
            return b''
        if not (self.start <= self.pos and self.pos+n <= self.start+len(self.cache)):
            start = self.pos
            end = min(self.size-1, start+max(n, 8*1024**2)-1)
            for attempt in range(5):
                try:
                    r = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, timeout=90, stream=True)
                    with r:
                        r.raise_for_status()
                        expected = f'bytes {start}-{end}/{self.size}'
                        if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                            raise RuntimeError(f'Server did not honor requested range: {r.status_code} {r.headers.get("Content-Range")}')
                        content = r.content
                        if len(content) != end-start+1:
                            raise RuntimeError('Incomplete range')
                    self.cache = content
                    self.start = start
                    self.downloaded += len(content)
                    break
                except (requests.RequestException, RuntimeError):
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
        result = self.cache[self.pos-self.start:self.pos-self.start+n]
        self.pos += len(result)
        return result


def valid_file(path, info):
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    crc = 0
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            crc = zlib.crc32(block, crc)
    return crc & 0xffffffff == info.CRC


def extract(source, root, url=None):
    url = url or SOURCES[source]
    remote = RemoteZipFile(url)
    with zipfile.ZipFile(remote) as archive:
        members = [i for i in archive.infolist() if '/sequences/08/' in '/'+i.filename and not i.is_dir()]
        if source == 'color':
            members = [i for i in members if '/image_2/' in i.filename or '/image_3/' in i.filename]
        if not members:
            sequences = sorted({i.filename.split('/sequences/')[1].split('/')[0]
                                for i in archive.infolist() if '/sequences/' in i.filename
                                and i.filename.split('/sequences/')[1]})
            status = {'source': url, 'complete': False, 'status': 'sequence_not_published',
                      'available_sequences': sequences,
                      'reason': 'The author archive does not contain sequence 08. Validation does not require depth supervision.'}
            (root/'downloads'/f'{source}-08-manifest.json').write_text(json.dumps(status, indent=2)+'\n')
            raise RuntimeError(f'No sequence 08 members found; available sequences: {sequences}')
        members.sort(key=lambda i: i.header_offset)
        target = root / ('KITTI_Odometry_Stereo_Depth' if source == 'depth' else 'semantic_kitti')
        paths = []
        for info in members:
            # Some archives wrap dataset/ in an extra top-level folder.
            name = info.filename[info.filename.index('dataset/'):] if 'dataset/' in info.filename else info.filename
            path = (target/name).resolve()
            if not path.is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe ZIP member path')
            paths.append(path)
        needed = sum(i.file_size for i,p in zip(members, paths) if not p.exists())
        if shutil.disk_usage(root).free < needed + 20*1024**3:
            raise RuntimeError(f'Insufficient space for {needed} bytes plus 20 GiB reserve')
        print(f'{source}: {len(members)} files, {sum(i.file_size for i in members)/1024**3:.2f} GiB extracted, archive {remote.size} bytes', flush=True)
        records = []
        for idx, (info, path) in enumerate(zip(members, paths)):
            if not valid_file(path, info):
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name+'.part')
                with archive.open(info) as src, temp.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024**2)
                if not valid_file(temp, info):
                    raise RuntimeError(f'CRC mismatch: {path}')
                os.replace(temp, path)
            records.append({'path': str(path.relative_to(root)), 'bytes': info.file_size, 'crc32': f'{info.CRC:08x}'})
            if idx % 100 == 0 or idx+1 == len(members):
                print(f'{source}: {idx+1}/{len(members)}, transferred {remote.downloaded/1024**3:.2f} GiB', flush=True)
        manifest = {'source': url, 'archive_bytes': remote.size, 'transferred_bytes_this_run': remote.downloaded, 'complete': True, 'files': records}
        (root/'downloads'/f'{source}-08-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=[*SOURCES, 'depth'])
    parser.add_argument('--url', help='Author-provided direct ZIP URL (required for depth)')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'data')
    args = parser.parse_args()
    if args.source == 'depth' and not args.url:
        parser.error('--url is required for depth')
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/'downloads').mkdir(exist_ok=True)
    extract(args.source, args.root, args.url)
