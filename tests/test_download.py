import io
import zipfile
import zlib

from tools.download_kitti08 import valid_file


def test_existing_download_requires_size_and_crc(tmp_path):
    payload = b'verified KITTI member'
    info = zipfile.ZipInfo('dataset/sequences/08/calib.txt')
    info.file_size = len(payload)
    info.CRC = zlib.crc32(payload)
    path = tmp_path/'calib.txt'
    assert not valid_file(path, info)
    path.write_bytes(payload)
    assert valid_file(path, info)
    path.write_bytes(b'x'*len(payload))
    assert not valid_file(path, info)


def test_color_extraction_never_overwrites_complete_calibration(tmp_path, monkeypatch):
    import tools.download_kitti08 as downloader
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('dataset/sequences/08/image_2/000000.png', b'left')
        z.writestr('dataset/sequences/08/image_3/000000.png', b'right')
        z.writestr('dataset/sequences/08/calib.txt', b'simplified')
    class LocalZip(io.BytesIO):
        size = len(archive.getvalue())
        downloaded = 0
    monkeypatch.setattr(downloader, 'RemoteZipFile', lambda _:LocalZip(archive.getvalue()))
    (tmp_path/'downloads').mkdir()
    calib = tmp_path/'semantic_kitti/dataset/sequences/08/calib.txt'
    calib.parent.mkdir(parents=True)
    calib.write_bytes(b'full calibration with Tr')
    downloader.extract('color', tmp_path)
    assert calib.read_bytes() == b'full calibration with Tr'
    assert (calib.parent/'image_2/000000.png').read_bytes() == b'left'
