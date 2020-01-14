import hashlib


def calc_scan_hash(image_file, label_file):
    hash_md5 = hashlib.md5()

    with open(image_file, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            hash_md5.update(chunk)

    with open(label_file, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            hash_md5.update(chunk)

    scan_hash = hash_md5.hexdigest()

    return scan_hash
