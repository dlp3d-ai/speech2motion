import hashlib


def str_to_md5(input_str: str) -> str:
    """Convert a string to MD5 hash value.

    Args:
        input_str (str):
            Input string to be hashed.

    Returns:
        str:
            MD5 hash value as a hexadecimal string.
    """
    hash_object = hashlib.md5()
    hash_object.update(input_str.encode('utf-8'))
    md5sum = hash_object.hexdigest()
    return md5sum
