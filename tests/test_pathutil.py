"""Tests for path normalization."""


from dedup.pathutil import get_read_chunk_size, is_unc_path


def test_is_unc_path():
    assert is_unc_path("//NAS/share/photo.jpg")
    assert is_unc_path("\\\\NAS\\share\\photo.jpg")
    assert not is_unc_path("C:/Users/photos/img.jpg")
    assert not is_unc_path("/home/user/photos/img.jpg")


def test_chunk_size_unc():
    assert get_read_chunk_size("//NAS/share/photo.jpg") == 65536


def test_chunk_size_local():
    assert get_read_chunk_size("C:/Users/photos/img.jpg") == 8192
