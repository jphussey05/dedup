"""Tests for database operations."""



def test_init_schema(db):
    """Schema creates all tables."""
    tables = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {row["name"] for row in tables}
    assert "images" in names
    assert "duplicate_groups" in names
    assert "duplicate_pairs" in names
    assert "scan_sessions" in names
    assert "schema_version" in names


def test_insert_and_check_image(db):
    """Can insert and query images."""
    img_id = db.insert_image(
        path="/test/photo.jpg",
        filename="photo.jpg",
        file_size=1024,
        width=100,
        height=100,
        fmt="JPEG",
    )
    assert img_id is not None
    assert db.image_exists("/test/photo.jpg")
    assert not db.image_exists("/test/other.jpg")


def test_update_sha256(db):
    img_id = db.insert_image("/test/a.jpg", "a.jpg", 1024)
    db.update_sha256(img_id, "abc123")
    db.conn.commit()

    row = db.conn.execute("SELECT sha256 FROM images WHERE id = ?", (img_id,)).fetchone()
    assert row["sha256"] == "abc123"


def test_status_empty(db):
    stats = db.get_status()
    assert stats["total_images"] == 0
    assert stats["hashed"] == 0
    assert stats["exact_groups"] == 0


def test_duplicate_group_operations(db):
    id1 = db.insert_image("/a.jpg", "a.jpg", 100)
    id2 = db.insert_image("/b.jpg", "b.jpg", 100)
    db.conn.commit()

    group_id = db.insert_duplicate_group("exact", 1.0, 2)
    db.insert_duplicate_pair(group_id, id1, id2, 1.0)
    db.conn.commit()

    stats = db.get_status()
    assert stats["exact_groups"] == 1

    db.clear_groups("exact")
    db.conn.commit()

    stats = db.get_status()
    assert stats["exact_groups"] == 0
