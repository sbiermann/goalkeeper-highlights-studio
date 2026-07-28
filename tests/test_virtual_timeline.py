from goalkeeper_highlights.sources import SourceItem, SourceManifest


def test_locate_maps_global_time_to_source_time():
    manifest = SourceManifest("directory", "/game", 30.0, [
        SourceItem("a.mp4", "/game/a.mp4", 10.0, 0.0, 10.0),
        SourceItem("b.mp4", "/game/b.mp4", 20.0, 10.0, 30.0),
    ])
    item, local = manifest.locate(12.5)
    assert item.name == "b.mp4"
    assert local == 2.5


def test_boundary_belongs_to_next_source():
    manifest = SourceManifest("directory", "/game", 20.0, [
        SourceItem("a.mp4", "/game/a.mp4", 10.0, 0.0, 10.0),
        SourceItem("b.mp4", "/game/b.mp4", 10.0, 10.0, 20.0),
    ])
    item, local = manifest.locate(10.0)
    assert item.name == "b.mp4"
    assert local == 0.0
