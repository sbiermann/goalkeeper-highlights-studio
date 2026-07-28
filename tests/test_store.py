from pathlib import Path
from goalkeeper_highlights.store import AnalysisStore
from goalkeeper_highlights.models import Candidate

def test_store_roundtrip(tmp_path: Path):
    store=AnalysisStore(tmp_path/'a.db')
    item=Candidate(1,2,1.5,0.2,7)
    store.replace_candidates([item])
    loaded=store.load_candidates()
    assert len(loaded)==1 and loaded[0].keeper_track_id==7
    store.append_detections([(1,0.04,7,0,0.9,1,2,3,4)])
    assert store.connection.execute('select count(*) from detections').fetchone()[0]==1
    store.close()
