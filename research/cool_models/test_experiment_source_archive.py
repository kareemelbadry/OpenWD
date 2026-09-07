import hashlib
import tarfile
import pytest
from experiment_source_archive import archive_sources


def test_archived_sources_match_recorded_hash_even_after_local_edit(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source=tmp_path/'model.py'
    source.write_text('version = 1\n')
    hashes=archive_sources(['model.py'],'sources.tar.gz')
    source.write_text('version = 2\n')
    with tarfile.open('sources.tar.gz') as archive:
        saved=archive.extractfile('model.py').read()
    assert saved==b'version = 1\n'
    assert hashes['model.py']==hashlib.sha256(saved).hexdigest()
    with pytest.raises(FileExistsError):archive_sources(['model.py'],'sources.tar.gz')


@pytest.mark.parametrize('paths',[['../escape.py'],['/absolute.py'],['a.py','a.py']])
def test_invalid_source_selection_rejected(tmp_path,paths):
    with pytest.raises(ValueError):archive_sources(paths,tmp_path/'sources.tar.gz')
