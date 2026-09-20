"""Contract and extraction boundaries for PoliScreen archives."""

import json
import shutil
import stat
import zipfile

import pytest

from shark.core.session import read_poliscreen_session


@pytest.fixture
def sample_session_zip(tmp_path):
    path = tmp_path / 'sample.poliscreen'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('manifest.json', json.dumps({'format': 1, 'project': 'Screening_Test', 'full': True}))
        z.writestr('ranking.csv', 'ligand,score\nLIG_A,-8.5\nLIG_B,-7.2\n')
        z.writestr('receptores/target.pdb', 'END\n')
        for name in ('LIG_A_pose_1', 'LIG_A_pose_2', 'LIG_B_pose_1'):
            z.writestr(f'poses/{name}.pdb', 'END\n')
    return path


def test_legacy_identity_preserved(sample_session_zip, tmp_path):
    session = read_poliscreen_session(sample_session_zip, tmp_path / 'unpacked')
    assert {p.ligand_id for p in session.poses} == {'LIG_A', 'LIG_B'}
    assert session.get_pose('LIG_A', 2).pose_file.name == 'LIG_A_pose_2.pdb'
    assert session.list_top_poses(1)[0].ligand_id == 'LIG_A'
    assert session.archive_sha256


def test_current_multi_pocket_contract(tmp_path):
    archive = tmp_path / 'modern.poliscreen'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('manifest.json', '{"format": 1, "full": true}')
        rows = ['receptor,pose_name,compound_name,docking_score,engine']
        for pocket in ('target_ready~Pk1', 'target_ready~Pk2'):
            name = f'docking_{pocket}_compounds_a_lig_with_underscores'
            for model in (1, 2):
                stem = f'{name}-model{model}'
                rows.append(f'{pocket},{stem},lig_with_underscores,{-5-model},vina')
                z.writestr(f'poses/{stem}.pdb', 'END\n')
                z.writestr(f'fused_complexes/Complex_{stem}.pdb', 'END\n')
            z.writestr(f'poses/{name}.pdbqt', 'MODEL 1\nENDMDL\nMODEL 2\nENDMDL')
        z.writestr('docking_results.csv', '\n'.join(rows))
        z.writestr('receptors/target_ready.pdb', 'END\n')
        z.writestr('receptors/target.pdb', 'END\n')
    session = read_poliscreen_session(archive, tmp_path / 'unpacked')
    assert len(session.poses) == 4
    assert {p.ligand_id for p in session.poses} == {'lig_with_underscores'}
    assert all(p.complex_file.is_file() for p in session.poses)
    with pytest.raises(ValueError, match='Ambiguous'):
        session.get_pose('lig_with_underscores')
    pose = session.get_pose('lig_with_underscores', 2, 'target_ready~Pk2')
    assert pose.receptor_id == 'target_ready~Pk2'
    assert session.receptor_for(pose.receptor_id).name == 'target_ready.pdb'


@pytest.mark.parametrize('name', ['../dest-other/file', '../../file', '/file', 'C:/file', r'..\file'])
def test_escaping_archive_paths(tmp_path, name):
    archive = tmp_path / 'bad.poliscreen'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(name, 'bad')
    with pytest.raises(ValueError, match='Insecure path'):
        read_poliscreen_session(archive, tmp_path / 'dest')
    assert not (tmp_path / 'dest').exists()


def test_links_limits_and_manifest_errors(tmp_path):
    archive = tmp_path / 'bad.poliscreen'
    member = zipfile.ZipInfo('link')
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(member, '../outside')
    with pytest.raises(ValueError, match='Insecure'):
        read_poliscreen_session(archive)
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('manifest.json', '{"format": 99}')
    with pytest.raises(ValueError, match='maximum unpacked'):
        read_poliscreen_session(archive, max_unpacked_bytes=1)
    with pytest.raises(ValueError, match='Unsupported'):
        read_poliscreen_session(archive)


def test_same_name_sessions_do_not_reuse_stale_files(sample_session_zip, tmp_path, monkeypatch):
    monkeypatch.setenv('SHARK_SCRATCH', str(tmp_path))
    first = read_poliscreen_session(sample_session_zip)
    second = read_poliscreen_session(sample_session_zip)
    assert first.extract_dir != second.extract_dir
    with pytest.raises(ValueError, match='empty directory'):
        read_poliscreen_session(sample_session_zip, first.extract_dir)
    shutil.rmtree(first.extract_dir)
    shutil.rmtree(second.extract_dir)


def test_producer_export_can_be_read(tmp_path):
    producer = pytest.importorskip('poliscreen.core.session')
    project = tmp_path / 'project'
    (project / 'input_ligands').mkdir(parents=True)
    (project / 'input_ligands' / 'candidate.mol').write_text('fixture')
    (project / 'ranking.csv').write_text('compound,best_dock\ncandidate,-1\n')
    archive = producer.save_session(project, tmp_path / 'export', full_=False)
    result = read_poliscreen_session(archive, tmp_path / 'unpacked')
    assert result.manifest['format'] == 1
    assert result.ligand_files['candidate'].name == 'candidate.mol'
    assert result.poses == []


def test_match_receptor_id():
    from shark.core.session import match_receptor_id
    assert match_receptor_id('8HTB_ready', '8HTB') is True
    assert match_receptor_id('8HTB_ready~Pk1', '8HTB') is True
    assert match_receptor_id('8HTB_ready~Pk1', '8HTB_ready') is True
    assert match_receptor_id('8HTB_clean', '8HTB') is True
    assert match_receptor_id('8HTB_prep', '8HTB') is True
    assert match_receptor_id('4D44_ready', '8HTB') is False
    assert match_receptor_id('8HTB', None) is True
    assert match_receptor_id('8HTB', '') is True
    assert match_receptor_id(None, '8HTB') is False


def test_case_insensitive_and_target_normalized_pose_lookup(sample_session_zip, tmp_path):
    session = read_poliscreen_session(sample_session_zip, tmp_path / 'unpacked_lookup')
    # Original pose stored with ligand_id='LIG_A' and receptor_id='target'
    pose = session.get_pose('lig_a', 1, 'target')
    assert pose is not None
    assert pose.ligand_id == 'LIG_A'
    # Check that receptor normalization works with get_pose
    pose_norm = session.get_pose('lig_a', 1, 'target_ready')
    assert pose_norm is not None
    assert pose_norm.ligand_id == 'LIG_A'

