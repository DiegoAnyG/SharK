"""Isolated-ligand chemistry, provenance and ORCA lifecycle tests."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import pytest
from rdkit import Chem

from shark.cli import main
from shark.core.runner import read_qm_results, run_orca_job
from shark.core.session import read_poliscreen_session
from shark.workflows.ligand_qm import prepare_ligand_jobs, select_ligands


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / 'input.poliscreen'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('manifest.json', '{"format": 1, "poliscreen": "test", "full": false}')
        z.writestr('ranking.csv', 'compound,receptor,best_dock,is_control,is_pareto\n'
                   'lig_A,rec~Pk1,-2,0,true\nlig_A,rec~Pk2,-3,0,true\n'
                   'lig_B,rec~Pk1,-1,0,false\nreference,rec~Pk1,-9,1,true\n')
        z.writestr('ligands_meta.csv', 'name,smiles\nlig_A,C[C@H](O)C(=O)[O-]\nlig_B,CO\nreference,O\n')
    return path


@pytest.fixture
def session(archive, tmp_path):
    return read_poliscreen_session(archive, tmp_path / 'unpacked')


def test_selection_is_target_specific_and_unique(session):
    selected = select_ligands(session, top=5)
    assert [s['ligand_id'] for s in selected] == ['lig_A', 'lig_B']
    assert set(selected[0]['targets']) == {'rec~Pk1', 'rec~Pk2'}
    assert len(select_ligands(session, pareto=True, top=5)) == 1
    assert select_ligands(session, target='rec~Pk1')[0]['targets'] == ['rec~Pk1']
    with pytest.raises(ValueError, match='absent'):
        select_ligands(session, compounds=['lig'])


def test_preparation_reproducible_and_preserves_chemistry(session, tmp_path):
    first = prepare_ligand_jobs(session, tmp_path / 'first')[0]
    second = prepare_ligand_jobs(session, tmp_path / 'second')[0]
    assert (first / 'calculation.inp').read_bytes() == (second / 'calculation.inp').read_bytes()
    data = json.loads((first / 'job.json').read_text())
    assert data['status'] == 'prepared' and data['results'] is None
    assert data['parameters']['charge'] == -1
    assert '@' in data['source']['smiles']
    assert data['source']['geometry'] == 'ETKDGv3'
    assert str(tmp_path) not in (first / 'job.json').read_text()
    assert data['inputs'] == json.loads((second / 'job.json').read_text())['inputs']
    with pytest.raises(ValueError, match='never overwritten'):
        prepare_ligand_jobs(session, tmp_path / 'first')


def test_invalid_charge_spin_and_resources_fail_before_writing(session, tmp_path):
    for kwargs in ({'charge': 1}, {'multiplicity': 2}, {'nprocs': 0}, {'maxcore': 0}, {'method': 'PBE\n*xyz 0 1'}):
        with pytest.raises(ValueError):
            prepare_ligand_jobs(session, tmp_path / 'invalid', **kwargs)
        assert not (tmp_path / 'invalid').exists()


def test_mismatched_input_chemistry_is_rejected(session, tmp_path):
    wrong = tmp_path / 'wrong.mol'
    Chem.MolToMolFile(Chem.MolFromSmiles('CC'), str(wrong))
    session.ligand_files['lig_a'] = wrong
    # Keep the input within the session so provenance validation is meaningful.
    moved = session.extract_dir / 'wrong.mol'
    moved.write_bytes(wrong.read_bytes())
    session.ligand_files['lig_a'] = moved
    with pytest.raises(ValueError, match='disagree'):
        prepare_ligand_jobs(session, tmp_path / 'invalid')


def test_cli_prepare_has_no_simulated_results(archive, tmp_path, monkeypatch):
    monkeypatch.setenv('SHARK_SCRATCH', str(tmp_path))
    out = tmp_path / 'jobs'
    assert main(['run-qm', '--session', str(archive), '--work-dir', str(out)]) == 0
    text = (out / 'dossier.html').read_text()
    assert 'Not calculated' in text and 'prepared' in text
    assert '-25.40' not in text and '2.94 +- 0.33' not in text
    assert not list(tmp_path.glob('shark-session-*'))


def test_parser_requires_final_converged_data(tmp_path):
    output = tmp_path / 'calculation.out'
    output.write_text('ORCA TERMINATED NORMALLY\n')
    with pytest.raises(ValueError, match='optimization'):
        read_qm_results(output, optimize=True, frequencies=False)
    with pytest.raises(ValueError, match='electronic energy'):
        read_qm_results(output, optimize=False, frequencies=False)
    output.write_text('ORBITAL ENERGIES\nNO OCC E(Eh) E(eV)\n0 2.0 -0.4 -10.0\n1 0.0 0.1 3.0\n\n'
                      'ORBITAL ENERGIES\nNO OCC E(Eh) E(eV)\n0 2.0 -0.3 -8.0\n1 0.0 0.0 0.0\n\n'
                      'FINAL SINGLE POINT ENERGY -1.0\nORCA TERMINATED NORMALLY\n')
    result = read_qm_results(output, optimize=False, frequencies=False)
    assert result['orbitals']['0']['homo']['energy_eV'] > -10
    assert result['orbitals']['0']['gap_ev'] == -result['orbitals']['0']['homo']['energy_eV']
    assert not result['stationary_minimum_verified']


def test_missing_orca_and_modified_input_are_not_success(session, tmp_path, monkeypatch):
    job = prepare_ligand_jobs(session, tmp_path / 'jobs')[0]
    monkeypatch.setenv('SHARK_ORCA', str(tmp_path / 'missing'))
    result = run_orca_job(job)
    assert result['status'] == 'failed' and result['results'] is None
    changed = prepare_ligand_jobs(session, tmp_path / 'changed')[0]
    (changed / 'calculation.inp').write_text('changed')
    with pytest.raises(ValueError, match='inputs changed'):
        run_orca_job(changed)


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process-group boundary')
def test_timeout_kills_child_even_when_parent_exits(session, tmp_path, monkeypatch):
    job = prepare_ligand_jobs(session, tmp_path / 'jobs')[0]
    executable = tmp_path / 'fake-orca'
    executable.write_text('#!' + sys.executable + '\n'
                          'import subprocess, sys, time\n'
                          'from pathlib import Path\n'
                          'p = subprocess.Popen([sys.executable, "-c", '
                          '"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])\n'
                          'Path("child.pid").write_text(str(p.pid))\n'
                          'time.sleep(60)\n')
    executable.chmod(0o755)
    monkeypatch.setenv('SHARK_ORCA', str(executable))
    result = run_orca_job(job, timeout=0.5)
    assert result['status'] == 'failed' and 'timed out' in result['error']
    pid = int((job / 'child.pid').read_text())
    time.sleep(0.1)
    proc_status = Path(f'/proc/{pid}/stat')
    if proc_status.exists():
        assert proc_status.read_text().split()[2] in ('Z', 'X', 'T')


def test_explicit_smiles_geometry_bypasses_unsupported_structure(session, tmp_path):
    invalid = session.extract_dir / 'invalid.mol2'
    invalid.write_text('Unsupported structure')
    session.ligand_files['lig_a'] = invalid
    job = prepare_ligand_jobs(session, tmp_path / 'jobs', geometry='smiles')[0]
    data = json.loads((job / 'job.json').read_text())
    assert data['source']['kind'] == 'recorded_smiles'
    assert data['parameters']['geometry'] == 'smiles'


def test_successful_execution_records_real_output_and_hashes(session, tmp_path, monkeypatch):
    job = prepare_ligand_jobs(session, tmp_path / 'jobs', orbital_grid=None)[0]
    executable = tmp_path / 'fixture-orca'
    fixture = ('Program Version fixture\nTHE OPTIMIZATION HAS CONVERGED\n'
               'FINAL SINGLE POINT ENERGY -1.0\nORBITAL ENERGIES\n'
               'NO OCC E(Eh) E(eV)\n0 2.0 -0.3 -8.0\n1 0.0 0.0 0.0\n\n'
               'ORCA TERMINATED NORMALLY\n')
    executable.write_text('#!' + sys.executable + '\nprint(' + repr(fixture) + ')\n')
    executable.chmod(0o755)
    monkeypatch.setenv('SHARK_ORCA', str(executable))
    result = run_orca_job(job)
    assert result['status'] == 'completed'
    assert result['results']['orca_version'] == 'fixture'
    assert result['outputs']['calculation.out']
    assert result['orca_executable_sha256']
    assert not result['results']['stationary_minimum_verified']
    with pytest.raises(ValueError, match='Only a prepared'):
        run_orca_job(job)


@pytest.mark.skipif(os.name != 'posix', reason='POSIX signal boundary')
def test_sigterm_preserves_interrupted_job(session, tmp_path):
    job = prepare_ligand_jobs(session, tmp_path / 'jobs')[0]
    executable = tmp_path / 'fixture-orca'
    executable.write_text('#!' + sys.executable + '\n'
                          'import time\nfrom pathlib import Path\n'
                          'Path("started").touch()\ntime.sleep(60)\n')
    executable.chmod(0o755)
    env = os.environ.copy()
    env['SHARK_ORCA'] = str(executable)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    child = subprocess.Popen([sys.executable, '-c',
                              'import sys; from shark.core.runner import run_orca_job; run_orca_job(sys.argv[1])',
                              str(job)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not (job / 'started').exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (job / 'started').exists()
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=8)
        record = json.loads((job / 'job.json').read_text())
        assert record['status'] == 'interrupted' and record['results'] is None
        assert 'calculation.out' in record['outputs']
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_dossier_escapes_session_fields(tmp_path):
    from shark.reports.dossier import generate_html_dossier
    attack = '</script><script>window.injected=true</script>'
    file = generate_html_dossier(attack, [{'ligand_id': attack, 'score': None}], tmp_path / 'report.html')
    text = file.read_text()
    assert attack not in text
    assert '<script src=' not in text
    assert 'N/A' in text
