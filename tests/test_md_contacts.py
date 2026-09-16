"""Periodic-contact analysis and interactive input boundaries."""
from pathlib import Path
import numpy as np
import pytest
from shark import cli


def test_contacts_use_periodic_box_and_preserve_source_directory(tmp_path):
    mda=pytest.importorskip('MDAnalysis')
    from shark.analysis.md_contacts import analyze_contacts
    u=mda.Universe.empty(2,n_residues=2,atom_resindex=[0,1],trajectory=True)
    u.add_TopologyAttr('names',['CA','C1'])
    u.add_TopologyAttr('resnames',['ALA','LIG'])
    u.add_TopologyAttr('resids',[7,99])
    u.add_TopologyAttr('ids',[1,2])
    u.dimensions=[20,20,20,90,90,90]
    u.atoms.positions=[[1,1,1],[19,1,1]]
    topology=tmp_path/'system.gro';trajectory=tmp_path/'run.xtc'
    u.atoms.write(str(topology))
    with mda.Writer(str(trajectory),n_atoms=2) as writer:
        writer.write(u)
        u.trajectory.ts.time=10
        u.atoms.positions=[[1,1,1],[10,1,1]]
        writer.write(u)
    before=set(tmp_path.iterdir())
    result=analyze_contacts(topology,trajectory,ligand_selection='resname LIG',cutoff_angstrom=3)
    assert set(tmp_path.iterdir())==before
    assert result['sampled_frame_count']==2
    contact=result['contacts'][0]
    assert contact['occupancy_fraction']==.5
    assert contact['minimum_distance_angstrom']==pytest.approx(2,abs=.01)
    assert contact['protein_atom']['resid']==7
    assert contact['ligand_atom']['resid']==99
    assert contact['protein_atom']['chain_id'] is None
    with pytest.raises(ValueError,match='nonempty'):
        analyze_contacts(topology,trajectory,ligand_selection='resname MISSING')
    with pytest.raises(ValueError,match='overlap'):
        analyze_contacts(topology,trajectory,ligand_selection='resname LIG',protein_selection='all')
    with pytest.raises(ValueError,match='Invalid'):
        analyze_contacts(topology,trajectory,ligand_selection='unrecognized_keyword')
    with pytest.raises(ValueError,match='No frames'):
        analyze_contacts(topology,trajectory,ligand_selection='resname LIG',start_ns=100)


def test_interactive_compound_and_frequency_selection(monkeypatch,tmp_path):
    monkeypatch.setattr(cli,'interactive_session_picker',lambda:Path('session.poliscreen'))
    replies=iter(['5','2','1','sample','target','','','1','smiles','y',str(tmp_path/'new')])
    monkeypatch.setattr('builtins.input',lambda _:next(replies))
    captured=[]
    monkeypatch.setattr(cli,'main',lambda args:captured.extend(args) or 0)
    assert cli.run_interactive()==0
    assert captured[captured.index('--compound')+1]=='sample'
    assert '--execute' in captured and '--frequencies' in captured


def test_interactive_existing_md_selection(monkeypatch,tmp_path):
    monkeypatch.setattr(cli,'interactive_session_picker',lambda:Path('session.poliscreen'))
    replies=iter(['4','system.gro','run.xtc','resname UNL and not name H*','4',str(tmp_path/'new')])
    monkeypatch.setattr('builtins.input',lambda _:next(replies))
    captured=[]
    monkeypatch.setattr(cli,'main',lambda args:captured.extend(args) or 0)
    assert cli.run_interactive()==0
    assert '--analyze-md' in captured and '--md' not in captured
    assert captured[captured.index('--ligand-selection')+1]=='resname UNL and not name H*'
