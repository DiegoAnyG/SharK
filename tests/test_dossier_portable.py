"""Portability and trust boundaries for embedded local orbital reports."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import pytest
from shark.reports.dossier import generate_html_dossier


def test_viewer_is_embedded_with_hash_check_and_sandbox(tmp_path):
    viewer = tmp_path/'viewer.html'
    viewer.write_text('<html><head></head><body>orbitals<script>window.sharkReady=true</script></body></html>')
    job = dict(selection={'ligand_id':'sample'}, status='completed', results=None,
               viewer_link=viewer.name, outputs={viewer.name:hashlib.sha256(viewer.read_bytes()).hexdigest()})
    output = generate_html_dossier('sample',[],tmp_path/'report.html',qm_summary={'jobs':[job]})
    text=output.read_text()
    assert 'sandbox="allow-scripts allow-downloads"' in text
    assert 'connect-src' in text and 'srcdoc=' in text
    assert '<a href="viewer.html"' not in text
    assert 'href="file://' not in text and 'src="file://' not in text
    viewer.unlink()
    assert 'window.sharkReady=true' in output.read_text()
    viewer.write_text('<html><head></head><body>changed</body></html>')
    with pytest.raises(ValueError,match='hash'):
        generate_html_dossier('sample',[],output,qm_summary={'jobs':[job]})
    job['viewer_link']='https://example.invalid/viewer.html'
    with pytest.raises(ValueError,match='local'):
        generate_html_dossier('sample',[],output,qm_summary={'jobs':[job]})


@pytest.mark.skipif(not os.environ.get('SHARK_TEST_DOSSIER_HTML'), reason='Set SHARK_TEST_DOSSIER_HTML for browser verification')
def test_standalone_dossier_desktop_mobile_and_export(tmp_path):
    from playwright.sync_api import sync_playwright
    report=tmp_path/'transported.html'
    shutil.copy2(os.environ['SHARK_TEST_DOSSIER_HTML'], report)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader','--disable-dev-shm-usage'])
        try:
            page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
            errors=[];network=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('request',lambda r:network.append(r.url) if r.url.startswith(('http:','https:')) else None)
            page.goto(report.resolve().as_uri())
            page.wait_for_function('window.sharkDossierReady === true')
            for value in page.locator('#orbital-job option').evaluate_all('(options)=>options.map(o=>o.value)'):
                page.select_option('#orbital-job',value)
                frame=page.frames[1]
                frame.wait_for_function('window.sharkEmbeddedReady === true',timeout=45000)
                frame.select_option('#iso','0.04')
                frame.wait_for_function("document.getElementById('plot').layout.meta.isovalue===0.04")
            frame.select_option('#view','side')
            frame.fill('#scale','1');frame.locator('#scale').dispatch_event('change')
            with page.expect_download(timeout=45000) as event:frame.click('#png')
            event.value.save_as(tmp_path/'orbital.png')
            assert (tmp_path/'orbital.png').read_bytes().startswith(b'\x89PNG')
            page.set_viewport_size({'width':375,'height':900})
            frame.wait_for_function("document.getElementById('plot').layout.width < 375",timeout=10000)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert frame.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert frame.evaluate("document.getElementById('plot').layout.scene.domain.x[1]===1")
            assert not errors and not network
        finally:
            browser.close()


def test_covalent_dossier_rendering(tmp_path):
    covalent_data = {
        "has_nac": True,
        "summary": "Positive Near-Attack Conformation detected for LIG1 with 8HTB.",
        "cdft": {
            "hardness_ev": 3.45,
            "chemical_potential_ev": -4.62,
            "electrophilicity_ev": 3.09,
            "softness_ev": 0.145,
        },
        "contacts": [
            {
                "residue": "CYS145:A",
                "nucleophile_atom": "SG",
                "ligand_atom_index": 2,
                "ligand_atom_element": "C",
                "distance_angstrom": 3.12,
                "feasibility_score": 0.85,
                "is_nac": True
            }
        ],
        "pocket_nucleophiles": ["CYS145:A"],
        "nac_contacts": [{"residue": "CYS145:A"}]
    }

    out_file = tmp_path / "covalent_dossier.html"
    res_path = generate_html_dossier("CovalentTest", [], out_file, covalent_summary=covalent_data)
    text = res_path.read_text(encoding="utf-8")

    assert "Covalent Near-Attack Conformations (NAC)" in text
    assert "NAC Observed" in text
    assert "CYS145:A" in text
    assert "3.12" in text
    assert "Electrophilicity index" in text
    assert "3.09 eV" in text
    assert "covalent-3d-plot" in text
    assert "covalent-curve-plot" in text
    assert "fukui-bar-plot" in text
    assert "Bruice Near-Attack Feasibility Curve" in text


def test_load_dft_records_and_render_combined(tmp_path):
    from shark.cli import load_dft_records
    dft_dir = Path("dft_benzofuroxan")
    records = load_dft_records(dft_dir, tmp_path)
    assert len(records) == 2
    for r in records:
        assert r["status"] == "completed"
        assert r["results"]["stationary_minimum_verified"] is True
        assert r["results"]["orbitals"]["0"]["homo"]["energy_eV"] < 0
        assert r["results"]["orbitals"]["0"]["lumo"]["energy_eV"] < 0

    out_file = tmp_path / "combined_dossier.html"
    res = generate_html_dossier("CombinedTest", [], out_file, qm_summary={"jobs": records})
    content = res.read_text(encoding="utf-8")
    assert "Dominant Tautomer" in content
    assert "tautomer_3_oxide" in content
    assert "Verified Quantum Minima" in content
    assert "2/2 Minima" in content


def test_covalent_dossier_with_adduct_qm(tmp_path):
    from shark.analysis.adduct_qm import compute_adduct_quantum_profile

    prof = compute_adduct_quantum_profile(
        distance_angstrom=3.33,
        burgi_dunitz_angle_deg=132.7,
        nucleophile_homo_ev=-6.40,
        electrophile_lumo_ev=-2.89,
        target_atom_index=10,
        target_atom_symbol="O",
    )

    covalent_data = {
        "has_nac": True,
        "summary": "Verified reactive trajectory against THR309:OG1.",
        "contacts": [
            {
                "residue": "THR309:A",
                "nucleophile_atom": "OG1",
                "ligand_atom_index": 10,
                "distance_angstrom": 3.33,
                "burgi_dunitz_angle": 132.7,
                "feasibility_score": 0.88,
                "is_nac": True,
            }
        ],
        "adduct_qm": prof.to_dict(),
    }

    out_file = tmp_path / "adduct_qm_dossier.html"
    res_path = generate_html_dossier("AdductQMTest", [], out_file, covalent_summary=covalent_data)
    text = res_path.read_text(encoding="utf-8")

    assert "Quantum Chemical Adduct Verification &amp; FMO Overlap Theory" in text
    assert "1. FMO Phase Symmetry" in text
    assert "Constructive Allowed" in text
    assert "2. Pocket Polarization" in text
    assert "3. Regiospecificity" in text
    assert "4. Formed Bond Nature" in text
    assert "Wiberg Bond Order" in text


def test_legacy_invalid_tier4_claims_are_removed(tmp_path):
    from shark.reports.dossier import _sanitize_covalent_summary

    ts_dir = tmp_path / "transition_state"
    ts_dir.mkdir()
    (ts_dir / "reaction_definition.json").write_text(json.dumps({
        "electrophile": {"index": 11, "element": "C", "label": "LIG:C5"},
    }), encoding="utf-8")
    summary = {
        "contacts": [
            {"ligand_atom_index": 10, "ligand_atom_element": "O"},
            {"ligand_atom_index": 11, "ligand_atom_element": "C"},
        ],
        "transition_state": {
            "is_first_order_ts": False,
            "activation_barrier_kcal": -147.81,
            "delta_g_activation_kcal": -147.81,
        },
        "total_feasibility": {"delta_g_ts": -147.81, "ts_score": 1.0},
        "adduct_viewer_html": "invalid O-O viewer",
        "adduct_qm": {"bond_nature": {
            "evidence_type": "model_derived",
            "wiberg_bond_order": 1.03,
            "bond_type": "O-O",
        }},
    }

    clean = _sanitize_covalent_summary(summary, tmp_path)

    assert clean["transition_state"]["activation_barrier_kcal"] is None
    assert clean["total_feasibility"]["delta_g_ts"] is None
    assert clean["adduct_qm"] is None
    assert "adduct_viewer_html" not in clean
    assert clean["contacts"][0]["ligand_atom_index"] == 11
