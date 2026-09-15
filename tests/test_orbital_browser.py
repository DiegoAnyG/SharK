"""Opt-in browser checks for a generated offline report.

Set SHARK_TEST_VIEWER_HTML and install Playwright/Chromium to run these checks.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('SHARK_TEST_VIEWER_HTML'), 'Set SHARK_TEST_VIEWER_HTML for browser verification')
class OrbitalBrowserTests(unittest.TestCase):
    def test_offline_rotation_settings_and_png_export(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, timeout=15000,
                                       args=['--no-sandbox', '--enable-unsafe-swiftshader', '--disable-dev-shm-usage'])
            try:
                page = browser.new_page(viewport={'width': 1500, 'height': 1100}, accept_downloads=True)
                errors, network = [], []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('request', lambda request: network.append(request.url) if request.url.startswith(('https:', 'http:')) else None)
                page.goto(Path(os.environ['SHARK_TEST_VIEWER_HTML']).resolve().as_uri())
                page.wait_for_function('window.sharkReady === true', timeout=45000)
                page.select_option('#iso', '0.04')
                page.wait_for_function("document.getElementById('plot').layout.meta.isovalue === 0.04")
                page.select_option('#view', 'side')
                page.wait_for_function("document.getElementById('plot').layout.scene.camera.eye.y < 0 && document.getElementById('plot').layout.scene.camera.eye.z === 0")
                cameras = page.evaluate("Object.keys(document.getElementById('plot').layout).filter(k=>/^scene[0-9]*$/.test(k)).map(k=>document.getElementById('plot').layout[k].camera)")
                self.assertTrue(all(camera == cameras[0] for camera in cameras))
                # Exercise a user drag, not only the predefined view buttons.
                rect = page.locator('#plot').bounding_box()
                page.mouse.move(rect['x']+rect['width']*.25, rect['y']+220)
                page.mouse.down()
                page.mouse.move(rect['x']+rect['width']*.25+75, rect['y']+250, steps=10)
                page.mouse.up()
                page.wait_for_timeout(500)
                with tempfile.TemporaryDirectory() as tmp:
                    with page.expect_download() as event:
                        page.click('#save')
                    state_path = Path(tmp)/'state.json'
                    event.value.save_as(state_path)
                    state = json.loads(state_path.read_text())
                    self.assertEqual(state['settings']['isovalue'], .04)
                    page.click('#reset')
                    page.set_input_files('#load', state_path)
                    page.wait_for_function("document.getElementById('status').textContent === 'Settings restored.'")
                    page.fill('#scale', '1')
                    page.locator('#scale').dispatch_event('change')
                    with page.expect_download(timeout=45000) as event:
                        page.click('#png')
                    image = Path(tmp)/'orbital.png'
                    event.value.save_as(image)
                    self.assertTrue(image.read_bytes().startswith(b'\x89PNG'))
                    self.assertGreater(image.stat().st_size, 10000)
                self.assertEqual(errors, [])
                self.assertEqual(network, [])
            finally:
                browser.close()


if __name__ == '__main__':
    unittest.main()
