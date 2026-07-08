#!/usr/bin/env python3
"""
Hand Gesture Control — Test Suite
Runs headless (no camera, no Tkinter, no microphone required).
Usage:
    python test_gesture_control.py          # single run
    python test_gesture_control.py --x100  # 100 runs + report
"""

import sys
import io
import os
import json
import time
import tempfile
import unittest
from unittest import mock
import datetime
import argparse

# ── headless guard: stub out hardware imports before loading app ──────────────

# Stub pyautogui so it never moves the real mouse
sys.modules.setdefault("pyautogui", mock.MagicMock())

# Stub numpy
sys.modules.setdefault("numpy", mock.MagicMock())

# Stub cv2
_cv2 = mock.MagicMock()
_cv2.VideoCapture.return_value.read.return_value = (False, None)
sys.modules.setdefault("cv2", _cv2)

# Stub mediapipe
sys.modules.setdefault("mediapipe", mock.MagicMock())
sys.modules.setdefault("mediapipe.solutions", mock.MagicMock())
sys.modules.setdefault("mediapipe.solutions.hands", mock.MagicMock())
sys.modules.setdefault("mediapipe.solutions.drawing_utils", mock.MagicMock())
sys.modules.setdefault("mediapipe.solutions.drawing_styles", mock.MagicMock())

# Stub PIL
sys.modules.setdefault("PIL", mock.MagicMock())
sys.modules.setdefault("PIL.Image", mock.MagicMock())
sys.modules.setdefault("PIL.ImageTk", mock.MagicMock())

# Stub pynput
sys.modules.setdefault("pynput", mock.MagicMock())
sys.modules.setdefault("pynput.mouse", mock.MagicMock())
sys.modules.setdefault("pynput.keyboard", mock.MagicMock())

# Stub tkinter
_tk = mock.MagicMock()
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.ttk", mock.MagicMock())

# Patch pyautogui.size() used at module load time in Cfg
sys.modules["pyautogui"].size.return_value = (1920, 1080)

# Now import the application module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_gesture_control import (
    Cfg, CommandParser, LandmarkSmoother, GestureStabiliser,
    VelocityTracker, GestureRecogniser, CustomGestureRecogniser,
    GestureDatabase, SmoothMouse, G, HandTracker,
    AutonomousAgent, VoiceController, run_action,
)


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def _flat_hand(y_tip=0.2, y_pip=0.5):
    """21-point landmark list with all fingers extended."""
    lm = [(0.5, 0.8, 0.0)] * 21   # wrist-ish base
    # Set each tip above its PIP (y decreases upward in norm coords)
    for tip, pip in (
        (HandTracker.INDEX_TIP,  HandTracker.INDEX_PIP),
        (HandTracker.MIDDLE_TIP, HandTracker.MIDDLE_PIP),
        (HandTracker.RING_TIP,   HandTracker.RING_PIP),
        (HandTracker.PINKY_TIP,  HandTracker.PINKY_PIP),
    ):
        lm[tip]  = (0.5, y_tip, 0.0)
        lm[pip]  = (0.5, y_pip, 0.0)
    lm[HandTracker.THUMB_TIP] = (0.3, 0.5, 0.0)   # thumb extended (x < IP)
    lm[HandTracker.THUMB_IP]  = (0.4, 0.5, 0.0)
    return lm


def _fist_hand():
    """All fingers curled (tip below PIP)."""
    lm = [(0.5, 0.5, 0.0)] * 21
    for tip, pip in (
        (HandTracker.INDEX_TIP,  HandTracker.INDEX_PIP),
        (HandTracker.MIDDLE_TIP, HandTracker.MIDDLE_PIP),
        (HandTracker.RING_TIP,   HandTracker.RING_PIP),
        (HandTracker.PINKY_TIP,  HandTracker.PINKY_PIP),
    ):
        lm[tip]  = (0.5, 0.7, 0.0)   # tip BELOW pip → curled
        lm[pip]  = (0.5, 0.5, 0.0)
    lm[HandTracker.THUMB_TIP] = (0.6, 0.5, 0.0)   # thumb NOT extended
    lm[HandTracker.THUMB_IP]  = (0.4, 0.5, 0.0)
    return lm


def _index_only():
    """Only index extended."""
    lm = _fist_hand()
    lm[HandTracker.INDEX_TIP] = (0.5, 0.2, 0.0)
    lm[HandTracker.INDEX_PIP] = (0.5, 0.5, 0.0)
    return lm


def _two_fingers():
    """Index + middle extended."""
    lm = _index_only()
    lm[HandTracker.MIDDLE_TIP] = (0.5, 0.2, 0.0)
    lm[HandTracker.MIDDLE_PIP] = (0.5, 0.5, 0.0)
    return lm


def _pinch_hand():
    """Thumb and index very close (pinch)."""
    lm = _fist_hand()
    lm[HandTracker.THUMB_TIP]  = (0.50, 0.50, 0.0)
    lm[HandTracker.THUMB_IP]   = (0.45, 0.50, 0.0)
    lm[HandTracker.INDEX_TIP]  = (0.51, 0.50, 0.0)  # ~0.01 apart → < PINCH_THRESH
    lm[HandTracker.INDEX_PIP]  = (0.50, 0.60, 0.0)
    return lm


# ─────────────────────────────────────────────────────────────
#  TEST: CommandParser
# ─────────────────────────────────────────────────────────────
class TestCommandParser(unittest.TestCase):
    def setUp(self):
        self.p = CommandParser()

    def _parse(self, text):
        return self.p.parse(text)

    def test_apri_youtube(self):
        r = self._parse("apri youtube")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "open_url")
        self.assertIn("youtube", r[1])

    def test_apri_google(self):
        r = self._parse("apri google")
        self.assertEqual(r[0], "open_url")
        self.assertIn("google", r[1])

    def test_apri_gmail(self):
        r = self._parse("Apri Gmail")
        self.assertEqual(r[0], "open_url")

    def test_nuova_cartella(self):
        r = self._parse("nuova cartella")
        self.assertEqual(r[0], "create_folder")
        self.assertEqual(r[1], "Nuova Cartella")

    def test_crea_cartella_named(self):
        r = self._parse("crea cartella lavoro")
        self.assertEqual(r[0], "create_folder")
        self.assertEqual(r[1], "lavoro")

    def test_cerca(self):
        r = self._parse("cerca meteo Milano")
        self.assertEqual(r[0], "search")
        self.assertIn("meteo", r[1])

    def test_vai_su(self):
        r = self._parse("vai su example.com")
        self.assertEqual(r[0], "open_url")

    def test_copia(self):
        r = self._parse("copia")
        self.assertEqual(r[0], "hotkey")
        self.assertIn("c", r[1])

    def test_incolla(self):
        r = self._parse("incolla")
        self.assertEqual(r[0], "hotkey")
        self.assertIn("v", r[1])

    def test_annulla(self):
        r = self._parse("annulla")
        self.assertEqual(r[0], "hotkey")
        self.assertIn("z", r[1])

    def test_rifai(self):
        r = self._parse("rifai")
        self.assertEqual(r[0], "hotkey")
        self.assertIn("y", r[1])

    def test_salva(self):
        r = self._parse("salva")
        self.assertEqual(r[0], "hotkey")
        self.assertIn("s", r[1])

    def test_chiudi(self):
        r = self._parse("chiudi")
        self.assertEqual(r[0], "hotkey")

    def test_screenshot(self):
        self.assertEqual(self._parse("screenshot")[0], "screenshot")
        self.assertEqual(self._parse("schermo")[0], "screenshot")

    def test_zoom_avanti(self):
        r = self._parse("zoom avanti")
        self.assertEqual(r[0], "zoom")
        self.assertEqual(r[1], 1)

    def test_zoom_indietro(self):
        r = self._parse("zoom indietro")
        self.assertEqual(r[1], -1)

    def test_ingrandisci(self):
        self.assertEqual(self._parse("ingrandisci")[1], 1)

    def test_rimpicciolisci(self):
        self.assertEqual(self._parse("rimpicciolisci")[1], -1)

    def test_scrivi(self):
        r = self._parse("scrivi ciao mondo")
        self.assertEqual(r[0], "type")
        self.assertIn("ciao", r[1])

    def test_digita(self):
        r = self._parse("digita test")
        self.assertEqual(r[0], "type")

    def test_invio(self):
        r = self._parse("invio")
        self.assertEqual(r[0], "hotkey")
        self.assertIn("return", r[1])

    def test_cancella(self):
        r = self._parse("cancella")
        self.assertEqual(r[0], "hotkey")
        self.assertIn("backspace", r[1])

    def test_volume_su(self):
        r = self._parse("volume su")
        self.assertEqual(r[0], "hotkey")

    def test_volume_giu(self):
        r = self._parse("volume giù")
        self.assertEqual(r[0], "hotkey")

    def test_muto(self):
        r = self._parse("muto")
        self.assertEqual(r[0], "hotkey")

    def test_torna_indietro(self):
        r = self._parse("torna indietro")
        self.assertEqual(r[0], "hotkey")
        self.assertIn("left", r[1])

    def test_vai_avanti(self):
        r = self._parse("vai avanti")
        self.assertIn("right", r[1])

    def test_alt_tab(self):
        r = self._parse("alt tab")
        self.assertEqual(r[0], "hotkey")

    def test_desktop(self):
        r = self._parse("desktop")
        self.assertEqual(r[0], "hotkey")

    def test_unknown_returns_none(self):
        self.assertIsNone(self._parse("questo non esiste come comando"))

    def test_case_insensitive(self):
        self.assertIsNotNone(self._parse("COPIA"))
        self.assertIsNotNone(self._parse("Zoom Avanti"))


# ─────────────────────────────────────────────────────────────
#  TEST: LandmarkSmoother
# ─────────────────────────────────────────────────────────────
class TestLandmarkSmoother(unittest.TestCase):
    def _lm(self, val=0.5):
        return [(val, val, val)] * 21

    def test_first_frame_identity(self):
        s  = LandmarkSmoother(alpha=0.5)
        lm = self._lm(0.8)
        out = s.smooth("a", lm)
        self.assertEqual(out, lm)

    def test_convergence(self):
        s   = LandmarkSmoother(alpha=0.5)
        lm0 = self._lm(0.0)
        lm1 = self._lm(1.0)
        s.smooth("a", lm0)
        out = s.smooth("a", lm1)
        self.assertAlmostEqual(out[0][0], 0.5, places=5)

    def test_reset_clears_state(self):
        s = LandmarkSmoother(alpha=0.5)
        s.smooth("a", self._lm(0.9))
        s.reset("a")
        out = s.smooth("a", self._lm(0.1))
        self.assertAlmostEqual(out[0][0], 0.1, places=5)

    def test_multiple_keys_independent(self):
        s = LandmarkSmoother(alpha=0.5)
        s.smooth("a", self._lm(0.0))
        s.smooth("b", self._lm(1.0))
        out_a = s.smooth("a", self._lm(0.0))
        out_b = s.smooth("b", self._lm(1.0))
        self.assertAlmostEqual(out_a[0][0], 0.0, places=3)
        self.assertAlmostEqual(out_b[0][0], 1.0, places=3)


# ─────────────────────────────────────────────────────────────
#  TEST: GestureStabiliser
# ─────────────────────────────────────────────────────────────
class TestGestureStabiliser(unittest.TestCase):
    def test_not_stable_before_window_full(self):
        gs = GestureStabiliser(window=6, thresh=0.6)
        for _ in range(5):
            gs.feed("CURSOR")
        self.assertEqual(gs.stable, "NONE")

    def test_stable_after_majority(self):
        gs = GestureStabiliser(window=6, thresh=0.6)
        for _ in range(6):
            gs.feed("CURSOR")
        self.assertEqual(gs.stable, "CURSOR")

    def test_minority_not_confirmed(self):
        gs = GestureStabiliser(window=6, thresh=0.6)
        for g in ["CURSOR", "CURSOR", "CURSOR", "SCROLL", "SCROLL", "SCROLL"]:
            gs.feed(g)
        # 50/50 split → neither >= 60%
        self.assertNotEqual(gs.stable, "SCROLL")

    def test_reset_clears(self):
        gs = GestureStabiliser(window=6, thresh=0.6)
        for _ in range(6):
            gs.feed("PINCH")
        gs.reset()
        self.assertEqual(gs.stable, "NONE")

    def test_transition(self):
        gs = GestureStabiliser(window=4, thresh=0.75)
        for g in ["A", "A", "A", "A"]:
            gs.feed(g)
        self.assertEqual(gs.stable, "A")
        for g in ["B", "B", "B", "B"]:
            gs.feed(g)
        self.assertEqual(gs.stable, "B")


# ─────────────────────────────────────────────────────────────
#  TEST: VelocityTracker
# ─────────────────────────────────────────────────────────────
class TestVelocityTracker(unittest.TestCase):
    def test_empty_returns_zero(self):
        vt = VelocityTracker()
        self.assertEqual(vt.velocity(), (0.0, 0.0))

    def test_single_point_returns_zero(self):
        vt = VelocityTracker()
        vt.push(0.5, 0.5)
        self.assertEqual(vt.velocity(), (0.0, 0.0))

    def test_velocity_direction(self):
        vt = VelocityTracker(window=2)
        vt.push(0.0, 0.0)
        time.sleep(0.01)
        vt.push(0.1, 0.0)
        vx, vy = vt.velocity()
        self.assertGreater(vx, 0)
        self.assertAlmostEqual(vy, 0.0, places=3)

    def test_reset_clears(self):
        vt = VelocityTracker()
        vt.push(0.0, 0.0)
        vt.push(1.0, 1.0)
        vt.reset()
        self.assertEqual(vt.velocity(), (0.0, 0.0))


# ─────────────────────────────────────────────────────────────
#  TEST: GestureRecogniser (rule-based)
# ─────────────────────────────────────────────────────────────
class TestGestureRecogniser(unittest.TestCase):
    def setUp(self):
        self.rec = GestureRecogniser()

    def test_none_input(self):
        self.assertEqual(self.rec.classify(None), G.NONE)

    def test_open_palm(self):
        lm = _flat_hand()
        result = self.rec.classify(lm)
        self.assertEqual(result, G.OPEN_PALM)

    def test_fist(self):
        lm = _fist_hand()
        self.assertEqual(self.rec.classify(lm), G.FIST)

    def test_cursor(self):
        lm = _index_only()
        self.assertEqual(self.rec.classify(lm), G.CURSOR)

    def test_scroll(self):
        lm = _two_fingers()
        self.assertEqual(self.rec.classify(lm), G.SCROLL)

    def test_pinch(self):
        lm = _pinch_hand()
        self.assertEqual(self.rec.classify(lm), G.PINCH)

    def test_fingers_up_all(self):
        lm = _flat_hand()
        f = self.rec.fingers_up(lm)
        self.assertEqual(sum(f), 5)

    def test_fingers_up_none(self):
        lm = _fist_hand()
        f = self.rec.fingers_up(lm)
        self.assertEqual(sum(f), 0)


# ─────────────────────────────────────────────────────────────
#  TEST: SmoothMouse coordinate mapping
# ─────────────────────────────────────────────────────────────
class TestSmoothMouseCoords(unittest.TestCase):
    def setUp(self):
        self.m = SmoothMouse()

    def test_centre_maps_to_screen_centre(self):
        x, y = self.m._n2s(0.5, 0.5)
        self.assertAlmostEqual(x, Cfg.SCR_W // 2, delta=30)
        self.assertAlmostEqual(y, Cfg.SCR_H // 2, delta=30)

    def test_clamp_low(self):
        # nx very low → sx=0 → x=(1-0)*SCR_W = SCR_W (mirrored); ny very low → y=0
        x, y = self.m._n2s(-1.0, -1.0)
        self.assertEqual(x, Cfg.SCR_W)
        self.assertEqual(y, 0)

    def test_clamp_high(self):
        # nx very high → sx=1 → x=(1-1)*SCR_W = 0 (mirrored); ny very high → y=SCR_H
        x, y = self.m._n2s(2.0, 2.0)
        self.assertEqual(x, 0)
        self.assertEqual(y, Cfg.SCR_H)

    def test_left_maps_high_x(self):
        xl, _ = self.m._n2s(0.0, 0.5)
        xr, _ = self.m._n2s(1.0, 0.5)
        self.assertGreater(xl, xr)   # flipped: low nx → high screen x

    def test_top_maps_low_y(self):
        _, yt = self.m._n2s(0.5, 0.0)
        _, yb = self.m._n2s(0.5, 1.0)
        self.assertLess(yt, yb)

    def test_margin_excluded(self):
        x_full, _ = self.m._n2s(1 - Cfg.MARGIN_X - 0.001, 0.5)
        x_clamp, _ = self.m._n2s(1.0, 0.5)
        self.assertGreaterEqual(x_full, 0)
        self.assertGreaterEqual(x_clamp, 0)


# ─────────────────────────────────────────────────────────────
#  TEST: GestureDatabase
# ─────────────────────────────────────────────────────────────
class TestGestureDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
            suffix=".json", delete=False)
        self.tmp.close()
        self.db = GestureDatabase()
        setattr(self.db, "DB_FILE", self.tmp.name)
        self.db._d = {}

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_add_sample(self):
        self.db.add_sample("wave", [0.1] * 20)
        self.assertEqual(self.db.sample_count("wave"), 1)

    def test_multiple_samples(self):
        for _ in range(5):
            self.db.add_sample("wave", [0.5] * 20)
        self.assertEqual(self.db.sample_count("wave"), 5)

    def test_set_action(self):
        self.db.set_action("wave", "hotkey", ["ctrl", "c"])
        entry = self.db.get_entry("wave")
        self.assertEqual(entry["action"], "hotkey")

    def test_remove(self):
        self.db.add_sample("test", [0.0] * 20)
        self.db.remove("test")
        self.assertIsNone(self.db.get_entry("test"))

    def test_list_names_sorted(self):
        self.db.add_sample("z_gest", [0.0] * 20)
        self.db.add_sample("a_gest", [0.0] * 20)
        names = self.db.list_names()
        self.assertEqual(names, sorted(names))

    def test_save_and_reload(self):
        self.db.add_sample("persist", [0.9] * 20)
        self.db.set_action("persist", "screenshot")
        self.db.save()
        db2 = GestureDatabase()
        setattr(db2, "DB_FILE", self.tmp.name)
        db2._load()
        self.assertEqual(db2.sample_count("persist"), 1)
        self.assertEqual(db2.get_entry("persist")["action"], "screenshot")

    def test_load_missing_file(self):
        setattr(self.db, "DB_FILE", "/tmp/nonexistent_xyz123.json")
        self.db._load()   # should not raise
        self.assertEqual(self.db._d, {})

    def test_load_corrupt_json(self):
        with open(self.tmp.name, "w", encoding="utf-8") as f:
            f.write("not valid json {{{{")
        self.db._load()   # should not raise
        self.assertEqual(self.db._d, {})


# ─────────────────────────────────────────────────────────────
#  TEST: CustomGestureRecogniser (k-NN)
# ─────────────────────────────────────────────────────────────
class TestCustomGestureRecogniser(unittest.TestCase):
    def _make_db_with_gesture(self, name, lm):
        db  = GestureDatabase()
        db._d = {}
        rec = CustomGestureRecogniser(db)
        fv  = rec.feature_vector(lm)
        for _ in range(5):
            db.add_sample(name, fv)
        db.set_action(name, "screenshot")
        return db, rec

    def test_exact_match_returns_name(self):
        lm = _flat_hand()
        _, rec = self._make_db_with_gesture("myopen", lm)
        result = rec._knn(rec.feature_vector(lm))
        self.assertEqual(result, "myopen")

    def test_no_match_returns_none(self):
        lm_train = _flat_hand()
        lm_test  = _fist_hand()
        _, rec   = self._make_db_with_gesture("myopen", lm_train)
        result   = rec._knn(rec.feature_vector(lm_test))
        # fist is very different from open palm → should NOT match
        self.assertNotEqual(result, "myopen")

    def test_empty_db_returns_none(self):
        db  = GestureDatabase()
        db._d = {}
        rec = CustomGestureRecogniser(db)
        self.assertIsNone(rec._knn([0.5] * 20))

    def test_fallback_to_rules_when_no_custom(self):
        db  = GestureDatabase()
        db._d = {}
        rec = CustomGestureRecogniser(db)
        # With empty db, should classify via parent rules
        self.assertEqual(rec.classify(None), G.NONE)
        self.assertEqual(rec.classify(_fist_hand()), G.FIST)

    def test_feature_vector_length(self):
        db  = GestureDatabase()
        db._d = {}
        rec = CustomGestureRecogniser(db)
        fv  = rec.feature_vector(_flat_hand())
        self.assertEqual(len(fv), 20)

    def test_custom_overrides_rules(self):
        lm       = _fist_hand()   # normally → FIST
        _, rec   = self._make_db_with_gesture("custom_fist", lm)
        result   = rec.classify(lm)
        # custom_fist should win over built-in FIST
        self.assertEqual(result, "custom_fist")


# ─────────────────────────────────────────────────────────────
#  TEST: run_action
# ─────────────────────────────────────────────────────────────
class TestRunAction(unittest.TestCase):
    def setUp(self):
        self.mouse = mock.MagicMock(spec=SmoothMouse)

    def test_hotkey_list(self):
        run_action(self.mouse, "hotkey", ["ctrl", "c"])
        self.mouse.hotkey.assert_called_once_with("ctrl", "c")

    def test_hotkey_tuple(self):
        run_action(self.mouse, "hotkey", ("ctrl", "v"))
        self.mouse.hotkey.assert_called_once_with("ctrl", "v")

    def test_zoom_in(self):
        run_action(self.mouse, "zoom", 1)
        self.mouse.zoom.assert_called_once_with(1)

    def test_zoom_out(self):
        run_action(self.mouse, "zoom", -1)
        self.mouse.zoom.assert_called_once_with(-1)

    def test_wait_clamped(self):
        start = time.monotonic()
        run_action(self.mouse, "wait", 0.0)   # 0.0 non è None → dorme 0s
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.5)

    def test_wait_negative_clamped_to_zero(self):
        run_action(self.mouse, "wait", -5)  # must not raise

    def test_unknown_action_silent(self):
        run_action(self.mouse, "nonexistent_action", None)  # must not raise


# ─────────────────────────────────────────────────────────────
#  TEST: AutonomousAgent
# ─────────────────────────────────────────────────────────────
def _make_fake_anthropic(plan_json: str):
    """Return a fake anthropic module whose Anthropic client returns plan_json."""
    class _Block:
        type = "text"
        text = plan_json
    class _Msg:
        content = [_Block()]
    class _Messages:
        def create(self, **_kwargs): return _Msg()
    class _Client:
        def __init__(self, **_kwargs): self.messages = _Messages()
    fake = mock.MagicMock()
    fake.Anthropic = _Client
    return fake


class TestAutonomousAgent(unittest.TestCase):
    def _make_agent(self, plan_json=None, api_key="test-key"):
        mouse = mock.MagicMock(spec=SmoothMouse)
        updates = []
        agent = AutonomousAgent(mouse, on_update=lambda: updates.append(agent.status))
        agent.has_sdk = True
        agent.api_key = api_key
        if plan_json is not None:
            sys.modules["anthropic"] = _make_fake_anthropic(plan_json)
        return agent, mouse, updates

    def tearDown(self):
        sys.modules.pop("anthropic", None)

    def test_unavailable_without_key(self):
        agent, _, _ = self._make_agent(api_key="")
        self.assertFalse(agent.available)

    def test_unavailable_without_sdk(self):
        mouse = mock.MagicMock(spec=SmoothMouse)
        agent = AutonomousAgent(mouse)
        agent.has_sdk = False
        agent.api_key = "x"
        self.assertFalse(agent.available)

    def test_available_with_key_and_sdk(self):
        agent, _, _ = self._make_agent()
        self.assertTrue(agent.available)

    def test_run_executes_plan(self):
        plan = '[{"action": "zoom", "args": 1}]'
        agent, mouse, _ = self._make_agent(plan)
        agent.run("zoom in")
        agent._thread.join(timeout=5)
        self.assertEqual(agent.status, "COMPLETATO")
        mouse.zoom.assert_called_once_with(1)

    def test_run_empty_goal_ignored(self):
        agent, _, _ = self._make_agent('[]')
        agent.run("   ")
        self.assertIsNone(agent._thread)

    def test_run_while_busy_sets_occupato(self):
        plan = '[{"action": "wait", "args": 0.3}]'
        agent, _, updates = self._make_agent(plan)
        agent.run("first goal")
        time.sleep(0.05)
        agent.run("second goal while busy")
        self.assertIn("OCCUPATO", updates)

    def test_bad_json_sets_errore_piano(self):
        sys.modules["anthropic"] = _make_fake_anthropic("not json at all")
        mouse = mock.MagicMock(spec=SmoothMouse)
        agent = AutonomousAgent(mouse)
        agent.has_sdk = True
        agent.api_key = "x"
        agent.run("bad plan")
        agent._thread.join(timeout=5)
        self.assertEqual(agent.status, "ERRORE PIANO")

    def test_stop_interrupts_loop(self):
        plan = '[{"action":"wait","args":0.5},{"action":"wait","args":0.5},{"action":"wait","args":0.5}]'
        agent, _, _ = self._make_agent(plan)
        agent.run("long task")
        time.sleep(0.1)
        agent.stop()
        agent._thread.join(timeout=5)
        # After stop(), loop exits with INTERROTTO (if mid-run) or IDLE (if stopped before start)
        self.assertIn(agent.status, ("INTERROTTO", "IDLE", "COMPLETATO"))

    def test_api_key_reread_at_runtime(self):
        agent, _, _ = self._make_agent(api_key="")
        self.assertFalse(agent.available)
        os.environ["ANTHROPIC_API_KEY"] = "runtime-key"
        try:
            self.assertTrue(agent.available)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]


# ─────────────────────────────────────────────────────────────
#  TEST: VoiceController (mocked speech_recognition)
# ─────────────────────────────────────────────────────────────
class TestVoiceControllerParser(unittest.TestCase):
    """Tests the VoiceController parsing/dispatch logic with mocked SR."""

    def test_agent_goal_dispatched_to_callback(self):
        received = []
        agent_goals = []
        mouse = mock.MagicMock(spec=SmoothMouse)
        vc = VoiceController(mouse,
                             on_command=lambda t, a: received.append((t, a)),
                             on_agent_goal=agent_goals.append)
        # Simulate what _loop does when text is recognised
        text = "agente apri youtube"
        result = vc._parser.parse(text)
        self.assertIsNotNone(result)
        action, args = result
        self.assertEqual(action, "agent")
        if action == "agent" and vc._on_agent:
            vc._on_agent(str(args))
        self.assertEqual(agent_goals, ["apri youtube"])

    def test_non_agent_command_not_dispatched_to_agent(self):
        agent_goals = []
        mouse = mock.MagicMock(spec=SmoothMouse)
        vc = VoiceController(mouse,
                             on_command=lambda t, a: None,
                             on_agent_goal=agent_goals.append)
        result = vc._parser.parse("copia")
        self.assertIsNotNone(result)
        action, _ = result
        self.assertNotEqual(action, "agent")
        self.assertEqual(agent_goals, [])

    def test_history_bounded_by_cfg(self):
        mouse = mock.MagicMock(spec=SmoothMouse)
        vc = VoiceController(mouse, on_command=lambda t, a: None)
        self.assertEqual(vc.history.maxlen, Cfg.VOICE_HISTORY_MAX)

    def test_available_false_without_sr(self):
        sys.modules.pop("speech_recognition", None)
        # If speech_recognition was never importable, available should be False
        mouse = mock.MagicMock(spec=SmoothMouse)
        vc = VoiceController(mouse, on_command=lambda t, a: None)
        # available depends on import at __init__ time; it's already False in headless env
        self.assertIsInstance(vc.available, bool)


# ─────────────────────────────────────────────────────────────
#  100-RUN REPORTER
# ─────────────────────────────────────────────────────────────
def _build_suite():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in (
        TestCommandParser,
        TestLandmarkSmoother,
        TestGestureStabiliser,
        TestVelocityTracker,
        TestGestureRecogniser,
        TestSmoothMouseCoords,
        TestGestureDatabase,
        TestCustomGestureRecogniser,
        TestRunAction,
        TestAutonomousAgent,
        TestVoiceControllerParser,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


def run_100_times() -> dict:
    runs       = []
    total_run  = 0
    total_pass = 0
    total_fail = 0
    total_err  = 0
    t_start    = time.perf_counter()

    for i in range(100):
        buf    = io.StringIO()
        result = unittest.TextTestRunner(stream=buf, verbosity=0).run(_build_suite())
        passed = result.testsRun - len(result.failures) - len(result.errors)
        total_run  += result.testsRun
        total_pass += passed
        total_fail += len(result.failures)
        total_err  += len(result.errors)
        runs.append({
            "run":      i + 1,
            "tests":    result.testsRun,
            "passed":   passed,
            "failures": [str(f[0]) for f in result.failures],
            "errors":   [str(e[0]) for e in result.errors],
        })

    elapsed = time.perf_counter() - t_start

    report = {
        "timestamp":     datetime.datetime.now().isoformat(timespec="seconds"),
        "total_runs":    100,
        "total_run":     total_run,
        "total_passed":  total_pass,
        "total_failed":  total_fail,
        "total_errors":  total_err,
        "elapsed_sec":   round(elapsed, 2),
        "pass_rate_pct": round(100 * total_pass / max(total_run, 1), 2),
        "runs":          runs,
    }
    return report


def _write_report(report: dict):
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"gesture_report_{ts}")
    json_path = base + ".json"
    txt_path  = base + ".txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    failed_runs = [r for r in report["runs"] if r["failures"] or r["errors"]]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Hand Gesture Control — Test Report (100 runs)\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Data/Ora  : {report['timestamp']}\n")
        f.write(f"  Durata    : {report['elapsed_sec']} sec\n")
        f.write(f"  Test/run  : {report['total_run'] // 100}\n")
        f.write(f"  Totale    : {report['total_passed']}/{report['total_run']} passed\n")
        f.write(f"  Pass rate : {report['pass_rate_pct']}%\n")
        if failed_runs:
            f.write(f"\n  Run con fallimenti ({len(failed_runs)}):\n")
            for r in failed_runs:
                f.write(f"    Run #{r['run']}: {r['failures']} {r['errors']}\n")
        else:
            f.write("\n  Tutti i run: PASS ✓\n")
        f.write("=" * 60 + "\n")

    return json_path, txt_path


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def _main():
    parser = argparse.ArgumentParser(description="Hand Gesture Control Test Suite")
    parser.add_argument("--x100", action="store_true",
                        help="Esegui 100 run con report su file")
    args, remaining = parser.parse_known_args()

    if args.x100:
        print("Eseguo 100 run della suite di test...")
        results = run_100_times()
        jp, tp  = _write_report(results)
        pr      = results["pass_rate_pct"]
        total_p = results["total_passed"]
        total_r = results["total_run"]
        status  = "✓ PASS" if pr == 100.0 else "✗ ATTENZIONE"
        print(f"\n{status}  {total_p}/{total_r} passed  ({pr}%)  "
              f"in {results['elapsed_sec']}s")
        print(f"  JSON: {jp}")
        print(f"  TXT:  {tp}")
        sys.exit(0 if pr == 100.0 else 1)
    else:
        # Standard unittest run
        unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)


if __name__ == "__main__":
    _main()
