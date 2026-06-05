"""
CODICE MADRE — Architettura Universale per Hand Gesture Control
═══════════════════════════════════════════════════════════════
Fondazione che abilita ogni realtà del sistema: gesti, azioni,
pipeline, skill, modalità, feature.

Principio unico: dichiarare cosa, non come.
  • Aggiungere un gesto   = 1 GestureSpec nella Registry
  • Aggiungere un'azione  = 1 SkillSpec nella SkillMatrix
  • Aggiungere uno stage  = 1 PipelineStage nella Registry
  • Attivare una feature  = 1 flag in FeatureFlags

Nessun file va toccato oltre a questo per estendere il sistema.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TIPI FONDAMENTALI
#    Enumerazioni immutabili come contratto tra ogni parte del sistema.
#    Mai stringhe sparse; ogni valore è tipizzato e documentato.
# ═══════════════════════════════════════════════════════════════════════════════

class GestureID(str, Enum):
    CURSOR        = "cursor"
    CLICK         = "click"
    DOUBLE_CLICK  = "double_click"
    RIGHT_CLICK   = "right_click"
    DRAG          = "drag"
    SCROLL_UP     = "scroll_up"
    SCROLL_DOWN   = "scroll_down"
    COPY          = "copy"
    PASTE         = "paste"
    UNDO          = "undo"
    SAVE          = "save"
    ZOOM_IN       = "zoom_in"
    ZOOM_OUT      = "zoom_out"
    FIST          = "fist"
    OPEN_PALM     = "open_palm"
    SWIPE_LEFT    = "swipe_left"
    SWIPE_RIGHT   = "swipe_right"
    CUSTOM        = "custom"
    NONE          = "none"


class ActionType(str, Enum):
    MOUSE_MOVE         = "mouse_move"
    MOUSE_CLICK        = "mouse_click"
    MOUSE_RIGHT_CLICK  = "mouse_right_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_DRAG         = "mouse_drag"
    MOUSE_SCROLL       = "mouse_scroll"
    HOTKEY             = "hotkey"
    KEYBOARD_TYPE      = "keyboard_type"
    CUSTOM_FN          = "custom_fn"
    NONE               = "none"


class HandRole(str, Enum):
    DOMINANT = "dominant"    # Mano principale: cursore/azioni
    MODIFIER = "modifier"    # Mano non dominante: cambia modalità
    ANY      = "any"         # Valido per entrambe


class Mode(str, Enum):
    NORMAL       = "normal"        # Modalità standard
    FREEZE       = "freeze"        # Cursore bloccato
    ZOOM         = "zoom"          # Ctrl+scroll
    SCROLL_H     = "scroll_h"      # Scorrimento orizzontale
    ALT_TAB      = "alt_tab"       # Cambio finestra
    MIDDLE_CLICK = "middle_click"  # Click centrale


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPECIFICHE DICHIARATIVE
#    Strutture dati frozen: definiscono il CONTRATTO di ogni componente.
#    Immutabili = nessun effetto collaterale nascosto.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GestureSpec:
    """
    Specifica completa e immutabile di un gesto.
    Contiene tutto: identità, presentazione UI, azione associata.
    """
    id: GestureID
    label: str
    description: str
    hand_role: HandRole       = HandRole.DOMINANT
    mode: Mode                = Mode.NORMAL
    priority: int             = 50
    icon: str                 = "●"
    action: ActionType        = ActionType.NONE
    action_params: Dict[str, Any] = field(default_factory=dict)
    custom_fn: Optional[str]  = None


@dataclass(frozen=True)
class SkillSpec:
    """
    Una skill = mappatura gesto → comportamento in un contesto dato.
    La matrice di tutte le SkillSpec è l'intero comportamento del sistema.
    Modifica qui → il sistema si riconfigura senza toccare altro codice.
    """
    gesture_id: GestureID
    mode: Mode
    hand_role: HandRole
    action_type: ActionType
    action_params: Dict[str, Any] = field(default_factory=dict)
    priority: int                 = 50


@dataclass
class PipelineStage:
    """
    Stage singolo della pipeline di elaborazione frame.
    order controlla la sequenza; enabled permette disattivazione live.
    """
    name: str
    processor: Callable[[Any], Any]
    order: int   = 0
    enabled: bool = True


@dataclass
class FeatureFlags:
    """
    Tutte le feature del sistema in un unico oggetto.
    Cambia un flag = cambia il comportamento a runtime.
    """
    voice_control:         bool = True
    dual_hand:             bool = True
    custom_gestures:       bool = True
    gesture_stabilization: bool = True
    landmark_smoothing:    bool = True
    velocity_tracking:     bool = True
    zoom_control:          bool = True
    virtual_keyboard:      bool = True
    debug_overlay:         bool = False
    performance_mode:      bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONFIGURAZIONE UNIVERSALE
#    Tutti i parametri numerici in un unico posto.
#    Nessun magic number disperso nel codice.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    # Camera
    cam_width:  int   = 640
    cam_height: int   = 480
    max_hands:  int   = 2

    # Smoothing
    ema_landmarks: float = 0.40   # Alpha EMA per landmark
    ema_cursor:    float = 0.28   # Alpha EMA per cursore

    # Gesture stabilization
    stab_window:    int   = 6     # Frame da votare
    stab_threshold: float = 0.60  # Soglia maggioranza (60 %)

    # Pinch / drag
    pinch_threshold: float = 0.06
    drag_threshold:  float = 0.07

    # Scroll
    scroll_step: int = 3

    # Screen margins (0–1)
    margin_x: float = 0.12
    margin_y: float = 0.12

    # k-NN custom gestures
    knn_k:            int = 3
    knn_samples:      int = 30
    knn_feature_dim:  int = 20

    # Voice
    voice_language: str = "it-IT"


# Istanza globale — importabile direttamente: `from core import CFG`
CFG = SystemConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REGISTRY UNIVERSALE (Singleton thread-safe)
#    Una sola sorgente di verità per gesti, skill, stage, handler, processor.
#    Fluent API: registry.register_gesture(x).register_skill(y).on("evt", fn)
# ═══════════════════════════════════════════════════════════════════════════════

class Registry:
    """
    Cuore del Codice Madre.
    Tutto ciò che esiste nel sistema è registrato qui.
    Nessun componente conosce un altro direttamente:
    comunicano tutti attraverso la Registry.
    """

    _instance: Optional[Registry] = None
    _lock = threading.Lock()

    def __new__(cls) -> Registry:
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
        return cls._instance

    def _init(self) -> None:
        self._gestures:  Dict[GestureID, GestureSpec]                          = {}
        self._skills:    Dict[Tuple[GestureID, Mode, HandRole], SkillSpec]     = {}
        self._stages:    List[PipelineStage]                                   = []
        self._handlers:  Dict[str, List[Callable]]                             = defaultdict(list)
        self._processors: Dict[str, Callable]                                  = {}
        self._features   = FeatureFlags()
        self._config     = CFG

    # ── Gesti ────────────────────────────────────────────────────────────────

    def register_gesture(self, spec: GestureSpec) -> Registry:
        self._gestures[spec.id] = spec
        return self

    def unregister_gesture(self, gid: GestureID) -> Registry:
        self._gestures.pop(gid, None)
        return self

    def gesture(self, gid: GestureID) -> Optional[GestureSpec]:
        return self._gestures.get(gid)

    def all_gestures(self) -> List[GestureSpec]:
        return sorted(self._gestures.values(), key=lambda g: g.priority, reverse=True)

    # ── Skill Matrix ─────────────────────────────────────────────────────────

    def register_skill(self, spec: SkillSpec) -> Registry:
        self._skills[(spec.gesture_id, spec.mode, spec.hand_role)] = spec
        return self

    def resolve_skill(
        self,
        gid: GestureID,
        mode: Mode = Mode.NORMAL,
        role: HandRole = HandRole.DOMINANT,
    ) -> Optional[SkillSpec]:
        """
        Risolve la skill più specifica disponibile.
        Ordine: (gid, mode, role) → (gid, mode, ANY) → None
        """
        return (
            self._skills.get((gid, mode, role))
            or self._skills.get((gid, mode, HandRole.ANY))
        )

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def register_stage(self, stage: PipelineStage) -> Registry:
        self._stages.append(stage)
        self._stages.sort(key=lambda s: s.order)
        return self

    def active_stages(self) -> List[PipelineStage]:
        return [s for s in self._stages if s.enabled]

    # ── Event Bus ────────────────────────────────────────────────────────────

    def on(self, event: str, handler: Callable) -> Registry:
        """Registra un handler per un evento. Supporta wildcard '*'."""
        self._handlers[event].append(handler)
        return self

    def emit(self, event: str, **payload: Any) -> None:
        """Emette un evento verso tutti gli handler registrati."""
        for handler in self._handlers.get(event, []) + self._handlers.get("*", []):
            try:
                handler(event=event, **payload)
            except Exception as exc:
                # L'error handler non può ricorrere su se stesso
                if event != "error":
                    self.emit("error", source=event, exc=exc)

    # ── Processori ───────────────────────────────────────────────────────────

    def register_processor(self, name: str, fn: Callable) -> Registry:
        self._processors[name] = fn
        return self

    def processor(self, name: str) -> Optional[Callable]:
        return self._processors.get(name)

    # ── Feature Flags ────────────────────────────────────────────────────────

    @property
    def features(self) -> FeatureFlags:
        return self._features

    def toggle(self, feature: str, value: Optional[bool] = None) -> Registry:
        current = getattr(self._features, feature, None)
        if current is not None:
            setattr(self._features, feature, (not current) if value is None else value)
        return self

    # ── Config ───────────────────────────────────────────────────────────────

    @property
    def config(self) -> SystemConfig:
        return self._config


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE COMPONIBILE
#    frame → stage₁ → stage₂ → … → stageₙ → risultato
#    Ogni stage è indipendente. Fallisce uno → gli altri continuano.
#    Aggiungi/rimuovi stage senza toccare nient'altro.
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    Pipeline data-flow componibile.
    Si costruisce dai PipelineStage registrati nella Registry.
    Resiliente: errori in uno stage vengono emessi sull'event bus
    senza interrompere il flusso degli altri stage.
    """

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def run(self, data: Any) -> Any:
        result = data
        for stage in self._registry.active_stages():
            try:
                result = stage.processor(result)
            except Exception as exc:
                self._registry.emit(
                    "pipeline_error", stage=stage.name, exc=exc, data=data
                )
        return result

    def run_until(self, data: Any, stop_stage: str) -> Any:
        """Esegue la pipeline fino allo stage indicato (escluso)."""
        result = data
        for stage in self._registry.active_stages():
            if stage.name == stop_stage:
                break
            try:
                result = stage.processor(result)
            except Exception as exc:
                self._registry.emit(
                    "pipeline_error", stage=stage.name, exc=exc, data=data
                )
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FACTORY
#    Costruisce ogni componente dalla Registry.
#    Nessun componente si istanzia direttamente: passa sempre dalla Factory.
#    Zero accoppiamento, zero dipendenze circolari.
# ═══════════════════════════════════════════════════════════════════════════════

class ComponentFactory:
    """
    Unico punto di costruzione per ogni componente del sistema.
    I controller hardware (mouse, keyboard) sono iniettati dall'esterno:
    la Factory non li conosce, li usa soltanto.
    """

    def __init__(self, registry: Registry) -> None:
        self._r = registry

    def build_pipeline(self) -> Pipeline:
        return Pipeline(self._r)

    def build_skill_resolver(self) -> Callable[[GestureID, Mode, HandRole], Optional[SkillSpec]]:
        return self._r.resolve_skill

    def build_action_dispatcher(
        self,
        mouse: Any,
        keyboard: Any,
    ) -> Callable[[SkillSpec, Dict[str, Any]], None]:
        """
        Ritorna una funzione dispatch(skill, context) pronta all'uso.
        mouse e keyboard sono controller con l'interfaccia del progetto.
        """
        registry = self._r

        def dispatch(skill: SkillSpec, context: Dict[str, Any]) -> None:
            at = skill.action_type
            p  = skill.action_params

            if at == ActionType.MOUSE_MOVE:
                mouse.move(context.get("x", 0), context.get("y", 0))
            elif at == ActionType.MOUSE_CLICK:
                mouse.click()
            elif at == ActionType.MOUSE_RIGHT_CLICK:
                mouse.right_click()
            elif at == ActionType.MOUSE_DOUBLE_CLICK:
                mouse.double_click()
            elif at == ActionType.MOUSE_DRAG:
                mouse.drag(context.get("x", 0), context.get("y", 0))
            elif at == ActionType.MOUSE_SCROLL:
                mouse.scroll(p.get("dy", registry.config.scroll_step))
            elif at == ActionType.HOTKEY:
                keyboard.hotkey(*p.get("keys", []))
            elif at == ActionType.KEYBOARD_TYPE:
                keyboard.type(p.get("text", ""))
            elif at == ActionType.CUSTOM_FN:
                fn = registry.processor(p.get("fn", ""))
                if fn:
                    fn(context)

            registry.emit("action_dispatched", skill=skill, context=context)

        return dispatch


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SKILL MATRIX — La mappa completa di ogni comportamento del sistema
#    Ogni gesto × ogni modalità × ogni ruolo mano → azione.
#    Modificare qui = modificare tutto il comportamento. Un solo posto.
# ═══════════════════════════════════════════════════════════════════════════════

def _build_default_registry() -> Registry:
    r = Registry()

    # ── Definizione dei 17 gesti integrati ──────────────────────────────────
    r.register_gesture(GestureSpec(
        id=GestureID.CURSOR, label="Cursore", icon="☝️", priority=90,
        description="Muovi il cursore con l'indice alzato",
        action=ActionType.MOUSE_MOVE,
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.CLICK, label="Click", icon="👌", priority=85,
        description="Pinch pollice-indice per cliccare",
        action=ActionType.MOUSE_CLICK,
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.DOUBLE_CLICK, label="Doppio Click", icon="👍", priority=80,
        description="Solo pollice alzato",
        action=ActionType.MOUSE_DOUBLE_CLICK,
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.RIGHT_CLICK, label="Click Destro", icon="✌️", priority=80,
        description="Pinch pollice-medio",
        action=ActionType.MOUSE_RIGHT_CLICK,
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.DRAG, label="Trascina", icon="✊", priority=85,
        description="Pizzica e mantieni per trascinare",
        action=ActionType.MOUSE_DRAG,
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.SCROLL_UP, label="Scorri Su", icon="☝️☝️", priority=75,
        description="Indice e medio alzati, muovi su",
        action=ActionType.MOUSE_SCROLL, action_params={"dy": CFG.scroll_step},
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.SCROLL_DOWN, label="Scorri Giù", icon="👇", priority=75,
        description="Indice e medio alzati, muovi giù",
        action=ActionType.MOUSE_SCROLL, action_params={"dy": -CFG.scroll_step},
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.COPY, label="Copia", icon="📋", priority=70,
        description="3 dita alzate",
        action=ActionType.HOTKEY, action_params={"keys": ["ctrl", "c"]},
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.PASTE, label="Incolla", icon="📌", priority=70,
        description="4 dita alzate",
        action=ActionType.HOTKEY, action_params={"keys": ["ctrl", "v"]},
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.UNDO, label="Annulla", icon="🤘", priority=65,
        description="Rock: indice + mignolo alzati",
        action=ActionType.HOTKEY, action_params={"keys": ["ctrl", "z"]},
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.SAVE, label="Salva", icon="💾", priority=65,
        description="Pollice + mignolo alzati",
        action=ActionType.HOTKEY, action_params={"keys": ["ctrl", "s"]},
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.ZOOM_IN, label="Zoom +", icon="🔍", priority=70,
        description="Allarga le due mani",
        action=ActionType.HOTKEY, action_params={"keys": ["ctrl", "+"]},
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.ZOOM_OUT, label="Zoom −", icon="🔎", priority=70,
        description="Avvicina le due mani",
        action=ActionType.HOTKEY, action_params={"keys": ["ctrl", "-"]},
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.FIST, label="Pugno", icon="✊", priority=60,
        description="Mano chiusa — pausa azioni",
        action=ActionType.NONE,
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.OPEN_PALM, label="Palmo Aperto", icon="🖐️", priority=60,
        description="Mano aperta — freeze cursore",
        action=ActionType.NONE,
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.SWIPE_LEFT, label="Swipe Sinistra", icon="👈", priority=55,
        description="Movimento rapido verso sinistra",
        action=ActionType.HOTKEY, action_params={"keys": ["alt", "left"]},
    ))
    r.register_gesture(GestureSpec(
        id=GestureID.SWIPE_RIGHT, label="Swipe Destra", icon="👉", priority=55,
        description="Movimento rapido verso destra",
        action=ActionType.HOTKEY, action_params={"keys": ["alt", "right"]},
    ))

    # ── Skill Matrix: comportamento per ogni contesto ────────────────────────
    # Modalità NORMAL — mano dominante
    _D, _N = HandRole.DOMINANT, Mode.NORMAL
    for spec in [
        SkillSpec(GestureID.CURSOR,       _N, _D, ActionType.MOUSE_MOVE),
        SkillSpec(GestureID.CLICK,        _N, _D, ActionType.MOUSE_CLICK),
        SkillSpec(GestureID.DOUBLE_CLICK, _N, _D, ActionType.MOUSE_DOUBLE_CLICK),
        SkillSpec(GestureID.RIGHT_CLICK,  _N, _D, ActionType.MOUSE_RIGHT_CLICK),
        SkillSpec(GestureID.DRAG,         _N, _D, ActionType.MOUSE_DRAG),
        SkillSpec(GestureID.SCROLL_UP,    _N, _D, ActionType.MOUSE_SCROLL,  {"dy":  CFG.scroll_step}),
        SkillSpec(GestureID.SCROLL_DOWN,  _N, _D, ActionType.MOUSE_SCROLL,  {"dy": -CFG.scroll_step}),
        SkillSpec(GestureID.COPY,         _N, _D, ActionType.HOTKEY,        {"keys": ["ctrl", "c"]}),
        SkillSpec(GestureID.PASTE,        _N, _D, ActionType.HOTKEY,        {"keys": ["ctrl", "v"]}),
        SkillSpec(GestureID.UNDO,         _N, _D, ActionType.HOTKEY,        {"keys": ["ctrl", "z"]}),
        SkillSpec(GestureID.SAVE,         _N, _D, ActionType.HOTKEY,        {"keys": ["ctrl", "s"]}),
        SkillSpec(GestureID.SWIPE_LEFT,   _N, _D, ActionType.HOTKEY,        {"keys": ["alt", "left"]}),
        SkillSpec(GestureID.SWIPE_RIGHT,  _N, _D, ActionType.HOTKEY,        {"keys": ["alt", "right"]}),
    ]:
        r.register_skill(spec)

    # Modalità ZOOM — scroll diventa Ctrl+/- per zoom documento
    _Z = Mode.ZOOM
    for spec in [
        SkillSpec(GestureID.CURSOR,      _Z, _D, ActionType.MOUSE_MOVE),
        SkillSpec(GestureID.SCROLL_UP,   _Z, _D, ActionType.HOTKEY, {"keys": ["ctrl", "+"]}),
        SkillSpec(GestureID.SCROLL_DOWN, _Z, _D, ActionType.HOTKEY, {"keys": ["ctrl", "-"]}),
    ]:
        r.register_skill(spec)

    # Modalità ALT_TAB — swipe cambia finestra attiva
    _A = Mode.ALT_TAB
    for spec in [
        SkillSpec(GestureID.SWIPE_LEFT,  _A, _D, ActionType.HOTKEY, {"keys": ["alt", "shift", "tab"]}),
        SkillSpec(GestureID.SWIPE_RIGHT, _A, _D, ActionType.HOTKEY, {"keys": ["alt", "tab"]}),
    ]:
        r.register_skill(spec)

    # Modalità SCROLL_H — scorrimento orizzontale
    r.register_skill(SkillSpec(
        GestureID.CURSOR, Mode.SCROLL_H, _D, ActionType.CUSTOM_FN,
        {"fn": "hscroll"},
    ))

    return r


# ═══════════════════════════════════════════════════════════════════════════════
# 8. BOOTSTRAP
#    Un'unica chiamata costruisce l'intero sistema configurato.
#    Questo è il punto di ingresso che ogni modulo del progetto usa.
# ═══════════════════════════════════════════════════════════════════════════════

_registry_instance: Optional[Registry] = None


def get_registry() -> Registry:
    """Ritorna la Registry singleton inizializzata con i default."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = _build_default_registry()
    return _registry_instance


def reset_registry() -> None:
    """Resetta la Registry. Usato esclusivamente nei test."""
    global _registry_instance
    _registry_instance = None
    Registry._instance = None


def bootstrap(features: Optional[Dict[str, bool]] = None) -> ComponentFactory:
    """
    Punto di ingresso unico per costruire l'intero sistema.

    Esempio d'uso:
        from core import bootstrap, GestureSpec, GestureID, ActionType

        factory = bootstrap({"voice_control": False})
        pipeline   = factory.build_pipeline()
        dispatcher = factory.build_action_dispatcher(mouse_ctrl, kb_ctrl)

    Per aggiungere un nuovo gesto al sistema:
        get_registry().register_gesture(GestureSpec(
            id=GestureID.CUSTOM,
            label="Il Mio Gesto",
            description="...",
            action=ActionType.HOTKEY,
            action_params={"keys": ["ctrl", "n"]},
        ))

    Per aggiungere uno stage alla pipeline:
        get_registry().register_stage(PipelineStage(
            name="my_filter",
            processor=my_fn,
            order=15,   # eseguito tra order=10 e order=20
        ))

    Per reagire a un evento:
        get_registry().on("action_dispatched", lambda event, skill, context: ...)
    """
    registry = get_registry()
    if features:
        for key, val in features.items():
            registry.toggle(key, val)
    return ComponentFactory(registry)


# ── Esposizione pubblica del modulo ──────────────────────────────────────────
__all__ = [
    # Tipi
    "GestureID", "ActionType", "HandRole", "Mode",
    # Specifiche
    "GestureSpec", "SkillSpec", "PipelineStage", "FeatureFlags", "SystemConfig",
    # Configurazione globale
    "CFG",
    # Architettura
    "Registry", "Pipeline", "ComponentFactory",
    # API pubblica
    "get_registry", "reset_registry", "bootstrap",
]
