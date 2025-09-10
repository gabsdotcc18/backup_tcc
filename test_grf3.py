import os
import glob
import csv
import json
import shutil
import math
import gc
from collections import deque

import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio

# (Opcional) se rodar headless e tiver erro de SDL:
# os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

import gfootball.env as football_env

# ====================================================================================
# Configurações principais (idênticas à sua base, com prints e criação de vídeo)
# ====================================================================================
SCENARIO_NAME = "three_vs_three"

OUTPUT_DIR = f"frames_{SCENARIO_NAME}"                  # PNGs + métricas
VIDEO_LOGDIR = os.path.join(OUTPUT_DIR, "video_logs")    # engine video (se suportado)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_LOGDIR, exist_ok=True)

# Limpa frames e vídeos antigos (sobrescrever sempre)
for old_png in glob.glob(os.path.join(OUTPUT_DIR, "frame_*.png")):
    try:
        os.remove(old_png)
    except Exception:
        pass
for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
    for old_vid in glob.glob(os.path.join(VIDEO_LOGDIR, ext)):
        try:
            os.remove(old_vid)
        except Exception:
            pass

# Limpa métricas antigas
for old_file in [
    os.path.join(OUTPUT_DIR, "events_legacy.csv"),
    os.path.join(OUTPUT_DIR, "events_user.csv"),
    os.path.join(OUTPUT_DIR, "touches.csv"),
    os.path.join(OUTPUT_DIR, "metrics.json"),
    os.path.join(OUTPUT_DIR, f"{SCENARIO_NAME}_fallback.mp4"),
    os.path.join(OUTPUT_DIR, "info_debug.jsonl"),
]:
    try:
        if os.path.isdir(old_file):
            shutil.rmtree(old_file, ignore_errors=True)
        else:
            os.remove(old_file)
    except Exception:
        pass

MAX_STEPS = 500
CHANNEL_W, CHANNEL_H = 640, 480   # (largura, altura) -> obs vem (H, W, 3)

# Fallback MP4 (você já instalou o encoder)
MAKE_MP4_FROM_PNGS = True
FPS_FALLBACK = 30

# Saídas
METRICS_JSON       = os.path.join(OUTPUT_DIR, "metrics.json")
EVENTS_LEGACY_CSV  = os.path.join(OUTPUT_DIR, "events_legacy.csv")
EVENTS_USER_CSV    = os.path.join(OUTPUT_DIR, "events_user.csv")
TOUCHES_CSV        = os.path.join(OUTPUT_DIR, "touches.csv")
INFO_DBG           = os.path.join(OUTPUT_DIR, "info_debug.jsonl")

print("="*80)
print(f"Iniciando execução - cenário: {SCENARIO_NAME}")
print(f"Saída de frames em: {OUTPUT_DIR}")
print(f"Log de vídeo (engine): {VIDEO_LOGDIR}")
print(f"Gerar MP4 de fallback pelos PNGs: {MAKE_MP4_FROM_PNGS} (FPS={FPS_FALLBACK})")
print("="*80)

# ====================================================================================
# Ambiente único (PIXELS) — força vídeo do motor se disponível
# ====================================================================================
def create_env_pixels():
    base_kwargs = dict(
        env_name=SCENARIO_NAME,
        representation="pixels",
        channel_dimensions=(CHANNEL_W, CHANNEL_H),
        render=True,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        write_video=True,
        logdir=VIDEO_LOGDIR,
    )
    try:
        return football_env.create_environment(
            **base_kwargs,
            dump_full_episodes=True,
            write_full_episode_dumps=True,
        )
    except TypeError:
        return football_env.create_environment(**base_kwargs)

# Aviso headless
if not os.environ.get("DISPLAY"):
    print("No DISPLAY defined, doing off-screen rendering")

env = create_env_pixels()
print("Ambiente criado com sucesso e render ativo.")
print("Verificando action_space:", env.action_space)

# ====================================================================================
# Utils — frames e estado interno
# ====================================================================================
def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return frame

def extract_frame(obs):
    if isinstance(obs, np.ndarray):
        return _to_uint8_rgb(obs)
    if isinstance(obs, (list, tuple)) and len(obs) > 0:
        return extract_frame(obs[0])
    if isinstance(obs, dict) and "frame" in obs:
        return _to_uint8_rgb(np.array(obs["frame"]))
    raise TypeError(f"Formato de observação inesperado para pixels: {type(obs)}; exemplo={str(obs)[:200]}")

def _get_internal_observation(env_obj):
    for attr in ["_env", "env"]:
        inner = getattr(env_obj.unwrapped, attr, None)
        if inner is None:
            continue
        for fn_name in ["_observation", "observation"]:
            fn = getattr(inner, fn_name, None)
            if callable(fn):
                try:
                    raw = fn()
                    if isinstance(raw, dict):
                        return raw
                except Exception:
                    pass
    return None

def collect_state(info, env_obj, dbg_writer=None, step=None):
    """Extrai estado padronizado do mesmo env (info + raw)."""
    out = {
        "score": info.get("score", None),
        "ball_owned_team": info.get("ball_owned_team", None),
        "ball_owned_player": info.get("ball_owned_player", None),
        "ball_position": info.get("ball_position", None),
        "left_team": info.get("left_team", None),
        "right_team": info.get("right_team", None),
    }
    need_raw = any(out[k] is None for k in ["score", "ball_owned_team", "ball_owned_player", "ball_position"]) \
               or out["left_team"] is None or out["right_team"] is None
    raw = None
    if need_raw:
        raw = _get_internal_observation(env_obj)
        if isinstance(raw, dict):
            if out["score"] is None:
                sc = raw.get("score")
                if isinstance(sc, (list, tuple)) and len(sc) >= 2:
                    out["score"] = (int(sc[0]), int(sc[1]))
            if out["ball_owned_team"] is None:
                bot = raw.get("ball_owned_team")
                if isinstance(bot, (int, np.integer)):
                    out["ball_owned_team"] = int(bot)
            if out["ball_owned_player"] is None:
                bop = raw.get("ball_owned_player")
                if isinstance(bop, (int, np.integer)):
                    out["ball_owned_player"] = int(bop)
            if out["ball_position"] is None:
                bp = None
                for key in ("ball_position", "ball", "ball_coords"):
                    val = raw.get(key)
                    if isinstance(val, (list, tuple, np.ndarray)):
                        bp = val
                        break
                if isinstance(bp, (list, tuple, np.ndarray)) and len(bp) >= 2:
                    out["ball_position"] = [float(bp[0]), float(bp[1])]
                else:
                    pr = raw.get("players_raw")
                    if isinstance(pr, (list, tuple)) and len(pr) > 0 and isinstance(pr[0], dict):
                        b = pr[0].get("ball")
                        if isinstance(b, (list, tuple, np.ndarray)) and len(b) >= 2:
                            out["ball_position"] = [float(b[0]), float(b[1])]
            if out["left_team"] is None:
                lt = raw.get("left_team")
                if isinstance(lt, (list, tuple, np.ndarray)) and len(lt) > 0:
                    out["left_team"] = np.array(lt, dtype=float).tolist()
            if out["right_team"] is None:
                rt = raw.get("right_team")
                if isinstance(rt, (list, tuple, np.ndarray)) and len(rt) > 0:
                    out["right_team"] = np.array(rt, dtype=float).tolist()

    # padrões
    out.setdefault("score", (0, 0))
    out.setdefault("ball_owned_team", -1)
    out.setdefault("ball_owned_player", -1)
    if out.get("ball_position") is None:
        out["ball_position"] = None
    if out.get("left_team") is None:
        out["left_team"] = None
    if out.get("right_team") is None:
        out["right_team"] = None

    if dbg_writer is not None:
        try:
            dbg_writer.write(json.dumps({
                "step": step,
                "keys_info": sorted(list(info.keys())) if isinstance(info, dict) else [],
                "sample": {
                    "score": out["score"],
                    "bot": out["ball_owned_team"],
                    "bop": out["ball_owned_player"],
                    "ball": out["ball_position"],
                    "lt_len": len(out["left_team"]) if isinstance(out["left_team"], list) else None,
                    "rt_len": len(out["right_team"]) if isinstance(out["right_team"], list) else None,
                }
            }, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return out

# ====================================================================================
# Geometria e helpers
# ====================================================================================
def vsub(a, b): return (a[0]-b[0], a[1]-b[1])
def vlen(v): return math.hypot(v[0], v[1])
def vdot(a, b): return a[0]*b[0] + a[1]*b[1]
def angle_between(a, b):
    la, lb = vlen(a), vlen(b)
    if la == 0 or lb == 0:
        return math.pi
    cos = max(-1.0, min(1.0, vdot(a, b)/(la*lb)))
    return math.acos(cos)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ====================================================================================
# Parâmetros (recalibrados para 3v3 com escala [-1,1])
# ====================================================================================
GOAL_CENTER = {0: ( 1.0, 0.0), 1: (-1.0, 0.0)}   # time 0 chuta à direita; time 1 chuta à esquerda
GOAL_CONE_DEG  = 12.0                            # direção ao gol no toque
PASS_CONE_DEG  = 28.0                            # direção ao companheiro no toque
BALL_MIN_SPEED = 0.009                           # velocidade mínima (após suavização) p/ considerar toque/chute
RECEIVE_RADIUS = 0.09                            # “bola no pé” ao receber
TOUCH_RADIUS   = 0.045                           # para possuidor proxy (quando BOT == -1)
WINDOW_PASS    = 40                              
WINDOW_SHOT    = 55                              
DEFENSE_RADIUS = 0.18                            # região considerada “defendida” perto do gol

EMA_ALPHA = 0.6                                  # suavização da velocidade (EMA sobre 3 frames efetivos)

# ====================================================================================
# Detector de toques (âncora única de eventos)
# ====================================================================================
class Touch:
    __slots__ = ("touch_id", "step", "team", "player", "ball_pos", "v_ema", "class_type",
                 "target_idx", "goal_angle_deg", "best_pass_angle_deg")
    def __init__(self, touch_id, step, team, player, ball_pos, v_ema,
                 class_type, target_idx, goal_angle_deg, best_pass_angle_deg):
        self.touch_id = touch_id
        self.step = step
        self.team = team
        self.player = player
        self.ball_pos = ball_pos
        self.v_ema = v_ema
        self.class_type = class_type        # "shot" | "pass" | "none"
        self.target_idx = target_idx        # índice do companheiro (se "pass")
        self.goal_angle_deg = goal_angle_deg
        self.best_pass_angle_deg = best_pass_angle_deg

class TouchDetector:
    def __init__(self):
        self.prev_ball = None
        self.v_raw = (0.0, 0.0)
        self.v_ema = (0.0, 0.0)
        self.touch_counter = 0
        self.last_touch_player = None  # (team, idx)

    def _ema_vec(self, prev, new):
        return (EMA_ALPHA*new[0] + (1-EMA_ALPHA)*prev[0],
                EMA_ALPHA*new[1] + (1-EMA_ALPHA)*prev[1])

    def nearest_player(self, ball, team_positions):
        if ball is None or team_positions is None:
            return None, None
        bx, by = ball
        dmin, idx = 1e9, None
        for i, p in enumerate(team_positions):
            d = (p[0]-bx)**2 + (p[1]-by)**2
            if d < dmin:
                dmin, idx = d, i
        return idx, math.sqrt(dmin)

    def classify_touch(self, team, player, ball_prev, v_ema, lt, rt):
        if ball_prev is None or vlen(v_ema) < BALL_MIN_SPEED:
            return "none", None, 999.0, 999.0
        # Vetores alvos
        goal_vec = vsub(GOAL_CENTER[team], ball_prev)
        ang_goal = math.degrees(angle_between(v_ema, goal_vec))
        if ang_goal <= GOAL_CONE_DEG:
            return "shot", None, ang_goal, 999.0

        # Pass: escolhe companheiro com menor ângulo
        team_positions = lt if team == 0 else rt
        best_idx, best_ang = None, 999.0
        if team_positions is not None:
            for i, p in enumerate(team_positions):
                if i == player:
                    continue
                to_teammate = vsub(p, ball_prev)
                ang = math.degrees(angle_between(v_ema, to_teammate))
                if ang < best_ang:
                    best_ang, best_idx = ang, i
        if best_idx is not None and best_ang <= PASS_CONE_DEG:
            return "pass", best_idx, ang_goal, best_ang
        return "none", None, ang_goal, best_ang

    def update(self, step, s, dbg_writer=None):
        """Retorna uma lista de toques detectados neste step (0 ou 1)."""
        touches = []
        ball = s["ball_position"]
        lt, rt = s["left_team"], s["right_team"]
        bot, bop = s["ball_owned_team"], s["ball_owned_player"]

        # velocidade
        if ball is not None and self.prev_ball is not None:
            self.v_raw = vsub(ball, self.prev_ball)
            self.v_ema = self._ema_vec(self.v_ema, self.v_raw)

        # Quem é o "possuidor" efetivo no instante, com fallback proxy
        poss_team, poss_player = bot, bop
        if poss_team == -1 or poss_player == -1:
            # possuidor proxy: jogador mais próximo (qualquer time) se muito perto
            idxL, dL = self.nearest_player(ball, lt) if lt is not None else (None, None)
            idxR, dR = self.nearest_player(ball, rt) if rt is not None else (None, None)
            if dL is None and dR is None:
                poss_team, poss_player = -1, -1
            else:
                # escolhe o mais perto
                if dR is None or (dL is not None and dL <= dR):
                    if dL is not None and dL <= TOUCH_RADIUS:
                        poss_team, poss_player = 0, idxL
                else:
                    if dR is not None and dR <= TOUCH_RADIUS:
                        poss_team, poss_player = 1, idxR

        # Regra de toque: (a) há possuidor efetivo, (b) velocidade ema acima de limiar, (c) mudou o possuidor desde o último toque OU é o mesmo mas velocidade subiu (chute/pass)
        speed = vlen(self.v_ema)
        touch_detected = False
        if poss_team in (0, 1) and poss_player is not None and poss_player != -1:
            # novo toque se trocou jogador, ou se é o mesmo jogador mas houve "impulso"
            if self.last_touch_player is None or self.last_touch_player != (poss_team, poss_player):
                if speed >= BALL_MIN_SPEED:
                    touch_detected = True
            else:
                # mesmo jogador: detecta novo toque se há novo pico de velocidade
                if speed >= BALL_MIN_SPEED and self.prev_ball is not None:
                    # pequeno histerese: o vetor mudou o suficiente?
                    if vlen(self.v_raw) >= BALL_MIN_SPEED * 0.6:
                        touch_detected = True

        if touch_detected:
            self.touch_counter += 1
            class_type, target_idx, ang_goal, ang_pass = self.classify_touch(
                poss_team, poss_player, self.prev_ball if self.prev_ball is not None else ball,
                self.v_ema, lt, rt
            )
            t = Touch(
                touch_id=self.touch_counter,
                step=step,
                team=poss_team,
                player=poss_player,
                ball_pos=(ball[0], ball[1]) if ball is not None else (None, None),
                v_ema=self.v_ema,
                class_type=class_type,
                target_idx=target_idx,
                goal_angle_deg=ang_goal,
                best_pass_angle_deg=ang_pass
            )
            touches.append(t)
            self.last_touch_player = (poss_team, poss_player)

        if dbg_writer is not None:
            try:
                dbg_writer.write(json.dumps({
                    "step": step,
                    "ball": ball,
                    "v_raw": list(self.v_raw),
                    "v_ema": list(self.v_ema),
                    "speed": speed,
                    "poss_team": poss_team,
                    "poss_player": poss_player,
                    "touches_this_step": [{
                        "touch_id": T.touch_id, "team": T.team, "player": T.player,
                        "class_type": T.class_type, "target_idx": T.target_idx,
                        "ang_goal": T.goal_angle_deg, "ang_pass": T.best_pass_angle_deg
                    } for T in touches]
                }, ensure_ascii=False) + "\n")
            except Exception:
                pass

        self.prev_ball = ball[:] if ball is not None else None
        return touches

# ====================================================================================
# Motor de eventos a partir dos toques
# ====================================================================================
class EventEngine:
    def __init__(self):
        # eventos textuais (para CSV)
        self.events_user = []     # (step, name, team, extra_dict)
        self.events_legacy = []   # (step, name, team, extra_dict)

        # métricas agregadas
        self.metrics_user = {
            0: {"passes_tentados": 0, "passes_certos": 0, "finalizacoes": 0,
                "finalizacoes_certas": 0, "finalizacoes_erradas": 0, "gols": 0},
            1: {"passes_tentados": 0, "passes_certos": 0, "finalizacoes": 0,
                "finalizacoes_certas": 0, "finalizacoes_erradas": 0, "gols": 0},
        }
        self.metrics_legacy = {
            0: {"passes_tentados": 0, "passes_certos": 0, "finalizacoes": 0,
                "finalizacoes_certas": 0, "finalizacoes_erradas": 0, "gols": 0},
            1: {"passes_tentados": 0, "passes_certos": 0, "finalizacoes": 0,
                "finalizacoes_certas": 0, "finalizacoes_erradas": 0, "gols": 0},
        }

        # estados para janelas (com lock por touch_id)
        self.pending_pass = None  # {"touch_id","team","from_idx","target_idx","start_step"}
        self.pending_shot = None  # {"touch_id","team","start_step"}

        self.prev_score = (0, 0)

    def log_user(self, step, name, team, **kw):
        self.events_user.append((step, name, team if team is not None else -1, kw))

    def log_legacy(self, step, name, team, **kw):
        self.events_legacy.append((step, name, team if team is not None else -1, kw))

    def crossed_goal_line(self, prev_ball, ball, team):
        if prev_ball is None or ball is None:
            return False
        x0, y0 = prev_ball
        x1, y1 = ball
        GOAL_Y = 0.065
        def y_at_x(x_target):
            if x1 == x0:
                return None
            t = (x_target - x0) / (x1 - x0)
            if t < 0 or t > 1:
                return None
            return y0 + t * (y1 - y0)
        if team == 0:
            y_cross = y_at_x(1.0)
            return (y_cross is not None) and (abs(y_cross) <= GOAL_Y + 0.02)
        else:
            y_cross = y_at_x(-1.0)
            return (y_cross is not None) and (abs(y_cross) <= GOAL_Y + 0.02)

    def process(self, step, s, touches, prev_ball):
        score = s["score"]
        bot, bop = s["ball_owned_team"], s["ball_owned_player"]
        ball = s["ball_position"]
        lt, rt = s["left_team"], s["right_team"]

        # ----------------------
        # 1) Gols (sempre contam)
        # ----------------------
        if score != self.prev_score:
            dL = score[0] - self.prev_score[0]
            dR = score[1] - self.prev_score[1]
            if dL > 0:
                self.metrics_user[0]["gols"] += dL
                self.metrics_user[0]["finalizacoes"] += dL
                self.metrics_user[0]["finalizacoes_certas"] += dL
                self.metrics_legacy[0]["gols"] += dL
                self.metrics_legacy[0]["finalizacoes"] += dL
                self.metrics_legacy[0]["finalizacoes_certas"] += dL
                self.log_user(step, "gol", 0, delta=dL, score=score)
                self.log_legacy(step, "gol", 0, delta=dL, score=score)
            if dR > 0:
                self.metrics_user[1]["gols"] += dR
                self.metrics_user[1]["finalizacoes"] += dR
                self.metrics_user[1]["finalizacoes_certas"] += dR
                self.metrics_legacy[1]["gols"] += dR
                self.metrics_legacy[1]["finalizacoes"] += dR
                self.metrics_legacy[1]["finalizacoes_certas"] += dR
                self.log_user(step, "gol", 1, delta=dR, score=score)
                self.log_legacy(step, "gol", 1, delta=dR, score=score)
            # um gol encerra janelas
            self.pending_pass = None
            self.pending_shot = None

        # ----------------------------------------
        # 2) Abrir janelas a partir dos TOQUES (lock)
        # ----------------------------------------
        for t in touches:
            if t.class_type == "shot":
                # abre shot somente se não há outro pendente
                if self.pending_shot is None:
                    self.pending_shot = {"touch_id": t.touch_id, "team": t.team, "start_step": step}
                    self.metrics_user[t.team]["finalizacoes"] += 1
                    self.metrics_legacy[t.team]["finalizacoes"] += 1
                    self.log_user(step, "finalizacao_tentada_user", t.team,
                                  kicker=t.player, touch_id=t.touch_id, ang_g=t.goal_angle_deg)
                    self.log_legacy(step, "finalizacao_tentada_legacy", t.team,
                                    kicker=t.player, touch_id=t.touch_id)
                # um toque que é chute cancela passe pendente
                self.pending_pass = None

            elif t.class_type == "pass":
                # abre passe somente se não há outro pendente (lock por toque)
                if self.pending_pass is None and self.pending_shot is None:
                    self.pending_pass = {"touch_id": t.touch_id, "team": t.team,
                                         "from_idx": t.player, "target_idx": t.target_idx, "start_step": step}
                    self.metrics_user[t.team]["passes_tentados"] += 1
                    self.metrics_legacy[t.team]["passes_tentados"] += 1
                    self.log_user(step, "passe_tentado_user", t.team,
                                  from_idx=t.player, to_idx=t.target_idx, touch_id=t.touch_id,
                                  ang_pass=t.best_pass_angle_deg)
                    self.log_legacy(step, "passe_tentado_legacy", t.team,
                                    from_idx=t.player, to_idx=t.target_idx, touch_id=t.touch_id)

        # --------------------------------------------------
        # 3) Desfechos: PASSE (user: alvo fixo | legacy: simples)
        # --------------------------------------------------
        if self.pending_pass is not None:
            pt = self.pending_pass
            # User regra: alvo fixo e bola no pé do receptor
            if bot == pt["team"] and bop != -1 and bop == pt["target_idx"]:
                team_positions = lt if bot == 0 else rt
                recv_pos = None
                if team_positions is not None and bop < len(team_positions):
                    recv_pos = team_positions[bop]
                ok = False
                if ball is not None and recv_pos is not None:
                    ok = vlen(vsub(ball, recv_pos)) <= RECEIVE_RADIUS
                if ok:
                    self.metrics_user[pt["team"]]["passes_certos"] += 1
                    self.log_user(step, "passe_certo_user", pt["team"],
                                  from_idx=pt["from_idx"], to_idx=bop, touch_id=pt["touch_id"])
                    # Legacy (conta também como correto, pois alvo bateu)
                    self.metrics_legacy[pt["team"]]["passes_certos"] += 1
                    self.log_legacy(step, "passe_certo_legacy", pt["team"],
                                    from_idx=pt["from_idx"], to_idx=bop, touch_id=pt["touch_id"])
                    self.pending_pass = None
            # Legacy fallback: troca de jogador no mesmo time (mesmo se alvo diferente), bola próxima ao receptor
            elif bot == pt["team"] and bop != -1 and bop != pt["from_idx"]:
                team_positions = lt if bot == 0 else rt
                recv_pos = None
                if team_positions is not None and bop < len(team_positions):
                    recv_pos = team_positions[bop]
                ok_legacy = False
                if ball is not None and recv_pos is not None:
                    ok_legacy = vlen(vsub(ball, recv_pos)) <= RECEIVE_RADIUS
                if ok_legacy:
                    self.metrics_legacy[pt["team"]]["passes_certos"] += 1
                    self.log_legacy(step, "passe_certo_legacy", pt["team"],
                                    from_idx=pt["from_idx"], to_idx=bop, touch_id=pt["touch_id"])
                    # não fecha a janela do user ainda; só fecha se for o target do user ou se expirar
            # Erro de passe se posse vira do adversário
            elif bot in (0, 1) and bot != pt["team"]:
                self.log_user(step, "passe_errado_user", pt["team"], reason="posse_adversario", touch_id=pt["touch_id"])
                self.log_legacy(step, "passe_errado_legacy", pt["team"], reason="posse_adversario", touch_id=pt["touch_id"])
                self.pending_pass = None
            # Expira janela
            elif step - pt["start_step"] > WINDOW_PASS:
                self.log_user(step, "passe_nao_concluido_user", pt["team"], reason="timeout", touch_id=pt["touch_id"])
                self.log_legacy(step, "passe_nao_concluido_legacy", pt["team"], reason="timeout", touch_id=pt["touch_id"])
                self.pending_pass = None

        # --------------------------------------------------
        # 4) Desfechos: FINALIZAÇÃO (gol / defesa / fora)
        # --------------------------------------------------
        if self.pending_shot is not None:
            st = self.pending_shot
            # gol já tratado acima (fecha)
            if score != self.prev_score:
                self.pending_shot = None
            elif step - st["start_step"] > WINDOW_SHOT:
                # defesa: adversário ganha posse PERTINHO do gol
                adv = 1 - st["team"]
                defended = False
                gx, gy = GOAL_CENTER[st["team"]]
                if ball is not None and bot == adv:
                    if abs(ball[0] - gx) < 0.15 and abs(ball[1] - gy) < DEFENSE_RADIUS:
                        defended = True
                if defended:
                    self.metrics_user[st["team"]]["finalizacoes_certas"] += 1
                    self.metrics_legacy[st["team"]]["finalizacoes_certas"] += 1
                    self.log_user(step, "finalizacao_defendida_user", st["team"])
                    self.log_legacy(step, "finalizacao_defendida_legacy", st["team"])
                else:
                    self.metrics_user[st["team"]]["finalizacoes_erradas"] += 1
                    self.metrics_legacy[st["team"]]["finalizacoes_erradas"] += 1
                    self.log_user(step, "finalizacao_fora_user", st["team"])
                    self.log_legacy(step, "finalizacao_fora_legacy", st["team"])
                self.pending_shot = None

        self.prev_score = score

# ====================================================================================
# Execução principal (um env; frames + toques + eventos; prints de progresso)
# ====================================================================================
obs = env.reset()
print("Ambiente resetado. Coleta de frames e métricas iniciada.")

done = False
step = 0

# arquivos de depuração
dbg = open(INFO_DBG, "w", encoding="utf-8")

# salva primeiro frame
frame0 = extract_frame(obs)
imageio.imwrite(os.path.join(OUTPUT_DIR, f"frame_{step:04d}.png"), frame0)

# estado inicial
state = collect_state({}, env, dbg_writer=dbg, step=step)

detector = TouchDetector()
engine = EventEngine()

prev_ball_for_goal = state["ball_position"][:] if state["ball_position"] is not None else None

# também vamos acumular TOQUES em CSV
touch_rows = []
# processa step 0 (não gera eventos ainda; só captura baseline)
step += 1

with tqdm(total=MAX_STEPS - 1, desc="Frames + métricas (1 env, motor por toques)", ncols=92) as pbar:
    while not done and step < MAX_STEPS:
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action])

        # salva PNG
        frame = extract_frame(obs)
        imageio.imwrite(os.path.join(OUTPUT_DIR, f"frame_{step:04d}.png"), frame)

        # estado para este step
        s = collect_state(info or {}, env, dbg_writer=dbg, step=step)

        # detecção de toque
        touches = detector.update(step, s, dbg_writer=dbg)
        for t in touches:
            touch_rows.append([t.touch_id, step, t.team, t.player,
                               round(t.ball_pos[0], 6) if t.ball_pos[0] is not None else None,
                               round(t.ball_pos[1], 6) if t.ball_pos[1] is not None else None,
                               round(vlen(t.v_ema), 6), t.class_type, t.target_idx,
                               round(t.goal_angle_deg, 3), round(t.best_pass_angle_deg, 3)])

        # processa eventos (user + legacy) a partir das janelas comuns
        engine.process(step, s, touches, prev_ball_for_goal)

        prev_ball_for_goal = s["ball_position"][:] if s["ball_position"] is not None else None

        # prints de marcos
        if step % 100 == 0:
            print(f"[INFO] Step {step} | user(L): {engine.metrics_user[0]} | user(R): {engine.metrics_user[1]}")

        step += 1
        pbar.update(1)

dbg.close()

# fecha e força flush do vídeo do motor
env.close()
del env
gc.collect()

print(f"🏁 Episódio terminou no step {step}")
print(f"✅ PNGs salvos em: {OUTPUT_DIR}/")

# ====================================================================================
# Checagem do vídeo do motor (substitui antigos pois limpamos antes)
# ====================================================================================
def find_engine_videos(root):
    vids = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        vids.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return vids

engine_videos = find_engine_videos(VIDEO_LOGDIR)
if engine_videos:
    print("🎬 Vídeo(s) do motor encontrado(s):")
    for v in engine_videos:
        print(" -", v)
else:
    print("⚠️ Nenhum vídeo do motor encontrado em", VIDEO_LOGDIR)

# ====================================================================================
# Fallback MP4 a partir dos PNGs (substitui antigo pois apagamos no início)
# ====================================================================================
if MAKE_MP4_FROM_PNGS:
    try:
        mp4_path = os.path.join(OUTPUT_DIR, f"{SCENARIO_NAME}_fallback.mp4")
        pngs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "frame_*.png")))
        if pngs:
            print(f"🎯 Gerando vídeo de fallback a partir dos PNGs: {mp4_path}")
            with imageio.get_writer(mp4_path, mode="I", fps=FPS_FALLBACK, codec="libx264") as writer:
                for p in pngs:
                    writer.append_data(imageio.imread(p))
            print("✅ Vídeo de fallback salvo em:", mp4_path)
        else:
            print("⚠️ Sem PNGs para montar vídeo de fallback.")
    except Exception as e:
        print("❌ Falha ao gerar MP4 de fallback:", repr(e))
        print("💡 Já que você instalou o render MP4, confirme também no container:  pip install 'imageio[ffmpeg]'  ou apt-get install -y ffmpeg")

# ====================================================================================
# Dump de eventos, toques e métricas (com tolerância a erros de IO)
# ====================================================================================
def safe_dump_events(csv_path, events):
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "evento", "time", "extra_json"])
            for (s, name, team, extra) in events:
                w.writerow([s, name, team, json.dumps(extra, ensure_ascii=False)])
        return True, None
    except Exception as e:
        return False, repr(e)

def safe_dump_touches(csv_path, rows):
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["touch_id", "step", "team", "player", "ball_x", "ball_y",
                        "speed_ema", "class_type", "target_idx", "ang_goal_deg", "ang_pass_deg"])
            for r in rows:
                w.writerow(r)
        return True, None
    except Exception as e:
        return False, repr(e)

ok_u, err_u = safe_dump_events(EVENTS_USER_CSV, engine.events_user)
ok_l, err_l = safe_dump_events(EVENTS_LEGACY_CSV, engine.events_legacy)
ok_t, err_t = safe_dump_touches(TOUCHES_CSV, touch_rows)

if not ok_u:
    print("❌ Falha ao salvar events_user.csv:", err_u)
else:
    print("📝 Eventos (user) salvos em:", EVENTS_USER_CSV)

if not ok_l:
    print("❌ Falha ao salvar events_legacy.csv:", err_l)
else:
    print("📝 Eventos (legacy) salvos em:", EVENTS_LEGACY_CSV)

if not ok_t:
    print("❌ Falha ao salvar touches.csv:", err_t)
else:
    print("📝 Toques (âncoras) salvos em:", TOUCHES_CSV)

totals = {
    "legacy": engine.metrics_legacy,
    "user": engine.metrics_user
}
try:
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(totals, f, ensure_ascii=False, indent=2)
    print("📊 Métricas finais (resumo):")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(f"🧾 Resumo em: {METRICS_JSON}")
except Exception as e:
    print("❌ Falha ao salvar metrics.json:", repr(e))
# ====================================================================================
# Pós-processamento: validação cruzada e timeline de eventos
# ====================================================================================
def recompute_metrics_from_events(csv_path):
    rec = {
        0: {"passes_tentados": 0, "passes_certos": 0, "finalizacoes": 0,
            "finalizacoes_certas": 0, "finalizacoes_erradas": 0, "gols": 0},
        1: {"passes_tentados": 0, "passes_certos": 0, "finalizacoes": 0,
            "finalizacoes_certas": 0, "finalizacoes_erradas": 0, "gols": 0},
    }
    if not os.path.exists(csv_path):
        return rec

    def bump(team, key, v=1):
        if team in rec:
            rec[team][key] += v

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("evento", "")
            try:
                team = int(row.get("time", -1))
            except Exception:
                team = -1

            if "passe_tentado" in name:
                bump(team, "passes_tentados")
            elif "passe_certo" in name:
                bump(team, "passes_certos")
            elif "finalizacao_tentada" in name:
                bump(team, "finalizacoes")
            elif ("finalizacao_defendida" in name) or ("finalizacao_no_alvo" in name):
                bump(team, "finalizacoes_certas")
            elif "finalizacao_fora" in name:
                bump(team, "finalizacoes_erradas")
            elif name == "gol":
                bump(team, "gols")
                bump(team, "finalizacoes")
                bump(team, "finalizacoes_certas")
    return rec

def _load_totals_from_disk(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def print_diff_metrics(tag, computed, recomputed):
    print(f"\n🔎 Validação via CSV ({tag})")
    for team in (0, 1):
        a = computed.get(str(team), computed.get(team, {}))
        b = recomputed.get(team, {})
        keys = sorted(set(a.keys()) | set(b.keys()))
        diffs = []
        for k in keys:
            try:
                va = int(a.get(k, 0))
            except Exception:
                va = 0
            try:
                vb = int(b.get(k, 0))
            except Exception:
                vb = 0
            status = "OK" if va == vb else f"DIFF [{va} != {vb}]"
            diffs.append(f"{k}: {va} / {vb} -> {status}")
        print(f"  Team {team}: " + " | ".join(diffs))

# Recarrega 'totals' do disco se não estiver no escopo
if "totals" not in globals():
    _t = _load_totals_from_disk(METRICS_JSON)
    if isinstance(_t, dict):
        totals = _t
    else:
        totals = {"legacy": {0:{},1:{}}, "user": {0:{},1:{}}}

# Reconstruir métricas a partir dos CSVs de eventos
re_user   = recompute_metrics_from_events(EVENTS_USER_CSV)
re_legacy = recompute_metrics_from_events(EVENTS_LEGACY_CSV)

# Comparar com o que ficou em 'totals'
print_diff_metrics("user", totals.get("user", {}), re_user)
print_diff_metrics("legacy", totals.get("legacy", {}), re_legacy)

# Avisos de inconsistência comuns
def warn_common_inconsistencies(tot):
    for team in (0, 1):
        for family in ("user", "legacy"):
            fam = tot.get(family, {})
            # suportar chaves str ou int
            bucket = fam.get(team, fam.get(str(team), {}))
            pc = int(bucket.get("passes_certos", 0))
            pt = int(bucket.get("passes_tentados", 0))
            fc = int(bucket.get("finalizacoes_certas", 0))
            fe = int(bucket.get("finalizacoes_erradas", 0))
            ft = int(bucket.get("finalizacoes", 0))
            if pc > pt:
                print(f"⚠️ {family}: passes_certos ({pc}) > passes_tentados ({pt}) no time {team}")
            if (fc + fe) > ft:
                print(f"⚠️ {family}: finalizacoes_certas+erradas ({fc+fe}) > finalizacoes ({ft}) no time {team}")

warn_common_inconsistencies(totals)

# Timeline consolidada simples (apenas eventos)
TIMELINE_JSONL = os.path.join(OUTPUT_DIR, "timeline.jsonl")

def _load_events(csv_path, source_tag):
    out = []
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                s = int(row.get("step", 0))
            except Exception:
                continue
            try:
                team = int(row.get("time", -1))
            except Exception:
                team = -1
            try:
                extra = json.loads(row.get("extra_json", "{}"))
            except Exception:
                extra = {"raw": row.get("extra_json")}
            out.append({
                "step": s,
                "type": "event",
                "source": source_tag,
                "name": row.get("evento", ""),
                "team": team,
                "extra": extra
            })
    return out

timeline = []
timeline += _load_events(EVENTS_USER_CSV, "user")
timeline += _load_events(EVENTS_LEGACY_CSV, "legacy")
timeline.sort(key=lambda x: (x["step"], x["source"]))

try:
    with open(TIMELINE_JSONL, "w", encoding="utf-8") as f:
        for item in timeline:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"🧵 Timeline consolidada salva em: {TIMELINE_JSONL}")
except Exception as e:
    print("❌ Falha ao salvar timeline.jsonl:", repr(e))

# Estatísticas finais dos arquivos gerados
try:
    num_png = len(sorted(glob.glob(os.path.join(OUTPUT_DIR, "frame_*.png"))))
    print(f"🖼 Total de frames PNG: {num_png}")
except Exception:
    pass

try:
    engine_videos = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        engine_videos.extend(glob.glob(os.path.join(VIDEO_LOGDIR, "**", ext), recursive=True))
    if engine_videos:
        print("🎬 Vídeo(s) localizado(s) no log do engine:")
        for v in engine_videos:
            print(" -", v)
    else:
        print("⚠️ Nenhum vídeo do engine localizado em", VIDEO_LOGDIR)
except Exception:
    pass

print("\n✅ Execução concluída.")
