from . import *

def build_scenario(builder):
    cfg = builder.config()
    cfg.game_duration = 3000  # duração em steps (~5 minutos)
    cfg.right_team_difficulty = 0.5
    cfg.left_team_difficulty = 1
    cfg.deterministic = False

    # Log simples da configuração
    try:
        print(f"[scenario] LEFT(diff={cfg.left_team_difficulty}) atk vs RIGHT(diff={cfg.right_team_difficulty}) def+GK")
    except Exception:
        pass

    # Alternar quem começa com a bola
    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left

    # Time 1 (esquerdo ou direito dependendo do episódio)
    builder.SetTeam(first_team)
    builder.AddPlayer(-0.9, 0.0, e_PlayerRole_GK, controllable=False)  # Goleiro fixo
    builder.AddPlayer(-0.4, 0.1, e_PlayerRole_CB, lazy = False)  # Meio campo
    builder.AddPlayer(-0.4, -0.1, e_PlayerRole_CB,lazy = False)  # Atacante
    builder.AddPlayer(-0.5, -0.1, e_PlayerRole_CB,lazy = False)  # Atacante

    # Time 2 (oposto)
    builder.SetTeam(second_team)
    builder.AddPlayer(0.9, 0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(0.4, 0.1, e_PlayerRole_CM)
    builder.AddPlayer(0.4, -0.1, e_PlayerRole_CF)
    builder.AddPlayer(0.5, -0.1, e_PlayerRole_CF)
