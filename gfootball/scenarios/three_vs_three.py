# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

from . import *

def build_scenario(builder):
    builder.config().game_duration = 3000  # duração em steps (~5 minutos)
    builder.config().right_team_difficulty = 0.05
    builder.config().left_team_difficulty = 0.05
    builder.config().deterministic = False

    # Alternar quem começa com a bola
    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left

    # Time 1 (esquerdo ou direito dependendo do episódio)
    builder.SetTeam(first_team)
    builder.AddPlayer(-0.9, 0.0, e_PlayerRole_GK, controllable=False)  # Goleiro fixo, não controlável
    builder.AddPlayer(-0.4, 0.1, e_PlayerRole_CM)  # Meio campo
    builder.AddPlayer(-0.4, -0.1, e_PlayerRole_CF)  # Atacante

    # Time 2 (oposto)
    builder.SetTeam(second_team)
    builder.AddPlayer(0.9, 0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(0.4, 0.1, e_PlayerRole_CM)
    builder.AddPlayer(0.4, -0.1, e_PlayerRole_CF)
