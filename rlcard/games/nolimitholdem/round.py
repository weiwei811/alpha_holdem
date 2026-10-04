# -*- coding: utf-8 -*-
"""Implement no limit texas holdem Round class"""
from enum import Enum

from rlcard.games.limitholdem import PlayerStatus


class Action(Enum):
    FOLD = 0
    CHECK_CALL = 1
    #CALL = 2
    # RAISE_3BB = 3
    RAISE_HALF_POT = 2
    RAISE_POT = 3
    # RAISE_2POT = 5
    ALL_IN = 4
    # SMALL_BLIND = 7
    # BIG_BLIND = 8


class NolimitholdemRound:
    """Track seats still owing a decision and full-raise reopening rights."""

    def __init__(self, num_players, init_raise_amount, dealer, np_random):
        self.num_players = num_players
        self.init_raise_amount = init_raise_amount
        self.dealer = dealer
        self.np_random = np_random
        self.start_new_round(0)

    def start_new_round(self, game_pointer, raised=None):
        self.game_pointer = game_pointer
        self.raised = list(raised) if raised is not None else [0] * self.num_players
        self.last_raise_amount = self.init_raise_amount
        self.pending = None
        self.acted_at = {}

    def _active(self, players):
        return {i for i, p in enumerate(players) if p.status == PlayerStatus.ALIVE}

    def _can_raise(self, pid):
        return (pid not in self.acted_at or
                max(self.raised) - self.acted_at[pid] >= self.last_raise_amount)

    def _quantity(self, action, players):
        player = players[self.game_pointer]
        call = max(self.raised) - self.raised[self.game_pointer]
        pot = sum(p.in_chips for p in players)
        if action == Action.CHECK_CALL:
            return min(call, player.remained_chips)
        if action == Action.ALL_IN:
            return player.remained_chips
        fraction = 0.5 if action == Action.RAISE_HALF_POT else 1.0
        return call + int((pot + call) * fraction)

    def proceed_round(self, players, action):
        pid = self.game_pointer
        player = players[pid]
        if action not in self.get_nolimit_legal_actions(players):
            raise ValueError('Action not allowed')
        if self.pending is None:
            self.pending = self._active(players)
        old_max = max(self.raised)
        if action == Action.FOLD:
            player.status = PlayerStatus.FOLDED
        else:
            quantity = self._quantity(action, players)
            player.bet(quantity)
            self.raised[pid] += quantity
            if player.remained_chips == 0:
                player.status = PlayerStatus.ALLIN
            increment = max(self.raised) - old_max
            if increment >= self.last_raise_amount:
                self.last_raise_amount = increment
                self.pending = self._active(players)
            elif increment > 0:
                # Short all-ins require calls, but do not reopen earlier callers.
                self.pending |= {i for i in self._active(players)
                                 if self.raised[i] < max(self.raised)}
            self.acted_at[pid] = max(self.raised)
        self.pending.discard(pid)
        active = self._active(players)
        self.pending &= active
        if len(active) == 1 and self.raised[next(iter(active))] >= max(self.raised):
            self.pending.clear()
        if len([p for p in players if p.status != PlayerStatus.FOLDED]) == 1:
            self.pending.clear()
        for offset in range(1, self.num_players + 1):
            candidate = (pid + offset) % self.num_players
            if candidate in self.pending:
                self.game_pointer = candidate
                break
        return self.game_pointer

    def get_nolimit_legal_actions(self, players):
        pid = self.game_pointer
        player = players[pid]
        if player.status != PlayerStatus.ALIVE:
            return []
        actions = [Action.FOLD, Action.CHECK_CALL]
        call = max(self.raised) - self.raised[pid]
        other_active = self._active(players) - {pid}
        if not other_active or not self._can_raise(pid) or player.remained_chips <= call:
            return actions
        for action in (Action.RAISE_HALF_POT, Action.RAISE_POT):
            quantity = self._quantity(action, players)
            if quantity <= player.remained_chips and quantity - call >= self.last_raise_amount:
                actions.append(action)
        actions.append(Action.ALL_IN)
        return actions

    def is_over(self):
        return self.pending is not None and not self.pending
