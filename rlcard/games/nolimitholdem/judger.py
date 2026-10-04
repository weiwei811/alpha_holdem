import numpy as np
from rlcard.games.limitholdem import Judger
from rlcard.games.limitholdem.utils import compare_hands


class NolimitholdemJudger(Judger):
    def judge_game(self, players, hands, dealer_id=0):
        hands = [[c.get_index() for c in h] if h is not None else None for h in hands]
        contributions = np.asarray([p.in_chips for p in players], dtype=int)
        allocated = np.zeros(len(players), dtype=int)
        previous = 0
        for level in sorted(set(contributions) - {0}):
            contributors = contributions >= level
            pot = int((level - previous) * contributors.sum())
            eligible = [i for i, h in enumerate(hands) if h is not None and contributors[i]]
            if not eligible:
                # Uncalled folded overbets are returned to their contributors.
                allocated[contributors] += level - previous
            else:
                winners = compare_hands([h if i in eligible else None for i,h in enumerate(hands)])
                seats = [i for i, won in enumerate(winners) if won]
                share, odd = divmod(pot, len(seats))
                allocated[seats] += share
                if odd:
                    ordered = sorted(seats, key=lambda seat: (seat - dealer_id - 1) % len(players))
                    allocated[ordered[:odd]] += 1
            previous = level
        payoffs = allocated - contributions
        assert payoffs.sum() == 0
        return payoffs.tolist()
