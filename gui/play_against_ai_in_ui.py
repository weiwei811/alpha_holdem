import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use legacy Keras 2 (tf_keras) for TF1 graph-mode compatibility with Ray RLlib
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tqdm
import pickle
import numpy as np

from flask import Flask, render_template
from flask_socketio import SocketIO,emit
import time
from threading import Thread
import threading
import random
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ray.rllib.models import ModelCatalog
from agi.nl_holdem_env import NlHoldemEnvWrapper
from agi.nl_holdem_net_tf import NlHoldemNet
ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
import numpy as np
from tqdm import tqdm
import pandas as pd
from agi.evaluation_tools import NNAgent,death_match

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conf_path = os.path.join(base_dir, "confs/nl_holdem.py")
conf = eval(open(conf_path).read().strip())
# 6 players is the default setting
num_players = conf.get("env_config", {}).get("custom_options", {}).get("num_players", 6)
conf["env_config"]["custom_options"]["num_players"] = num_players
env = NlHoldemEnvWrapper(
    conf
)
weight_index = 1048
nn_agent = NNAgent(env.observation_space,
                        env.action_space,
                        conf,
                        os.path.join(base_dir, f"weights/c_{weight_index}.pkl"),
                        f"oppo_c{weight_index}")


class MyThread():
    def __init__(self, args=(), kwargs=None):
        self.daemon = True
        self.messages = []
        self._stop_event = threading.Event()
        self.num_players = num_players
        self.human_id = 0  # Seat 0 is Hero (Human), Seats 1..5 are AI opponents

        self.env = NlHoldemEnvWrapper(
            conf
        )

    def gen_obs(self, r, d):
        legal_actions = [
            ["Fold", 0],
            ["Check/Call", 0],
            ["Raise Half Pot", 0],
            ["Raise Pot", 0],
            ["Allin", 0],
            ["Next Game", 0],
        ]

        hands = []
        for p in range(self.num_players):
            try:
                st = self.env.env.get_state(p)
                hands.append(st["raw_obs"]["hand"])
            except Exception:
                hands.append([])

        state0 = self.env.env.get_state(0)
        public = state0["raw_obs"]["public_cards"]
        all_chip = [int(i) for i in state0["raw_obs"]["all_chips"]]
        stakes = [int(i) for i in state0["raw_obs"]["stakes"]]
        pot = int(state0["raw_obs"]["pot"])
        actions = state0["action_record"]
        current_player = int(self.env.my_agent()) if not d else -1

        action_recoards = []
        for pid, one_action in actions:
            a_name = one_action.name
            if a_name == "CHECK_CALL":
                a_name = "check/call"
            else:
                a_name = a_name.replace("_", " ").lower()
            p_label = "Hero" if pid == self.human_id else f"AI {pid}"
            action_recoards.append([int(pid), a_name, p_label])

        if d:
            legal_actions[-1][1] = 1
            payoffs = [int(i) for i in r]
        else:
            hero_state = self.env.env.get_state(self.human_id)
            for one_action in hero_state["raw_obs"]["legal_actions"]:
                legal_actions[one_action.value][1] = 1
            payoffs = [0] * self.num_players

        message = {
            "text": "game action",
            "data": {
                "num_players": self.num_players,
                "human_id": self.human_id,
                "current_player": current_player,
                "hands": hands,
                "hand_p0": hands[0] if len(hands) > 0 else [],
                "hand_p1": hands[1] if len(hands) > 1 else [],
                "public": public,
                "chip": all_chip,
                "stakes": stakes,
                "pot": pot,
                "legal_actions": legal_actions,
                "done": d,
                "payoffs": payoffs,
                "action_recoards": action_recoards,
            }
        }
        return message

    def _reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs

    def _step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, r, d, _, i = res
        else:
            obs, r, d, i = res
        return obs, r, d, i

    def run(self):
        obs = self._reset()
        d = False
        r = [0] * self.num_players
        while self.env.my_agent() != self.human_id and not d:
            action_ind = nn_agent.make_action(obs)
            obs, r, d, i = self._step(action_ind)

        socketio.emit('message_from_server', self.gen_obs(r, d))

    def send_message(self, message):
        action_id = message["action_id"]
        if action_id != 5:
            obs, r, d, i = self._step(message["action_id"])
            while self.env.my_agent() != self.human_id and not d:
                action_ind = nn_agent.make_action(obs)
                obs, r, d, i = self._step(action_ind)
            socketio.emit('message_from_server', self.gen_obs(r, d))
        else:
            d = False
            r = [0] * self.num_players
            self.env = NlHoldemEnvWrapper(
                conf
            )
            obs = self._reset()
            while self.env.my_agent() != self.human_id and not d:
                action_ind = nn_agent.make_action(obs)
                obs, r, d, i = self._step(action_ind)
            socketio.emit('message_from_server', self.gen_obs(r, d))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, async_mode='threading', engineio_logger=False)

t = None

# Display the HTML Page & pass in a username parameter
@app.route('/')
def html():
    return render_template('index.html', username="tester")

# Receive a message from the front end HTML
@socketio.on('send_message')
def message_recieved(data):
    global t
    if data['text'] == "start":
        if t is None:
            t = MyThread()
            t.run()
        else:
            t = MyThread()
            t.run()
    if data['text'] == "restart":
        t = MyThread()
        t.run()
    elif data['text'] == "load":
        t.resend_last_message()
    else:
        if "action_id" in data:
            t.send_message(data)

# Actually Start the App
if __name__ == '__main__':
    """ Run the app. """
    #import webbrowser
    #webbrowser.open("http://localhost:8000")
    socketio.run(app,host="0.0.0.0", port=8000, debug=False)
