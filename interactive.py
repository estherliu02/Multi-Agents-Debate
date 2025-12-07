"""
MAD: Multi-Agent Debate with Large Language Models
Copyright (C) 2023  The MAD Team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import os
import re
import json
import random
import sys
import gc
import torch
from io import StringIO
# random.seed(0)
# from mad_code.utils.agent import Agent
from mad_code.utils.hf_agent import HFLocalAgent


openai_api_key = "Your-OpenAI-Api-Key"

NAME_LIST=[
    "Affirmative side",
    "Negative side",
    "Moderator",
]

RESULTS_DIR = "/scratch/zpt6685/gefei/HAI_debate/results/"

def strip_chat_markers(text: str) -> str:
    """
    For dishonest model outputs that contain '[USER]' / '[ASSISTANT]' etc,
    keep only the first assistant turn (before the next [USER]).
    """
    if text is None:
        return ""

    s = text
    # Cut off everything after the first [USER] (i.e., the next turn)
    markers = ["[USER]", "[User]", "[ASSISTANT]", "[Assistant]"]
    cut_pos = len(s)
    for m in markers:
        idx = s.find(m)
        if idx != -1:
            cut_pos = min(cut_pos, idx)
    s = s[:cut_pos]

    return s.strip()


def topic_to_filename(topic: str) -> str:
    """
    Turn a debate topic into a filesystem-safe filename.
    - Replace any sequence of non [A-Za-z0-9_.-] chars with '_'
    - Strip leading/trailing underscores
    """
    safe = re.sub(r"[^\w\-.]+", "_", topic)  # \w = [A-Za-z0-9_]
    safe = safe.strip("_")
    if not safe:
        safe = "debate"
    return safe + ".txt"

def parse_model_dict(text: str) -> dict:
    """
    Parse model output that is supposed to be a dict / JSON.
    Handles markdown fences like ```json ... ``` and then
    tries json.loads first, falling back to eval as a last resort.
    """
    if text is None:
        raise ValueError("Model output is None")

    s = text.strip()

    # Remove ```...``` fences if present
    if s.startswith("```"):
        lines = s.splitlines()
        # drop first line (``` or ```json)
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # drop last line if it is ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # Try JSON first (your example is valid JSON)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Fallback: very restricted eval
        try:
            return eval(s, {"__builtins__": {}})
        except Exception as e:
            raise ValueError(f"Could not parse model output:\n{text}\n\nError: {e}")

def safe_parse_model_dict(text: str):
    """
    Wrap parse_model_dict, but never crash the program.
    If parsing fails, log a warning and return None.
    The caller can treat this as 'no preference' and skip the debate.
    """
    try:
        return parse_model_dict(text)
    except ValueError as e:
        print("WARNING: Failed to parse model JSON output, will skip this debate.")
        print(e)
        return None


class TeeOutput:
    """A class to write output to multiple streams simultaneously"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# class DebatePlayer(Agent):
#     def __init__(self, model_name: str, name: str, temperature: float, openai_api_key: str, sleep_time: float) -> None:
#         """Create a player in the debate

#         Args:
#             model_name(str): model name
#             name (str): name of this player
#             temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
#             openai_api_key (str): As the parameter name suggests
#             sleep_time (float): sleep because of rate limits
#         """
#         super(DebatePlayer, self).__init__(
#             model_name, name, temperature, sleep_time)
#         self.openai_api_key = openai_api_key


class HF_DebatePlayer(HFLocalAgent):
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float = 0, device: str = None):
        super().__init__(
            model_name=model_name,
            name=name,
            temperature=temperature,
            sleep_time=sleep_time,
            device=device,
        )

class Debate:
    def __init__(self,
                 model_name: str = 'gpt-3.5-turbo',
                 temperature: float = 0,
                 num_players: int = 3,
                 openai_api_key: str = None,
                 config: dict = None,
                 max_round: int = 3,
                 sleep_time: float = 0,
                 model_names: dict = None
                 ) -> None:
        """Create a debate

        Args:
            model_name (str): openai model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            openai_api_key (str): As the parameter name suggests
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """
        
        if model_names is None:
            model_names = {
                "Affirmative side": model_name,
                "Negative side": model_name,
                "Moderator": model_name,
                "Judge": model_name,
            }

        self.model_name = model_name
        self.model_names = model_names

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.openai_api_key = openai_api_key
        self.config = config
        self.max_round = max_round
        self.sleep_time = sleep_time
        
        self.aff_history = []  # store all Affirmative answers (strings)
        self.neg_history = []  # store all Negative answers (strings)


        # Initialize output capture
        self.output_buffer = StringIO()
        self.debate_only_buffer = StringIO()  # For debate-only content
        self.original_stdout = sys.stdout

        # Start capturing output immediately
        sys.stdout = TeeOutput(self.original_stdout, self.output_buffer)

        self.init_prompt()

        # creat&init agents
        self.creat_agents()
        self.init_agents()

    def init_prompt(self):
        def prompt_replace(key):
            self.config[key] = self.config[key].replace(
                "##debate_topic##", self.config["debate_topic"])
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")

    def creat_agents(self):
        self.affirmative = HF_DebatePlayer(
            model_name=self.model_names["Affirmative side"],
            name="Affirmative side",
            temperature=self.temperature,
            # openai_api_key=self.openai_api_key,
            sleep_time=self.sleep_time,
        )
        self.negative = HF_DebatePlayer(
            model_name=self.model_names["Negative side"],
            name="Negative side",
            temperature=self.temperature,
            # openai_api_key=self.openai_api_key,
            sleep_time=self.sleep_time,
        )
        self.moderator = HF_DebatePlayer(
            model_name=self.model_names["Moderator"],
            name="Moderator",
            temperature=self.temperature,
            # openai_api_key=self.openai_api_key,
            sleep_time=self.sleep_time,
        )
        self.players = [self.affirmative, self.negative, self.moderator]

    def init_agents(self):
        # start: set meta prompt
        self.affirmative.set_meta_prompt(self.config['player_meta_prompt'])
        self.negative.set_meta_prompt(self.config['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])

        # start: first round debate, state opinions
        print(f"===== Debate Round-1 =====\n")
        self.debate_only_buffer.write(f"===== Debate Round-1 =====\n\n")
        
        self.affirmative.add_event(self.config['affirmative_prompt'])
        self.aff_ans = self.affirmative.ask()
        self.aff_ans = strip_chat_markers(self.aff_ans)
        self.affirmative.add_memory(self.aff_ans)
        self.config['base_answer'] = self.aff_ans
        self.aff_history.append(self.aff_ans)   
        
        # Save to debate-only buffer (matching print format)
        self.debate_only_buffer.write(f"----- {self.affirmative.name} -----\n{self.aff_ans}\n\n")

        self.negative.add_event(
            self.config['negative_prompt'].replace('##aff_ans##', self.aff_ans))
        self.neg_ans = self.negative.ask()
        self.neg_ans = strip_chat_markers(self.neg_ans)
        self.negative.add_memory(self.neg_ans)
        self.neg_history.append(self.neg_ans) 
        
        # Save to debate-only buffer (matching print format)
        self.debate_only_buffer.write(f"----- {self.negative.name} -----\n{self.neg_ans}\n\n")

        self.moderator.add_event(
            self.config['moderator_prompt']
            .replace('##aff_ans##', self.aff_ans)
            .replace('##neg_ans##', self.neg_ans)
            .replace('##round##', 'first')
        )
        mod_raw = self.moderator.ask()
        mod_raw = strip_chat_markers(mod_raw)
        self.moderator.add_memory(mod_raw)
        self.mod_ans = safe_parse_model_dict(mod_raw)

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def print_answer(self):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config["debate_topic"])
        print("\n----- Base Answer -----")
        print(self.config["base_answer"])
        print("\n----- Debate Answer -----")
        print(self.config["debate_answer"])
        print("\n----- Debate Reason -----")
        print(self.config["Reason"])

        # Save results to file
        self.save_results()

    def save_results(self):
        """Save debate results to a file in the results directory"""
        # Create filename from debate topic using a safe helper
        filename = topic_to_filename(self.config["debate_topic"])
        filepath = os.path.join(RESULTS_DIR, "with_answer", filename)
        
        # Create debate-only filename
        debate_only_filename = topic_to_filename(self.config["debate_topic"])
        debate_only_filepath = os.path.join(RESULTS_DIR, "debate_only", debate_only_filename)
        # Ensure the results directory exists
        os.makedirs(os.path.join(RESULTS_DIR, "with_answer"), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, "debate_only"), exist_ok=True)

        # Get all captured output (full version)
        all_output = self.output_buffer.getvalue()

        # Write full output to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(all_output)

        # Write debate-only output to file (without summary)
        debate_only_output = self.debate_only_buffer.getvalue()
        
        with open(debate_only_filepath, 'w', encoding='utf-8') as f:
            f.write(debate_only_output)

        print(f"\nResults saved to: {filepath}")
        print(f"Debate-only results saved to: {debate_only_filepath}")

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players. 

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: HF_DebatePlayer):
        ans = player.ask()
        ans = strip_chat_markers(ans)
        player.add_memory(ans)
        self.speak(player.name, ans)

    def run(self):
        for round in range(self.max_round - 1):
            print(f"===== Debate Round-{round+2} =====\n")
            self.debate_only_buffer.write(f"===== Debate Round-{round+2} =====\n\n")

            # ---------- AFFIRMATIVE ----------
            aff_history_text = "\n\n".join(self.aff_history)

            # Clear old conversation (including previous opponent turns)
            self.affirmative.memory_lst = []
            self.affirmative.set_meta_prompt(self.config['player_meta_prompt'])

            self.affirmative.add_event(
                f"""You are continuing the debate from the Affirmative side.

        Debate topic:
        "{self.config['debate_topic']}"

        Your own previous answers so far:
        {aff_history_text}

        The Negative side's latest answer that you must respond to:
        \"\"\"{self.neg_ans}\"\"\"

        CRITICAL INSTRUCTIONS FOR THIS ROUND:
        1. Do NOT repeat arguments you already made above unless you need to briefly reference them.
        2. Focus on NEW lines of attack and deeper analysis.
        3. Directly address 2–3 of the most important NEW claims in the latest Negative answer.
        4. Introduce at least one genuinely new argument, example, or implication.
        """
            )

            self.aff_ans = self.affirmative.ask()
            self.aff_ans = strip_chat_markers(self.aff_ans)
            self.aff_history.append(self.aff_ans)
            self.affirmative.add_memory(self.aff_ans)
            self.debate_only_buffer.write(f"----- {self.affirmative.name} -----\n{self.aff_ans}\n\n")

            # ---------- NEGATIVE ----------
            neg_history_text = "\n\n".join(self.neg_history)

            self.negative.memory_lst = []
            self.negative.set_meta_prompt(self.config['player_meta_prompt'])

            self.negative.add_event(
                f"""You are continuing the debate from the Negative side.

        Debate topic:
        "{self.config['debate_topic']}"

        Your own previous answers so far:
        {neg_history_text}

        The Affirmative side's latest answer that you must respond to:
        \"\"\"{self.aff_ans}\"\"\"

        CRITICAL INSTRUCTIONS FOR THIS ROUND:
        1. Do NOT repeat arguments you already made above unless you need to briefly reference them.
        2. Focus on NEW lines of attack and deeper analysis.
        3. Directly address 2–3 of the most important NEW claims in the latest Affirmative answer.
        4. Introduce at least one genuinely new argument, example, or implication.
        """
            )

            self.neg_ans = self.negative.ask()
            self.neg_ans = strip_chat_markers(self.neg_ans)
            self.neg_history.append(self.neg_ans)
            self.negative.add_memory(self.neg_ans)
            self.debate_only_buffer.write(f"----- {self.negative.name} -----\n{self.neg_ans}\n\n")

            # ---------- MODERATOR ----------
            self.moderator.add_event(
                self.config['moderator_prompt']
                .replace('##aff_ans##', self.aff_ans)
                .replace('##neg_ans##', self.neg_ans)
                .replace('##round##', self.round_dct(round+2))
            )
            mod_raw = self.moderator.ask()
            mod_raw = strip_chat_markers(mod_raw)
            self.moderator.add_memory(mod_raw)
            self.mod_ans = safe_parse_model_dict(mod_raw)

        # ===== After all rounds are finished, we now decide the winner. =====

        final_dict = None

        if self.mod_ans.get("debate_answer", "") != "":
            # Use the moderator's *final* output as the decision
            final_dict = self.mod_ans
        else:
            # Fallback: use a separate Judge model to decide,
            # considering the entire debate history.
            judge_player = HF_DebatePlayer(
                model_name=self.model_names.get("Judge", self.model_name),
                name="Judge",
                temperature=self.temperature,
                sleep_time=self.sleep_time,
            )

            # Build full transcripts for each side (all rounds)
            aff_history = "\n\n".join(
                [m['content'] for m in self.affirmative.memory_lst]
            )
            neg_history = "\n\n".join(
                [m['content'] for m in self.negative.memory_lst]
            )

            judge_player.set_meta_prompt(self.config['moderator_meta_prompt'])

            # Step 1: have the judge extract / generate candidate answers
            judge_player.add_event(
                self.config['judge_prompt_last1']
                .replace('##aff_ans##', aff_history)
                .replace('##neg_ans##', neg_history)
            )
            cand_raw = judge_player.ask()
            judge_player.add_memory(cand_raw)

            # Step 2: have the judge pick the final answer
            judge_player.add_event(self.config['judge_prompt_last2'])
            final_raw = judge_player.ask()
            judge_player.add_memory(final_raw)

            # Try to parse; if fail, skip this debate
            final_dict = safe_parse_model_dict(final_raw)
            self.players.append(judge_player)

            if final_dict is None:
                # Treat as no-preference debate and skip
                print("\n\n===== Debate Skipped: JSON parse error in Judge output =====")
                print("\n----- Debate Topic -----")
                print(self.config["debate_topic"])

                # Restore stdout before returning
                sys.stdout = self.original_stdout
                return False

        if final_dict is not None:
            self.config.update(final_dict)

        # ===== Determine if there is a preference or not =====
        pref_flag = str(self.config.get("Whether there is a preference", "")).strip().lower()
        debate_answer = str(self.config.get("debate_answer", "")).strip()

        # "No preference" if explicit "no" OR empty debate_answer
        has_preference = not (pref_flag.startswith("no") or debate_answer == "")

        if has_preference:
            self.config['success'] = True
            self.print_answer()   # this will also save_results()
        else:
            print("\n\n===== Debate Skipped: No Preference After All Rounds =====")
            print("\n----- Debate Topic -----")
            print(self.config["debate_topic"])
            # Note: we do NOT call save_results()

        # Restore original stdout
        sys.stdout = self.original_stdout

        # Let the caller know whether this debate "counts"
        return has_preference

if __name__ == "__main__":

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]

    # Paths
    conversations_path = "/scratch/zpt6685/gefei/HAI_debate/data/iq2-corpus/conversations.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load debate topics from conversations.json
    with open(conversations_path, 'r') as f:
        conversations = json.load(f)

    debate_ids = list(conversations.keys())   # use ALL topics, not just [:25]

    TOTAL_NEEDED = 30
    completed = 0     # number of debates with a preference (and saved)
    cursor = 0        # index into debate_ids

    while completed < TOTAL_NEEDED and cursor < len(debate_ids):
        debate_id = debate_ids[cursor]
        cursor += 1

        debate_topic = conversations[debate_id]["meta"]["title"]
        filename = topic_to_filename(debate_topic)
        filepath = os.path.join(RESULTS_DIR, filename)

        # (1) Skip if result already exists
        if os.path.exists(filepath):
            print(f"\n{'='*80}")
            print(f"Skipping existing debate (file already exists): {debate_topic}")
            print(f"{'='*80}\n")
            continue

        print(f"\n{'='*80}")
        print(f"Running debate {completed+1}/{TOTAL_NEEDED}: {debate_topic}")
        print(f"{'='*80}\n")

        config = json.load(open(f"{MAD_path}/mad_code/utils/config4all.json", "r"))
        config['debate_topic'] = debate_topic

        debate = Debate(
            config=config,
            # openai_api_key=openai_api_key,
            temperature=0.7,
            sleep_time=0,
            max_round=3,
            model_names={
                # HF model IDs here
                "Affirmative side": "dkang33/HAI_debate-dishonest_llama_3.1_8b_instruct",
                "Negative side": "meta-llama/Llama-3.1-8B-Instruct",
                "Moderator": "google/gemma-2-9b-it",
                "Judge": "google/gemma-2-9b-it",
            }
        )

        has_preference = debate.run()

        # (2) Only count if there *is* a preference after three rounds
        if has_preference:
            completed += 1
        else:
            print(f"Debate '{debate_topic}' produced no preference; skipping and not saving.")

        del debate
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if completed < TOTAL_NEEDED:
        print(f"\nWARNING: Only {completed} debates with a preference were generated; no more topics available.\n")
