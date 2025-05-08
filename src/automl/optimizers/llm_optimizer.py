import os
import json
import time
import random
import logging
import sqlite3
from typing import Dict, List, Tuple, Optional

import numpy as np
from dotenv import load_dotenv

import google.generativeai as genai 
import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from automl.enums import Task
from automl.optimizers.base_optimizer import BaseOptimizer
from automl.config import get_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)



class _PromptCache:
    def __init__(self, db_path: str = "llm_cache.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                model   TEXT NOT NULL,
                prompt  TEXT NOT NULL,
                response TEXT NOT NULL,
                ts      REAL NOT NULL,
                PRIMARY KEY (model, prompt)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                model   TEXT NOT NULL,
                prompt  TEXT NOT NULL,
                response TEXT NOT NULL,
                ts      REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def lookup(self, model: str, prompt: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT response FROM cache WHERE model=? AND prompt=?",
            (model, prompt),
        ).fetchone()
        return row[0] if row else None

    def store(self, model: str, prompt: str, response: str) -> None:
        ts = time.time()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO cache (model, prompt, response, ts)
            VALUES (?,?,?,?)
            """,
            (model, prompt, response, ts),
        )
        self.conn.execute(
            """
            INSERT INTO requests (model, prompt, response, ts)
            VALUES (?,?,?,?)
            """,
            (model, prompt, response, ts),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class _RateLimiter:
    def __init__(self, max_calls: int, window: float = 60.0):
        self.max_calls = max_calls
        self.window = window
        self._queue: List[float] = []

    def wait(self):
        now = time.time()
        self._queue = [t for t in self._queue if now - t < self.window]
        if len(self._queue) >= self.max_calls:
            sleep_for = self.window - (now - self._queue[0]) + 0.05
            logger.debug("Rate limit reached → sleeping %.2fs", sleep_for)
            time.sleep(max(sleep_for, 0))
        self._queue.append(time.time())




class LLMOptimizer(BaseOptimizer):
    """
    Implementation of *Sequential Large Language Model-Based Bayesian Optimization* (Mahammadli & Ertekin, 2024).
    """

    EXAMPLE = {
        "update": False,
        "new_param_ranges": {},
        "next_params": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 40,
            "max_depth": 10,
            "min_child_samples": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "model": "LightGBM",
        },
    }

    EXAMPLE_UPDATE = {
        "update": True,
        "new_param_ranges": {
            "LightGBM": {
                "learning_rate": [0.001, 0.1],
                "n_estimators": [25, 200],
                "num_leaves": [10, 50],
                "max_depth": [1, 20],
                "min_child_samples": [5, 50],
                "subsample": [0.5, 1.0],
                "colsample_bytree": [0.5, 1.0],
            },
            "Ridge": {
                "alpha": [0.001, 1000],
            },
            "HistGradientBoosting": {
                "learning_rate": [0.001, 0.1],
                "max_iter": [100, 500],
                "max_depth": [5, 20],
                "max_leaf_nodes": [10, 100],
            },
            "Lasso": {
                "alpha": [0.001, 1000],
            },
        },
        "next_params": {
            "model": "Lasso",
            "alpha": 0.01,
            "max_iter": 100,
        },
    }

    SYSTEM_MSG = (
        "You are an expert AutoML assistant. Respond *only* with valid JSON. "
        "Don't wrap it in ```json or any other identifiers; your response must be "
        "only valid JSON, nothing else. This JSON must be compatible with Python's "
        "json library (e.g. use null, not None)."
        "Your message will be directly parsed as JSON, so don't add any extra text or predict ``` "
        "Answer in minified JSON format. "
        "Add an extra key in the response called 'rationale' with a short explanation of the response. "
        "The 'rationale' must be the first key in the response, and it should be a string. "
        "You can take multiple lines and steps to reach the final response, but the final response must be a single JSON object. "
        "For example you can add 'rationale2' as a key if you need to add more explanation. "
    )

    INIT_PROMPT = (
        "Define hyper-parameters to tune for a {task} task using models in this list: {models}.\n"
        "Return two keys: 'param_ranges' (dict of param:range_or_list) and 'initial_params' (dict).\n"
        "If numeric give [min,max]; if categorical give list."
    )

    OPT_PROMPT = (
        "Current best RMSE: {best:.4f}. Decide if ranges need update.\n"
        "Your main goal is to improve the RMSE.\n"
        "Remeber this is negative RMSE, so higher is better.\n"
        "Recent trials (JSON list) → {trials}\n"
        "Respond JSON with keys: 'update', 'new_param_ranges' (dict) "
        "'next_params' (json).\n"
        "Suggest only one set of hyper-parameters, value of next_params should be directly the suggest list of hyperparameters without the model name as it's key, the model name should be a key in the hyperparameters suggested.\n"
        "For example: {example}.\n"
        "You must follow the format of the example.\n"
        "Another example for updating: {example_update}.\n"
        "If you suggest a new range, include it in 'new_param_ranges' (dict of param:range_or_list).\n"
        "next_params must strictly follow the example format, suggest only one model type.\n"
        "Types of models you are allowed to suggest: {models}\n"
        "The model name should be a key in the hyperparameters suggested, it shouldn't be included elsewhere. And it can be accessed by response['next_params']['model']'\n"
    )

    SUMMARISE_PROMPT = (
        "Summarise these trials for hyper-parameter tuning context (keep best/worst, trends):\n{raw}"
    )

    def __init__(
        self,
        task: str,
        time_budget: int,
        problem_description: str,
        llm_model: str = "models/gemini-1.5-pro",
        max_llm_calls: int = 100,
        max_calls_per_min: int = 15,
        patience: int = 25,
        verbose: bool = False,
        cache_db: str = "llm_cache.db",
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Set GEMINI_API_KEY env var (or .env) for Gemini access"
            )
        genai.configure(api_key=api_key)
        self._llm = genai.GenerativeModel(llm_model)
        self._rate = _RateLimiter(max_calls=max_calls_per_min)
        self._llm_calls = 0
        self._max_llm_calls = max_llm_calls

        self._cache = _PromptCache(cache_db)

        super().__init__(
            task, time_budget, metric="rmse", verbose=verbose, config={}
        )  

        self.problem_description = problem_description
        self._history: List[Tuple[Dict[str, object], float]] = []
        self.best_score = -np.inf
        self.patience = patience
        self._since_improve = 0

        self.param_ranges: Dict[str, object] = {}

        self._study: Optional[optuna.study.Study] = None
        self._distributions: Dict[str, optuna.distributions.BaseDistribution] = {}
        self._tpe_sampler = optuna.samplers.TPESampler(seed=42)
        self._pending_trial: Optional[Tuple[optuna.trial.Trial, optuna.trial.Trial]] = None

    def _call_llm(self, prompt: str) -> str:
        full_prompt = f"<system>{self.SYSTEM_MSG}</system>\n<user>{prompt}</user>"
        cached = self._cache.lookup(self._llm.model_name, full_prompt)
        if cached is not None:
            logger.info("-  Cache hit - skipping Gemini call.")
            return cached

        self._rate.wait()
        self._llm_calls += 1
        resp = self._llm.generate_content(full_prompt)
        self._cache.store(self._llm.model_name, full_prompt, resp.text)
        return resp.text

    def fit(self, X, y): 
        start = time.time()
        self._zero_shot_initialize()
        logger.info("Initial params from LLM: %s", self._last_llm_params)
        for model in self._last_llm_params.keys():
            params = self._last_llm_params[model]
            params["model"] = model
            logger.info("Initialising %s with %s", model, params)
            initial_score = self.evaluate_candidate(self.build_model, params, X, y)
            logger.info("Initial score for %s: %.4f", model, initial_score)
            self._register_trial(self._last_llm_params, initial_score)

        logger.info("Init done → RMSE %.4f", initial_score)

        while (time.time() - start) < self.time_budget:
            logger.info("LLMBO iteration %d", len(self._history))
            if self._since_improve >= self.patience:
                logger.info("Early stopping triggered.")
                break
            if len(self._history) % 2 == 0 and self._llm_calls < self._max_llm_calls:
                candidate = self._llm_suggest()
            else:
                candidate = self._tpe_suggest()
                print("TPE candidate:", candidate)
            logger.info("Starting Evaluation for Candidate: %s", candidate)
            score = self.evaluate_candidate(self.build_model, candidate, X, y)
            self._register_trial(candidate, score)
            logger.info(
                "Iter %d | RMSE %.4f | best %.4f",
                len(self._history),
                score,
                self.best_score,
            )
            if self._since_improve >= self.patience:
                logger.info("Early stopping triggered.")
                break
            if (time.time() - start) >= self.time_budget:
                break
        best_params = max(self._history, key=lambda t: t[1])[0]
        self.optimal_model = self.build_model(best_params)
        self.optimal_model.fit(X, y)
        self.metric_value = self.best_score
        return self.optimal_model

    def _zero_shot_initialize(self):
        models = get_config(task=self.task)
        models = [m for m in models["models"].keys()]
        prompt = self.INIT_PROMPT.format(task=self.task, models=models)

        resp_text = self._call_llm(prompt)
        data = json.loads(resp_text)

        self.param_ranges = data["param_ranges"]
        self.models_config = {
            m: {
                k: self._convert_range(v)
                for k, v in self.param_ranges.items()
                if k != "model"
            }
            for m in models
        }
        self.param_ranges = data["param_ranges"]         
        self._build_optuna_dists()                    
        self._last_llm_params = data["initial_params"]

    def _convert_range(self, rng):
        if (
            isinstance(rng, list)
            and len(rng) == 2
            and all(isinstance(i, (int, float)) for i in rng)
        ):
            lo, hi = rng
            return list(np.linspace(lo, hi, 20))
        return rng 

    def _build_optuna_dists(self):
        model_names = sorted(self.param_ranges.keys())
        self._model_dist = CategoricalDistribution(model_names)

        self._sub_dists: Dict[str, Dict[str, optuna.distributions.BaseDistribution]] = {}
        for model in model_names:
            hp_dict = self.param_ranges.get(model, {})
            sub: Dict[str, optuna.distributions.BaseDistribution] = {}
            for hp, rng in hp_dict.items():
                if isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, (int, float)) for x in rng):
                    lo, hi = rng
                    sub[hp] = IntDistribution(lo, hi) if all(isinstance(x, int) for x in rng) \
                                                    else FloatDistribution(lo, hi)
                else:
                    sub[hp] = CategoricalDistribution(rng)
            if not sub:
                logger.warning("Model %s has no hyper-parameter search space.", model)
            else:
                self._sub_dists[model] = sub

        self._study = optuna.create_study(direction="maximize", sampler=self._tpe_sampler)

    def _llm_suggest(self) -> Dict[str, object]:
        prompt = self.OPT_PROMPT.format(
            best=self.best_score,
            trials=self._compact_history(10),
            example=self.EXAMPLE,
            example_update=self.EXAMPLE_UPDATE,
            models=get_config(task=self.task)["models"].keys(),
        )
        try:
            resp_text = self._call_llm(prompt)
            logger.info(resp_text)
            data = json.loads(resp_text)
            suggestion = {
                k: self._coerce(k, v) for k, v in data["next_params"].items()
            }
            logger.info("LLM suggestion: %s", suggestion)
            return suggestion
        except Exception as exc:
            logger.warning("LLM failure → random (%s)", exc)
        return self._random_candidate()

    def _maybe_update_ranges(self, new_ranges: Dict[str, object]):
        if not new_ranges:
            return
        for hp, rng in new_ranges.items():
            self.param_ranges[hp] = rng
        self._build_optuna_dists() 

    def _tpe_suggest(self) -> Dict[str, object]:
        assert hasattr(self, "_study") and hasattr(self, "_model_dist")

        trial_model = self._study.ask({"model": self._model_dist})
        chosen = trial_model.params.get("model")
        if chosen not in self._sub_dists:
            logger.error("Chosen model %r has no sub-distributions!", chosen)
            chosen = random.choice(list(self._sub_dists.keys()))

        trial_hp = self._study.ask(self._sub_dists[chosen])

        params = {"model": chosen, **trial_hp.params}

        self._pending_trial = (trial_model, trial_hp)
        logger.info("TPE suggestion: %s", params)
        return params

    def _register_trial(self, params: Dict[str, object], score: float):
        self._history.append((params, score))
        self.best_score = max(self.best_score, score)
        self._since_improve = 0 if score == self.best_score else self._since_improve + 1

        if isinstance(self._pending_trial, tuple):
            for tr in self._pending_trial:
                self._study.tell(tr, score)
        elif self._pending_trial is not None:
            self._study.tell(self._pending_trial, score)

        self._pending_trial = None


    def _compact_history(self, n: int) -> str:
        recent = self._history[-n:]
        return json.dumps([{"params": p, "rmse": s} for p, s in recent])

    def _random_candidate(self) -> Dict[str, object]:
        cand: Dict[str, object] = {}
        cand["model"] = random.choice(self.param_ranges.get("model", ["RandomForest"]))
        for hp, rng in self.param_ranges.items():
            if hp == "model":
                continue
            if (
                isinstance(rng, list)
                and len(rng) == 2
                and all(isinstance(i, (int, float)) for i in rng)
            ):
                cand[hp] = random.uniform(*rng)
            else:
                cand[hp] = random.choice(rng)
        return cand

    def _coerce(self, hp: str, val):
        rng = self.param_ranges.get(hp)
        if rng is None:
            return val
        if (
            isinstance(rng, list)
            and len(rng) == 2
            and all(isinstance(i, (int, float)) for i in rng)
        ):
            lo, hi = rng
            val = float(val)
            val = max(min(val, hi), lo)
            if isinstance(lo, int) and isinstance(hi, int):
                val = int(round(val))
            return val
        if val not in rng:
            val = random.choice(rng)
        return val

    def __del__(self):
        try:
            self._cache.close()
        except Exception:
            pass


if __name__ == "__main__":
    import pandas as pd
    from automl import createPipeline

    df = pd.read_csv("./Train.csv")
    pipeline = createPipeline(df, "Sales_Quantity")
    df = pipeline.transform(df)

    hpo = LLMOptimizer(
        task=Task.REGRESSION,
        time_budget=600,
        problem_description="Predicting sales quantities across time for different stores",
    )

    X = df.drop(columns=["Sales_Quantity"])
    Y = df["Sales_Quantity"]
    hpo.fit(X, Y)
    accuracy = hpo.get_metric_value()
    print("Best RMSE:", accuracy)
