# Copyright 2023 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#######################################################################################

# This example scripts runs openWakeWord in a simple web server receiving audio
# from a web page using websockets.

#######################################################################################

# Imports
import aiohttp
from aiohttp import web
import numpy as np
from openwakeword import Model
import resampy
import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def _get_config_candidates(model_path, explicit_config_path=""):
    if explicit_config_path:
        return [Path(explicit_config_path)]

    path = Path(model_path)
    candidates = [
        path.with_name(f"{path.stem}_config.yml"),
        path.with_name(f"{path.stem}_config.yaml"),
        path.with_suffix(".yml"),
        path.with_suffix(".yaml"),
    ]

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _load_target_phrases(model_path, explicit_config_path=""):
    if not model_path or yaml is None:
        return []

    model_name = Path(model_path).stem
    for config_path in _get_config_candidates(model_path, explicit_config_path):
        if not config_path.exists():
            continue

        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        config_model_name = str(config.get("model_name", "")).strip()
        target_phrase = config.get("target_phrase", [])
        if isinstance(target_phrase, str):
            target_phrase = [target_phrase]

        target_phrase = [i.strip() for i in target_phrase if isinstance(i, str) and i.strip()]
        if not target_phrase:
            continue

        if config_model_name and config_model_name != model_name:
            continue

        return list(dict.fromkeys(target_phrase))

    return []


def _build_loaded_model_metadata(model, custom_model_path="", custom_model_config=""):
    loaded_models = []
    model_labels = {}
    custom_model_name = Path(custom_model_path).stem if custom_model_path else ""
    custom_target_phrases = _load_target_phrases(custom_model_path, custom_model_config)

    for model_name in model.models.keys():
        if model.model_outputs[model_name] == 1:
            display_label = model_name
            if custom_target_phrases and model_name == custom_model_name:
                display_label = " / ".join(custom_target_phrases)
            loaded_models.append(model_name)
            model_labels[model_name] = display_label
            continue

        class_mapping = model.class_mapping.get(model_name, {})
        if not class_mapping:
            loaded_models.append(model_name)
            model_labels[model_name] = model_name
            continue

        for int_label in sorted(class_mapping.keys(), key=int):
            class_name = class_mapping[int_label]
            loaded_models.append(class_name)
            model_labels[class_name] = class_name

    return loaded_models, model_labels

# Define websocket handler
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Send loaded models
    await ws.send_str(json.dumps({
        "loaded_models": loaded_models,
        "model_labels": model_labels
    }))

    # Start listening for websocket messages
    async for msg in ws:
        # Get the sample rate of the microphone from the browser
        if msg.type == aiohttp.WSMsgType.TEXT:
            sample_rate = int(msg.data)
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print(f"WebSocket error: {ws.exception()}")
        else:
            # Get audio data from websocket
            audio_bytes = msg.data

            # Add extra bytes of silence if needed
            if len(msg.data) % 2 == 1:
                audio_bytes += (b'\x00')

            # Convert audio to correct format and sample rate
            data = np.frombuffer(audio_bytes, dtype=np.int16)
            if sample_rate != 16000:
                data = resampy.resample(data, sample_rate, 16000)

            # Get openWakeWord predictions and set to browser client
            predictions = owwModel.predict(data)

            activations = []
            for key in predictions:
                if predictions[key] >= 0.5:
                    activations.append(key)

            if activations != []:
                await ws.send_str(json.dumps({"activations": activations}))

    return ws

# Define static file handler
async def static_file_handler(request):
    return web.FileResponse('./streaming_client.html')

app = web.Application()
app.add_routes([web.get('/ws', websocket_handler), web.get('/', static_file_handler)])

if __name__ == '__main__':
    # Parse CLI arguments
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--chunk_size",
        help="How much audio (in number of samples) to predict on at once",
        type=int,
        default=1280,
        required=False
    )
    parser.add_argument(
        "--model_path",
        help="The path of a specific model to load",
        type=str,
        default="",
        required=False
    )
    parser.add_argument(
        "--inference_framework",
        help="The inference framework to use (either 'onnx' or 'tflite'",
        type=str,
        default='tflite',
        required=False
    )
    parser.add_argument(
        "--model_config",
        help="Optional path to the training config YAML for a custom model; if omitted, adjacent config files are auto-discovered",
        type=str,
        default="",
        required=False
    )
    parser.add_argument(
        "--port",
        help="Port to bind the web server to",
        type=int,
        default=9000,
        required=False
    )
    args=parser.parse_args()

    # Load openWakeWord models
    if args.model_path != "":
        owwModel = Model(wakeword_models=[args.model_path], inference_framework=args.inference_framework)
    else:
        owwModel = Model(inference_framework=args.inference_framework)

    loaded_models, model_labels = _build_loaded_model_metadata(
        owwModel,
        custom_model_path=args.model_path,
        custom_model_config=args.model_config
    )

    # Start webapp
    web.run_app(app, host='localhost', port=args.port)
