#!/usr/bin/env python3
"""Validate Gemma reasoning and image follow-ups against uninterrupted synthetic conversations. Never launches an app in /Applications."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import shutil
import subprocess
import sys
import time
import uuid
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Scripts'))
from benchmark_conversation_cache import Service, answer, thinking, model_options, source_identity
from benchmark_reporting import machine_state, require_idle_inference

PROMPTS = {
    'reasoning': ['What is 7 plus 8? Think briefly, then answer with just the number.',
                  'What is twice that number? Think briefly, then answer with just the number.',
                  'Add one to that last answer. Think briefly, then answer with just the number.'],
    'image': ['Name the shape and its color in this image in one short sentence.',
              'What color was the shape? Answer with just its color.',
              'Was it a circle or a square? Answer with just its shape.'],
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def synthetic_png(path):
    # The same smoke-fixture geometry: white canvas and a plainly colored shape.
    width = height = 256
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((255, 0, 0) if 48 <= x < 208 and 48 <= y < 208 else (255, 255, 255))
        rows.append(bytes(row))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(b''.join(rows))) + chunk(b'IEND', b''))


@contextmanager
def staged_synthetic_image(output):
    # Decode-service attachment admission uses the user's real Darwin temp
    # directory, not a child process's isolated TMPDIR override.
    darwin_temp = Path(subprocess.check_output(['getconf', 'DARWIN_USER_TEMP_DIR'], text=True).strip()).resolve()
    attachments = darwin_temp / 'TUFF-Attachments' / ('qualification-' + str(uuid.uuid4()))
    attachments.mkdir(parents=True, exist_ok=False)
    try:
        path = attachments / 'synthetic-red-square.png'
        synthetic_png(path)
        shutil.copyfile(path, output / path.name)
        yield dict(id=str(uuid.uuid4()), path=str(path.resolve()), displayName=path.name,
                   encodedBytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    finally:
        # Delete only this run's UUID directory, never the shared root or
        # another run's attachments. Also runs on exceptions and interruption.
        shutil.rmtree(attachments, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service', type=Path, required=True, help='Packaged TUFFDecodeService outside /Applications')
    p.add_argument('--model-root', type=Path, required=True, help='Installed Gemma 26B model and optional image companion directory')
    p.add_argument('--output', type=Path, required=True, help='New directory for synthetic conversations and validation evidence')
    p.add_argument('--workloads', default='reasoning,image', help='Comma-separated workloads: reasoning,image; image requires the installed companion')
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--max-new', type=int, default=256)
    p.add_argument('--timeout', type=int, default=1200)
    args = p.parse_args()
    service_path = args.service.absolute()
    assert service_path.is_file() and not service_path.resolve().is_relative_to('/Applications')
    workloads = args.workloads.split(','); assert all(w in PROMPTS for w in workloads)
    model, options = model_options(args, 'gemma4')
    companion = args.model_root / 'gemma4.vision.gturbo'
    if 'image' in workloads: assert (companion / 'manifest.json').is_file(), 'Installed Gemma26B image companion required'
    require_idle_inference()
    args.output.mkdir(parents=True, exist_ok=False)
    home = args.output / 'isolated-home'; home.mkdir()
    temp = args.output / 'temporary'; temp.mkdir()
    with staged_synthetic_image(args.output) as image:
        save(args.output / 'identity.json', {'service': str(service_path), 'service_sha256': hashlib.sha256(service_path.read_bytes()).hexdigest(),
            'source': source_identity(), 'machine_before': machine_state(), 'model_directory': str((args.model_root / model).resolve()),
            'model_manifest_sha256': hashlib.sha256((args.model_root / model / 'manifest.json').read_bytes()).hexdigest(),
            'vision_companion': str(companion.resolve()), 'vision_manifest_sha256': hashlib.sha256((companion / 'manifest.json').read_bytes()).hexdigest() if companion.exists() else None,
            'image': image, 'note': 'Gemma has no preserve-thinking UI toggle. Requests explicitly use preserveThinking=false; reasoning=on still disables text bridging; reuse may only resume where the previous prompt ended.'})
        rows, failures = [], []
        def request_for(workload, turn, history, distractor=False):
            value = dict(prompt='Say hello in one short sentence.' if distractor else PROMPTS[workload][turn],
                         systemPrompt='Answer concisely.', history=[] if distractor else history,
                         maxNewTokens=args.max_new, maxContextTokens=args.context,
                         reasoning='on' if workload == 'reasoning' and not distractor else 'off', preserveThinking=False,
                         temperature=0, topK=None, topP=None, repetitionPenalty=1, seed=20261006,
                         runtimeOptions=options, generationID=str(uuid.uuid4()), conversationKey='B' if distractor else 'A')
            if workload == 'image' and turn == 0 and not distractor: value['imageAttachments'] = [image]
            return value
        try:
            for workload in workloads:
                reference = {}
                for schedule in ['uninterrupted', 'interrupted']:
                    require_idle_inference()
                    label = workload + '-' + schedule
                    environment = dict(CFFIXED_USER_HOME=str(home.resolve()), HOME=str(home.resolve()), TMPDIR=str(temp.resolve()) + '/', TFF_LOG_CACHE='1')
                    service = Service(service_path, environment, args.output / (label + '.service.log'))
                    history = []
                    previous_prompt = 0
                    try:
                        load = dict(modelPath=str((args.model_root / model).resolve()), maxContextTokens=args.context, runtimeOptions=options,
                                    forceLogitsHead=False, requestID=str(uuid.uuid4()))
                        save(args.output / (label + '.load.request.json'), load)
                        ready = service.request({'load': {'_0': load}}, args.timeout)
                        save(args.output / (label + '.load.events.json'), ready)
                        assert ready[-1]['kind'] == 'ready', ready
                        for turn in range(3):
                            if schedule == 'interrupted' and turn:
                                distraction = request_for(workload, turn, [], distractor=True)
                                name = label + '-B' + str(turn)
                                save(args.output / (name + '.request.json'), distraction)
                                distraction_started = time.monotonic()
                                events = service.request({'generate': {'_0': distraction}}, args.timeout)
                                rows.append(dict(workload=workload, schedule=schedule, turn='B' + str(turn), wall_seconds=time.monotonic() - distraction_started, terminal=events[-1], output=answer(events), thinking=thinking(events)))
                                save(args.output / 'rows.json', rows)
                                save(args.output / (name + '.events.json'), events)
                                assert events[-1]['kind'] == 'finished', events[-1]
                            request = request_for(workload, turn, list(history))
                            name = label + '-A' + str(turn + 1)
                            save(args.output / (name + '.request.json'), request)
                            started = time.monotonic()
                            events = service.request({'generate': {'_0': request}}, args.timeout)
                            elapsed = time.monotonic() - started
                            save(args.output / (name + '.events.json'), events)
                            terminal = events[-1]
                            row = dict(workload=workload, schedule=schedule, turn=turn + 1, wall_seconds=elapsed,
                                       terminal=terminal, output=answer(events), thinking=thinking(events))
                            rows.append(row); save(args.output / 'rows.json', rows)
                            assert terminal['kind'] == 'finished' and row['output'].strip(), terminal
                            assert not terminal.get('droppedTurns'), terminal
                            assert terminal.get('stopReason') not in ['max-tokens', 'maxTokens', 'max_tokens'], terminal
                            for field in ['prefillSeconds', 'decodeSeconds', 'tokensPerSecond']:
                                assert math.isfinite(terminal[field]) and terminal[field] >= 0, terminal
                            cached = terminal.get('cachedPromptTokens') or 0
                            if workload == 'reasoning':
                                assert row['thinking'].strip(), 'Reasoning-on was not exercised'
                                # A finished thinking turn is dropped from the next
                                # render, so reuse may resume where the previous
                                # turn's prompt ended but never cover its
                                # generated reasoning, which a text bridge would.
                                assert cached <= previous_prompt, \
                                    f'Gemma thinking history reused {cached} tokens, past the previous prompt ({previous_prompt})'
                            elif turn:
                                assert cached > 0, 'Expected image-history continuation reuse'
                                assert terminal.get('conversationCacheSource') == ('active' if schedule == 'uninterrupted' else 'retained'), terminal
                            expected = ['15', '30', '31'][turn] if workload == 'reasoning' else ['red', 'red', 'square'][turn]
                            assert expected in row['output'].lower(), row
                            if schedule == 'uninterrupted': reference[turn] = (row['output'], row['thinking'])
                            else: assert reference[turn] == (row['output'], row['thinking']), 'Interrupted continuation differs from uninterrupted reference'
                            entry = dict(prompt=request['prompt'], response=row['output'], thinking=row['thinking'] or None)
                            if request.get('imageAttachments'): entry['images'] = request['imageAttachments']
                            history.append(entry)
                            previous_prompt = terminal.get('promptTokenCount') or 0
                            print(name, 'cached=', cached, 'source=', terminal.get('conversationCacheSource'), 'wall=', elapsed, 'output=', repr(row['output']), flush=True)
                    except Exception as error:
                        failures.append(dict(workload=workload, schedule=schedule, error=repr(error)))
                        save(args.output / 'failures.json', failures)
                        print('FAIL', label, repr(error), flush=True)
                    finally:
                        service.close()
        finally:
            save(args.output / 'summary.json', dict(requested_workloads=workloads, rows=len(rows), failures=failures,
                  machine_after=machine_state(), state_check='Greedy continuation text and thinking compared to uninterrupted execution; GPU state buffers were not extracted.'))
        return 1 if failures else 0

if __name__ == '__main__': raise SystemExit(main())
