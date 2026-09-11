#!/usr/bin/env python3
"""Serial Paris and photo smoke tests using a packaged app's inference runner.

Keep output under benchmark-results/: photo paths and model responses are private.
Resume is allowed only with the same runner, harness, photo, and run settings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
VISION_MODELS = {'gemma4-e2b', 'gemma4-e4b', 'gemma4-12b-qat', 'gemma4',
                 'qwen36', 'qwen38-flash-next'}
FOOTER = re.compile(r'\[stop=(\S+) prefill=(\d+)tok/([0-9.]+)s new=(\d+)tok decode=([0-9.]+)s tok/s=([0-9.]+)\]')
PHOTO_PROMPT = ('Describe the visible objects and clothing in this photo in two short sentences. '
                'Include what is on the person\'s head and face. Do not guess their identity.')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', required=True, type=Path)
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.repeat < 1 or args.timeout < 1:
        parser.error('repeat and timeout must be positive')
    cli = args.app.resolve() / 'Contents/Resources/bin/TUFFCLI'
    image = args.image.resolve()
    model_root = args.model_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(subprocess.check_output(
        ['ruby', '-rjson', '-e', 'require File.expand_path("Scripts/benchmark_models", Dir.pwd); '
         'puts JSON.generate({models: BENCHMARK_MODELS, labels: BENCHMARK_MODEL_LABELS})'], cwd=ROOT))
    identity = dict(cli_sha256=digest(cli), harness_sha256=digest(__file__),
                    source_sha256=hashlib.sha256(''.join(
                        str(p.relative_to(ROOT)) + digest(p)
                        for p in sorted((ROOT / 'Sources').rglob('*')) if p.is_file()).encode()).hexdigest(),
                    shaders_sha256=hashlib.sha256(''.join(
                        str(p.relative_to(args.app.resolve())) + digest(p)
                        for p in sorted(args.app.resolve().rglob('*'))
                        if p.suffix in {'.metal', '.metallib'}).encode()).hexdigest(),
                    model_table_sha256=digest(ROOT / 'Scripts/benchmark_models.rb'),
                    image_sha256=digest(image), model_root=str(model_root), repeat=args.repeat,
                    timeout=args.timeout, prompt_sha256=digest(ROOT / 'docs/benchmark-prompts/capital-of-france.json'))
    result_path = output / 'results.json'
    if result_path.exists():
        report = json.loads(result_path.read_text())
        if not args.resume or report['identity'] != identity:
            parser.error('existing results require --resume and identical inputs/binary')
    else:
        report = dict(identity=identity, started=time.time(),
                      commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      worktree_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                      hardware=subprocess.check_output(['sysctl', '-n', 'hw.model', 'hw.memsize'], text=True).strip(),
                      results=[])
    completed = {(r['model'], r['kind'], r['attempt']) for r in report['results']}
    for row in report['results']:
        if 'manifest_sha256' in row:
            current = model_root / Path(metadata['models'][row['model']]['path']).name / 'manifest.json'
            if digest(current) != row['manifest_sha256']:
                parser.error('installed model metadata changed since the previous run')
        if 'vision_manifest_sha256' in row:
            name = Path(metadata['models'][row['model']]['path']).name.removesuffix('.gturbo')
            if digest(model_root / (name + '.vision.gturbo') / 'manifest.json') != row['vision_manifest_sha256']:
                parser.error('installed image pack metadata changed since the previous run')
    for kind in ('paris', 'vision'):
        for model, config in metadata['models'].items():
            if kind == 'vision' and model not in VISION_MODELS:
                continue
            model_dir = model_root / Path(config['path']).name
            manifest = model_dir / 'manifest.json'
            for attempt in range(args.repeat if kind == 'paris' else 1):
                if (model, kind, attempt) in completed:
                    continue
                row = dict(model=model, label=metadata['labels'][model], kind=kind, attempt=attempt,
                           started=time.time(), status='failed')
                prefix = output / f'{kind}-{model}-{attempt + 1}'
                command = ['/usr/bin/time', '-l', str(cli), '--model', str(model_dir),
                           '--max-context', '4096', '--max-new', str(512 if kind == 'vision' else (256 if model == 'minimax-m2.7' else 128)), '--seed', '20260721',
                           *config['chat'], *config['sampling'], *config['runtime']]
                if kind == 'paris':
                    command += ['--messages-file', str(ROOT / 'docs/benchmark-prompts/capital-of-france.json')]
                else:
                    command += ['--chat-prompt', PHOTO_PROMPT, '--image', str(image)]
                row['command'] = command
                row['max_new_tokens'] = int(command[command.index('--max-new') + 1])
                try:
                    row['manifest_sha256'] = digest(manifest)
                    if kind == 'vision':
                        pack = model_root / (model_dir.name.removesuffix('.gturbo') + '.vision.gturbo')
                        row['vision_manifest_sha256'] = digest(pack / 'manifest.json')
                    with Path(str(prefix) + '.stdout.txt').open('w') as stdout, Path(str(prefix) + '.stderr.txt').open('w') as stderr:
                        proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                                start_new_session=True, env=dict(os.environ, TUFF_PHASES='1'))
                        try:
                            proc.wait(timeout=args.timeout)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                                proc.wait()
                            raise
                    row['exit_code'] = proc.returncode
                    answer = Path(str(prefix) + '.stdout.txt').read_text()
                    stderr = Path(str(prefix) + '.stderr.txt').read_text()
                    row['stdout_sha256'] = digest(Path(str(prefix) + '.stdout.txt'))
                    footer = FOOTER.search(stderr)
                    if footer:
                        row.update(stop=footer[1], prompt_tokens=int(footer[2]), prefill_seconds=float(footer[3]),
                                   generated_tokens=int(footer[4]), decode_seconds=float(footer[5]), tps=float(footer[6]))
                    rss = re.search(r'(\d+)\s+maximum resident set size', stderr)
                    if rss:
                        row['peak_rss_bytes'] = int(rss[1])
                    checks = ({'names_paris': bool(re.search(r'\bParis\b', answer, re.I))} if kind == 'paris' else
                              {'glasses': bool(re.search(r'\b(glasses|spectacles|eyeglasses)\b', answer, re.I)),
                               'towel': bool(re.search(r'\btowel\b', answer, re.I))})
                    row['checks'] = checks
                    # Keywords are a smoke check; keep every full response for human review.
                    row['status'] = 'passed' if proc.returncode == 0 and footer and all(checks.values()) else 'needs_review'
                except FileNotFoundError as exc:
                    row.update(status='unavailable', error=str(exc))
                except subprocess.TimeoutExpired:
                    row.update(status='timeout', error=f'exceeded {args.timeout} seconds')
                row['wall_seconds'] = time.time() - row['started']
                report['results'].append(row)
                atomic_json(result_path, report)
                print(f"{kind} {model}: {row['status']}, prefill={row.get('prefill_seconds', '?')}s, TPS={row.get('tps', '?')}", flush=True)
    report['finished'] = time.time()
    atomic_json(result_path, report)
    return 0 if all(r['status'] == 'passed' for r in report['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
