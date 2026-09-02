#!/usr/bin/env ruby
# frozen_string_literal: true

# One short question, one fresh process per model, one decode rate each.
# This is the quick launch-lineup check. Scripts/benchmark_v2.rb remains the
# long three-run matrix for workload-by-workload reporting.

require "fileutils"
require "json"
require "open3"
require "optparse"
require "shellwords"
require "time"

ROOT = File.expand_path("..", __dir__)
require_relative "benchmark_models"

CLI = File.join(ROOT, ".build", "release", "TUFFCLI")
PROMPT = File.join(ROOT, "docs", "benchmark-prompts", "capital-of-france.json")
SEED = 20_260_721
FOOTER = /\[stop=(\S+) prefill=(\d+)tok\/([0-9.]+)s new=(\d+)tok decode=([0-9.]+)s tok\/s=([0-9.]+)\]/
MAX_RSS = /\s*(\d+)\s+maximum resident set size/

def safe_capture(*command)
  stdout, _stderr, status = Open3.capture3(*command, chdir: ROOT)
  status.success? ? stdout.strip : ""
end

def system_report
  {
    "commit" => safe_capture("git", "rev-parse", "HEAD"),
    "worktree_status" => safe_capture("git", "status", "--short"),
    "cli_sha256" => safe_capture("shasum", "-a", "256", CLI).split.first,
    "macos" => safe_capture("sw_vers", "-productVersion"),
    "swift" => safe_capture("swift", "--version").lines.first.to_s.strip,
    "physical_memory_bytes" => safe_capture("sysctl", "-n", "hw.memsize").to_i,
    "hardware" => safe_capture(
      "system_profiler", "SPHardwareDataType", "-detailLevel", "mini"
    ).lines.grep(/^\s*(Model Name|Model Identifier|Chip|Memory):/).map(&:strip),
    "captured_at" => Time.now.iso8601
  }
end

# Where the .gturbo installs live. Defaults to the repository's scratch
# directory; --model-root points the harness at another copy of the same
# installs, such as the ones the packaged app downloaded.
def model_path(config, root)
  return File.expand_path(config.fetch(:path), ROOT) if root.nil?

  File.join(root, File.basename(config.fetch(:path)))
end

def command_for(config, max_new, root)
  [
    "/usr/bin/time", "-l", CLI,
    "--model", model_path(config, root),
    "--messages-file", PROMPT,
    "--max-new", max_new.to_s,
    "--max-context", "4096",
    "--seed", SEED.to_s,
    *config.fetch(:chat),
    *config.fetch(:sampling),
    *config.fetch(:runtime)
  ]
end

# One measured run. `attempt` distinguishes the saved artifacts when a model is
# measured more than once.
def run_once(model, command, output_dir, attempt)
  FileUtils.mkdir_p(File.join(output_dir, model))
  suffix = attempt.zero? ? "run" : "run-#{attempt + 1}"
  prefix = File.join(output_dir, model, suffix)
  File.write("#{prefix}.command.txt", Shellwords.join(command) + "\n")

  started = Time.now
  stdout, stderr, status = Open3.capture3(*command, chdir: ROOT)
  File.binwrite("#{prefix}.stdout.txt", stdout)
  File.binwrite("#{prefix}.stderr.txt", stderr)
  raise "#{model} exited #{status.exitstatus}" unless status.success?

  footer = stderr.match(FOOTER)
  raise "#{model} printed no timing footer" unless footer

  rss = stderr.match(MAX_RSS)
  raise "#{model} printed no peak RSS" unless rss

  [footer, rss, stdout.strip, (Time.now - started).round(2)]
end

# Decode rate is reported as the best of `repeat` runs rather than the mean.
# This Mac shares its GPU with the window server, so a run that overlapped
# other drawing measures the interference, not the runtime. The best run is the
# one least disturbed; averaging would fold the disturbance into the figure.
def measure(model, config, output_dir, max_new, root, repeat)
  command = command_for(config, max_new, root)

  print "#{model}: "
  $stdout.flush
  best = nil
  rates = []
  repeat.times do |attempt|
    footer, rss, answer, wall = run_once(model, command, output_dir, attempt)
    rates << footer[6].to_f
    next unless best.nil? || footer[6].to_f > best[0][6].to_f

    best = [footer, rss, answer, wall]
  end
  footer, rss, answer, wall = best
  row = {
    "model" => model,
    "label" => BENCHMARK_MODEL_LABELS.fetch(model, model),
    "stop_reason" => footer[1],
    "prompt_tokens" => footer[2].to_i,
    "prefill_seconds" => footer[3].to_f,
    "generated_tokens" => footer[4].to_i,
    "decode_seconds" => footer[5].to_f,
    "decode_tokens_per_second" => footer[6].to_f,
    "decode_tokens_per_second_all_runs" => rates,
    "runs" => repeat,
    "peak_rss_bytes" => rss[1].to_i,
    "wall_seconds" => wall,
    "answers_paris" => answer.downcase.include?("paris"),
    "answer" => answer,
    "command" => Shellwords.join(command)
  }
  puts format(
    "%.3f tok/s (best of %d: %s), prefill %.2f s, peak RSS %.0f MiB, %s",
    row.fetch("decode_tokens_per_second"),
    repeat,
    rates.map { |rate| format("%.2f", rate) }.join(", "),
    row.fetch("prefill_seconds"),
    row.fetch("peak_rss_bytes") / 1_048_576.0,
    row.fetch("answers_paris") ? "answered Paris" : "DID NOT SAY PARIS"
  )
  row
end

def write_markdown(path, report)
  lines = [
    "# TUFF launch-lineup decode rates",
    "",
    "Commit: `#{report.dig("system", "commit")}`",
    "",
    "One fresh process per model answering \"What is the capital of France?\"",
    "from `docs/benchmark-prompts/capital-of-france.json`, seed #{SEED}, 4,096-token",
    "context. Decode rate excludes install, load, and prefill. These are single",
    "runs on one Mac. Each rate is the best of #{report.fetch("runs_per_model")} "\
    "run(s): this Mac shares its GPU with the window server, so a slower run "\
    "measured that interference rather than the runtime.",
    "",
    "| Model | Decode rate | Prefill | Generated | Peak RSS | Answer |",
    "| --- | ---: | ---: | ---: | ---: | --- |"
  ]
  report.fetch("results").each do |row|
    lines << format(
      "| %s | %.2f tok/s | %.2f s | %d tok | %.0f MiB | %s |",
      row.fetch("label"),
      row.fetch("decode_tokens_per_second"),
      row.fetch("prefill_seconds"),
      row.fetch("generated_tokens"),
      row.fetch("peak_rss_bytes") / 1_048_576.0,
      row.fetch("answers_paris") ? "correct" : "did not name Paris"
    )
  end
  File.write(path, lines.join("\n") + "\n")
end

options = {
  models: BENCHMARK_MODELS.keys,
  model_root: nil,
  repeat: 1,
  max_new: 128,
  output: File.join(ROOT, "benchmark-results", "simple-#{Time.now.strftime("%Y%m%d-%H%M%S")}")
}

OptionParser.new do |parser|
  parser.banner = "Usage: Scripts/benchmark_simple.rb [options]"
  parser.on("--model NAME", BENCHMARK_MODELS.keys, "Measure one launch model") do |model|
    options[:models] = [model]
  end
  parser.on("--models NAMES", Array, "Measure these launch models") do |names|
    unknown = names - BENCHMARK_MODELS.keys
    raise OptionParser::InvalidArgument, unknown.join(",") unless unknown.empty?

    options[:models] = names
  end
  parser.on("--model-root DIR", "Directory holding the .gturbo installs") do |dir|
    options[:model_root] = File.expand_path(dir)
  end
  parser.on("--repeat N", Integer, "Runs per model; the best is reported (default 1)") do |n|
    raise OptionParser::InvalidArgument, n.to_s unless n.positive?

    options[:repeat] = n
  end
  parser.on("--max-new N", Integer, "Generated-token cap (default 128)") { |n| options[:max_new] = n }
  parser.on("--output PATH", "Result directory") { |path| options[:output] = File.expand_path(path) }
end.parse!

abort "release CLI is missing; run swift build -c release" unless File.executable?(CLI)
abort "benchmark prompt is missing: #{PROMPT}" unless File.file?(PROMPT)
options[:models].each do |model|
  path = model_path(BENCHMARK_MODELS.fetch(model), options[:model_root])
  abort "installed model is missing: #{path}" unless File.directory?(path)
end
busy = safe_capture("pgrep", "-fl", "TUFFServer|TUFFDecodeService|TUFFCLI")
abort "a TUFF model process is already running:\n#{busy}" unless busy.empty?

FileUtils.mkdir_p(options[:output])
report = { "prompt" => "What is the capital of France?", "seed" => SEED,
           "max_new_tokens" => options[:max_new], "model_root" => options[:model_root],
           "runs_per_model" => options[:repeat],
           "system" => system_report, "results" => [] }

options[:models].each do |model|
  report["results"] << measure(model, BENCHMARK_MODELS.fetch(model), options[:output],
                               options[:max_new], options[:model_root], options[:repeat])
  File.write(File.join(options[:output], "results.json"), JSON.pretty_generate(report) + "\n")
  write_markdown(File.join(options[:output], "summary.md"), report)
end

puts "\nwrote #{File.join(options[:output], "summary.md")}"
