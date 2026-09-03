#!/usr/bin/env ruby
# frozen_string_literal: true

# Reproducible greedy speculative-decoding benchmark. The model paths are
# explicit because TUFF model installs are intentionally outside the source
# tree and a "resident" versus "streamed" classification is hardware-specific.

require "fileutils"
require "json"
require "open3"
require "optparse"
require "shellwords"
require "time"

ROOT = File.expand_path("..", __dir__)
CLI = File.join(ROOT, ".build", "release", "TUFFCLI")
PROMPT_ROOT = File.join(ROOT, "docs", "benchmark-prompts", "speculative-v1")
PROMPTS = {
  "repetitive" => File.join(PROMPT_ROOT, "repetitive.txt"),
  "normal-prose" => File.join(PROMPT_ROOT, "normal-prose.txt"),
  "code-continuation" => File.join(PROMPT_ROOT, "code-continuation.txt"),
  "long-context" => File.join(PROMPT_ROOT, "long-context.txt")
}.freeze
BLOCKS = [nil, "auto", 2, 4, 6, 8].freeze

FOOTER = /\[stop=(\S+) prefill=(\d+)tok\/([0-9.]+)s new=(\d+)tok decode=([0-9.]+)s tok\/s=([0-9.]+)\]/
SPEC_FOOTER = /\[spec rounds=(\d+) proposed=(\d+) accepted=(\d+) acceptance=([0-9.]+) rejected=(\d+) corrections=(\d+) verifyTokens=(\d+) verifyMs=([0-9.]+) draftMs=([0-9.]+) verifyReads=(\d+) verifyBytes=(\d+) verifyCacheHits=(\d+) verifyCacheMisses=(\d+) verifyCBs=(\d+) blockMin=(\d+) blockMax=(\d+) fallbacks=(\d+) autoDisabled=(true|false)\]/
PHASE = /  (cb1 encode\+commit|expert io await|cb2 encode\+commit|gpu layer cbs|gpu head cbs|gpu routed cbs|gpu shared cbs):\s+([0-9.]+) ms(?: over (\d+) cbs)?/
EXPERT_IO = /  routed expert (reads|bytes|cache hits|cache misses):\s+(\d+)/
MAX_RSS = /\s*(\d+)\s+maximum resident set size/

def median(values)
  sorted = values.sort
  middle = sorted.length / 2
  sorted.length.odd? ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2.0
end

def capture(*command)
  stdout, stderr, status = Open3.capture3(*command, chdir: ROOT)
  [stdout.strip, stderr.strip, status.success?]
end

def system_report
  commit, = capture("git", "rev-parse", "HEAD")
  status, = capture("git", "status", "--short")
  swift, = capture("swift", "--version")
  macos, = capture("sw_vers")
  hardware, = capture("system_profiler", "SPHardwareDataType", "-detailLevel", "mini")
  cli_sha256, = capture("shasum", "-a", "256", CLI)
  {
    "commit" => commit,
    "cli_sha256" => cli_sha256.split.first,
    "worktree_status" => status,
    "swift" => swift,
    "macos" => macos,
    "hardware" => hardware.lines.grep(/^\s*(Model Name|Model Identifier|Chip|Total Number of Cores|Memory):/).map(&:strip),
    "captured_at" => Time.now.iso8601
  }
end

def command_for(model_path, prompt, block_size, max_new, max_context)
  gpt_oss = File.basename(model_path).start_with?("gpt-oss-")
  command = [
    "/usr/bin/time", "-l", CLI,
    "--model", model_path,
    gpt_oss ? "--chat-prompt" : "--prompt", prompt,
    "--max-new", max_new.to_s,
    "--max-context", max_context.to_s,
    "--temperature", "0",
    "--top-k", "1",
    "--top-p", "1.0",
    "--expert-cache-slots", "16",
    "--prefill", "on",
    "--prefill-chunk-tokens", "128",
    "--rdadvise", "off"
  ]
  command.insert(7, "--reasoning", "low") if gpt_oss
  if block_size == "auto"
    command += ["--speculative", "auto", "--speculative-draft-tokens", "8"]
  elsif block_size
    command += ["--speculative", "greedy", "--speculative-draft-tokens", block_size.to_s]
  end
  command
end

def parse_measurement(stderr, block_size)
  footer = stderr.match(FOOTER)
  raise "TUFFCLI did not print a timing footer" unless footer
  rss = stderr.match(MAX_RSS)
  raise "/usr/bin/time did not print peak RSS" unless rss
  spec = stderr.match(SPEC_FOOTER)
  if block_size && !spec
    raise "speculative run did not print a speculative footer"
  end

  phase_rows = {}
  stderr.scan(PHASE) do |label, milliseconds, command_buffers|
    phase_rows[label] = {
      "milliseconds" => milliseconds.to_f,
      "command_buffers" => command_buffers&.to_i
    }
  end
  expert_io = {}
  stderr.scan(EXPERT_IO) do |label, value|
    expert_io[label] = value.to_i
  end
  {
    "stop_reason" => footer[1],
    "prompt_tokens" => footer[2].to_i,
    "prefill_seconds" => footer[3].to_f,
    "generated_tokens" => footer[4].to_i,
    "decode_seconds" => footer[5].to_f,
    "decode_tokens_per_second" => footer[6].to_f,
    "peak_rss_bytes" => rss[1].to_i,
    "routed_expert_reads" => expert_io.fetch("reads", 0),
    "routed_expert_bytes" => expert_io.fetch("bytes", 0),
    "routed_expert_cache_hits" => expert_io.fetch("cache hits", 0),
    "routed_expert_cache_misses" => expert_io.fetch("cache misses", 0),
    "speculative" => if spec
      {
        "rounds" => spec[1].to_i,
        "proposed_tokens" => spec[2].to_i,
        "accepted_tokens" => spec[3].to_i,
        "acceptance_rate" => spec[4].to_f,
        "rejected_tokens" => spec[5].to_i,
        "correction_tokens" => spec[6].to_i,
        "verification_tokens" => spec[7].to_i,
        "verification_ms" => spec[8].to_f,
        "draft_ms" => spec[9].to_f,
        "verification_expert_reads" => spec[10].to_i,
        "verification_expert_bytes" => spec[11].to_i,
        "verification_expert_cache_hits" => spec[12].to_i,
        "verification_expert_cache_misses" => spec[13].to_i,
        "verification_command_buffers" => spec[14].to_i,
        "minimum_block_tokens" => spec[15].to_i,
        "maximum_block_tokens" => spec[16].to_i,
        "fallback_decodes" => spec[17].to_i,
        "adaptive_disabled" => spec[18] == "true"
      }
    else
      nil
    end,
    "phases" => phase_rows
  }
end

def run_once(output_dir, model, prompt_name, label, command, block_size)
  run_dir = File.join(output_dir, model, prompt_name)
  FileUtils.mkdir_p(run_dir)
  prefix = File.join(run_dir, label)
  File.write("#{prefix}.command.txt", Shellwords.join(command) + "\n")
  stdout, stderr, status = Open3.capture3({"TUFF_PHASES" => "1"}, *command, chdir: ROOT)
  File.binwrite("#{prefix}.stdout.txt", stdout)
  File.binwrite("#{prefix}.stderr.txt", stderr)
  raise "#{model} #{prompt_name} #{label} exited #{status.exitstatus}" unless status.success?
  measurement = parse_measurement(stderr, block_size)
  unless %w[endOfTurn eos maxTokens].include?(measurement.fetch("stop_reason"))
    raise "#{model} #{prompt_name} #{label} stopped with #{measurement.fetch("stop_reason")}"
  end
  measurement.merge(
    "label" => label,
    "command" => Shellwords.join(command),
    "stdout_file" => File.basename("#{prefix}.stdout.txt"),
    "stderr_file" => File.basename("#{prefix}.stderr.txt")
  )
end

def existing_run(output_dir, model, prompt_name, label, command, block_size)
  prefix = File.join(output_dir, model, prompt_name, label)
  command_path = "#{prefix}.command.txt"
  stdout_path = "#{prefix}.stdout.txt"
  stderr_path = "#{prefix}.stderr.txt"
  return nil unless [command_path, stdout_path, stderr_path].all? { |path| File.file?(path) }
  return nil unless File.read(command_path).strip == Shellwords.join(command)
  measurement = parse_measurement(File.read(stderr_path), block_size)
  return nil unless %w[endOfTurn eos maxTokens].include?(measurement.fetch("stop_reason"))
  measurement.merge(
    "label" => label,
    "command" => Shellwords.join(command),
    "stdout_file" => File.basename(stdout_path),
    "stderr_file" => File.basename(stderr_path)
  )
rescue RuntimeError
  nil
end

def summarize(measurements)
  spec_rows = measurements.map { |row| row["speculative"] }.compact
  {
    "prompt_tokens" => measurements.first.fetch("prompt_tokens"),
    "generated_tokens" => measurements.map { |row| row.fetch("generated_tokens") },
    "stop_reasons" => measurements.map { |row| row.fetch("stop_reason") },
    "median_prefill_seconds" => median(measurements.map { |row| row.fetch("prefill_seconds") }),
    "median_decode_seconds" => median(measurements.map { |row| row.fetch("decode_seconds") }),
    "median_decode_tokens_per_second" => median(measurements.map { |row| row.fetch("decode_tokens_per_second") }),
    "median_peak_rss_bytes" => median(measurements.map { |row| row.fetch("peak_rss_bytes") }),
    "median_routed_expert_reads" => median(measurements.map { |row| row.fetch("routed_expert_reads") }),
    "median_routed_expert_bytes" => median(measurements.map { |row| row.fetch("routed_expert_bytes") }),
    "median_routed_expert_cache_hits" => median(measurements.map { |row| row.fetch("routed_expert_cache_hits") }),
    "median_routed_expert_cache_misses" => median(measurements.map { |row| row.fetch("routed_expert_cache_misses") }),
    "median_speculative" => if spec_rows.empty?
      nil
    else
      {
        "rounds" => median(spec_rows.map { |row| row.fetch("rounds") }),
        "proposed_tokens" => median(spec_rows.map { |row| row.fetch("proposed_tokens") }),
        "accepted_tokens" => median(spec_rows.map { |row| row.fetch("accepted_tokens") }),
        "acceptance_rate" => median(spec_rows.map { |row| row.fetch("acceptance_rate") }),
        "rejected_tokens" => median(spec_rows.map { |row| row.fetch("rejected_tokens") }),
        "correction_tokens" => median(spec_rows.map { |row| row.fetch("correction_tokens") }),
        "verification_tokens" => median(spec_rows.map { |row| row.fetch("verification_tokens") }),
        "verification_ms" => median(spec_rows.map { |row| row.fetch("verification_ms") }),
        "draft_ms" => median(spec_rows.map { |row| row.fetch("draft_ms") }),
        "verification_expert_reads" => median(spec_rows.map { |row| row.fetch("verification_expert_reads") }),
        "verification_expert_bytes" => median(spec_rows.map { |row| row.fetch("verification_expert_bytes") }),
        "verification_expert_cache_hits" => median(spec_rows.map { |row| row.fetch("verification_expert_cache_hits") }),
        "verification_expert_cache_misses" => median(spec_rows.map { |row| row.fetch("verification_expert_cache_misses") }),
        "verification_command_buffers" => median(spec_rows.map { |row| row.fetch("verification_command_buffers") }),
        "minimum_block_tokens" => median(spec_rows.map { |row| row.fetch("minimum_block_tokens") }),
        "maximum_block_tokens" => median(spec_rows.map { |row| row.fetch("maximum_block_tokens") }),
        "fallback_decodes" => median(spec_rows.map { |row| row.fetch("fallback_decodes") }),
        "adaptive_disabled" => spec_rows.any? { |row| row.fetch("adaptive_disabled") }
      }
    end
  }
end

def write_markdown(path, report)
  lines = [
    "# TUFF speculative decoding benchmark",
    "",
    "Commit: `#{report.dig("system", "commit")}`",
    "",
    "Each condition has one warmup and repeated fresh-process measurements; medians are reported.",
    "The baseline is scalar greedy decode. Speculative conditions use the prompt-lookup drafter.",
    "",
    "| Model | Prompt | Condition | Prompt tokens | Decode tok/s | vs baseline | Decode s | Peak RSS MiB | Acceptance | Verify ms | Total expert reads | Total expert bytes |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
  ]
  report.fetch("models").each do |model, prompts|
    prompts.each do |prompt_name, conditions|
      baseline_tps = conditions.fetch("baseline").fetch("summary").fetch("median_decode_tokens_per_second")
      conditions.each do |label, detail|
        summary = detail.fetch("summary")
        spec = summary["median_speculative"]
        lines << format(
          "| %s | %s | %s | %d | %.3f | %.3f | %.1f | %s | %s | %s | %s |",
          model,
          prompt_name,
          label,
          summary.fetch("prompt_tokens"),
          summary.fetch("median_decode_tokens_per_second"),
          summary.fetch("median_decode_tokens_per_second") / baseline_tps,
          summary.fetch("median_decode_seconds"),
          summary.fetch("median_peak_rss_bytes") / 1_048_576.0,
          spec ? format("%.3f", spec.fetch("acceptance_rate")) : "n/a",
          spec ? format("%.2f", spec.fetch("verification_ms")) : "n/a",
          summary.fetch("median_routed_expert_reads"),
          summary.fetch("median_routed_expert_bytes")
        )
      end
    end
  end
  File.write(path, lines.join("\n") + "\n")
end

options = {
  resident: nil,
  streamed: nil,
  prompt_names: [],
  warmups: 1,
  runs: 3,
  max_new: 128,
  max_context: 4096,
  plan: false,
  resume: false,
  output: File.join(ROOT, "benchmark-results", "speculative-#{Time.now.strftime("%Y%m%d-%H%M%S")}")
}

OptionParser.new do |parser|
  parser.banner = "Usage: Scripts/benchmark_speculative.rb --resident-model PATH --streamed-model PATH [options]"
  parser.on("--resident-model PATH", "Resident control model directory") { |path| options[:resident] = File.expand_path(path) }
  parser.on("--streamed-model PATH", "SSD-streamed MoE model directory") { |path| options[:streamed] = File.expand_path(path) }
  parser.on("--prompt NAME", "Run only this prompt fixture (repeatable)") { |name| options[:prompt_names] << name }
  parser.on("--warmups N", Integer, "Warmups per condition (default 1)") { |n| options[:warmups] = n }
  parser.on("--runs N", Integer, "Measured runs per condition (default 3)") { |n| options[:runs] = n }
  parser.on("--max-new N", Integer, "Generated-token cap (default 128)") { |n| options[:max_new] = n }
  parser.on("--max-context N", Integer, "Context cap (default 4096)") { |n| options[:max_context] = n }
  parser.on("--output PATH", "Result directory") { |path| options[:output] = File.expand_path(path) }
  parser.on("--plan", "Print commands without requiring the models or CLI") { options[:plan] = true }
  parser.on("--resume", "Continue a matching interrupted result directory") { options[:resume] = true }
end.parse!

abort "--warmups must be at least 1" if options[:warmups] < 1
abort "--runs must be at least 3" if options[:runs] < 3
abort "--max-new must be positive" if options[:max_new] < 1
abort "--max-context must be positive" if options[:max_context] < 1

models = { "resident" => options[:resident], "streamed" => options[:streamed] }.compact
unknown_prompts = options[:prompt_names] - PROMPTS.keys
abort "unknown prompt: #{unknown_prompts.join(", ")}" unless unknown_prompts.empty?
selected_prompts = options[:prompt_names].empty? ? PROMPTS : PROMPTS.select {
  |name, _path| options[:prompt_names].include?(name)
}
if options[:plan]
  models = { "resident" => options[:resident] || "<resident-model>", "streamed" => options[:streamed] || "<streamed-model>" }
  models.each do |model, path|
    selected_prompts.each do |prompt_name, prompt_path|
      BLOCKS.each do |block_size|
        prompt = File.read(prompt_path).strip
        puts Shellwords.join(command_for(path, prompt, block_size, options[:max_new], options[:max_context]))
      end
    end
  end
  exit 0
end

abort "--resident-model and --streamed-model are required" unless models.length == 2
abort "release CLI is missing; run swift build -c release --product TUFFCLI" unless File.executable?(CLI)
models.each do |model, path|
  abort "#{model} model is missing: #{path}" unless File.directory?(path)
end
selected_prompts.each_value { |path| abort "benchmark prompt is missing: #{path}" unless File.file?(path) }

protocol = {
  "warmups_per_condition" => options[:warmups],
  "measured_runs_per_condition" => options[:runs],
  "max_new_tokens" => options[:max_new],
  "max_context_tokens" => options[:max_context],
  "blocks" => BLOCKS.map { |value| value || "baseline" },
  "prompts" => selected_prompts.keys
}
results_path = File.join(options[:output], "results.json")
if options[:resume]
  abort "resume results are missing: #{results_path}" unless File.file?(results_path)
  report = JSON.parse(File.read(results_path))
  abort "resume protocol does not match this invocation" unless report.fetch("protocol") == protocol
  current_system = system_report
  abort "resume commit does not match the current checkout" unless report.dig("system", "commit") == current_system.fetch("commit")
  abort "resume CLI binary does not match the original run" if report.dig("system", "cli_sha256") && report.dig("system", "cli_sha256") != current_system.fetch("cli_sha256")
else
  abort "output directory is not empty; choose another path or pass --resume" if File.directory?(options[:output]) && !Dir.empty?(options[:output])
  FileUtils.mkdir_p(options[:output])
  report = { "protocol" => protocol, "system" => system_report, "models" => {} }
end
File.write(File.join(options[:output], "system.json"), JSON.pretty_generate(report.fetch("system")) + "\n")

models.each do |model, model_path|
  report["models"][model] ||= {}
  selected_prompts.each do |prompt_name, prompt_path|
    prompt = File.read(prompt_path).strip
    report["models"][model][prompt_name] ||= {}
    BLOCKS.each do |block_size|
      label = block_size ? "block-#{block_size}" : "baseline"
      command = command_for(model_path, prompt, block_size, options[:max_new], options[:max_context])
      options[:warmups].times do |index|
        warmup_label = "#{label}-warmup-#{index + 1}"
        existing_run(options[:output], model, prompt_name, warmup_label, command, block_size) ||
          run_once(options[:output], model, prompt_name, warmup_label, command, block_size)
      end
      measured = options[:runs].times.map do |index|
        measured_label = "#{label}-measured-#{index + 1}"
        existing_run(options[:output], model, prompt_name, measured_label, command, block_size) ||
          run_once(options[:output], model, prompt_name, measured_label, command, block_size)
      end
      baseline_stdout = report.dig("models", model, prompt_name, "baseline", "stdout")
      stdout_values = measured.map { |row| File.read(File.join(options[:output], model, prompt_name, row.fetch("stdout_file"))) }
      if block_size.nil?
        baseline_stdout = stdout_values.first
      elsif baseline_stdout && stdout_values.any? { |value| value != baseline_stdout }
        raise "#{model} #{prompt_name} #{label} changed greedy output from baseline"
      end
      report["models"][model][prompt_name][label] = {
        "block_size" => block_size,
        "stdout" => baseline_stdout,
        "measurements" => measured,
        "summary" => summarize(measured)
      }
      File.write(results_path, JSON.pretty_generate(report) + "\n")
      write_markdown(File.join(options[:output], "summary.md"), report)
    end
  end
end

puts "Results: #{options[:output]}"
