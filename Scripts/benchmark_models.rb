#!/usr/bin/env ruby
# frozen_string_literal: true

# Shared launch-model table for the benchmark scripts. Each entry pins the
# chat, sampling, and runtime flags a model is qualified with, so every
# harness measures the same configuration users actually run.

BENCHMARK_MODELS = {
  "gemma4-e2b" => {
    path: "scratch/gemma4-e2b.gturbo",
    chat: %w[--thinking off],
    sampling: %w[--temperature 1.0 --top-k 64 --top-p 0.95],
    runtime: %w[--expert-cache-slots 16 --prefill on --prefill-chunk-tokens auto --rdadvise off]
  },
  "gemma4-e4b" => {
    path: "scratch/gemma4-e4b.gturbo",
    chat: %w[--thinking off],
    sampling: %w[--temperature 1.0 --top-k 64 --top-p 0.95],
    runtime: %w[--expert-cache-slots 16 --prefill on --prefill-chunk-tokens auto --rdadvise off]
  },
  "gemma4-12b-qat" => {
    path: "scratch/gemma4-12b-qat.gturbo",
    chat: %w[--thinking off],
    sampling: %w[--temperature 1.0 --top-k 64 --top-p 0.95],
    runtime: %w[--expert-cache-slots 16 --prefill on --prefill-chunk-tokens auto --rdadvise off]
  },
  "gemma4" => {
    path: "scratch/gemma4.gturbo",
    chat: %w[--thinking off],
    sampling: %w[--temperature 0.2 --top-k 64 --top-p 0.95],
    runtime: %w[--expert-cache-slots 16 --prefill on --prefill-chunk-tokens auto --rdadvise off]
  },
  "qwen36" => {
    path: "scratch/qwen36.gturbo",
    chat: %w[--thinking off],
    sampling: %w[--temperature 0.2 --top-k 64 --top-p 0.95],
    runtime: %w[--expert-cache-slots 16 --prefill on --prefill-chunk-tokens auto --rdadvise off]
  },
  "gpt-oss-20b" => {
    path: "scratch/gpt-oss-20b.gturbo",
    chat: %w[--reasoning low],
    sampling: %w[--temperature 1.0 --top-k 0 --top-p 1.0],
    runtime: %w[--expert-cache-slots 16 --prefill on --prefill-chunk-tokens auto --rdadvise off]
  },
  "gpt-oss-120b" => {
    path: "scratch/gpt-oss-120b.gturbo",
    chat: %w[--reasoning low],
    sampling: %w[--temperature 1.0 --top-k 0 --top-p 1.0],
    runtime: %w[--expert-cache-slots 16 --prefill on --prefill-chunk-tokens auto --rdadvise bounded]
  },
  "minimax-m2.7" => {
    path: "scratch/minimax-m2.7.gturbo",
    chat: [],
    sampling: %w[--temperature 1.0 --top-k 40 --top-p 0.95],
    runtime: %w[--expert-cache-slots 16 --prefill on --prefill-chunk-tokens auto --rdadvise bounded]
  }
}.freeze

BENCHMARK_MODEL_LABELS = {
  "gemma4-e2b" => "Gemma 4 E2B IT",
  "gemma4-e4b" => "Gemma 4 E4B IT",
  "gemma4-12b-qat" => "Gemma 4 12B IT QAT",
  "gemma4" => "Gemma 4 26B-A4B IT",
  "qwen36" => "Qwen3.6 35B-A3B",
  "gpt-oss-20b" => "GPT-OSS 20B",
  "gpt-oss-120b" => "GPT-OSS 120B",
  "minimax-m2.7" => "MiniMax M2.7 4-bit"
}.freeze
