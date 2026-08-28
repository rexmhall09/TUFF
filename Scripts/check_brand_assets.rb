#!/usr/bin/env ruby

require "json"

ROOT = File.expand_path("..", __dir__)

def fail_check(message)
  warn "brand asset check failed: #{message}"
  exit 1
end

def read(relative_path)
  File.binread(File.join(ROOT, relative_path))
end

def png_header(relative_path)
  data = read(relative_path)
  fail_check("#{relative_path} is not a PNG") unless data.start_with?("\x89PNG\r\n\x1A\n".b)
  fail_check("#{relative_path} has no IHDR") unless data.byteslice(12, 4) == "IHDR"

  width, height, depth, color_type = data.byteslice(16, 10).unpack("NNCC")
  { width: width, height: height, depth: depth, color_type: color_type }
end

theme = read("Sources/TUFFApp/MacPresentation/TUFFMacTheme.swift")
[
  "srgbRed: 111.0 / 255.0",
  "green: 77.0 / 255.0",
  "blue: 255.0 / 255.0",
  "srgbRed: 167.0 / 255.0",
  "green: 139.0 / 255.0",
  "blue: 250.0 / 255.0",
].each do |component|
  fail_check("the two-violet mark palette changed") unless theme.include?(component)
end

logo = read("Brand/TUFFLogo.svg")
fail_check("the flat logo is missing its black field") unless
  logo.match?(/<rect\b[^>]*fill="#000000"/i)
fail_check("the flat logo must have one TUFF Violet body") unless
  logo.scan(/<path\b[^>]*fill="#6F4DFF"/i).length == 1
fail_check("the flat logo must have one light-violet wing") unless
  logo.scan(/<path\b[^>]*fill="#A78BFA"/i).length == 1
fail_check("the flat logo must not use outlined artwork") if
  logo.match?(/<path\b[^>]*stroke=/i)
fail_check("the flat logo must not bake in a gradient or filter") if
  logo.match?(/<(?:linearGradient|radialGradient|filter)\b/i)

manifest_path = File.join(ROOT, "Brand/TUFF.icon/icon.json")
manifest = JSON.parse(File.read(manifest_path))
groups = manifest.fetch("groups")
fail_check("Icon Composer source must separate body and wing") unless groups.length == 2
fail_check("Icon Composer source must support Mac square renditions") unless
  manifest.dig("supported-platforms", "squares") == "shared"
fail_check("Icon Composer canvas must be black") unless
  manifest.dig("fill", "automatic-gradient") ==
    "extended-srgb:0.00000,0.00000,0.00000,1.00000"

assets_directory = File.join(ROOT, "Brand/TUFF.icon/Assets")
approved_violets = %w[6F4DFF A78BFA].freeze
seen_violets = []
groups.each do |group|
  layers = group.fetch("layers")
  fail_check("each mark group must contain one layer") unless layers.length == 1
  asset = layers.first.fetch("image-name")
  asset_path = File.join(assets_directory, asset)
  fail_check("Icon Composer asset #{asset} is missing") unless File.file?(asset_path)
  asset_source = File.binread(asset_path)

  # Icon Composer rims each layer's own silhouette, so shapes that must not have
  # a glass edge between them have to share a layer. That means a layer can carry
  # more than one violet, but every path in it still has to be brand colour.
  paths = asset_source.scan(/<path\b[^>]*>/i)
  fail_check("Icon Composer asset #{asset} has no artwork") if paths.empty?
  paths.each do |path|
    violet = path[/fill="#([0-9A-F]{6})"/i, 1]&.upcase
    fail_check("Icon Composer asset #{asset} must fill every path with an approved violet") unless
      approved_violets.include?(violet)
    seen_violets << violet
  end
  fail_check("Icon Composer artwork must not use outlined paths") if
    asset_source.match?(/<path\b[^>]*stroke=/i)
end

fail_check("the icon must use both approved violet shades") unless
  seen_violets.uniq.sort == approved_violets.sort

{
  "Brand/TUFFLogo.png" => true,
  "docs/assets/tuff-logo.png" => true,
  "Sources/TUFFApp/Mac/Resources/tuff-app-icon.png" => true,
}.each do |path, requires_alpha|
  header = png_header(path)
  fail_check("#{path} must be 1024 by 1024") unless
    header[:width] == 1024 && header[:height] == 1024
  fail_check("#{path} must be 8-bit for iconutil") unless header[:depth] == 8
  fail_check("#{path} must retain transparency") if
    requires_alpha && ![4, 6].include?(header[:color_type])
end

puts "brand assets valid"
