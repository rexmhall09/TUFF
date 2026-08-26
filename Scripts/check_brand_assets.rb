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

theme = read("Sources/TurboFieldfareApp/MacPresentation/TurboFieldfareMacTheme.swift")
[
  "srgbRed: 111.0 / 255.0",
  "green: 77.0 / 255.0",
  "blue: 255.0 / 255.0",
].each do |component|
  fail_check("TUFF Violet is not #6F4DFF") unless theme.include?(component)
end

logo = read("Brand/TUFFLogo.svg")
fail_check("the flat logo is missing TUFF Violet") unless logo.include?("#6F4DFF")
fail_check("every flat logo facet must use TUFF Violet") unless
  logo.scan(/<path\b[^>]*fill="#6F4DFF"/i).length == 3
fail_check("the flat logo must keep a transparent canvas") if logo.match?(/<rect\b/i)

manifest_path = File.join(ROOT, "Brand/TUFF.icon/icon.json")
manifest = JSON.parse(File.read(manifest_path))
groups = manifest.fetch("groups")
fail_check("Icon Composer source must have three facet groups") unless groups.length == 3
fail_check("Icon Composer source must support Mac square renditions") unless
  manifest.dig("supported-platforms", "squares") == "shared"

assets_directory = File.join(ROOT, "Brand/TUFF.icon/Assets")
groups.each do |group|
  layers = group.fetch("layers")
  fail_check("each facet group must contain one layer") unless layers.length == 1
  asset = layers.first.fetch("image-name")
  fail_check("Icon Composer asset #{asset} is missing") unless
    File.file?(File.join(assets_directory, asset))
  asset_source = File.binread(File.join(assets_directory, asset))
  fail_check("Icon Composer asset #{asset} must use TUFF Violet") unless
    asset_source.scan(/<path\b[^>]*fill="#6F4DFF"/i).length == 1
end

material_settings = groups.map do |group|
  [group.dig("shadow", "opacity"), group.dig("translucency", "value")]
end
fail_check("all Icon Composer facets must use matching material settings") unless
  material_settings.uniq.length == 1

{
  "Brand/TUFFLogo.png" => true,
  "docs/assets/tuff-logo.png" => true,
  "Sources/TurboFieldfareApp/Mac/Resources/tuff-app-icon.png" => true,
}.each do |path, requires_alpha|
  header = png_header(path)
  fail_check("#{path} must be 1024 by 1024") unless
    header[:width] == 1024 && header[:height] == 1024
  fail_check("#{path} must be 8-bit for iconutil") unless header[:depth] == 8
  fail_check("#{path} must retain transparency") if
    requires_alpha && ![4, 6].include?(header[:color_type])
end

puts "brand assets valid"
