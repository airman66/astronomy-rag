"""Render the Russian text layer of the source time-zone vector map."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)


def inherited_number(
    element: ET.Element, parents: dict[ET.Element, ET.Element], name: str, default: float
) -> float:
    current: ET.Element | None = element
    while current is not None:
        style = dict(
            part.strip().split(":", 1) for part in current.get("style", "").split(";") if ":" in part
        )
        value = current.get(name, style.get(name))
        if value is not None:
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group())
        current = parents.get(current)
    return default


def approximate_width(label: str, size: float, letter_spacing: float) -> float:
    glyphs = sum(0.34 if char.isspace() else 0.6 for char in label)
    return size * glyphs + max(0, len(label) - 1) * letter_spacing


def localize(source: Path, translations: dict[str, str], destination: Path) -> tuple[int, int]:
    tree = ET.parse(source)
    parents = {child: parent for parent in tree.iter() for child in parent}
    translated = 0
    shared = 0
    for switch in tree.findall(f".//{{{SVG}}}switch"):
        original = next(
            (child for child in switch if "en" in child.get("systemLanguage", "").split(",")),
            None,
        )
        if original is None:
            raise ValueError("A language switch lacks its English source")
        label = "".join(original.itertext()).strip()
        if original.tag == f"{{{SVG}}}text" and label:
            if label not in translations:
                raise ValueError(f"Missing Russian label: {label!r}")
            russian = copy.deepcopy(original)
            russian.set("systemLanguage", "ru")
            if "id" in russian.attrib:
                russian.set("id", russian.get("id", "") + "-ru")
            russian.text = translations[label]
            size = inherited_number(original, parents, "font-size", 6.0)
            spacing = inherited_number(original, parents, "letter-spacing", 0.0)
            english_width = approximate_width(label, size, spacing)
            russian_width = approximate_width(russian.text, size, spacing)
            if russian_width > english_width * 1.08 and english_width > 0:
                if spacing > 0:
                    russian.set("letter-spacing", "0")
                    russian_width = approximate_width(russian.text, size, 0)
                if russian_width > english_width * 1.05:
                    scale = max(0.80, math.sqrt(english_width / russian_width))
                    if scale < 0.97:
                        russian.set("style", russian.get("style", "") + f";font-size:{size * scale:.3f}px")
                    scaled_width = approximate_width(russian.text, size * scale, 0)
                    stretch = min(1.0, english_width / scaled_width)
                    x = russian.get("x", "").split()[0]
                    if stretch < 0.98 and x and not russian.get("transform"):
                        russian.set("transform", f"translate({x},0) scale({stretch:.4f},1) translate(-{x},0)")
            switch.insert(0, russian)
            translated += 1
        else:
            languages = original.get("systemLanguage", "").split(",")
            if "ru" not in languages:
                original.set("systemLanguage", ",".join([*languages, "ru"]))
            shared += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return translated, shared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_svg", type=Path)
    parser.add_argument("translations_json", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=2157)
    args = parser.parse_args()
    translations = json.loads(args.translations_json.read_text())
    with tempfile.TemporaryDirectory() as temporary:
        localized_svg = Path(temporary) / "timezones-ru.svg"
        translated, shared = localize(args.source_svg, translations, localized_svg)
        output = args.output_png.with_suffix(".tmp.png")
        subprocess.run(
            [
                "rsvg-convert",
                "--accept-language",
                "ru",
                "--background-color",
                "white",
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--output",
                str(output),
                str(localized_svg),
            ],
            check=True,
        )
        output.replace(args.output_png)
    print(
        json.dumps(
            {"translated_labels": translated, "shared_shapes": shared, "output": str(args.output_png)},
            ensure_ascii=False,
        )
    )
