"""Convert the Roboto OpenType-CFF fonts of TeX Live to TrueType (quadratic glyf outlines) for matplotlib.
matplotlib embeds TrueType fonts cleanly as Type 42 in PDF; CFF fonts give a font-type mismatch warning.
Usage: python otf2ttf.py   (writes Roboto-*.ttf next to this script; Roboto is Apache-2.0 licensed)."""
import os
import subprocess
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

# the Roboto OpenType fonts of TeX Live (texmf-dist/fonts/opentype/google/roboto), or $ROBOTO_OTF_DIR
SRC = os.environ.get("ROBOTO_OTF_DIR") or os.path.join(subprocess.run(["kpsewhich", "-var-value", "TEXMFDIST"], capture_output=True, text=True).stdout.strip(), "fonts", "opentype", "google", "roboto")
OUT = os.path.dirname(os.path.abspath(__file__))
MAX_ERR = 1.0          # font units
REVERSE = True         # PostScript (cubic, counter-clockwise) -> TrueType (clockwise) contour direction


def convert(src, dst):
    font = TTFont(src)
    glyph_order = font.getGlyphOrder()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = glyph_order
    glyf.glyphs = {}
    gs = font.getGlyphSet()
    for name in glyph_order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, MAX_ERR, reverse_direction=REVERSE))
        glyf[name] = pen.glyph()
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    for attr in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
                 "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, attr, 0)
    maxp.maxComponentDepth = 0
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = glyph_order
    font["head"].indexToLocFormat = 0
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    font.sfntVersion = "\000\001\000\000"
    font.save(dst)


if __name__ == "__main__":
    for stem in ("Roboto-Regular", "Roboto-Medium", "Roboto-Bold", "Roboto-Italic"):
        convert(os.path.join(SRC, stem + ".otf"), os.path.join(OUT, stem + ".ttf"))
        print("wrote", stem + ".ttf")
