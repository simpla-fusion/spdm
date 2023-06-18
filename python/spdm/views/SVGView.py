import collections.abc
import typing

import numpy as np
import uuid
from ..utils.logger import logger
from .View import View


@View.register("svg")
class SVGView(View):

    TEMPLATE = """<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns:xlink="http://www.w3.org/1999/xlink" width="{p_width}pt" height="{p_height}pt" viewBox="{xmin:.2f} {ymin:.2f} {width:.2f} {height:.2f}"  xmlns="http://www.w3.org/2000/svg" version="1.1">
<style>
    path {{
        fill: none; 
        stroke: #0000ff; 
        stroke-linejoin: miter; 
        stroke-width:0.05
    }}
</style>
    <g id="{name}">
    {contents}
    </g>
</svg>"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def display(self, *objs, **kwargs) -> typing.Any:

        bbox = []
        contents = []
        if len(objs) == 1 and isinstance(objs[0], collections.abc.Sequence):
            objs = objs[0]

        for obj in objs:
            if isinstance(obj, tuple):
                obj, opts = obj
            else:
                opts = {}

            if hasattr(obj, "__svg__"):
                contents.append(obj.__svg__(**opts))
                if hasattr(obj, "bbox"):
                    bbox.append(obj.bbox)
            else:
                logger.error(f"Can not plot {obj}")

        if bbox is None or len(bbox) == 0:
            raise RuntimeError("SVGView.show() requires a bbox attribute")

        bbox = np.bitwise_or.reduce(bbox)

        (xmin, ymin), (xmax, ymax) = bbox
        width = xmax-xmin
        height = ymax-ymin
        p_width = 300
        p_height = int(p_width*height/width+10)
        contents = "\n\t".join(contents)

        return self.TEMPLATE.format(
            p_width=p_width,
            p_height=p_height,
            xmin=xmin,
            ymin=ymin,
            width=width,
            height=height,
            contents=contents,
            signature=self.signature,
            name=kwargs.pop("name", f"spdm_{uuid.uuid()}"),
            title=kwargs.pop("title", ""),
            xlabel=kwargs.pop("xlabel", ""),
            ylabel=kwargs.pop("ylabel", ""),)
