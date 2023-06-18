


def intersection(self, other: GeoObject) -> typing.List[GeoObject]:
        """ Return the intersection of self with other. """
        return [bbox_intersection(self.bbox, other.bbox)]

def reflect(self, line: GeoObject) -> GeoObject: return bbox_reflect(self.bbox, line)
""" reflect  by line"""

def rotate(self, angle, axis=None) -> GeoObject: return bbox_rotate(self.bbox, angle, axis=axis)
""" rotate  by angle and axis"""

def scale(self, *s, point=None) -> GeoObject: return bbox_scale(self.bbox, *s, point=point)
""" scale self by *s, point """

def translate(self, *shift) -> GeoObject: return bbox_translate(self.bbox, *shift)

def trim(self): raise NotImplementedError(f"")