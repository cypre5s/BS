"""Unlabelled development diagnostics; mapping availability is not accuracy."""
from itertools import combinations

import numpy as np

from .canonical_grid import CanonicalGrid, map_view
from .grid_detector import GridBoundaryDetector
from .views import VisualViewGenerator


def diagnose(image, source_id, predictions, config):
    detection = GridBoundaryDetector(config).detect(image)
    views = VisualViewGenerator(config).generate(image,source_id)
    anchor = CanonicalGrid.from_otsl(predictions['global_0'].raw,'global','global_0')
    geometry = detection.geometry(anchor.native.rows,anchor.native.columns)
    mappings=[]
    for view in views:
        if view.local and view.view_id in predictions:
            helper=CanonicalGrid.from_otsl(predictions[view.view_id].raw,view.channel_id,view.view_id)
            mapping=map_view(anchor,helper,view,geometry,config.boundary_tolerance)
            mappings.append({'view_id':view.view_id,'channel':view.channel_id,'quality':view.mapping_quality,**mapping})
    coverage={}
    for channel in ('horizontal','vertical','tiles'):
        selected=[v for v in views if v.channel_id==channel]
        mask=np.zeros((image.height,image.width),dtype=np.uint16)
        for view in selected:
            own=np.zeros_like(mask,dtype=bool)
            for piece in view.pieces:
                x0,y0,x1,y1=piece.source_bbox;own[y0:y1,x0:x1]=True
            mask+=own
        coverage[channel]={'crop_coverage':float((mask>0).mean()),'overlap_coverage':float((mask>1).mean()),
                           'views':len(selected)}
    vertical=[v for v in views if v.channel_id=='vertical']
    similarities=[]
    for a,b in combinations(vertical,2):
        x0,y0,x1,y1=a.requested_box;u0,v0,u1,v1=b.requested_box
        intersection=max(0,min(x1,u1)-max(x0,u0))*max(0,min(y1,v1)-max(y0,v0))
        union=(x1-x0)*(y1-y0)+(u1-u0)*(v1-v0)-intersection
        similarities.append(intersection/union if union else 1.)
    local=[v for v in views if v.local]
    keys=[(v.channel_id,tuple(p.source_bbox for p in v.pieces),v.plan.actual_size) for v in local]
    snapped=[v for v in local if v.channel_id!='vertical']
    success=sum(m['status']=='MAPPED' for m in mappings)
    return {'source_table_id':source_id,'grid_detection':detection.as_dict(),
            'grid_detection_coverage':int(detection.reliable),'anchor_grid_geometry_matches':geometry.confident,
            'grid_snap_rate':sum(v.requested_box!=v.crop_box for v in snapped)/len(snapped) if snapped else None,
            'grid_snap_denominator':len(snapped),'crop_mapping_success_rate':success/len(mappings) if mappings else None,
            'unresolved_rate':1-success/len(mappings) if mappings else None,'mapping_denominator':len(mappings),
            'vertical_view_diversity':1-float(np.mean(similarities)) if similarities else None,
            'vertical_view_count':len(vertical),'duplicate_view_rate':1-len(set(keys))/len(keys) if keys else None,
            'coverage':coverage,'mappings':mappings,
            'canonical_grid_mapping_accuracy':None,
            'mapping_accuracy_reason':'No independent cell-coordinate labels on development images; report synthetic mapping correctness separately, never equate MAPPED with correct.',
            'formal_TEDS':None}
