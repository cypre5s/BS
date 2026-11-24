#!/usr/bin/env python3
"""Independent synthetic-coordinate diagnostics; no model or training data."""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from mesa.canonical_grid import CanonicalGrid, map_view
from mesa.config import MESAConfig
from mesa.crop_mapping import snap_box
from mesa.grid_detector import GridBoundaryDetector
from mesa.views import VisualViewGenerator
from orchestration.storage import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    config=MESAConfig();checked=correct=before_cuts=after_cuts=snapped=0
    for case in range(36):
        xs=[10,90,170,250,330];ys=[10]
        for row in range(12+case%9):ys.append(ys[-1]+20+(row%3)*8)
        image=Image.new('RGB',(341,ys[-1]+11),'white');draw=ImageDraw.Draw(image)
        for x in xs:draw.line((x,ys[0],x,ys[-1]),fill='black',width=2)
        for y in ys:draw.line((xs[0],y,xs[-1],y),fill='black',width=2)
        detection=GridBoundaryDetector(config).detect(image)
        box=(103,ys[1]+3,244,ys[-2]-3)
        adjusted,_,quality=snap_box(box,image.size,detection,64)
        before_cuts+=sum(value not in lines for value,lines in zip(box,(xs,ys,xs,ys)))
        after_cuts+=sum(value not in lines for value,lines in zip(adjusted,(xs,ys,xs,ys)))
        snapped+=quality=='HIGH'
        def raw(rows):return ''.join(''.join(f'<fcel>R{r}C{c}' for c in range(4))+'<nl>' for r in rows)
        global_grid=CanonicalGrid.from_otsl(raw(range(len(ys)-1)),'global','g')
        for view in VisualViewGenerator(config).generate(image,str(case)):
            if view.channel_id!='vertical':continue
            local=CanonicalGrid.from_otsl(raw(view.row_context),'vertical',view.view_id)
            m=map_view(global_grid,local,view,detection.geometry(len(ys)-1,4))
            assert m['status']=='MAPPED'
            for cell in m['cells']:
                checked+=1
                # Expected row comes from independently encoded synthetic text,
                # not from the mapping's proposed output coordinate.
                c=next(c for c in local.cells if (c.row_start,c.col_start)==tuple(cell['local']))
                r,col=map(int,c.text.replace('R','').split('C'))
                correct+=tuple(cell['global'])==(r,col)
    report={'synthetic_cases':36,'candidate_boxes':36,'snapped_boxes':snapped,
            'cut_boundaries_before':before_cuts,'cut_boundaries_after':after_cuts,
            'mapped_cells_checked':checked,'correct_mappings':correct,
            'canonical_grid_mapping_accuracy_synthetic':correct/checked,
            'not_a_development_recognition_accuracy':True,'training_data_generated':False}
    assert correct==checked and after_cuts<before_cuts
    atomic_json(args.output,report)
    print(json.dumps(report))


if __name__=='__main__':main()
