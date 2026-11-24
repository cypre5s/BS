"""Synthetic geometry fixtures are unit inputs, never training examples."""
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mesa.canonical_grid import CanonicalGrid, map_view
from mesa.config import MESAConfig
from mesa.crop_mapping import UnresolvedMapping, snap_box
from mesa.grid_detector import GridBoundaryDetector
from mesa.views import VisualViewGenerator


def table_image(rows=16, mode="full"):
    title=35 if mode=="title" else 0
    ys=[10+title]
    for i in range(rows):ys.append(ys[-1]+(20+(i%3)*10 if mode=="irregular" else 26))
    xs=[10,90,170,250,330]
    image=Image.new("RGB",(341,ys[-1]+11),"white")
    draw=ImageDraw.Draw(image)
    if mode not in ("borderless",):
        for x in xs:
            if mode=="partial" and x in xs[1:-1]:continue
            draw.line((x,ys[0],x,ys[-1]),fill="black",width=2)
        for i,y in enumerate(ys):
            left=170 if mode=="multilevel" and i==1 else xs[0]
            draw.line((left,y,xs[-1],y),fill="black",width=2)
    if mode=="broken_h":
        for y in ys:
            for x in range(30,310,40):draw.rectangle((x,y,x+2,y+1),fill="white")
    if mode=="broken_v":
        for x in xs:
            for y in range(30,ys[-1]-5,40):draw.rectangle((x,y,x+1,y+2),fill="white")
    if mode=="noise":
        array=np.array(image); rng=np.random.default_rng(320)
        y=rng.integers(0,array.shape[0],500);x=rng.integers(0,array.shape[1],500)
        array[y,x]=0;image=Image.fromarray(array)
    if mode=="shadow":
        a=np.array(image,dtype=float);a[:,:8]*=np.linspace(.25,.8,8)[None,:,None]
        image=Image.fromarray(a.astype('uint8'))
    if mode=="title":draw.text((40,8),"Independent title",fill="black")
    return image,xs,ys


@pytest.mark.parametrize("mode",["full","broken_h","broken_v","noise","shadow","multilevel","irregular","partial","borderless"])
def test_grid_boundary_families(mode):
    image,xs,ys=table_image(mode=mode)
    d=GridBoundaryDetector(MESAConfig()).detect(image)
    if mode=="borderless":
        assert not d.reliable and not d.rows
    elif mode=="partial":
        assert len(d.columns)==2  # Missing interior columns are never invented.
    else:
        assert d.reliable
        assert len(d.rows)==len(ys) and len(d.columns)==len(xs)
        assert np.allclose(d.rows,ys,atol=1) and np.allclose(d.columns,xs,atol=1)
    for boundary in (*d.horizontal,*d.vertical):
        assert all(0<=getattr(boundary,k)<=1 for k in ("line_support","continuity","intersection_support","confidence"))
        assert boundary.source_method


@pytest.mark.parametrize("rows,mode",[(16,"full"),(16,"multilevel"),(2,"full"),(60,"full"),(19,"irregular"),(16,"borderless"),(16,"partial"),(16,"title")])
def test_vertical_diversity_header_and_complete_pixel_coverage(rows,mode):
    image,_,_=table_image(rows,mode)
    views=[v for v in VisualViewGenerator(MESAConfig()).generate(image,"synthetic") if v.channel_id=="vertical"]
    assert 1<=len(views)<=3
    coverage=np.zeros((image.height,image.width),bool)
    regions=[]
    for view in views:
        assert view.pieces[0].source_bbox[1]==0
        assert "header" in view.header_context
        assert view.prepare(image).size==view.plan.actual_size
        regions.append(tuple(p.source_bbox for p in view.pieces))
        for piece in view.pieces:
            x0,y0,x1,y1=piece.source_bbox;coverage[y0:y1,x0:x1]=True
    assert coverage.all() and len(set(regions))==len(regions)
    if len(views)>1:
        assert all(not (len(v.pieces)==1 and v.pieces[0].source_bbox==(0,0,image.width,image.height)) for v in views)
        assert views[-1].pieces[-1].source_bbox[1]>views[0].pieces[-1].source_bbox[1]
    if mode=="borderless":assert all(v.mapping_quality=="LOW" for v in views)
    if mode=="multilevel":assert views[-1].pieces[0].global_rows==(0,1)


@pytest.mark.parametrize("case",range(36))
def test_at_least_thirty_piecewise_coordinate_roundtrips(case):
    image,_,_=table_image(12+case%8,"irregular")
    cfg=MESAConfig(padding=case%7,detail_scale=(1.,1.25,1.5,1.75)[case%4],tile_grid="3x3")
    views=[v for v in VisualViewGenerator(cfg).generate(image,str(case)) if v.local]
    view=views[case%len(views)]
    for i,p in enumerate(view.pieces):
        x0,y0,x1,y1=p.canvas_bbox
        points=[(x0,y0),(x1,y0),(x0,y1),(x1,y1),((x0+x1)/2,(y0+y1)/2),
                (x0-view.padding,y0-view.padding),(x1+view.padding,y1+view.padding)]
        for processed in (False,True):
            size=view.plan.processed_size if processed else view.plan.actual_size
            for x,y in points:
                local=(x*size[0]/view.plan.original_size[0],y*size[1]/view.plan.original_size[1])
                original=view.to_original(local,processed=processed,piece_index=i)
                assert view.from_original(original,processed=processed,piece_index=i)==pytest.approx(local,abs=1e-9)


def otsl(rows,columns=4):
    return ''.join(''.join(f'<fcel>r{r}c{c}' for c in range(columns))+'<nl>' for r in rows)


@pytest.mark.parametrize("case",range(24))
def test_twenty_four_otsl_mapping_cases(case):
    n=12+case%9
    image,_,_=table_image(n)
    cfg=MESAConfig()
    detection=GridBoundaryDetector(cfg).detect(image)
    anchor=CanonicalGrid.from_otsl(otsl(range(n)),"global","g")
    views=[v for v in VisualViewGenerator(cfg).generate(image,'s') if v.channel_id=='vertical']
    view=views[case%len(views)]
    raw=otsl(view.row_context)
    # Different structure families, including blank slots, colspan, rowspan and xcel.
    if case%4==1:raw=raw.replace('<fcel>r0c0','<ecel>')
    if case%4==2:raw=raw.replace('<fcel>r0c1','<lcel>')
    if case%4==3 and len(view.row_context)>2 and view.row_context[1]==1:
        raw=raw.replace('<fcel>r0c1','<lcel>').replace('<fcel>r1c0','<ucel>').replace('<fcel>r1c1','<xcel>')
    local=CanonicalGrid.from_otsl(raw,'vertical',view.view_id)
    assert local.native.status=='RESOLVED'
    mapping=map_view(anchor,local,view,detection.geometry(n,4))
    assert mapping['status']=='MAPPED'
    for cell in mapping['cells']:
        assert cell['global']==(view.row_context[cell['local'][0]],cell['local'][1])
        expected=next(c for c in local.cells if (c.row_start,c.col_start)==tuple(cell['local']))
        assert (cell['rowspan'],cell['colspan'])==(expected.rowspan,expected.colspan)


def test_seam_spans_and_partial_rows_rejected():
    image,_,_=table_image(18)
    cfg=MESAConfig();d=GridBoundaryDetector(cfg).detect(image)
    view=[v for v in VisualViewGenerator(cfg).generate(image,'s') if v.channel_id=='vertical'][-1]
    anchor=CanonicalGrid.from_otsl(otsl(range(18)),'global','g')
    raw=otsl(view.row_context).replace(f'<fcel>r{view.row_context[1]}c0','<ucel>')
    local=CanonicalGrid.from_otsl(raw,'vertical','v')
    assert map_view(anchor,local,view,d.geometry(18,4))['reason']=='local_span_crosses_header_data_seam'
    local=CanonicalGrid.from_otsl(otsl(view.row_context[:-1]),'vertical','v')
    assert map_view(anchor,local,view,d.geometry(18,4))['status']=='UNRESOLVED'
    assert map_view(anchor,local,replace(view,mapping_quality='LOW'),d.geometry(18,4))['status']=='UNRESOLVED'
    gap=(view.pieces[0].canvas_bbox[3]+1)*view.plan.actual_size[1]/view.plan.original_size[1]
    with pytest.raises(UnresolvedMapping):view.to_original((20,gap))


def test_snapping_never_loses_target_and_overlap_maps_same_cells():
    image,_,_=table_image(18)
    cfg=MESAConfig();d=GridBoundaryDetector(cfg).detect(image)
    assert snap_box((83,52,244,287),image.size,d,64)[2]=='LOW'
    target=(103,52,244,287)
    box,ids,quality=snap_box(target,image.size,d,64)
    assert quality=='HIGH' and ids and box[0]<=target[0] and box[1]<=target[1] and box[2]>=target[2] and box[3]>=target[3]
    v=[v for v in VisualViewGenerator(cfg).generate(image,'s') if v.channel_id=='vertical']
    assert set(v[0].row_context)&set(v[1].row_context)-{0}
    for left,right in pairwise(v):
        for r in set(left.row_context)&set(right.row_context):
            assert r in range(18)  # Both views retain the same explicit original row ID.


def test_right_hand_crop_maps_nonzero_columns_and_retains_blanks():
    image,_,_=table_image(12)
    cfg=MESAConfig();d=GridBoundaryDetector(cfg).detect(image)
    base=CanonicalGrid.from_otsl(otsl(range(12)),'global','g')
    view=[v for v in VisualViewGenerator(cfg).generate(image,'s') if v.channel_id=='horizontal'][-1]
    # Explicit known synthetic right half, with trusted source boundary evidence.
    view=replace(view,crop_box=(170,0,341,image.height),mapping_quality='HIGH')
    raw=''.join(f'<ecel><fcel>r{r}c3<nl>' for r in range(12))
    local=CanonicalGrid.from_otsl(raw,'horizontal','right')
    m=map_view(base,local,view,d.geometry(12,4))
    assert m['status']=='MAPPED' and m['column_indices']==(2,3)
    assert m['cells'][0]['global']==(0,2) and m['cells'][-1]['global']==(11,3)


def test_double_header_rules_are_one_separator_not_an_extra_row():
    image,xs,ys=table_image(12)
    draw=ImageDraw.Draw(image);draw.line((xs[0],ys[1]+3,xs[-1],ys[1]+3),fill='black',width=2)
    d=GridBoundaryDetector(MESAConfig()).detect(image)
    assert len(d.rows)==len(ys) and len(d.columns)==len(xs)
    assert d.rows[1]==pytest.approx(ys[1]+1.5)
    assert d.geometry(12,4).confident


@pytest.mark.parametrize('mode',['fixed_overlap','grid_aligned','header_preserved'])
def test_all_vertical_modes(mode):
    image,_,_=table_image(20)
    views=[v for v in VisualViewGenerator(MESAConfig(vertical_segmentation_mode=mode)).generate(image,'s')
           if v.channel_id=='vertical']
    assert len(views)==3
    assert all(v.mapping_quality==('LOW' if mode=='fixed_overlap' else 'HIGH') for v in views)
